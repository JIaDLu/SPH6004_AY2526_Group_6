"""
ts_preprocessor.py – Feature engineering for the time-series table.

Missing-rate driven strategy (computed on TRAIN data only):
  ─────────────────────────────────────────────────────────
  Low    (< 15%) : clip → LOCF → head-fill with train-median → add obs-mask
  Mid-Hi (15-90%): same pipeline as Low
  Extreme(≥ 90%) : clip (if in range) → ever_measured[t] binary
                   (1 if the feature was observed at any step ≤ t)
  ─────────────────────────────────────────────────────────

Output shape per stay: (T, feature_dim)
  feature_dim = (n_low + n_mid_hi) * 2   [value + mask per feature]
              + n_extreme                 [ever_measured binary]

Anomaly clipping is done BEFORE missing-value handling.
Val/test use the train medians (no re-fitting).
"""

import numpy as np
import pandas as pd

from src.utils.constants import (
    CLINICAL_CLIP_RANGES,
    LOW_MISSING_THRESH,
    HIGH_MISSING_THRESH,
)


class TimeSeriesPreprocessor:
    """
    Call fit() on the training TS DataFrame, then transform_stay() per stay.
    """

    def __init__(self):
        self.feature_names:    list[str]        = []  # all 24 clinical cols
        self.missing_rates:    dict[str, float]  = {}
        self.train_medians:    dict[str, float]  = {}
        self.feature_cats: dict[str, list[str]] = {
            "low":      [],
            "mid_hi":   [],
            "extreme":  [],
        }
        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, ts_df: pd.DataFrame) -> "TimeSeriesPreprocessor":
        """
        Compute missing rates and per-feature medians from training data.
        Must be called before any transform_stay().
        """
        exclude = {"stay_id", "hour_ts"}
        self.feature_names = [c for c in ts_df.columns if c not in exclude]

        for col in self.feature_names:
            rate = ts_df[col].isna().mean()
            self.missing_rates[col] = rate

            # Median of observed values only (after clipping)
            series = ts_df[col].dropna()
            if col in CLINICAL_CLIP_RANGES:
                lo, hi = CLINICAL_CLIP_RANGES[col]
                series = series.clip(lo, hi)
            self.train_medians[col] = float(series.median()) if len(series) > 0 else 0.0

            # Categorise
            if rate >= HIGH_MISSING_THRESH:
                self.feature_cats["extreme"].append(col)
            elif rate <= LOW_MISSING_THRESH:
                self.feature_cats["low"].append(col)
            else:
                self.feature_cats["mid_hi"].append(col)

        self._fitted = True
        self._print_summary()
        return self

    def _print_summary(self):
        print("[TSPreprocessor] Feature categories:")
        for cat, cols in self.feature_cats.items():
            print(f"  {cat:8s} ({len(cols):2d}): {cols}")
        print(f"  → output feature_dim = {self.feature_dim}")

    # ── transform_stay ────────────────────────────────────────────────────────

    def transform_stay(self, stay_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Process all time steps for a single stay.  因为有两种类型的缺失：t=1 NaN → 没有历史 → 用 median     t=5 NaN → 有历史 → 用 ffill

        Returns
        -------
        arr   : float32 array of shape (T, feature_dim)
        cols  : column name list matching axis-1 of arr
        """
        assert self._fitted, "Call fit() first"
        stay_df = stay_df.sort_values("hour_ts").reset_index(drop=True)
        T       = len(stay_df)

        result: dict[str, np.ndarray] = {}

        # ── Continuous features (low + mid_hi) ────────────────────────────
        for col in self.feature_cats["low"] + self.feature_cats["mid_hi"]:
            raw = stay_df[col].values.astype(float)

            # 1. Clip
            if col in CLINICAL_CLIP_RANGES:
                lo, hi = CLINICAL_CLIP_RANGES[col]
                raw = np.clip(raw, lo, hi)  #type:ignore

            # 2. Observation mask (before filling)
            obs_mask = (~np.isnan(raw)).astype(np.float32)

            # 3. LOCF with head-fill from train median
            head_fill = self.train_medians[col]
            filled    = _locf(raw, head_fill) #type:ignore

            result[col]            = filled.astype(np.float32)
            result[f"mask_{col}"]  = obs_mask

        # ── Extreme-missing features → ever_measured binary ───────────────
        for col in self.feature_cats["extreme"]:
            raw = stay_df[col].values.astype(float)

            # Clip where applicable (values do exist occasionally)
            if col in CLINICAL_CLIP_RANGES:
                lo, hi = CLINICAL_CLIP_RANGES[col]
                raw = np.clip(raw, lo, hi) #type:ignore

            ever = np.zeros(T, dtype=np.float32)
            seen = False
            for t in range(T):
                if not np.isnan(raw[t]):
                    seen = True
                ever[t] = 1.0 if seen else 0.0

            result[col] = ever

        # ── Assemble ordered output ────────────────────────────────────────
        cols_ordered: list[str] = []
        for col in self.feature_cats["low"] + self.feature_cats["mid_hi"]:
            cols_ordered.extend([col, f"mask_{col}"])
        for col in self.feature_cats["extreme"]:
            cols_ordered.append(col)

        arr = np.stack([result[c] for c in cols_ordered], axis=1)  # (T, D)
        return arr, cols_ordered

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        n_cont    = len(self.feature_cats["low"]) + len(self.feature_cats["mid_hi"])
        n_extreme = len(self.feature_cats["extreme"])
        return n_cont * 2 + n_extreme  # value + mask for continuous; 1 for extreme


# ── Helper ────────────────────────────────────────────────────────────────────

def _locf(values: np.ndarray, head_fill: float) -> np.ndarray:
    """
    Forward-fill NaN values.
    Positions before the first observation are filled with head_fill.
    """
    out  = values.copy()
    last = head_fill
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out