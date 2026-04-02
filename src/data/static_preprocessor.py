"""
static_preprocessor.py – Feature engineering for the static (demographic)
table.  All categorical decisions are defined here; downstream code only
calls fit() / transform().

Features produced (in order):
  gender        (1)  M=1 F=0
  language      (1)  English=1 else=0
  age           (1)  StandardScaler
  race          (8)  one-hot
  insurance     (6)  one-hot
  marital_status(5)  one-hot
  ─────────────────────────────
  Total         22
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    RACE_MAPPING_RULES,
    RACE_CATEGORIES,
    INSURANCE_CATEGORIES,
    MARITAL_CATEGORIES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_race(value) -> str:
    """Map a raw race string to one of the standard categories."""
    if pd.isna(value):
        return "Unknown"
    upper = str(value).upper()
    for keyword, category in RACE_MAPPING_RULES:
        if keyword in upper:
            return category
    return "Other"


def _one_hot(value: str, categories: list[str]) -> list[float]:
    """Return a one-hot list; all zeros if value not in categories."""
    return [1.0 if value == cat else 0.0 for cat in categories]


# ── Preprocessor ─────────────────────────────────────────────────────────────

class StaticPreprocessor:
    """
    Fit on train data, then call transform() on any split without re-fitting.
    No data leakage: only .fit() touches the scaler, val/test only .transform().
    """

    def __init__(self):
        self.age_scaler = StandardScaler()
        self._fitted    = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "StaticPreprocessor":
        """Fit age scaler on training data."""
        self.age_scaler.fit(df[["age"]].values.astype(float))
        self._fitted = True
        print(f"[StaticPreprocessor] fitted on {len(df):,} rows  "
              f"age μ={self.age_scaler.mean_[0]:.1f}")  #type:ignore
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return float32 array of shape (N, feature_dim).
        Safe to call on val/test without re-fitting.
        """
        assert self._fitted, "Call fit() before transform()"
        rows = [self._encode_row(row) for _, row in df.iterrows()]
        return np.array(rows, dtype=np.float32)

    def _encode_row(self, row: pd.Series) -> list[float]:
        feats: list[float] = []

        # gender
        feats.append(1.0 if row.get("gender") == "M" else 0.0)

        # language
        lang = str(row.get("language", "")).strip().lower()
        feats.append(1.0 if lang == "english" else 0.0)

        # age (scaled)
        age_scaled = self.age_scaler.transform([[float(row["age"])]])[0][0]
        feats.append(float(age_scaled))

        # race one-hot
        race = _map_race(row.get("race"))
        feats.extend(_one_hot(race, RACE_CATEGORIES))

        # insurance one-hot
        ins = row.get("insurance", None)
        ins = "Unknown" if (ins is None or (isinstance(ins, float) and np.isnan(ins))) else str(ins)
        feats.extend(_one_hot(ins, INSURANCE_CATEGORIES))

        # marital_status one-hot
        ms = row.get("marital_status", None)
        ms = "Unknown" if (ms is None or (isinstance(ms, float) and np.isnan(ms))) else str(ms)
        feats.extend(_one_hot(ms, MARITAL_CATEGORIES))

        return feats

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        return (
            1  # gender
            + 1  # language
            + 1  # age
            + len(RACE_CATEGORIES)
            + len(INSURANCE_CATEGORIES)
            + len(MARITAL_CATEGORIES)
        )

    @property
    def feature_names(self) -> list[str]:
        names = ["gender", "language", "age"]
        names += [f"race_{c}" for c in RACE_CATEGORIES]
        names += [f"ins_{c}"  for c in INSURANCE_CATEGORIES]
        names += [f"ms_{c}"   for c in MARITAL_CATEGORIES]
        return names