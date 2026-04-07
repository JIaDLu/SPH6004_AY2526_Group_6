"""
dataset.py – PyTorch Dataset and DataLoader construction.

Each sample = one time-step t in a patient's ICU stay.

Windowing
─────────
At step t (0-indexed), the input window contains the MOST RECENT min(t+1, W)
time steps, right-padded with zeros so total length = W.

    window = [step_{t−k+1}, …, step_t,  0, …, 0]
              ←── valid_len ──→  ← pad →
    ts_length = valid_len  (passed to pack_padded_sequence)

Label
─────
RLOS(t) = icu_los_hours − elapsed_hours(t)   [clamped to ≥ 0]

where elapsed_hours(t) = hours since the first recorded time step of this stay.

Each __getitem__ returns a 6-tuple:
    (ts_window, ts_length, static_feat, text_emb, no_note_flag, label)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.constants import WINDOW_SIZE
from src.data.static_preprocessor import StaticPreprocessor
from src.data.ts_preprocessor     import TimeSeriesPreprocessor


class _TextPrepProtocol(Protocol):
    """Duck-type interface: any object with get_embedding() works here."""
    embedding_dim: int
    def get_embedding(self, stay_id: int, current_hour: float) -> tuple[np.ndarray, bool]: ...


class ICUDataset(Dataset):

    def __init__(
        self,
        static_df:    pd.DataFrame,
        text_df:      pd.DataFrame,
        ts_df:        pd.DataFrame,
        static_prep:  StaticPreprocessor,
        ts_prep:      TimeSeriesPreprocessor,
        text_prep:    _TextPrepProtocol,
        window_size:  int = WINDOW_SIZE,
    ):
        self.window_size = window_size
        self.samples: list[dict] = []
        self._build(static_df, text_df, ts_df, static_prep, ts_prep, text_prep)

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self, static_df, text_df, ts_df, static_prep, ts_prep, text_prep):
        # Index static and text by stay_id for O(1) lookup
        static_by_id = {
            int(row["stay_id"]): row
            for _, row in static_df.iterrows()
        }
        text_by_id = {
            int(row["stay_id"]): row
            for _, row in text_df.iterrows()
        }

        # Pre-compute static features (one vector per stay)
        static_feats: dict[int, np.ndarray] = {}
        for sid, row in static_by_id.items():
            row_df = pd.DataFrame([row])
            static_feats[sid] = static_prep.transform(row_df)[0]  # (D_static,)

        # Iterate over stays
        grouped = ts_df.groupby("stay_id")
        total_samples = 0

        for stay_id, group in grouped:
            stay_id = int(stay_id)
            if stay_id not in static_by_id:
                continue

            # ── TS processing ────────────────────────────────────────────────
            ts_arr, _ = ts_prep.transform_stay(group)     # (T, D_ts)
            T, D_ts   = ts_arr.shape

            # ── Elapsed hours per step ───────────────────────────────────────
            group_sorted  = group.sort_values("hour_ts").reset_index(drop=True)
            elapsed_hours = _compute_elapsed_hours(group_sorted["hour_ts"])

            # ── ICU LOS label source ─────────────────────────────────────────
            icu_los_hours = float(static_by_id[stay_id]["icu_los_hours"])

            s_feat = static_feats[stay_id]

            # ── Windowing ────────────────────────────────────────────────────
            for t in range(T):
                start     = max(0, t - self.window_size + 1)
                real_data = ts_arr[start : t + 1]            # (valid_len, D_ts)
                valid_len = len(real_data)

                # Right-pad with zeros so shape = (W, D_ts)
                window = np.zeros((self.window_size, D_ts), dtype=np.float32)
                window[:valid_len] = real_data

                # Text embedding at current hour
                curr_hour          = elapsed_hours[t]
                text_emb, no_note  = text_prep.get_embedding(stay_id, curr_hour)  # 这里没有兼容在production phase的处理 但目前只评估数据集的结果 可以忽略！

                # RLOS label
                label = float(max(0.0, icu_los_hours - curr_hour))

                self.samples.append({
                    "ts_window":    window,
                    "ts_length":    valid_len,           # int
                    "static_feat":  s_feat,
                    "text_emb":     text_emb,
                    "no_note_flag": 1.0 if no_note else 0.0,
                    "label":        np.float32(label),
                })

            total_samples += T

        print(f"[ICUDataset] Built {len(self.samples):,} samples "
              f"from {len(grouped):,} stays")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["ts_window"]),                              # (W, D_ts)
            torch.tensor(s["ts_length"],    dtype=torch.long),             # scalar
            torch.from_numpy(s["static_feat"]),                            # (D_static,)
            torch.from_numpy(s["text_emb"]),                               # (D_text,)
            torch.tensor(s["no_note_flag"], dtype=torch.float32),          # scalar
            torch.tensor(s["label"],        dtype=torch.float32),          # scalar
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    train_splits: dict,
    val_splits:   dict,
    test_splits:  dict,
    static_prep:  StaticPreprocessor,
    ts_prep:      TimeSeriesPreprocessor,
    text_prep:    _TextPrepProtocol,
    batch_size:   int = 64,
    num_workers:  int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).
    Prints dataset sizes and a sample shape report.
    """

    def _make_dataset(splits: dict) -> ICUDataset:
        return ICUDataset(
            static_df   = splits["static"],
            text_df     = splits["text"],
            ts_df       = splits["ts"],
            static_prep = static_prep,
            ts_prep     = ts_prep,
            text_prep   = text_prep,
        )

    print("\n[DataLoader] Building datasets …")
    train_ds = _make_dataset(train_splits)
    val_ds   = _make_dataset(val_splits)
    test_ds  = _make_dataset(test_splits)

    print(f"\n{'─'*40}")
    print(f"Dataset sizes:  train={len(train_ds):,}  "
          f"val={len(val_ds):,}  test={len(test_ds):,}")

    # Print shape from first sample
    sample_names = [
        "ts_window   ", "ts_length   ", "static_feat ",
        "text_emb    ", "no_note_flag", "label       ",
    ]
    print("\nSample tensor shapes (first train sample):")
    for name, tensor in zip(sample_names, train_ds[0]):
        print(f"  {name} : {tuple(tensor.shape) if tensor.dim() > 0 else 'scalar'}")
    print(f"{'─'*40}\n")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ── Utility ───────────────────────────────────────────────────────────────────

def _compute_elapsed_hours(hour_ts_series: pd.Series) -> list[float]:
    """
    Convert hour_ts column to elapsed hours from the first step.
    Handles both numeric (integer/float hours) and datetime strings.
    """
    # Try numeric first
    try:
        vals   = pd.to_numeric(hour_ts_series).values.astype(float)
        return (vals - vals[0]).tolist()
    except (ValueError, TypeError):
        pass

    # Fallback: datetime strings
    dt     = pd.to_datetime(hour_ts_series)
    t0     = dt.iloc[0]
    return [(t - t0).total_seconds() / 3600.0 for t in dt]