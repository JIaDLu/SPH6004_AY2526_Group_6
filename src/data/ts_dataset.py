"""
ts_dataset.py – Lightweight Dataset for Stage 1 (time-series only).

No text processing needed → builds ~10x faster than ICUDataset.
Windowing logic is identical to ICUDataset to guarantee label consistency.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.constants           import WINDOW_SIZE
from src.data.ts_preprocessor     import TimeSeriesPreprocessor
from src.data.dataset              import _compute_elapsed_hours   # shared helper


class TSOnlyDataset(Dataset):
    """
    Returns (ts_window, ts_length, label) per windowed time step.

    ts_window : float32  (W, D_ts)   zero-padded on the left
    ts_length : int64    scalar       number of valid (non-padded) steps
    label     : float32  scalar       raw RLOS in hours (≥ 0)
    """

    def __init__(
        self,
        static_df:   pd.DataFrame,
        ts_df:       pd.DataFrame,
        ts_prep:     TimeSeriesPreprocessor,
        window_size: int = WINDOW_SIZE,
    ):
        self.window_size = window_size
        self.samples: list[tuple[np.ndarray, int, np.float32]] = []
        self._build(static_df, ts_df, ts_prep)

    def _build(
        self,
        static_df: pd.DataFrame,
        ts_df:     pd.DataFrame,
        ts_prep:   TimeSeriesPreprocessor,
    ) -> None:
        # Index LOS label by stay_id (O(1) lookup)
        los_by_sid: dict[int, float] = {
            int(r["stay_id"]): float(r["icu_los_hours"])
            for _, r in static_df.iterrows()
        }

        grouped = ts_df.groupby("stay_id")
        for stay_id, group in grouped:
            stay_id = int(stay_id)  # type:ignore
            if stay_id not in los_by_sid:
                continue

            icu_los = los_by_sid[stay_id]
            ts_arr, _ = ts_prep.transform_stay(group)          # (T, D_ts)
            T, D_ts   = ts_arr.shape

            group_sorted  = group.sort_values("hour_ts").reset_index(drop=True)
            elapsed_hours = _compute_elapsed_hours(group_sorted["hour_ts"])

            for t in range(T):
                start     = max(0, t - self.window_size + 1)
                valid_len = t - start + 1

                window = np.zeros((self.window_size, D_ts), dtype=np.float32)
                window[:valid_len] = ts_arr[start : t + 1]

                label = np.float32(max(0.0, icu_los - elapsed_hours[t]))
                self.samples.append((window, valid_len, label))

        print(f"[TSOnlyDataset] Built {len(self.samples):,} samples "
              f"from {len(grouped):,} stays")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, valid_len, label = self.samples[idx]
        return (
            torch.from_numpy(window),                              # (W, D_ts)
            torch.tensor(valid_len, dtype=torch.long),             # scalar
            torch.tensor(label,     dtype=torch.float32),          # scalar
        )


def build_ts_dataloaders(
    train_splits: dict,
    val_splits:   dict,
    test_splits:  dict,
    ts_prep:      TimeSeriesPreprocessor,
    batch_size:   int = 512,
    num_workers:  int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for Stage 1."""

    def _make(splits: dict) -> TSOnlyDataset:
        return TSOnlyDataset(splits["static"], splits["ts"], ts_prep)

    train_ds = _make(train_splits)
    val_ds   = _make(val_splits)
    test_ds  = _make(test_splits)

    print(f"\nTS-only dataset sizes: "
          f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True,
    )
    return (
        make_loader(train_ds, True),
        make_loader(val_ds,   False),
        make_loader(test_ds,  False),
    )