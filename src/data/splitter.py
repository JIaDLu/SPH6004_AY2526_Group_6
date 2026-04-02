"""
splitter.py – Randomly split stay_ids into train / val / test sets
and partition all three DataFrames accordingly.
"""

import numpy as np
import pandas as pd

from src.utils.constants import RANDOM_SEED


def split_stay_ids(
    stay_ids,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle stay_ids and split into (train, val, test) arrays.
    test_ratio = 1 - train_ratio - val_ratio.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(list(stay_ids))
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids   = ids[n_train : n_train + n_val]
    test_ids  = ids[n_train + n_val :]

    print(f"[splitter] Train={len(train_ids):,}  Val={len(val_ids):,}  "
          f"Test={len(test_ids):,}  stays")
    return train_ids, val_ids, test_ids


def split_dataframes(
    static:    pd.DataFrame,
    text:      pd.DataFrame,
    ts:        pd.DataFrame,
    train_ids: np.ndarray,
    val_ids:   np.ndarray,
    test_ids:  np.ndarray,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Return a dict:
        splits["train"] = {"static": ..., "text": ..., "ts": ...}
        splits["val"]   = {...}
        splits["test"]  = {...}
    """
    def _filter(df: pd.DataFrame, ids) -> pd.DataFrame:
        return df[df["stay_id"].isin(ids)].copy().reset_index(drop=True)

    splits = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        splits[name] = {
            "static": _filter(static, ids),
            "text":   _filter(text,   ids),
            "ts":     _filter(ts,     ids),
        }
    return splits