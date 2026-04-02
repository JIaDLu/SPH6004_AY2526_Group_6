"""
loader.py – Load the three raw MIMIC-IV CSV tables and apply
the first-pass filter (keep only discharged-alive patients).
"""

import os
import pandas as pd

from src.utils.constants import DATA_DIR, STATIC_FILE, TEXT_FILE, TS_FILE


def load_raw_data(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load static, text, and time-series tables from disk."""
    static = pd.read_csv(os.path.join(data_dir, STATIC_FILE))
    text   = pd.read_csv(os.path.join(data_dir, TEXT_FILE))
    ts     = pd.read_csv(os.path.join(data_dir, TS_FILE))

    print(f"[loader] Loaded  static={len(static):,}  text={len(text):,}  ts={len(ts):,} rows")
    return static, text, ts


def filter_discharged_patients(
    static: pd.DataFrame,
    text:   pd.DataFrame,
    ts:     pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only ICU stays where the patient was discharged alive
    (icu_death_flag == 0).  The same stay_ids are removed from the
    text and time-series tables.
    """
    before = len(static)
    static = static[static["icu_death_flag"] == 0].copy().reset_index(drop=True)
    after  = len(static)
    print(f"[loader] After icu_death_flag==0 filter: {after:,} stays "
          f"(removed {before - after:,})")

    valid_ids = set(static["stay_id"].unique())
    text = text[text["stay_id"].isin(valid_ids)].copy().reset_index(drop=True)
    ts   = ts  [ts  ["stay_id"].isin(valid_ids)].copy().reset_index(drop=True)

    print(f"[loader] Synced  text={len(text):,}  ts={len(ts):,} rows")
    return static, text, ts