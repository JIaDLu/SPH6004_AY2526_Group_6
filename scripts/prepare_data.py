"""
scripts/prepare_data.py – Step 0: fit and save all preprocessors.

Loads the three raw MIMIC-IV tables, applies the cohort filter, splits
stay_ids, fits all feature preprocessors on the training split, and
saves the fitted state to disk.

Must be run once before any training script.

Outputs
-------
checkpoints/preprocessors.pkl   ← StaticPreprocessor, TimeSeriesPreprocessor,
                                   TextPreprocessor (fitted state)
data/cache/text_embeddings.pkl  ← ClinicalBERT embeddings for all notes
                                   (auto-created by TextPreprocessor)

Usage
-----
python -m scripts.prepare_data
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader               import load_raw_data, filter_discharged_patients
from src.data.splitter             import split_stay_ids, split_dataframes
from src.data.static_preprocessor import StaticPreprocessor
from src.data.ts_preprocessor     import TimeSeriesPreprocessor
from src.data.text_preprocessor   import TextPreprocessor
from src.utils.persistence        import save_preprocessors
from src.utils.constants          import CACHE_DIR, RANDOM_SEED


def main() -> None:
    print("=" * 55)
    print("  Step 0 — Fit & Save Preprocessors")
    print("=" * 55)

    # ── Load & filter ─────────────────────────────────────────────────────────
    print("\n[1/4] Load & filter raw data")
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)

    # ── Split (same seed used everywhere) ────────────────────────────────────
    print("\n[2/4] Split stay_ids (8 : 1 : 1)")
    train_ids, val_ids, test_ids = split_stay_ids(
        static["stay_id"].unique(), seed=RANDOM_SEED
    )
    splits = split_dataframes(static, text, ts, train_ids, val_ids, test_ids)

    # ── Fit preprocessors on train split only ─────────────────────────────────
    print("\n[3/4] Fit preprocessors on training data")

    static_prep = StaticPreprocessor()
    static_prep.fit(splits["train"]["static"])

    ts_prep = TimeSeriesPreprocessor()
    ts_prep.fit(splits["train"]["ts"])

    # TextPreprocessor: BERT is frozen — fitting = pre-computing embeddings.
    # Uses the full text table so every stay_id (including val/test) is covered.
    text_prep = TextPreprocessor(
        cache_path=str(ROOT / CACHE_DIR / "text_embeddings.pkl")
    )
    text_prep.fit(text)

    # ── Persist ───────────────────────────────────────────────────────────────
    print("\n[4/4] Save preprocessors")
    prep_path = ROOT / "checkpoints" / "preprocessors.pkl"
    prep_path.parent.mkdir(exist_ok=True)
    save_preprocessors(static_prep, ts_prep, text_prep, path=str(prep_path))

    print(f"\n  Done.  Preprocessors saved → {prep_path}")
    print("  Re-run this script only if the raw data changes.\n")


if __name__ == "__main__":
    main()