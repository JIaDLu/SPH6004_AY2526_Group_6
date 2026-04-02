"""
persistence.py – Save and load all fitted preprocessors in one call.

After fitting on training data, call save_preprocessors() to persist state.
On the next run (or at inference time), call load_preprocessors() to restore
all three preprocessors without re-fitting — and without touching the data.

Storage format: a single .pkl file containing a dict of the three objects.
The BERT embedding cache is a separate file managed by TextPreprocessor itself.
"""

import os
import pickle

from src.data.static_preprocessor import StaticPreprocessor
from src.data.ts_preprocessor     import TimeSeriesPreprocessor
from src.data.text_preprocessor   import TextPreprocessor

_DEFAULT_PATH = "checkpoints/preprocessors.pkl"


def save_preprocessors(
    static_prep: StaticPreprocessor,
    ts_prep:     TimeSeriesPreprocessor,
    text_prep:   TextPreprocessor,
    path:        str = _DEFAULT_PATH,
) -> None:
    """Persist all three fitted preprocessors to a single file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "static_prep": static_prep,
        "ts_prep":     ts_prep,
        "text_prep":   text_prep,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[persistence] Preprocessors saved → {path}")


def load_preprocessors(
    path: str = _DEFAULT_PATH,
) -> tuple[StaticPreprocessor, TimeSeriesPreprocessor, TextPreprocessor]:
    """Restore all three preprocessors from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[persistence] No preprocessor file found at '{path}'. "
            "Run the full pipeline first to fit and save them."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    static_prep = bundle["static_prep"]
    ts_prep     = bundle["ts_prep"]
    text_prep   = bundle["text_prep"]
    print(f"[persistence] Preprocessors loaded ← {path}")
    return static_prep, ts_prep, text_prep