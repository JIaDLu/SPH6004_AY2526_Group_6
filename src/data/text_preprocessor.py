"""
text_preprocessor.py – Encode radiology notes with Bio_ClinicalBERT.

Key design decisions
────────────────────
1. Notes per stay are split by NOTE_SEPARATOR, then assigned timestamps
   uniformly in [t_min, t_max].
2. At query time t, only notes with assigned_time ≤ t are used
   (no future information).
3. Multiple valid notes → time-aware weighted sum:
       weight_i = exp(−(t − note_time_i))   (recency-weighted)
4. No valid notes → zero vector + no_note_flag = True.
   The model has a learnable NO_NOTE parameter to handle this explicitly.
5. Embeddings are computed ONCE and cached to disk (BERT is fixed / not trained).
   Call fit() on the FULL text table (all splits) so every stay_id is covered.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.utils.constants import (
    BERT_MODEL_NAME,
    BERT_EMBED_DIM,
    NOTE_SEPARATOR,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    CACHE_DIR,
)


class TextPreprocessor:
    """
    fit(text_df) → compute & cache ClinicalBERT embeddings for every note.
    get_embedding(stay_id, current_hour) → (np.ndarray[768], no_note_flag: bool)
    """

    def __init__(self, cache_path: str | None = None):
        self.cache_path    = cache_path or os.path.join(CACHE_DIR, "text_embeddings.pkl")
        self.embedding_dim = BERT_EMBED_DIM

        # {stay_id: [(assigned_hour, emb_vector), ...]}  sorted by time
        self._stay_notes: dict[int, list[tuple[float, np.ndarray]]] = {}
        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, text_df: pd.DataFrame) -> "TextPreprocessor":
        """
        Compute or load ClinicalBERT embeddings for all rows in text_df.
        Pass the FULL text table (all splits) to ensure every stay_id
        has embeddings available at transform time.
        """
        if os.path.exists(self.cache_path):
            print(f"[TextPreprocessor] Loading cached embeddings from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self._stay_notes = pickle.load(f)
            self._fitted = True
            print(f"[TextPreprocessor] Loaded {len(self._stay_notes):,} stays from cache")
            return self

        print(f"[TextPreprocessor] Computing ClinicalBERT embeddings "
              f"for {len(text_df):,} stays …")

        # 1. Parse all notes and collect (stay_id, assigned_hour, text)
        all_notes: list[tuple[int, float, str]] = []
        for _, row in text_df.iterrows():
            stay_id = int(row["stay_id"])
            t_min   = _to_float(row.get("radiology_note_time_min", 0.0))
            t_max   = _to_float(row.get("radiology_note_time_max", 0.0))
            notes   = _split_notes(row.get("radiology_note_text", ""))
            if not notes:
                continue
            times = _uniform_times(t_min, t_max, len(notes))
            for note_text, t in zip(notes, times):
                all_notes.append((stay_id, t, note_text))

        if not all_notes:
            print("[TextPreprocessor] No notes found — all stays will use NO_NOTE.")
            self._fitted = True
            return self

        # 2. Batch-encode with ClinicalBERT
        print(f"[TextPreprocessor] Encoding {len(all_notes):,} notes …")
        texts      = [n[2] for n in all_notes]
        embeddings = self._batch_encode(texts)

        # 3. Organise by stay_id
        for (stay_id, t, _), emb in zip(all_notes, embeddings):
            self._stay_notes.setdefault(stay_id, []).append((t, emb))

        for sid in self._stay_notes:
            self._stay_notes[sid].sort(key=lambda x: x[0])

        # 4. Persist cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._stay_notes, f)
        print(f"[TextPreprocessor] Cached embeddings → {self.cache_path}")

        self._fitted = True
        return self

    # ── get_embedding ─────────────────────────────────────────────────────────

    def get_embedding(
        self,
        stay_id:      int,
        current_hour: float,
    ) -> tuple[np.ndarray, bool]:
        """
        Return (embedding, no_note_flag).
        embedding  : float32 array of shape (BERT_EMBED_DIM,)
        no_note_flag: True  → no notes available at this time step
                      False → at least one note was available
        """
        assert self._fitted, "Call fit() first"

        notes = self._stay_notes.get(stay_id, [])
        valid = [(t, e) for t, e in notes if t <= current_hour]

        if not valid:
            return np.zeros(self.embedding_dim, dtype=np.float32), True

        times  = np.array([t for t, _ in valid], dtype=float)
        embs   = np.stack([e for _, e in valid])           # (K, D)
        delta  = current_hour - times                       # (K,)

        # exp(-delta) can underflow to 0 when delta is very large.
        # Stabilise with log-sum-exp shift: subtract max(delta) before exp
        # so the most-recent note always has weight exp(0) = 1.
        delta_shifted = delta - delta.min()                # shift so min=0
        w = np.exp(-delta_shifted)
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else np.ones(len(valid)) / len(valid)

        result = (w[:, None] * embs).sum(axis=0)           # (D,)

        # Final safety: replace any residual NaN with 0
        if np.isnan(result).any():
            result = np.zeros(self.embedding_dim, dtype=np.float32)

        return result.astype(np.float32), False

    # ── internal: ClinicalBERT encoding ──────────────────────────────────────

    def _batch_encode(self, texts: list[str]) -> np.ndarray:
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
        model.eval()

        all_embs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), BERT_BATCH_SIZE), desc="BERT encoding"):
            batch  = texts[i : i + BERT_BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=BERT_MAX_LENGTH,
                padding=True,
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            # [CLS] token as sentence embedding
            embs = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embs.append(embs)

        # Free GPU memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.vstack(all_embs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_notes(raw_text) -> list[str]:
    """Split concatenated radiology notes by the separator."""
    if raw_text is None or (isinstance(raw_text, float) and np.isnan(raw_text)):
        return []
    parts = str(raw_text).split(NOTE_SEPARATOR)
    return [p.strip() for p in parts if p.strip()]


def _uniform_times(t_min: float, t_max: float, n: int) -> list[float]:
    """Assign n timestamps uniformly in [t_min, t_max]."""
    if n == 1:
        return [t_min]
    return list(np.linspace(t_min, t_max, n))


def _to_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default