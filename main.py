"""
main.py – Quick verification of the best experiment result.

Best configuration (determined from Stage 1 + Stage 2 experiments):
  TS Encoder : MHA (Multi-Head Attention)
  Variant    : ts_static_text  (TS + Static + Radiology Text)

What this script does
─────────────────────
1. Fit preprocessors        — skipped if checkpoints/preprocessors.pkl exists
2. Train Stage 1 (MHA)      — skipped if checkpoints/ts_mha_best.pt exists
3. Train Stage 2 (multimodal) — skipped if checkpoints/mm_ts_static_text_mha_best.pt exists
4. Evaluate best model on the held-out test set and print results

Each step is idempotent: re-running main.py is safe and fast once checkpoints exist.

Usage
─────
python main.py
"""

import os
import sys
import time
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.loader                import load_raw_data, filter_discharged_patients
from src.data.splitter              import split_stay_ids, split_dataframes
from src.data.static_preprocessor  import StaticPreprocessor
from src.data.ts_preprocessor      import TimeSeriesPreprocessor
from src.data.text_preprocessor    import TextPreprocessor
from src.data.ts_dataset           import build_ts_dataloaders
from src.data.dataset              import build_dataloaders
from src.models.encoders.ts_encoder import build_ts_encoder
from src.models.ts_only            import TSOnlyModel
from src.models.multimodal         import MultimodalICUModel
from src.training.loss             import LogMSELoss, decode_log_pred
from src.utils.metrics             import compute_all, format_metrics
from src.utils.persistence        import save_preprocessors, load_preprocessors
from src.utils.constants           import CACHE_DIR, RANDOM_SEED


# ── Best-run configuration ────────────────────────────────────────────────────
# These are the verified best hyperparameters. Edit here if you re-tune.

ARCH    = "mha"
VARIANT = "ts_static_text"

ENCODER_CFG = {
    "arch":       "mha",
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads":  4,
    "dropout":    0.2,
}

TRAIN_CFG = {
    # Stage 1
    "s1_epochs":      100,
    "s1_batch_size":  512,
    "s1_lr":          3e-4,
    # Stage 2
    "s2_epochs":      100,
    "s2_batch_size":  256,
    "s2_lr":          3e-4,
    # Shared
    "weight_decay":   1e-4,
    "patience":       10,
    "grad_clip":      1.0,
    "num_workers":    4,
    "seed":           RANDOM_SEED,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR   = ROOT / "checkpoints"
PREP_PATH  = CKPT_DIR / "preprocessors.pkl"
S1_CKPT    = CKPT_DIR / f"ts_{ARCH}_best.pt"
S2_CKPT    = CKPT_DIR / f"mm_{VARIANT}_{ARCH}_best.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Reproducibility ───────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_and_split(static_prep, ts_prep):
    """Load raw tables, filter, split. Returns (splits, static, text, ts)."""
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)
    train_ids, val_ids, test_ids = split_stay_ids(
        static["stay_id"].unique(), seed=TRAIN_CFG["seed"]
    )
    splits = split_dataframes(static, text, ts, train_ids, val_ids, test_ids)
    return splits, text


# ── Step 0: Fit preprocessors ─────────────────────────────────────────────────

def step0_prepare() -> tuple:
    """Fit and save preprocessors, or load from disk if already done."""
    CKPT_DIR.mkdir(exist_ok=True)

    if PREP_PATH.exists():
        print(f"[Step 0] Preprocessors found — loading from {PREP_PATH}")
        return load_preprocessors(str(PREP_PATH))

    print("[Step 0] Fitting preprocessors …")
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)
    train_ids, val_ids, _ = split_stay_ids(
        static["stay_id"].unique(), seed=TRAIN_CFG["seed"]
    )
    splits = split_dataframes(static, text, ts, train_ids, val_ids,
                              static["stay_id"].unique()[len(train_ids)+len(val_ids):])

    static_prep = StaticPreprocessor().fit(splits["train"]["static"])
    ts_prep     = TimeSeriesPreprocessor().fit(splits["train"]["ts"])
    text_prep   = TextPreprocessor(
        cache_path=str(ROOT / CACHE_DIR / "text_embeddings.pkl")
    )
    text_prep.fit(text)

    save_preprocessors(static_prep, ts_prep, text_prep, path=str(PREP_PATH))
    return static_prep, ts_prep, text_prep


# ── Stage 1: MHA TS-only training ─────────────────────────────────────────────

def step1_train_ts(static_prep, ts_prep) -> None:
    """Train Stage 1 MHA encoder. Skipped if checkpoint already exists."""
    if S1_CKPT.exists():
        print(f"[Step 1] Stage 1 checkpoint found — skipping ({S1_CKPT.name})")
        return

    print(f"[Step 1] Training Stage 1 — arch={ARCH}")
    splits, _ = _load_and_split(static_prep, ts_prep)

    train_loader, val_loader, _ = build_ts_dataloaders(
        splits["train"], splits["val"], splits["test"],
        ts_prep      = ts_prep,
        batch_size   = TRAIN_CFG["s1_batch_size"],
        num_workers  = TRAIN_CFG["num_workers"],
    )

    model     = TSOnlyModel.from_config(ENCODER_CFG, ts_prep.feature_dim).to(DEVICE)
    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = TRAIN_CFG["s1_lr"],
        weight_decay = TRAIN_CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    _run_loop(model, train_loader, val_loader, optimizer, scheduler,
              criterion, S1_CKPT,
              n_epochs   = TRAIN_CFG["s1_epochs"],
              patience   = TRAIN_CFG["patience"],
              grad_clip  = TRAIN_CFG["grad_clip"],
              variant    = None)


# ── Stage 2: ts_static_text multimodal training ───────────────────────────────

def step2_train_multimodal(static_prep, ts_prep, text_prep) -> None:
    """Train Stage 2 multimodal model. Skipped if checkpoint already exists."""
    if S2_CKPT.exists():
        print(f"[Step 2] Stage 2 checkpoint found — skipping ({S2_CKPT.name})")
        return

    if not S1_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {S1_CKPT}. "
            "Run Step 1 first (or run main.py from scratch)."
        )

    print(f"[Step 2] Training Stage 2 — variant={VARIANT}, ts_arch={ARCH}")
    splits, _ = _load_and_split(static_prep, ts_prep)

    train_loader, val_loader, _ = build_dataloaders(
        splits["train"], splits["val"], splits["test"],
        static_prep  = static_prep,
        ts_prep      = ts_prep,
        text_prep    = text_prep,
        batch_size   = TRAIN_CFG["s2_batch_size"],
        num_workers  = TRAIN_CFG["num_workers"],
    )

    encoder = build_ts_encoder(input_dim=ts_prep.feature_dim, **ENCODER_CFG)
    model   = MultimodalICUModel(
        ts_encoder      = encoder,
        static_feat_dim = static_prep.feature_dim,
        text_emb_dim    = text_prep.embedding_dim,
        variant         = VARIANT,
        freeze_ts       = True,
    ).to(DEVICE)
    model.load_ts_encoder(str(S1_CKPT))

    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = TRAIN_CFG["s2_lr"],
        weight_decay = TRAIN_CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    _run_loop(model, train_loader, val_loader, optimizer, scheduler,
              criterion, S2_CKPT,
              n_epochs   = TRAIN_CFG["s2_epochs"],
              patience   = TRAIN_CFG["patience"],
              grad_clip  = TRAIN_CFG["grad_clip"],
              variant    = VARIANT)


# ── Final evaluation ──────────────────────────────────────────────────────────

def step3_evaluate(static_prep, ts_prep, text_prep) -> dict:
    """Load the best Stage 2 checkpoint and evaluate on the test set."""
    print(f"\n[Step 3] Evaluating best model on test set …")
    splits, _ = _load_and_split(static_prep, ts_prep)

    _, _, test_loader = build_dataloaders(
        splits["train"], splits["val"], splits["test"],
        static_prep  = static_prep,
        ts_prep      = ts_prep,
        text_prep    = text_prep,
        batch_size   = TRAIN_CFG["s2_batch_size"],
        num_workers  = TRAIN_CFG["num_workers"],
    )

    encoder = build_ts_encoder(input_dim=ts_prep.feature_dim, **ENCODER_CFG)
    model   = MultimodalICUModel(
        ts_encoder      = encoder,
        static_feat_dim = static_prep.feature_dim,
        text_emb_dim    = text_prep.embedding_dim,
        variant         = VARIANT,
        freeze_ts       = True,
    ).to(DEVICE)
    model.load_ts_encoder(str(S1_CKPT))
    model.load_state_dict(
        torch.load(S2_CKPT, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for ts_w, ts_l, s_f, t_e, n_f, labels in test_loader:
            ts_w, ts_l = ts_w.to(DEVICE), ts_l.to(DEVICE)
            s_f, t_e, n_f = s_f.to(DEVICE), t_e.to(DEVICE), n_f.to(DEVICE)
            pred = decode_log_pred(model(ts_w, ts_l, s_f, t_e, n_f))
            all_preds.append(pred.cpu())
            all_labels.append(labels)

    metrics = compute_all(torch.cat(all_preds), torch.cat(all_labels))
    return metrics


# ── Generic training loop ─────────────────────────────────────────────────────

def _run_loop(model, train_loader, val_loader, optimizer, scheduler,
              criterion, save_path, n_epochs, patience, grad_clip, variant):
    """Shared train + early-stop loop used by both Stage 1 and Stage 2."""
    best_mae, no_improve = float("inf"), 0

    print(f"{'─'*60}")
    for epoch in range(1, n_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        total_loss, n_batches, n_skip = 0.0, 0, 0
        for batch in train_loader:
            loss = _batch_loss(model, batch, criterion, variant)
            if loss is None:
                n_skip += 1; continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        if n_skip:
            print(f"  [warn] {n_skip} non-finite batches skipped")

        # ── validate ──────────────────────────────────────────────────────
        val_metrics = _evaluate(model, val_loader, variant)
        val_mae     = val_metrics["MAE"]
        scheduler.step(val_mae)

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"loss={train_loss:.4f}  {format_metrics(val_metrics)}")

        if val_mae < best_mae:
            best_mae, no_improve = val_mae, 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best val MAE={best_mae:.4f}  saved → {save_path.name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"{'─'*60}")


def _batch_loss(model, batch, criterion, variant):
    """Compute loss for one batch; returns None if non-finite."""
    if variant is None:
        # Stage 1: (ts_w, ts_l, label)
        ts_w, ts_l, labels = [x.to(DEVICE) for x in batch]
        pred = model(ts_w, ts_l)
    else:
        # Stage 2: (ts_w, ts_l, s_f, t_e, n_f, label)
        ts_w, ts_l, s_f, t_e, n_f, labels = [x.to(DEVICE) for x in batch]
        s_f2 = s_f if "static" in variant else None
        t_e2 = t_e if "text"   in variant else None
        n_f2 = n_f if "text"   in variant else None
        pred = model(ts_w, ts_l, s_f2, t_e2, n_f2)

    loss = criterion(pred, labels)
    return loss if torch.isfinite(loss) else None


@torch.no_grad()
def _evaluate(model, loader, variant) -> dict:
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        if variant is None:
            ts_w, ts_l, labels = [x.to(DEVICE) for x in batch]
            pred = decode_log_pred(model(ts_w, ts_l))
        else:
            ts_w, ts_l, s_f, t_e, n_f, labels = [x.to(DEVICE) for x in batch]
            s_f2 = s_f if "static" in variant else None
            t_e2 = t_e if "text"   in variant else None
            n_f2 = n_f if "text"   in variant else None
            pred = decode_log_pred(model(ts_w, ts_l, s_f2, t_e2, n_f2))
        preds.append(pred.cpu()); labels_all.append(labels.cpu())
    return compute_all(torch.cat(preds), torch.cat(labels_all))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _set_seed(TRAIN_CFG["seed"])

    print("=" * 60)
    print("  ICU RLOS  |  Best Model Quick Verify")
    print(f"  arch={ARCH}  variant={VARIANT}  device={DEVICE}")
    print("=" * 60 + "\n")

    static_prep, ts_prep, text_prep = step0_prepare()
    step1_train_ts(static_prep, ts_prep)
    step2_train_multimodal(static_prep, ts_prep, text_prep)
    test_metrics = step3_evaluate(static_prep, ts_prep, text_prep)

    print("\n" + "=" * 60)
    print("  Final Test Results")
    print("  " + format_metrics(test_metrics))
    print("=" * 60)


if __name__ == "__main__":
    main()