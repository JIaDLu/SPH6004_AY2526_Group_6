"""
main.py – End-to-end pipeline entry point.

Steps
─────
1. Load & filter raw data (icu_death_flag == 0)
2. Split stay_ids 8 : 1 : 1
3. Fit preprocessors on training data  (no data leakage)
   – or restore from disk if --resume is passed
4. Build ICUDataset + DataLoader for train / val / test
5. Initialise MultimodalICUModel
6. Train with Trainer (early stopping, checkpointing)
7. Evaluate best model on test set

Usage
─────
    python main.py                        # full run
    python main.py --resume               # skip fitting; load saved preprocessors
    python main.py --epochs 50 --lr 5e-4
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader               import load_raw_data, filter_discharged_patients
from src.data.splitter             import split_stay_ids, split_dataframes
from src.data.static_preprocessor import StaticPreprocessor
from src.data.ts_preprocessor     import TimeSeriesPreprocessor
from src.data.text_preprocessor   import TextPreprocessor
from src.data.dataset              import build_dataloaders
from src.models.multimodal         import MultimodalICUModel
from src.training.trainer          import Trainer
from src.utils.metrics             import format_metrics
from src.utils.persistence         import save_preprocessors, load_preprocessors
from src.utils.constants           import CACHE_DIR, RANDOM_SEED


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ICU RLOS multimodal pipeline")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints")
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument(
        "--resume", action="store_true",
        help="Skip fitting; reload preprocessors from checkpoints/preprocessors.pkl"
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("  ICU Remaining Length-of-Stay  |  Multimodal Pipeline")
    print("=" * 65)

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    print("\n[1/7] Load & filter")
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n[2/7] Split stay_ids (8 : 1 : 1)")
    stay_ids                      = static["stay_id"].unique()
    train_ids, val_ids, test_ids  = split_stay_ids(stay_ids, seed=RANDOM_SEED)
    splits                        = split_dataframes(
        static, text, ts, train_ids, val_ids, test_ids
    )

    # ── 3. Fit or restore preprocessors ──────────────────────────────────────
    prep_path = os.path.join(args.ckpt_dir, "preprocessors.pkl")

    if args.resume:
        print("\n[3/7] Restore preprocessors from disk  (--resume)")
        static_prep, ts_prep, text_prep = load_preprocessors(prep_path)
    else:
        print("\n[3/7] Fit preprocessors (train data only)")

        static_prep = StaticPreprocessor()
        static_prep.fit(splits["train"]["static"])

        ts_prep = TimeSeriesPreprocessor()
        ts_prep.fit(splits["train"]["ts"])

        # BERT is frozen → fitting = pre-compute embeddings, no label leakage
        text_prep = TextPreprocessor(
            cache_path=os.path.join(CACHE_DIR, "text_embeddings.pkl")
        )
        text_prep.fit(text)   # full text table so every stay_id is covered

        save_preprocessors(static_prep, ts_prep, text_prep, path=prep_path)

    # ── 4. Build DataLoaders ──────────────────────────────────────────────────
    print("\n[4/7] Build DataLoaders")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_splits = splits["train"],
        val_splits   = splits["val"],
        test_splits  = splits["test"],
        static_prep  = static_prep,
        ts_prep      = ts_prep,
        text_prep    = text_prep,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )

    # ── 5. Initialise model ───────────────────────────────────────────────────
    print("\n[5/7] Initialise model")
    model = MultimodalICUModel(
        ts_feat_dim     = ts_prep.feature_dim,
        static_feat_dim = static_prep.feature_dim,
        text_emb_dim    = text_prep.embedding_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ts_feat_dim     = {ts_prep.feature_dim}")
    print(f"  static_feat_dim = {static_prep.feature_dim}")
    print(f"  text_emb_dim    = {text_prep.embedding_dim}")
    print(f"  Total params    = {n_params:,}")

    _verify_shapes(model, train_loader)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[6/7] Train")
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        grad_clip      = args.grad_clip,
        patience       = args.patience,
        checkpoint_dir = args.ckpt_dir,
    )
    history = trainer.fit(args.epochs)

    # ── 7. Test evaluation ────────────────────────────────────────────────────
    print("\n[7/7] Test evaluation")
    trainer.load_best()
    test_metrics = trainer.evaluate(test_loader)
    print(f"\n  Test results:  {format_metrics(test_metrics)}")
    print(f"\n{'=' * 65}")
    print("  Pipeline complete.")
    print("=" * 65)

    return model, history, test_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _verify_shapes(model: torch.nn.Module, loader) -> None:
    """One forward pass to confirm tensor shapes match the model."""
    device = next(model.parameters()).device
    batch  = next(iter(loader))
    ts_w, ts_l, s_f, t_e, n_f, lab = [x.to(device) for x in batch]

    names = ["ts_window", "ts_lengths", "static_feat", "text_emb", "no_note_flag", "labels"]
    print("\n  Batch shapes:")
    for name, t in zip(names, [ts_w, ts_l, s_f, t_e, n_f, lab]):
        shape = tuple(t.shape) if t.dim() > 0 else "scalar"
        print(f"    {name:<14}: {shape}")

    with torch.no_grad():
        pred = model(ts_w, ts_l, s_f, t_e, n_f)
    print(f"    {'output':<14}: {tuple(pred.shape)}")


if __name__ == "__main__":
    main()