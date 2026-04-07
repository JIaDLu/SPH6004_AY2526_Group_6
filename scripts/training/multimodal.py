"""
scripts/training/multimodal.py – Stage 2: multimodal ablation training.

Trains one variant of the multimodal model per run.  The TS encoder is
always loaded from a Stage 1 checkpoint and frozen.

Usage
-----
# After Stage 1, identify the best arch (e.g. lstm) and run:
python -m scripts.training.multimodal --ts_arch lstm --variant ts_static
python -m scripts.training.multimodal --ts_arch lstm --variant ts_text
python -m scripts.training.multimodal --ts_arch lstm --variant ts_static_text

# Optional overrides:
python -m scripts.training.multimodal --ts_arch lstm --variant ts_static_text \
    --lr 1e-4 --batch_size 128

Outputs (auto-created per run)
-------------------------------
outputs/{variant}_{arch}_{timestamp}/
    config.yaml
    metrics.json
    history.json
    model_best.pt
checkpoints/mm_{variant}_{arch}_best.pt   ← canonical Stage 2 checkpoint
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader               import load_raw_data, filter_discharged_patients
from src.data.splitter             import split_stay_ids, split_dataframes
from src.data.dataset              import build_dataloaders
from src.models.encoders.ts_encoder import build_ts_encoder
from src.models.multimodal         import MultimodalICUModel, VALID_VARIANTS
from src.training.loss             import LogMSELoss, decode_log_pred
from src.utils.metrics             import compute_all, format_metrics
from src.utils.persistence         import load_preprocessors
from src.utils.constants           import RANDOM_SEED


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 – Multimodal training")
    p.add_argument("--ts_arch",      required=True, choices=["lstm", "gru", "mha"],
                   help="TS encoder arch (must match Stage 1 winner)")
    p.add_argument("--variant",      required=True, choices=list(VALID_VARIANTS),
                   help="Modality combination to train")
    p.add_argument("--ts_ckpt",      default=None,
                   help="Stage 1 checkpoint path "
                        "(default: checkpoints/ts_{arch}_best.pt)")
    p.add_argument("--encoder_cfg",  default=None,
                   help="Encoder yaml path (default: configs/model/ts_encoder/{arch}.yaml)")
    p.add_argument("--train_cfg",    default=str(ROOT / "configs/training/multimodal.yaml"))
    p.add_argument("--prep_path",    default=None)
    # Inline overrides
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience",     type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--num_workers",  type=int,   default=None)
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(args: argparse.Namespace) -> dict:
    enc_cfg_path = args.encoder_cfg or str(
        ROOT / "configs/model/ts_encoder" / f"{args.ts_arch}.yaml"
    )
    enc_cfg   = load_yaml(enc_cfg_path)
    train_cfg = load_yaml(args.train_cfg)

    for key in ("epochs", "batch_size", "lr", "weight_decay",
                "patience", "seed", "num_workers"):
        val = getattr(args, key, None)
        if val is not None:
            train_cfg[key] = val

    return {
        "ts_arch":  args.ts_arch,
        "variant":  args.variant,
        "encoder":  enc_cfg,
        "training": train_cfg,
        "stage":    "multimodal",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_output_dir(cfg: dict) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = ROOT / cfg["training"]["output_root"] / \
          f"{cfg['variant']}_{cfg['ts_arch']}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Training / evaluation ─────────────────────────────────────────────────────

def _unpack_batch(batch, variant: str, device: str):
    """
    Full ICUDataset batch: (ts_w, ts_l, static_feat, text_emb, no_note_flag, label)
    Selectively nullify modalities not used in this variant.
    """
    ts_w, ts_l, s_f, t_e, n_f, labels = [x.to(device) for x in batch]
    s_f = s_f if "static" in variant else None
    t_e = t_e if "text"   in variant else None
    n_f = n_f if "text"   in variant else None
    return ts_w, ts_l, s_f, t_e, n_f, labels


def train_one_epoch(model, loader, optimizer, criterion,
                    grad_clip, device, variant) -> float:
    model.train()
    total_loss, n_batches, n_skipped = 0.0, 0, 0

    for batch in loader:
        ts_w, ts_l, s_f, t_e, n_f, labels = _unpack_batch(batch, variant, device)
        optimizer.zero_grad()
        pred = model(ts_w, ts_l, s_f, t_e, n_f)
        loss = criterion(pred, labels)

        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if n_skipped:
        print(f"  [warn] {n_skipped} non-finite batches skipped")
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, variant) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        ts_w, ts_l, s_f, t_e, n_f, labels = _unpack_batch(batch, variant, device)
        pred_orig = decode_log_pred(model(ts_w, ts_l, s_f, t_e, n_f)).cpu()
        all_preds.append(pred_orig)
        all_labels.append(labels.cpu())
    return compute_all(torch.cat(all_preds), torch.cat(all_labels))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    cfg     = merge_configs(args)
    t_cfg   = cfg["training"]
    e_cfg   = cfg["encoder"]
    variant = cfg["variant"]

    set_seed(t_cfg.get("seed", RANDOM_SEED))
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = make_output_dir(cfg)

    print("=" * 65)
    print(f"  Stage 2 Multimodal  |  variant={variant}  "
          f"ts_arch={cfg['ts_arch']}  device={device}")
    print(f"  Output → {out_dir}")
    print("=" * 65)

    # ── Preprocessors ─────────────────────────────────────────────────────────
    prep_path = args.prep_path or str(
        ROOT / t_cfg.get("ckpt_dir", "checkpoints") / "preprocessors.pkl"
    )
    static_prep, ts_prep, text_prep = load_preprocessors(prep_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Load & split data")
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)
    train_ids, val_ids, test_ids = split_stay_ids(
        static["stay_id"].unique(), seed=t_cfg.get("seed", RANDOM_SEED)
    )
    splits = split_dataframes(static, text, ts, train_ids, val_ids, test_ids)

    print("\n[2/4] Build DataLoaders (full multimodal)")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_splits = splits["train"],
        val_splits   = splits["val"],
        test_splits  = splits["test"],
        static_prep  = static_prep,
        ts_prep      = ts_prep,
        text_prep    = text_prep,
        batch_size   = t_cfg.get("batch_size", 256),
        num_workers  = t_cfg.get("num_workers", 4),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Build model")
    encoder = build_ts_encoder(
        input_dim = ts_prep.feature_dim,
        **{**e_cfg, "arch": cfg["ts_arch"]},
    )
    model = MultimodalICUModel(
        ts_encoder      = encoder,
        static_feat_dim = static_prep.feature_dim,
        text_emb_dim    = text_prep.embedding_dim,
        variant         = variant,
        freeze_ts       = True,
    ).to(device)

    # Load Stage 1 TS encoder weights
    ts_ckpt = args.ts_ckpt or str(
        ROOT / t_cfg.get("ckpt_dir", "checkpoints") / f"ts_{cfg['ts_arch']}_best.pt"
    )
    model.load_ts_encoder(ts_ckpt)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,} total  |  {trainable:,} trainable  "
          f"({total - trainable:,} frozen TS)")

    # ── Optimiser (only trainable params) ─────────────────────────────────────
    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = t_cfg.get("lr", 3e-4),
        weight_decay = t_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = _init_wandb(cfg)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[4/4] Train")
    best_val_mae    = float("inf")
    patience_count  = 0
    patience        = t_cfg.get("patience", 10)
    grad_clip       = t_cfg.get("grad_clip", 1.0)
    n_epochs        = t_cfg.get("epochs", 100)

    ckpt_dir   = ROOT / t_cfg.get("ckpt_dir", "checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt  = out_dir / "model_best.pt"
    canon_ckpt = ckpt_dir / f"mm_{variant}_{cfg['ts_arch']}_best.pt"

    history: dict[str, list] = {"train_loss": [], "val_mae": [], "val_rmse": []}

    print(f"\n{'─'*65}")
    for epoch in range(1, n_epochs + 1):
        t0          = time.time()
        train_loss  = train_one_epoch(
            model, train_loader, optimizer, criterion,
            grad_clip, device, variant,
        )
        val_metrics = evaluate(model, val_loader, device, variant)
        val_mae     = val_metrics["MAE"]
        elapsed     = time.time() - t0

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"loss={train_loss:.4f}  {format_metrics(val_metrics)}  "
              f"({elapsed:.1f}s)")

        history["train_loss"].append(round(train_loss, 6))
        history["val_mae"].append(round(val_mae, 4))
        history["val_rmse"].append(round(val_metrics["RMSE"], 4))

        if wandb_run:
            wandb_run.log({ #type:ignore
                "train/loss": train_loss,
                "val/MAE":    val_mae,
                "val/RMSE":   val_metrics["RMSE"],
                "val/MedAE":  val_metrics["MedAE"],
                "val/R2":     val_metrics["R2"],
                "epoch":      epoch,
            })

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae   = val_mae
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
            torch.save(model.state_dict(), canon_ckpt)
            print(f"  ✓ Best val MAE={best_val_mae:.4f} — saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stop at epoch {epoch}")
                break

    print(f"{'─'*65}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device, variant)
    print(f"\n  Test:  {format_metrics(test_metrics)}")

    if wandb_run:
        wandb_run.summary.update({f"test/{k}": v for k, v in test_metrics.items()}) #type:ignore

    # ── Save artefacts ────────────────────────────────────────────────────────
    metrics_payload = {
        "run_name":     f"{variant}_{cfg['ts_arch']}",
        "ts_arch":      cfg["ts_arch"],
        "variant":      variant,
        "stage":        "multimodal",
        "best_val_mae": round(best_val_mae, 4),
        "test_MAE":     round(test_metrics["MAE"],   4),
        "test_RMSE":    round(test_metrics["RMSE"],  4),
        "test_MedAE":   round(test_metrics["MedAE"], 4),
        "test_R2":      round(test_metrics["R2"],    4),
        "config":       cfg,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n  Artefacts → {out_dir}")
    print(f"  Canonical ckpt → {canon_ckpt}")
    if wandb_run:
        wandb_run.finish()  #type:ignore


def _init_wandb(cfg: dict) -> Optional[object]:
    try:
        import wandb
        w_cfg = cfg["training"].get("wandb", {})
        run = wandb.init(
            project = w_cfg.get("project", "sph6004"),
            entity  = w_cfg.get("entity",  "lujiadong-nus"),
            name    = f"{cfg['variant']}_{cfg['ts_arch']}",
            config  = cfg,
            tags    = ["stage2", "multimodal", cfg["variant"], cfg["ts_arch"]],
        )
        try:
            import weave
            weave.init(f"{w_cfg.get('entity','lujiadong-nus')}/{w_cfg.get('project','sph6004')}")
        except Exception:
            pass
        return run
    except Exception as e:
        print(f"  [wandb] Skipped ({e})")
        return None


if __name__ == "__main__":
    main()