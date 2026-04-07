"""
scripts/training/baseline.py – Stage 1: train a single TS-only baseline run.

Usage
-----
# Train all three architectures (run each separately):
python -m scripts.training.baseline --arch lstm
python -m scripts.training.baseline --arch gru
python -m scripts.training.baseline --arch mha

# Override any training config key:
python -m scripts.training.baseline --arch lstm --lr 1e-3 --epochs 50

Outputs (auto-created per run)
-------------------------------
outputs/{arch}_{timestamp}/
    config.yaml       complete merged config snapshot
    metrics.json      final test metrics + best val metrics
    history.json      per-epoch train_loss / val_mae
    model_best.pt     best checkpoint (val MAE, original space)
checkpoints/ts_{arch}_best.pt   ← canonical path used by Stage 2
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

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader               import load_raw_data, filter_discharged_patients
from src.data.splitter             import split_stay_ids, split_dataframes
from src.data.ts_dataset           import build_ts_dataloaders
from src.models.ts_only            import TSOnlyModel
from src.training.loss             import LogMSELoss, decode_log_pred
from src.utils.metrics             import compute_all, format_metrics
from src.utils.persistence         import load_preprocessors
from src.utils.constants           import RANDOM_SEED


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 – TS-only baseline training")
    p.add_argument("--arch",         required=True, choices=["lstm", "gru", "mha"],
                   help="Encoder architecture")
    p.add_argument("--encoder_cfg",  default=None,
                   help="Path to encoder yaml (default: configs/model/ts_encoder/{arch}.yaml)")
    p.add_argument("--train_cfg",    default=str(ROOT / "configs/training/baseline.yaml"),
                   help="Path to training yaml")
    p.add_argument("--prep_path",    default=None,
                   help="Path to preprocessors.pkl (default: checkpoints/preprocessors.pkl)")
    # Allow inline overrides for any training config key
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience",     type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--num_workers",  type=int,   default=None)
    return p.parse_args()


# ── Config helpers ────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(args: argparse.Namespace) -> dict:
    """Merge encoder yaml + training yaml + CLI overrides."""
    encoder_cfg_path = args.encoder_cfg or str(
        ROOT / "configs/model/ts_encoder" / f"{args.arch}.yaml"
    )
    enc_cfg   = load_yaml(encoder_cfg_path)
    train_cfg = load_yaml(args.train_cfg)

    # CLI overrides
    for key in ("epochs", "batch_size", "lr", "weight_decay", "patience",
                "seed", "num_workers"):
        val = getattr(args, key)
        if val is not None:
            train_cfg[key] = val

    return {
        "arch":     args.arch,
        "encoder":  enc_cfg,
        "training": train_cfg,
        "stage":    "ts_only",
    }


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Output directory ──────────────────────────────────────────────────────────

def make_output_dir(cfg: dict) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = ROOT / cfg["training"]["output_root"] / f"{cfg['arch']}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       nn.Module,
    loader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    grad_clip:   float,
    device:      str,
) -> float:
    model.train()
    total_loss, n_batches, n_skipped = 0.0, 0, 0

    for ts_w, ts_l, labels in loader:
        ts_w, ts_l, labels = ts_w.to(device), ts_l.to(device), labels.to(device)

        optimizer.zero_grad()
        pred_log = model(ts_w, ts_l)
        loss     = criterion(pred_log, labels)

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
def evaluate(
    model:  nn.Module,
    loader,
    device: str,
) -> dict[str, float]:
    """Evaluate in original RLOS space (decode from log)."""
    model.eval()
    all_preds, all_labels = [], []

    for ts_w, ts_l, labels in loader:
        ts_w, ts_l = ts_w.to(device), ts_l.to(device)
        pred_log   = model(ts_w, ts_l)
        pred_orig  = decode_log_pred(pred_log).cpu()
        all_preds.append(pred_orig)
        all_labels.append(labels)

    return compute_all(
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = merge_configs(args)
    t_cfg = cfg["training"]
    e_cfg = cfg["encoder"]

    set_seed(t_cfg.get("seed", RANDOM_SEED))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = make_output_dir(cfg)

    print("=" * 65)
    print(f"  Stage 1 Baseline  |  arch={cfg['arch']}  device={device}")
    print(f"  Output → {out_dir}")
    print("=" * 65)

    # ── Load preprocessors ────────────────────────────────────────────────────
    prep_path = args.prep_path or str(
        ROOT / t_cfg.get("ckpt_dir", "checkpoints") / "preprocessors.pkl"
    )
    static_prep, ts_prep, _ = load_preprocessors(prep_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Load & split data")
    static, text, ts = load_raw_data()
    static, text, ts = filter_discharged_patients(static, text, ts)
    train_ids, val_ids, test_ids = split_stay_ids(
        static["stay_id"].unique(), seed=t_cfg.get("seed", RANDOM_SEED)
    )
    splits = split_dataframes(static, text, ts, train_ids, val_ids, test_ids)

    print("\n[2/4] Build DataLoaders (TS-only, no BERT)")
    train_loader, val_loader, test_loader = build_ts_dataloaders(
        train_splits = splits["train"],
        val_splits   = splits["val"],
        test_splits  = splits["test"],
        ts_prep      = ts_prep,
        batch_size   = t_cfg.get("batch_size", 512),
        num_workers  = t_cfg.get("num_workers", 4),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Build model")
    model = TSOnlyModel.from_config(
        encoder_cfg = {**e_cfg, "arch": cfg["arch"]},
        input_dim   = ts_prep.feature_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  arch={cfg['arch']}  hidden={e_cfg['hidden_dim']}  "
          f"layers={e_cfg['num_layers']}  params={n_params:,}")

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    criterion = LogMSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t_cfg.get("lr", 3e-4),
        weight_decay = t_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── wandb (optional) ──────────────────────────────────────────────────────
    wandb_run = _init_wandb(cfg)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[4/4] Train")
    best_val_mae   = float("inf")
    patience_count = 0
    patience       = t_cfg.get("patience", 10)
    grad_clip      = t_cfg.get("grad_clip", 1.0)
    n_epochs       = t_cfg.get("epochs", 100)

    ckpt_dir  = ROOT / t_cfg.get("ckpt_dir", "checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = out_dir / "model_best.pt"
    canon_ckpt = ckpt_dir / f"ts_{cfg['arch']}_best.pt"  # used by Stage 2

    history: dict[str, list] = {"train_loss": [], "val_mae": [], "val_rmse": []}

    print(f"\n{'─'*65}")
    for epoch in range(1, n_epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, grad_clip, device
        )
        val_metrics = evaluate(model, val_loader, device)
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
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n  Test:  {format_metrics(test_metrics)}")

    if wandb_run:
        wandb_run.summary.update({f"test/{k}": v for k, v in test_metrics.items()}) #type:ignore

    # ── Save artefacts ────────────────────────────────────────────────────────
    metrics_payload = {
        "run_name":      f"{cfg['arch']}_baseline",
        "arch":          cfg["arch"],
        "stage":         "ts_only",
        "best_val_mae":  round(best_val_mae, 4),
        "test_MAE":      round(test_metrics["MAE"],   4),
        "test_RMSE":     round(test_metrics["RMSE"],  4),
        "test_MedAE":    round(test_metrics["MedAE"], 4),
        "test_R2":       round(test_metrics["R2"],    4),
        "config":        cfg,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2)
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2)
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n  Artefacts saved → {out_dir}")
    print(f"  Canonical ckpt  → {canon_ckpt}")

    if wandb_run:
        wandb_run.finish() #type:ignore


# ── wandb helper ──────────────────────────────────────────────────────────────

def _init_wandb(cfg: dict) -> Optional[object]:
    """Initialise wandb + weave; returns run or None if not installed."""
    try:
        import wandb
        w_cfg = cfg["training"].get("wandb", {})
        run = wandb.init(
            project = w_cfg.get("project", "sph6004"),
            entity  = w_cfg.get("entity",  "lujiadong-nus"),
            name    = f"{cfg['arch']}_baseline",
            config  = cfg,
            tags    = ["stage1", "ts_only", cfg["arch"]],
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