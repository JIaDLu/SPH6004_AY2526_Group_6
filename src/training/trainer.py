"""
trainer.py – Training loop for the multimodal ICU RLOS model.

Features
────────
• MSE loss with optional gradient clipping
• Per-epoch validation evaluation
• Early stopping on val MAE
• Best-model checkpointing (saves/loads state_dict only)
• Clean per-epoch logging
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.metrics import compute_all, format_metrics


class Trainer:

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        grad_clip:      float = 1.0,
        patience:       int   = 10,
        checkpoint_dir: str   = "checkpoints",
        device:         str | None = None,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.grad_clip     = grad_clip
        self.patience      = patience
        self.checkpoint_dir = checkpoint_dir
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._best_val_mae     = float("inf")
        self._epochs_no_improve = 0
        self._best_ckpt_path   = os.path.join(checkpoint_dir, "best_model.pt")

        print(f"[Trainer] device={self.device}  lr={lr}  patience={patience}")

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, n_epochs: int) -> dict[str, list]:
        """
        Run the training loop for n_epochs.
        Returns history dict with per-epoch train_loss and val metrics.
        """
        history: dict[str, list] = {
            "train_loss": [], "val_MAE": [], "val_RMSE": [],
            "val_MedAE":  [], "val_R2":  [],
        }

        print(f"\n{'─'*65}")
        print(f"  Training for up to {n_epochs} epochs  (early stop patience={self.patience})")
        print(f"{'─'*65}")

        for epoch in range(1, n_epochs + 1):
            t0         = time.time()
            train_loss = self._train_one_epoch()
            val_metrics = self.evaluate(self.val_loader)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"loss={train_loss:.4f}  "
                f"{format_metrics(val_metrics)}  "
                f"({elapsed:.1f}s)"
            )

            # Record history
            history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                history[f"val_{k}"].append(v)

            # LR scheduler
            self.scheduler.step(val_metrics["MAE"])

            # Early stopping & checkpointing
            if val_metrics["MAE"] < self._best_val_mae:
                self._best_val_mae      = val_metrics["MAE"]
                self._epochs_no_improve = 0
                torch.save(self.model.state_dict(), self._best_ckpt_path)
                print(f"  ✓ New best val MAE={self._best_val_mae:.4f} — checkpoint saved")
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping triggered at epoch {epoch}")
                    break

        print(f"{'─'*65}")
        print(f"  Training complete.  Best val MAE = {self._best_val_mae:.4f}")
        return history

    def load_best(self):
        """Restore the best checkpoint weights into self.model."""
        self.model.load_state_dict(
            torch.load(self._best_ckpt_path, map_location=self.device)
        )
        print(f"[Trainer] Loaded best checkpoint from {self._best_ckpt_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Run inference over loader and return regression metrics."""
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in loader:
            ts_w, ts_l, s_f, t_e, n_f, labels = self._to_device(batch)
            preds = self.model(ts_w, ts_l, s_f, t_e, n_f)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return compute_all(all_preds, all_labels)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches, n_skipped = 0.0, 0, 0

        for batch in self.train_loader:
            ts_w, ts_l, s_f, t_e, n_f, labels = self._to_device(batch)

            self.optimizer.zero_grad()
            preds = self.model(ts_w, ts_l, s_f, t_e, n_f)
            loss  = self.criterion(preds, labels)

            # Skip batch if loss is NaN/Inf (e.g. corrupted embedding)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            loss.backward()

            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if n_skipped > 0:
            print(f"  [warn] Skipped {n_skipped} non-finite batches this epoch")

        return total_loss / max(n_batches, 1)

    def _to_device(self, batch):
        return [x.to(self.device) for x in batch]