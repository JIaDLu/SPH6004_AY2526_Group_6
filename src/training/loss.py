"""
loss.py – Loss function and label transforms for the log-space training strategy.

Why log-space?
──────────────
RLOS has a heavy right tail (some stays last hundreds of hours).
Naïve MSE would focus training entirely on these rare extreme samples,
hurting predictions for the majority of patients.

Strategy
────────
  Train : predict  ŷ = log1p(RLOS)   → MSE(ŷ_pred, ŷ_true)
  Eval  : decode   RLOS_pred = expm1(ŷ_pred).clamp(min=0)
          then compute MAE / RMSE on original-space values.

Both functions are used in scripts/training/baseline.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMSELoss(nn.Module):
    """
    MSE loss computed in log1p space.

    The model predicts log1p(RLOS); this loss transforms raw labels on the fly.
    Labels must be raw RLOS values in hours (≥ 0).
    """

    def forward(self, pred_log: torch.Tensor, label_raw: torch.Tensor) -> torch.Tensor:
        label_log = torch.log1p(label_raw.clamp(min=0))
        return F.mse_loss(pred_log, label_log)


def decode_log_pred(pred_log: torch.Tensor) -> torch.Tensor:
    """
    Convert model output (log1p space) back to original RLOS hours.
    Clamps to ≥ 0 to prevent negative predictions.
    """
    return torch.expm1(pred_log).clamp(min=0)