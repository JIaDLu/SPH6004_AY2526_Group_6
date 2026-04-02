"""
metrics.py – Regression metrics for RLOS prediction.

All functions accept numpy arrays or torch tensors.
"""

import numpy as np
import torch


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()


def mae(preds, labels) -> float:
    p, l = _to_np(preds), _to_np(labels)
    return float(np.mean(np.abs(p - l)))


def rmse(preds, labels) -> float:
    p, l = _to_np(preds), _to_np(labels)
    return float(np.sqrt(np.mean((p - l) ** 2)))


def median_ae(preds, labels) -> float:
    p, l = _to_np(preds), _to_np(labels)
    return float(np.median(np.abs(p - l)))


def r2(preds, labels) -> float:
    p, l   = _to_np(preds), _to_np(labels)
    ss_res = np.sum((l - p) ** 2)
    ss_tot = np.sum((l - np.mean(l)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_all(preds, labels) -> dict[str, float]:
    return {
        "MAE":      mae(preds, labels),
        "RMSE":     rmse(preds, labels),
        "MedAE":    median_ae(preds, labels),
        "R2":       r2(preds, labels),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())