"""
ts_only.py – Stage 1 model: a TSEncoder followed by a scalar regression head.

Deliberately minimal so Stage 1 comparisons are purely about encoder architecture.
Stage 2 reuses the same TSEncoder (loaded from Stage 1 checkpoint) inside
MultimodalICUModel.
"""

import torch
import torch.nn as nn

from src.models.encoders.ts_encoder import build_ts_encoder


class TSOnlyModel(nn.Module):
    """
    TSEncoder  →  Linear(hidden_dim → 1)
    Outputs raw log1p(RLOS) during training; decoding is handled by loss.py.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(encoder.hidden_dim, 1) # type:ignore

    def forward(
        self,
        ts_window:  torch.Tensor,   # (B, W, D_ts)
        ts_lengths: torch.Tensor,   # (B,)
    ) -> torch.Tensor:              # (B,)
        h    = self.encoder(ts_window, ts_lengths)   # (B, hidden_dim)
        pred = self.head(h).squeeze(-1)              # (B,)
        return pred

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, encoder_cfg: dict, input_dim: int) -> "TSOnlyModel":
        """Build from a yaml-loaded config dict."""
        encoder = build_ts_encoder(input_dim=input_dim, **encoder_cfg)
        return cls(encoder)

    @classmethod
    def load(cls, ckpt_path: str, encoder_cfg: dict, input_dim: int) -> "TSOnlyModel":
        """Restore a saved Stage 1 model."""
        model = cls.from_config(encoder_cfg, input_dim)
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        return model