"""
ts_encoder.py – Unified time-series encoder supporting three backbone architectures:
  • LSTM   – pack_padded_sequence; last-layer hidden state
  • GRU    – pack_padded_sequence; last-layer hidden state
  • MHA    – Transformer encoder layers; mean-pool over valid (non-padded) steps

All three expose the same interface:
    encoder(ts_window, ts_lengths) → (B, hidden_dim)

This module contains ONLY model code.  Training logic lives in scripts/.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ── Recurrent (LSTM / GRU) ────────────────────────────────────────────────────

class RecurrentEncoder(nn.Module):
    """Shared implementation for LSTM and GRU."""

    def __init__(
        self,
        arch:       str,    # "lstm" | "gru"
        input_dim:  int,
        hidden_dim: int,
        num_layers: int,
        dropout:    float,
    ):
        super().__init__()
        assert arch in ("lstm", "gru"), f"Unknown arch: {arch}"
        rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, W, input_dim)
        lengths : (B,)  number of valid (non-padded) steps per sample
        returns : (B, hidden_dim)  — last-layer hidden state at final valid step
        """
        packed = pack_padded_sequence(
            x,
            lengths.clamp(min=1).cpu(),
            batch_first    = True,
            enforce_sorted = False,
        )
        _, hidden = self.rnn(packed)
        # LSTM returns (h_n, c_n); GRU returns h_n directly
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]   # (B, hidden_dim)


# ── Multi-Head Attention (Transformer Encoder) ────────────────────────────────

class MHAEncoder(nn.Module):
    """
    Lightweight Transformer encoder.
    Padding mask ensures padded positions never influence attention.
    Output: mean-pool over valid (non-padded) positions.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        num_layers: int,
        num_heads:  int,
        dropout:    float,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model        = hidden_dim,
                nhead          = num_heads,
                dim_feedforward= hidden_dim * 4,
                dropout        = dropout,
                batch_first    = True,
                norm_first     = True,   # Pre-LN: more stable training
            )
            for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, W, input_dim)
        lengths : (B,)
        returns : (B, hidden_dim)
        """
        B, W, _ = x.shape
        h = self.input_proj(x)   # (B, W, hidden_dim)

        # key_padding_mask: True → position is PADDING (ignore in attention)
        idx  = torch.arange(W, device=x.device).unsqueeze(0)   # (1, W)
        mask = idx >= lengths.unsqueeze(1)                      # (B, W)

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=mask)             # (B, W, hidden_dim)

        # Mean-pool over valid positions only
        valid = (~mask).float().unsqueeze(-1)                   # (B, W, 1)
        h = (h * valid).sum(1) / valid.sum(1).clamp(min=1)     # (B, hidden_dim)
        return h


# ── Factory ───────────────────────────────────────────────────────────────────

def build_ts_encoder(
    arch:       str,
    input_dim:  int,
    hidden_dim: int,
    num_layers: int,
    dropout:    float,
    num_heads:  int = 4,    # only used by MHA
    **_kwargs,              # absorb extra yaml keys gracefully
) -> nn.Module:
    """
    Build and return the appropriate TSEncoder.
    `input_dim` is always injected at runtime from ts_prep.feature_dim.

    Usage
    -----
    cfg = yaml.safe_load(open("configs/model/ts_encoder/lstm.yaml"))
    encoder = build_ts_encoder(input_dim=ts_prep.feature_dim, **cfg)
    """
    arch = arch.lower()
    if arch in ("lstm", "gru"):
        return RecurrentEncoder(arch, input_dim, hidden_dim, num_layers, dropout)
    if arch == "mha":
        return MHAEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
    raise ValueError(f"Unknown arch '{arch}'. Choose from: lstm, gru, mha")