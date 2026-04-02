"""
multimodal.py – Multimodal ICU length-of-stay prediction model.

Architecture
────────────
  Time series  →  LSTM               →  h_ts     (lstm_hidden,)
  Static       →  Linear + ReLU      →  h_static (static_hidden,)
  Text         →  [NO_NOTE swap]
               →  Linear + ReLU      →  h_text   (text_hidden,)
                                                    ↓ concat
                                        Fusion MLP  →  scalar (RLOS)

NO_NOTE handling
────────────────
When no_note_flag == 1 the pre-computed BERT embedding is a zero vector.
We replace it with a learnable nn.Parameter (no_note_emb) so the model
can learn an explicit "no report yet" signal, rather than seeing silence.
"""

import torch
import torch.nn as nn


class MultimodalICUModel(nn.Module):

    def __init__(
        self,
        ts_feat_dim:    int,
        static_feat_dim: int,
        text_emb_dim:   int,
        lstm_hidden:    int = 64,
        lstm_layers:    int = 2,
        static_hidden:  int = 32,
        text_hidden:    int = 64,
        fusion_hidden:  int = 128,
        dropout:        float = 0.2,
    ):
        super().__init__()

        # ── Time-series branch ────────────────────────────────────────────
        self.ts_lstm = nn.LSTM(
            input_size  = ts_feat_dim,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Static branch ─────────────────────────────────────────────────
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feat_dim, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Text branch ───────────────────────────────────────────────────
        # Learnable embedding used when no note is available at time t
        self.no_note_emb = nn.Parameter(torch.randn(text_emb_dim) * 0.01)

        self.text_encoder = nn.Sequential(
            nn.Linear(text_emb_dim, text_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Fusion head ───────────────────────────────────────────────────
        fusion_in = lstm_hidden + static_hidden + text_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ts_window:    torch.Tensor,   # (B, W, D_ts)
        ts_lengths:   torch.Tensor,   # (B,)  int — valid time steps
        static_feat:  torch.Tensor,   # (B, D_static)
        text_emb:     torch.Tensor,   # (B, D_text)   zeros when no note
        no_note_flag: torch.Tensor,   # (B,)  1.0 = no note
    ) -> torch.Tensor:                # (B,)  predicted RLOS

        # ── Time series ───────────────────────────────────────────────────
        packed = nn.utils.rnn.pack_padded_sequence(
            ts_window,
            ts_lengths.clamp(min=1).cpu(),
            batch_first    = True,
            enforce_sorted = False,
        )
        _, (hn, _) = self.ts_lstm(packed)
        h_ts = hn[-1]                               # last-layer hidden (B, H)

        # ── Static ────────────────────────────────────────────────────────
        h_static = self.static_encoder(static_feat) # (B, static_hidden)

        # ── Text  (swap zeros for learnable NO_NOTE embedding) ────────────
        # When flag=1: effective_emb = 0 + 1 * no_note_emb = no_note_emb
        # When flag=0: effective_emb = text_emb + 0         = text_emb
        flag         = no_note_flag.unsqueeze(-1)                  # (B, 1)
        eff_text_emb = text_emb + flag * self.no_note_emb          # (B, D_text)
        h_text       = self.text_encoder(eff_text_emb)             # (B, text_hidden)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = torch.cat([h_ts, h_static, h_text], dim=-1)       # (B, fusion_in)
        pred  = self.fusion(fused).squeeze(-1)                     # (B,)
        return pred