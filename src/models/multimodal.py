"""
multimodal.py – Stage 2 multimodal ICU RLOS model.

Supports three modality variants controlled by `variant`:
  "ts_static"      : frozen TS encoder + static branch
  "ts_text"        : frozen TS encoder + text branch
  "ts_static_text" : frozen TS encoder + static + text  (full multimodal)

The TS encoder is always loaded from a Stage 1 checkpoint and frozen by
default, guaranteeing a fair isolated contribution-analysis of each modality.

Usage
-----
    encoder = build_ts_encoder(input_dim=ts_prep.feature_dim, **enc_cfg)
    model   = MultimodalICUModel(
        ts_encoder      = encoder,
        static_feat_dim = static_prep.feature_dim,
        text_emb_dim    = text_prep.embedding_dim,
        variant         = "ts_static_text",
        freeze_ts       = True,
    )
    model.load_ts_encoder("checkpoints/ts_lstm_best.pt")
"""

import torch
import torch.nn as nn

VALID_VARIANTS = ("ts_static", "ts_text", "ts_static_text")


class MultimodalICUModel(nn.Module):

    def __init__(
        self,
        ts_encoder:      nn.Module,
        static_feat_dim: int,
        text_emb_dim:    int,
        variant:         str   = "ts_static_text",
        static_hidden:   int   = 32,
        text_hidden:     int   = 64,
        fusion_hidden:   int   = 128,
        dropout:         float = 0.2,
        freeze_ts:       bool  = True,
    ):
        super().__init__()
        assert variant in VALID_VARIANTS, \
            f"variant must be one of {VALID_VARIANTS}, got '{variant}'"

        self.variant      = variant
        self.ts_encoder   = ts_encoder
        ts_hidden         = ts_encoder.hidden_dim
        fusion_in         = ts_hidden

        # ── Static branch (optional) ──────────────────────────────────────
        self.use_static = "static" in variant
        if self.use_static:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_feat_dim, static_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_in += static_hidden

        # ── Text branch (optional) ────────────────────────────────────────
        self.use_text = "text" in variant
        if self.use_text:
            self.no_note_emb  = nn.Parameter(torch.randn(text_emb_dim) * 0.01)
            self.text_encoder = nn.Sequential(
                nn.Linear(text_emb_dim, text_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_in += text_hidden

        # ── Fusion head ───────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

        if freeze_ts:
            self.freeze_ts_encoder()

    # ── Weight loading & freezing ─────────────────────────────────────────────

    def load_ts_encoder(self, ckpt_path: str) -> None:
        """Load TSEncoder weights from a Stage 1 TSOnlyModel checkpoint."""
        state     = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc_state = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.ts_encoder.load_state_dict(enc_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"TS encoder load mismatch.\n"
                f"  Missing   : {missing}\n"
                f"  Unexpected: {unexpected}"
            )
        print(f"[MultimodalICUModel] TS encoder loaded ← {ckpt_path}")

    def freeze_ts_encoder(self) -> None:
        for p in self.ts_encoder.parameters():
            p.requires_grad = False
        print("[MultimodalICUModel] TS encoder frozen.")

    def unfreeze_ts_encoder(self) -> None:
        for p in self.ts_encoder.parameters():
            p.requires_grad = True
        print("[MultimodalICUModel] TS encoder unfrozen.")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ts_window:    torch.Tensor,          # (B, W, D_ts)
        ts_lengths:   torch.Tensor,          # (B,)
        static_feat:  torch.Tensor | None,   # (B, D_static) or None
        text_emb:     torch.Tensor | None,   # (B, D_text)   or None
        no_note_flag: torch.Tensor | None,   # (B,)          or None
    ) -> torch.Tensor:                       # (B,)  log1p(RLOS)

        # TS (frozen branch uses no_grad internally)
        ts_frozen = not any(p.requires_grad for p in self.ts_encoder.parameters())
        ctx = torch.no_grad() if ts_frozen else torch.enable_grad()
        with ctx:
            h_ts = self.ts_encoder(ts_window, ts_lengths)      # (B, ts_hidden)

        parts = [h_ts]

        if self.use_static:
            parts.append(self.static_encoder(static_feat))

        if self.use_text:
            flag         = no_note_flag.unsqueeze(-1)
            eff_emb      = text_emb + flag * self.no_note_emb
            parts.append(self.text_encoder(eff_emb))

        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused).squeeze(-1)                   # (B,)