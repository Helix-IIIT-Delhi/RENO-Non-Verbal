"""Rényi Divergence-based Heterogeneous Fusion"""

import torch
import torch.nn as nn
from .renyi import RenyiDivergenceLoss


class RenyiFusion(nn.Module):
    """
    Heterogeneous fusion of Mamba and other foundation models using Rényi divergence.
    """

    def __init__(
        self,
        d_model: int = 512,
        use_mamba: bool = True,
        use_other_fm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_mamba = use_mamba
        self.use_other_fm = use_other_fm

        if use_mamba and use_other_fm:
            # Gating networks for adaptive weighting
            self.gate_mamba = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )

            self.gate_other = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )

            # Rényi divergence loss
            self.renyi_loss = RenyiDivergenceLoss(alpha=0.5)

            # Self-attention for refinement
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )

    def forward(self, mamba_feat, other_fm_feat):
        """Fuse using Rényi divergence alignment"""

        if self.use_mamba and self.use_other_fm:
            # Compute gating weights
            gate_m = self.gate_mamba(mamba_feat)
            gate_o = self.gate_other(other_fm_feat)

            # Weighted fusion
            fused = gate_m * mamba_feat + (1 - gate_o) * other_fm_feat

            # Self-attention refinement
            if fused.dim() == 2:
                fused = fused.unsqueeze(1)

            attn_out, _ = self.self_attention(fused, fused, fused)
            fused = fused + attn_out

            if attn_out.size(1) == 1:
                fused = fused.squeeze(1)

            return fused

        elif self.use_mamba:
            return mamba_feat
        elif self.use_other_fm:
            return other_fm_feat
        else:
            raise ValueError("At least one feature stream must be enabled")
