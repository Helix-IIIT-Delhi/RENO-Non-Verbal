"""RENO Model Architecture"""

import torch
import torch.nn as nn
from .fusion import RenyiFusion


class RENOModel(nn.Module):
    """
    RENO: Rényi Divergence-based Heterogeneous Foundation Model Fusion

    Fuses Mamba-based and other audio foundation models for non-verbal emotion recognition.

    Args:
        num_emotions: Number of emotion classes
        mamba_dim: Mamba foundation model dimension
        other_fm_dim: Other foundation model dimension
        hidden_dim: Hidden dimension
        use_mamba: Enable Mamba models
        use_other_fm: Enable other foundation models
    """

    def __init__(
        self,
        num_emotions: int = 8,
        mamba_dim: int = 768,
        other_fm_dim: int = 768,
        hidden_dim: int = 512,
        use_mamba: bool = True,
        use_other_fm: bool = True,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.use_mamba = use_mamba
        self.use_other_fm = use_other_fm

        # Feature projections
        if use_mamba:
            self.mamba_proj = nn.Linear(mamba_dim, hidden_dim)
        if use_other_fm:
            self.other_fm_proj = nn.Linear(other_fm_dim, hidden_dim)

        # Fusion module
        self.fusion = RenyiFusion(
            hidden_dim,
            use_mamba=use_mamba,
            use_other_fm=use_other_fm
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, mamba_feat=None, other_fm_feat=None):
        """Forward pass for non-verbal emotion recognition"""
        # Project features
        if mamba_feat is not None and self.use_mamba:
            mamba = self.mamba_proj(mamba_feat)
        else:
            mamba = None

        if other_fm_feat is not None and self.use_other_fm:
            other_fm = self.other_fm_proj(other_fm_feat)
        else:
            other_fm = None

        # Fuse representations
        fused = self.fusion(mamba, other_fm)

        # Global pooling if sequence
        if fused.dim() > 2:
            fused = fused.mean(dim=1)

        # Classify
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=1),
            "confidence": torch.softmax(logits, dim=1).max(dim=1).values,
            "emotion": torch.argmax(logits, dim=1),
        }
