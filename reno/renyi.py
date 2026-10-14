"""Rényi Divergence Loss for Distribution Alignment"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenyiDivergenceLoss(nn.Module):
    """
    Rényi divergence loss for optimal alignment of foundation model distributions.

    D_α(P||Q) = (1/(α-1)) * log(Σ p(x)^α q(x)^(1-α))
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, p_dist, q_dist):
        """
        Compute Rényi divergence between two distributions.

        Args:
            p_dist: Distribution from Mamba models [batch, features]
            q_dist: Distribution from other foundation models [batch, features]

        Returns:
            Rényi divergence scalar
        """
        # Normalize distributions
        p = F.softmax(p_dist, dim=-1)
        q = F.softmax(q_dist, dim=-1)

        # Compute Rényi divergence
        pq = (p ** self.alpha) * (q ** (1 - self.alpha))
        divergence = torch.log(torch.sum(pq, dim=-1) + self.eps)

        # Average over batch
        loss = torch.mean(divergence) / (self.alpha - 1) if self.alpha != 1 else divergence.mean()

        return loss
