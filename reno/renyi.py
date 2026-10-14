"""Rényi Divergence Loss for Distribution Alignment"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenyiDivergenceLoss(nn.Module):
    """
    Rényi divergence for distribution alignment.

    D_α(P||Q) = (1/(α-1)) * log(∫ p(x)^α q(x)^(1-α) dx)

    For empirical distributions:
    D_α(P||Q) = (1/(α-1)) * log(Σ_i p_i^α q_i^(1-α))
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8):
        super().__init__()
        assert alpha > 0 and alpha != 1, "alpha must be > 0 and != 1"
        self.alpha = alpha
        self.eps = eps

    def forward(self, p_dist, q_dist):
        """
        Compute Rényi divergence between two distributions.

        Args:
            p_dist: Distribution from model P [batch, ...]
            q_dist: Distribution from model Q [batch, ...]

        Returns:
            Rényi divergence scalar
        """
        # Normalize distributions properly
        p = F.softmax(p_dist.view(p_dist.size(0), -1), dim=-1)
        q = F.softmax(q_dist.view(q_dist.size(0), -1), dim=-1)

        # Ensure numerical stability
        p = torch.clamp(p, min=self.eps)
        q = torch.clamp(q, min=self.eps)

        # Compute Rényi divergence
        # D_α(P||Q) = (1/(α-1)) * log(Σ p^α * q^(1-α))
        pq_weighted = (p ** self.alpha) * (q ** (1 - self.alpha))
        sum_pq = torch.sum(pq_weighted, dim=-1)

        # Add epsilon for numerical stability
        sum_pq = torch.clamp(sum_pq, min=self.eps)

        # Compute final divergence
        divergence = torch.log(sum_pq)

        # Divide by (α-1) with proper handling of sign
        # When α < 1, (α-1) is negative, so we use absolute value and negate
        # to ensure divergence is non-negative
        alpha_factor = 1.0 / abs(self.alpha - 1)
        if self.alpha < 1:
            loss = -alpha_factor * torch.mean(divergence)
        else:
            loss = alpha_factor * torch.mean(divergence)

        return loss
