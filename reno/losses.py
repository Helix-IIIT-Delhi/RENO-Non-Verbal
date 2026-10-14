"""Training Losses for RENO"""

import torch
import torch.nn as nn


class RENOLoss(nn.Module):
    """Combined loss: classification + Rényi divergence"""

    def __init__(self, lambda_renyi: float = 0.1):
        super().__init__()
        self.lambda_renyi = lambda_renyi
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, renyi_div=0.0):
        """Compute combined loss"""
        # Classification loss
        ce = self.ce_loss(logits, targets)

        # Rényi regularization
        renyi = renyi_div if renyi_div > 0 else torch.tensor(0.0)

        # Combined loss
        total_loss = ce + self.lambda_renyi * renyi

        return total_loss
