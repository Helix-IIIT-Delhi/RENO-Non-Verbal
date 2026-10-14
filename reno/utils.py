"""Utility Functions for RENO"""

import torch
import numpy as np


def extract_nonverbal_features(audio, model_type='mamba'):
    """Extract non-verbal audio features"""
    # Placeholder for feature extraction
    return torch.randn(1, 768)


def normalize_features(features, eps=1e-8):
    """L2 normalize feature vectors"""
    return features / (torch.norm(features, dim=-1, keepdim=True) + eps)


def compute_accuracy(predictions, targets):
    """Compute classification accuracy"""
    correct = (predictions == targets).float().mean()
    return correct.item()


def seed_everything(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
