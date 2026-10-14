"""Unit tests for RENO models"""

import torch
from reno import RENOModel


def test_reno_model():
    """Test RENO model initialization and forward pass"""
    model = RENOModel(num_emotions=8)

    # Create dummy inputs
    mamba = torch.randn(4, 768)
    other_fm = torch.randn(4, 768)

    # Forward pass
    output = model(mamba, other_fm)

    assert output['logits'].shape == (4, 8)
    assert output['predictions'].shape == (4,)
    assert output['confidence'].shape == (4,)
    print("✓ RENO model test passed")


if __name__ == '__main__':
    test_reno_model()
