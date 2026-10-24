"""
Unit tests for deep learning model architectures and compound loss function.
Verifies forward pass shapes, 2-channel adaptation, MC Dropout modes, and loss gradients.
"""

import pytest
import torch
import torch.nn as nn

from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.models.losses import MaskedBCEDiceLoss


def test_unet_forward_and_shapes():
    """Verify U-Net accepts 2-channel input and outputs correct binary logits shape."""
    model = FloodUNet(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1)
    x = torch.randn(2, 2, 64, 64)
    logits = model(x)

    assert logits.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {logits.shape}"

    probs = model.predict_probability(x)
    assert probs.shape == (2, 1, 64, 64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_unet_mc_dropout_activation():
    """Verify enable_mc_dropout() sets dropout to train while BatchNorm stays in eval mode."""
    model = FloodUNet(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, decoder_dropout=0.2)
    model.enable_mc_dropout()

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            assert m.training is True, "Dropout layer should be in training mode for MC sampling"
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            assert m.training is False, "BatchNorm layer must remain in eval mode during MC sampling"


def test_deeplabv3p_forward_and_shapes():
    """Verify DeepLabV3+ accepts 2-channel input and outputs correct binary logits shape."""
    model = FloodDeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1)
    x = torch.randn(2, 2, 64, 64)
    logits = model(x)

    assert logits.shape == (2, 1, 64, 64), f"Expected (2, 1, 64, 64), got {logits.shape}"

    probs = model.predict_probability(x)
    assert probs.shape == (2, 1, 64, 64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_masked_bce_dice_loss_gradients():
    """Verify compound loss ignores -1 pixels and computes valid gradients."""
    criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=1.0, ignore_index=-1)

    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 32, 32))
    # Inject ignore labels
    target[0, :8, :] = -1
    target[1, 24:, :] = -1

    loss = criterion(logits, target)
    assert not torch.isnan(loss), "Loss computed NaN"
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "Found NaN gradients"
