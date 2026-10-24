"""
DeepLabV3+ architecture for 2-channel Sentinel-1 SAR flood segmentation.

Features:
- ResNet-34 encoder adapted to 2 input channels (VV, VH)
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
- enable_mc_dropout() sets dropout layers to train mode while keeping BatchNorm in eval mode
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FloodDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ segmentation model with ResNet-34 encoder and ASPP module.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 2,
        classes: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Output raw unscaled logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 2, H, W).
        Returns:
            Logits of shape (B, 1, H, W).
        """
        return self.model(x)

    def enable_mc_dropout(self):
        """
        Activates dropout layers for Monte Carlo test-time sampling,
        freezing BatchNorm layers in eval mode.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Standard deterministic probability map (sigmoid applied)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
