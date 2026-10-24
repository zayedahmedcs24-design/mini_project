"""
U-Net architecture for 2-channel Sentinel-1 SAR flood segmentation.

Features:
- ResNet-34 encoder adapted to 2 input channels (VV, VH)
- decoder_dropout=0.2 for Monte Carlo Dropout at inference time
- enable_mc_dropout() sets dropout layers to train mode while keeping BatchNorm in eval mode
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FloodUNet(nn.Module):
    """
    U-Net segmentation model with ResNet-34 encoder and Monte Carlo Dropout support.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 2,
        classes: int = 1,
        decoder_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_dropout_rate = decoder_dropout

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # Output raw unscaled logits
            decoder_dropout=decoder_dropout,
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
        while strictly freezing BatchNorm layers in eval mode to prevent
        stochastic drift in running mean/variance.
        """
        self.eval()  # Freeze BatchNorm and other layers
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()  # Activate dropout

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Standard deterministic probability map (sigmoid applied)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
