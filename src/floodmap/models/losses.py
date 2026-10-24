"""
Loss functions for flood segmentation with strict support for ignore labels (-1).

Compound Loss:
    L_total = alpha * L_BCE_masked + beta * L_Dice_masked
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft-Dice Loss,
    strictly ignoring pixels where target == ignore_index (-1).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 1.0,
        ignore_index: int = -1,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, H, W) or (B, 1, H, W) with {0, 1, -1}.
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Create binary valid mask
        valid_mask = (target != self.ignore_index).float()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target_clean = torch.where(target == self.ignore_index, torch.zeros_like(target), target).float()

        # 1. Masked BCE Loss
        bce_unreduced = F.binary_cross_entropy_with_logits(logits, target_clean, reduction="none")
        bce_loss = (bce_unreduced * valid_mask).sum() / (valid_mask.sum() + 1e-7)

        # 2. Masked Soft-Dice Loss
        probs = torch.sigmoid(logits)
        probs_v = probs * valid_mask
        target_v = target_clean * valid_mask

        intersection = (probs_v * target_v).sum(dim=(2, 3))
        union = probs_v.sum(dim=(2, 3)) + target_v.sum(dim=(2, 3))

        dice_per_batch = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_per_batch.mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss
