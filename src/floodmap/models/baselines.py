"""
Classical baseline segmentation models for SAR flood mapping.

Includes:
1. Otsu thresholding on VV polarization
2. Otsu thresholding on VH polarization
3. Otsu thresholding on Dual-Pol Mean: (VV + VH) / 2

In SAR imagery, standing water exhibits low backscatter due to specular reflection.
Therefore:
    pixels <= threshold  --> Water / Flood (1)
    pixels >  threshold  --> Dry Land (0)
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
from skimage.filters import threshold_otsu
import torch


def otsu_segment_chip(
    s1_chip: Union[np.ndarray, torch.Tensor],
    variant: str = "vh",
    threshold_override: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Performs Otsu thresholding on a single 2-channel SAR chip.

    Args:
        s1_chip: 2-band SAR array of shape (2, H, W) or (H, W, 2), where band 0 = VV, band 1 = VH.
                 Values can be raw dB or normalized dB.
        variant: 'vv', 'vh', or 'mean'.
        threshold_override: If provided, uses this fixed threshold instead of computing per-chip.

    Returns:
        Tuple of (binary_pred_mask, threshold_value).
        binary_pred_mask: uint8 array of shape (H, W) with values {0, 1}.
    """
    if isinstance(s1_chip, torch.Tensor):
        s1_chip = s1_chip.detach().cpu().numpy()

    # Ensure shape is (2, H, W)
    if s1_chip.ndim == 3 and s1_chip.shape[2] == 2:
        s1_chip = np.transpose(s1_chip, (2, 0, 1))

    # Extract chosen polarization band
    if variant == "vv":
        band = s1_chip[0]
    elif variant == "vh":
        band = s1_chip[1]
    elif variant == "mean":
        band = 0.5 * (s1_chip[0] + s1_chip[1])
    else:
        raise ValueError(f"Unknown variant '{variant}'. Must be 'vv', 'vh', or 'mean'.")

    # Filter out NaNs and Infs for threshold determination
    valid_pixels = band[np.isfinite(band)]
    if len(valid_pixels) == 0:
        # Fallback if entire chip is invalid
        return np.zeros_like(band, dtype=np.uint8), 0.0

    if threshold_override is not None:
        thresh = threshold_override
    else:
        try:
            thresh = float(threshold_otsu(valid_pixels))
        except Exception:
            thresh = float(np.median(valid_pixels))

    # Water has low backscatter (dark pixels in radar)
    pred_mask = (band <= thresh).astype(np.uint8)

    return pred_mask, thresh


class GlobalOtsuBaseline:
    """
    Calibrates a single global threshold from the training split histogram,
    and applies it uniformly to validation and test chips.
    """

    def __init__(self, variant: str = "vh"):
        self.variant = variant
        self.global_threshold: Optional[float] = None

    def fit(self, dataset) -> float:
        """
        Samples pixel values across dataset to compute a global Otsu threshold.
        """
        sampled_values = []
        # Sample evenly across chips
        step = max(1, len(dataset) // 50)
        for idx in range(0, len(dataset), step):
            s1, _, _ = dataset[idx]
            if isinstance(s1, torch.Tensor):
                s1 = s1.numpy()
            if self.variant == "vv":
                band = s1[0]
            elif self.variant == "vh":
                band = s1[1]
            else:
                band = 0.5 * (s1[0] + s1[1])
            valid = band[np.isfinite(band)]
            if len(valid) > 0:
                # Subsample 5,000 pixels per chip for speed
                subsample = np.random.choice(valid, size=min(5000, len(valid)), replace=False)
                sampled_values.append(subsample)

        all_vals = np.concatenate(sampled_values)
        self.global_threshold = float(threshold_otsu(all_vals))
        return self.global_threshold

    def predict(self, s1_chip: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if self.global_threshold is None:
            raise RuntimeError("Global threshold has not been fitted. Call fit() first.")
        mask, _ = otsu_segment_chip(
            s1_chip,
            variant=self.variant,
            threshold_override=self.global_threshold,
        )
        return mask
