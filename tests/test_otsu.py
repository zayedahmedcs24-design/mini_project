"""
Unit tests for Otsu baseline segmentation module.
Verifies that low radar backscatter is correctly mapped to water (1).
"""

import pytest
import numpy as np
import torch

from src.floodmap.models.baselines import otsu_segment_chip, GlobalOtsuBaseline


def test_otsu_bimodal_separation():
    """Verify that a synthetic bimodal distribution (dark water vs bright land) is segmented."""
    # Create synthetic SAR chip where bottom half is water (-25 dB) and top half is land (-10 dB)
    s1 = np.zeros((2, 64, 64), dtype=np.float32)
    s1[0, :32, :] = -10.0  # Land VV
    s1[0, 32:, :] = -25.0  # Water VV
    s1[1, :32, :] = -15.0  # Land VH
    s1[1, 32:, :] = -30.0  # Water VH

    pred_mask, thresh = otsu_segment_chip(s1, variant="vh")

    # Water is where band <= thresh
    assert thresh > -30.0 and thresh < -15.0
    # Top half should be dry land (0)
    assert np.all(pred_mask[:32, :] == 0)
    # Bottom half should be water (1)
    assert np.all(pred_mask[32:, :] == 1)


def test_otsu_variants():
    """Verify all 3 variants (vv, vh, mean) execute without error."""
    s1 = np.random.uniform(-30.0, -5.0, size=(2, 64, 64)).astype(np.float32)

    for var in ["vv", "vh", "mean"]:
        mask, thresh = otsu_segment_chip(s1, variant=var)
        assert mask.shape == (64, 64)
        assert set(np.unique(mask)).issubset({0, 1})
        assert isinstance(thresh, float)


def test_otsu_torch_tensor_input():
    """Verify PyTorch tensor input is accepted."""
    s1 = torch.randn(2, 64, 64)
    mask, thresh = otsu_segment_chip(s1, variant="mean")
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (64, 64)
