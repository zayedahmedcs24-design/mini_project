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


def test_global_otsu_baseline():
    """Verify GlobalOtsuBaseline fits once on mock dataset ignoring -1 labels."""
    class MockDataset:
        def __init__(self):
            # Create synthetic samples: 10 water, 10 land, with ignore regions
            s1 = np.zeros((2, 32, 32), dtype=np.float32)
            s1[:, :16, :] = -5.0   # Land
            s1[:, 16:, :] = -25.0  # Water
            label = np.zeros((32, 32), dtype=np.int64)
            label[16:, :] = 1
            label[0:4, :] = -1  # Ignore region
            self.sample = (torch.from_numpy(s1), torch.from_numpy(label), "mock_chip")
            self.samples = [("mock_chip", "dummy_s1", "dummy_label")]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.sample

    mock_ds = MockDataset()
    # Mock existence check for testing
    from unittest.mock import patch
    with patch("pathlib.Path.exists", return_value=True):
        model = GlobalOtsuBaseline(variant="mean")
        thresh = model.fit(mock_ds)
        assert -25.0 < thresh < -5.0
        mask = model.predict(mock_ds.sample[0])
        assert mask.shape == (32, 32)
        assert mask[20, 0] == 1  # Water
        assert mask[10, 0] == 0  # Land
