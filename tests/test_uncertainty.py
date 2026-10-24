"""
Unit tests for uncertainty quantification modules:
- MC Dropout stochastic sampling
- Expected Calibration Error (ECE)
- Error detection AUROC
"""

import numpy as np
import pytest
import torch

from src.floodmap.models.unet import FloodUNet
from src.floodmap.uncertainty.mc_dropout import predict_mc_dropout
from src.floodmap.uncertainty.calibration import compute_ece
from src.floodmap.uncertainty.error_analysis import compute_error_detection_auroc


class TestMCDropout:
    def test_mc_dropout_shapes_and_bounds(self):
        model = FloodUNet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=2,
            classes=1,
            decoder_dropout=0.2,
        )
        x = torch.randn(2, 64, 64)
        res = predict_mc_dropout(model, x, num_samples=4, device=torch.device("cpu"))

        assert "mean" in res
        assert "variance" in res
        assert "entropy" in res
        assert "mutual_info" in res
        assert "samples" in res

        assert res["mean"].shape == (64, 64)
        assert res["variance"].shape == (64, 64)
        assert res["entropy"].shape == (64, 64)
        assert res["mutual_info"].shape == (64, 64)
        assert res["samples"].shape == (4, 64, 64)

        # Probabilities bounded in [0, 1]
        assert np.all(res["mean"] >= 0.0) and np.all(res["mean"] <= 1.0)
        # Variance non-negative
        assert np.all(res["variance"] >= 0.0)
        # Entropy non-negative and <= 1.0 for binary log2
        assert np.all(res["entropy"] >= 0.0) and np.all(res["entropy"] <= 1.0 + 1e-5)
        # Mutual info non-negative
        assert np.all(res["mutual_info"] >= 0.0)


class TestCalibrationECE:
    def test_ece_perfect_calibration(self):
        # 100 samples with p=0.8 and y=1 for 80, y=0 for 20
        p = np.full(100, 0.8, dtype=np.float32)
        y = np.zeros(100, dtype=np.int32)
        y[:80] = 1

        res = compute_ece(p, y, num_bins=10)
        assert pytest.approx(res["ece"], abs=1e-4) == 0.0

    def test_ece_high_overconfidence(self):
        # Model is 99% confident but label is always 0
        p = np.full(100, 0.99, dtype=np.float32)
        y = np.zeros(100, dtype=np.int32)

        res = compute_ece(p, y, num_bins=10)
        assert pytest.approx(res["ece"], abs=0.05) == 0.99

    def test_ece_ignore_index_exclusion(self):
        p = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        y = np.array([0, 1, -1], dtype=np.int32)

        res = compute_ece(p, y, num_bins=5, ignore_index=-1)
        assert res["total_evaluated_pixels"] == 2


class TestErrorDetectionAUROC:
    def test_perfect_error_detection_auroc(self):
        # 4 pixels:
        # Ground truth: 0, 0, 1, 1
        # Predictions:  0, 1, 1, 0  => errors at index 1 and 3
        # Uncertainty: 0.1, 0.9, 0.2, 0.8 => higher uncertainty on errors
        probs = np.array([0.1, 0.9, 0.8, 0.2])
        targets = np.array([0, 0, 1, 1])
        uncertainty = np.array([0.1, 0.9, 0.2, 0.8])

        res = compute_error_detection_auroc(probs, targets, uncertainty, threshold=0.5)
        assert res["error_pixels"] == 2
        assert res["total_valid_pixels"] == 4
        assert pytest.approx(res["auroc"], abs=1e-5) == 1.0

    def test_error_detection_ignore_index(self):
        probs = np.array([0.1, 0.9, 0.5])
        targets = np.array([0, 0, -1])
        uncertainty = np.array([0.1, 0.9, 0.5])

        res = compute_error_detection_auroc(probs, targets, uncertainty, threshold=0.5, ignore_index=-1)
        assert res["total_valid_pixels"] == 2
