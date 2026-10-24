"""
Unit tests for ignore-aware evaluation metrics.
Verifies mathematically exact behavior, ignore label handling (-1),
the class-imbalance accuracy trap, and PyTorch / NumPy interoperability.
"""

import pytest
import numpy as np
import torch

from src.floodmap.metrics import (
    compute_confusion_matrix,
    compute_metrics_from_counts,
    evaluate_predictions,
    MetricTracker,
)


def test_perfect_prediction():
    """Verify that identical prediction and target yields 1.0 across all positive metrics."""
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])

    metrics = evaluate_predictions(y_pred, y_true, threshold=0.5)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 2
    assert metrics["valid_pixels"] == 4
    assert pytest.approx(metrics["flood_iou"], 1e-6) == 1.0
    assert pytest.approx(metrics["flood_f1"], 1e-6) == 1.0
    assert pytest.approx(metrics["flood_precision"], 1e-6) == 1.0
    assert pytest.approx(metrics["flood_recall"], 1e-6) == 1.0
    assert pytest.approx(metrics["mean_iou"], 1e-6) == 1.0


def test_ignore_index_exclusion():
    """
    CRITICAL CHECK:
    Adding pixels with label -1 (ignore) must NOT change TP, FP, FN, or TN counts.
    """
    y_true_clean = np.array([[1, 0], [0, 1]])
    y_pred_clean = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Ground truth with ignored pixels around it
    y_true_with_ignore = np.array([
        [-1, -1, -1, -1],
        [-1,  1,  0, -1],
        [-1,  0,  1, -1],
        [-1, -1, -1, -1],
    ])
    # Predictions even with random noise on the ignored pixels
    y_pred_with_noise = np.array([
        [0.9, 0.1, 0.8, 0.2],
        [0.4, 1.0, 0.0, 0.7],
        [0.1, 0.0, 1.0, 0.3],
        [0.8, 0.2, 0.6, 0.4],
    ])

    m_clean = evaluate_predictions(y_pred_clean, y_true_clean)
    m_ignore = evaluate_predictions(y_pred_with_noise, y_true_with_ignore, ignore_index=-1)

    assert m_ignore["tp"] == m_clean["tp"] == 2
    assert m_ignore["fp"] == m_clean["fp"] == 0
    assert m_ignore["fn"] == m_clean["fn"] == 0
    assert m_ignore["tn"] == m_clean["tn"] == 2
    assert m_ignore["valid_pixels"] == 4
    assert pytest.approx(m_ignore["flood_iou"], 1e-6) == m_clean["flood_iou"]


def test_predict_all_no_flood_trap():
    """
    Tests the fundamental reason why accuracy is misleading in remote sensing:
    Imbalanced scene: 100 flood pixels (10%), 900 dry land pixels (90%).
    Dummy model predicts all 0s (never detects a flood).
    Accuracy is 90% (0.90), but flood IoU and Recall are EXACTLY 0.0.
    """
    # 100 flood pixels, 900 dry land pixels
    y_true = np.zeros(1000, dtype=np.int64)
    y_true[:100] = 1

    # Predict all 0s
    y_pred = np.zeros(1000, dtype=np.float32)

    metrics = evaluate_predictions(y_pred, y_true, threshold=0.5)

    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 100
    assert metrics["tn"] == 900
    assert pytest.approx(metrics["accuracy_valid"], 1e-6) == 0.90
    assert pytest.approx(metrics["flood_iou"], 1e-6) == 0.0
    assert pytest.approx(metrics["flood_f1"], 1e-6) == 0.0
    assert pytest.approx(metrics["flood_recall"], 1e-6) == 0.0
    assert pytest.approx(metrics["flood_precision"], 1e-6) == 0.0


def test_pytorch_tensor_support():
    """Ensure PyTorch tensors with arbitrary shapes are correctly processed."""
    # Batch of 2, 1 channel, 4x4
    y_true = torch.tensor([[[[1, 0, -1, 0],
                             [0, 1,  0, 0],
                             [0, 0,  1, 1],
                             [-1, 0, 0, 0]]]])
    y_pred = torch.tensor([[[[0.9, 0.1, 0.8, 0.2],
                             [0.1, 0.8, 0.2, 0.1],
                             [0.3, 0.1, 0.7, 0.6],
                             [0.9, 0.1, 0.2, 0.1]]]])

    metrics = evaluate_predictions(y_pred, y_true, threshold=0.5, ignore_index=-1)
    assert isinstance(metrics["flood_iou"], float)
    assert metrics["valid_pixels"] == 14  # 16 - 2 ignored pixels
    assert metrics["tp"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 10
    assert pytest.approx(metrics["flood_iou"], 1e-6) == 1.0


def test_metric_tracker_accumulation():
    """Verify that MetricTracker aggregates multiple chips without division bias."""
    tracker = MetricTracker(ignore_index=-1, threshold=0.5)

    # Chip 1: 5 TP, 0 FP, 0 FN, 5 TN
    t1 = np.array([1]*5 + [0]*5)
    p1 = np.array([1.0]*5 + [0.0]*5)
    tracker.update(p1, t1)

    # Chip 2: 0 TP, 5 FP, 5 FN, 0 TN
    t2 = np.array([1]*5 + [0]*5)
    p2 = np.array([0.0]*5 + [1.0]*5)
    tracker.update(p2, t2)

    summary = tracker.compute_summary()
    glob = summary["global"]

    assert glob["tp"] == 5
    assert glob["fp"] == 5
    assert glob["fn"] == 5
    assert glob["tn"] == 5
    # Global IoU = TP / (TP + FP + FN) = 5 / (5 + 5 + 5) = 5 / 15 = 1/3 ≈ 0.333333
    assert pytest.approx(glob["flood_iou"], 1e-5) == (1.0 / 3.0)

    # Chip 1 flood_iou = 1.0, Chip 2 flood_iou = 0.0 -> median = 0.5
    assert pytest.approx(summary["per_chip_flood_iou"]["median"], 1e-5) == 0.5
