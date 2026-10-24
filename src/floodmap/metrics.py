"""
Evaluation metrics for flood mapping with explicit support for ignore labels (-1).

Labels:
    1 = Flooded / Water
    0 = Non-Flooded / Dry Land
   -1 = Ignore / Invalid / Cloud / NoData

CRITICAL RULE:
    Pixels with ground truth == -1 must be strictly excluded from all
    confusion matrix tallies (TP, FP, FN, TN), losses, and metric calculations.
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np
import torch


def compute_confusion_matrix(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    ignore_index: int = -1,
) -> Dict[str, int]:
    """
    Computes confusion matrix counts (TP, FP, FN, TN) for the flood class (1),
    strictly excluding pixels where y_true == ignore_index.

    Args:
        y_pred: Predicted probabilities or binary predictions.
        y_true: Ground truth label array or tensor (contains 0, 1, or ignore_index).
        threshold: Decision threshold for continuous probabilities in [0, 1].
        ignore_index: Integer label for ignored pixels (default: -1).

    Returns:
        Dict with integer keys: 'tp', 'fp', 'fn', 'tn', 'valid_pixels'.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Align dimensions (e.g. (B, 1, H, W) vs (B, H, W))
    if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[1] == 1:
        y_pred = np.squeeze(y_pred, axis=1)
    elif y_true.ndim == y_pred.ndim + 1 and y_true.shape[1] == 1:
        y_true = np.squeeze(y_true, axis=1)

    # Binarize predictions
    if y_pred.dtype in [np.float32, np.float64, np.float16]:
        y_pred_bin = (y_pred >= threshold).astype(np.int64)
    else:
        y_pred_bin = (y_pred > 0).astype(np.int64)

    y_true_int = y_true.astype(np.int64)

    # Valid mask: exclude ignore_index (-1) and any NaNs
    valid_mask = (y_true_int != ignore_index) & (~np.isnan(y_true))
    if not np.any(valid_mask):
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid_pixels": 0}

    pred_v = y_pred_bin[valid_mask]
    true_v = y_true_int[valid_mask]

    tp = int(np.sum((true_v == 1) & (pred_v == 1)))
    fp = int(np.sum((true_v == 0) & (pred_v == 1)))
    fn = int(np.sum((true_v == 1) & (pred_v == 0)))
    tn = int(np.sum((true_v == 0) & (pred_v == 0)))
    valid_pixels = int(np.sum(valid_mask))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": valid_pixels,
    }


def compute_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Calculates segmentation metrics from aggregated confusion matrix counts.

    Returns:
        Dict with keys:
            - flood_iou: IoU for the flood class (TP / (TP + FP + FN))
            - nonflood_iou: IoU for the dry land class (TN / (TN + FP + FN))
            - mean_iou: Average of flood_iou and nonflood_iou
            - flood_f1: Dice / F1-Score for the flood class
            - flood_precision: Precision for the flood class
            - flood_recall: Recall for the flood class
            - accuracy_valid: Raw accuracy on valid pixels (for trap illustration)
    """
    flood_denom = tp + fp + fn
    flood_iou = (tp / (flood_denom + eps)) if flood_denom > 0 else 0.0

    nonflood_denom = tn + fp + fn
    nonflood_iou = (tn / (nonflood_denom + eps)) if nonflood_denom > 0 else 0.0

    mean_iou = 0.5 * (flood_iou + nonflood_iou)

    prec_denom = tp + fp
    flood_precision = (tp / (prec_denom + eps)) if prec_denom > 0 else 0.0

    rec_denom = tp + fn
    flood_recall = (tp / (rec_denom + eps)) if rec_denom > 0 else 0.0

    f1_denom = 2 * tp + fp + fn
    flood_f1 = (2 * tp / (f1_denom + eps)) if f1_denom > 0 else 0.0

    total_valid = tp + fp + fn + tn
    accuracy_valid = (tp + tn) / (total_valid + eps) if total_valid > 0 else 0.0

    return {
        "flood_iou": float(flood_iou),
        "nonflood_iou": float(nonflood_iou),
        "mean_iou": float(mean_iou),
        "flood_f1": float(flood_f1),
        "flood_precision": float(flood_precision),
        "flood_recall": float(flood_recall),
        "accuracy_valid": float(accuracy_valid),
    }


def evaluate_predictions(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """
    Direct end-to-end evaluation of predicted mask against ground truth.
    """
    counts = compute_confusion_matrix(
        y_pred=y_pred,
        y_true=y_true,
        threshold=threshold,
        ignore_index=ignore_index,
    )
    metrics = compute_metrics_from_counts(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        tn=counts["tn"],
    )
    metrics["valid_pixels"] = counts["valid_pixels"]
    metrics["tp"] = counts["tp"]
    metrics["fp"] = counts["fp"]
    metrics["fn"] = counts["fn"]
    metrics["tn"] = counts["tn"]
    return metrics


class MetricTracker:
    """
    Accumulator for calculating dataset-level metrics across batches or chips.
    Aggregates global TP, FP, FN, TN to prevent small-chip divide-by-zero bias,
    while also storing per-chip metrics for distribution analysis (median/IQR).
    """

    def __init__(self, ignore_index: int = -1, threshold: float = 0.5):
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.total_valid = 0
        self.per_chip_metrics = []

    def update(
        self,
        y_pred: Union[torch.Tensor, np.ndarray],
        y_true: Union[torch.Tensor, np.ndarray],
    ):
        """Update counts from a single chip or batch."""
        counts = compute_confusion_matrix(
            y_pred=y_pred,
            y_true=y_true,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
        )
        self.total_tp += counts["tp"]
        self.total_fp += counts["fp"]
        self.total_fn += counts["fn"]
        self.total_tn += counts["tn"]
        self.total_valid += counts["valid_pixels"]

        # Also record per-chip metrics
        chip_metrics = compute_metrics_from_counts(
            tp=counts["tp"],
            fp=counts["fp"],
            fn=counts["fn"],
            tn=counts["tn"],
        )
        self.per_chip_metrics.append(chip_metrics)

    def compute_global(self) -> Dict[str, float]:
        """Compute aggregated metrics over the entire evaluated dataset."""
        metrics = compute_metrics_from_counts(
            tp=self.total_tp,
            fp=self.total_fp,
            fn=self.total_fn,
            tn=self.total_tn,
        )
        metrics["total_valid_pixels"] = self.total_valid
        metrics["tp"] = self.total_tp
        metrics["fp"] = self.total_fp
        metrics["fn"] = self.total_fn
        metrics["tn"] = self.total_tn
        return metrics

    def compute_summary(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Returns global metrics alongside per-chip median and IQR for flood IoU.
        """
        global_m = self.compute_global()
        if not self.per_chip_metrics:
            return global_m

        chip_ious = [m["flood_iou"] for m in self.per_chip_metrics]
        q25, median, q75 = np.percentile(chip_ious, [25, 50, 75])

        return {
            "global": global_m,
            "per_chip_flood_iou": {
                "median": float(median),
                "iqr": float(q75 - q25),
                "q25": float(q25),
                "q75": float(q75),
                "count": len(chip_ious),
            },
        }
