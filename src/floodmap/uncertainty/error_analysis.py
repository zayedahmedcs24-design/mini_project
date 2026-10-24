"""
Error detection and uncertainty validation for flood extent segmentation.

Validates the core research hypothesis:
Does predictive uncertainty correlate with actual model errors under domain shift?
Computes Error-Detection AUROC (treating prediction errors as the positive class).
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def compute_error_detection_auroc(
    probs: Union[np.ndarray, Any],
    targets: Union[np.ndarray, Any],
    uncertainty_map: Union[np.ndarray, Any],
    threshold: float = 0.5,
    ignore_index: int = -1,
) -> Dict[str, Any]:
    """
    Computes Error-vs-Uncertainty AUROC.

    Args:
        probs: Array of predicted probabilities in [0.0, 1.0].
        targets: Array of binary ground truth labels (0, 1, or ignore_index).
        uncertainty_map: Array of uncertainty values (e.g., entropy, variance, mutual info).
        threshold: Decision threshold for binary flood classification (default 0.5).
        ignore_index: Pixel value to exclude (-1).

    Returns:
        Dict with keys:
            - "auroc": Area Under the ROC Curve for error detection in [0.5, 1.0]
            - "total_valid_pixels": Number of evaluated pixels
            - "error_pixels": Number of misclassified pixels (FP + FN)
            - "error_rate": Percentage of valid pixels that are misclassified
            - "fpr": False Positive Rates for ROC curve
            - "tpr": True Positive Rates for ROC curve
    """
    if hasattr(probs, "cpu"):
        probs = probs.cpu().numpy()
    if hasattr(targets, "cpu"):
        targets = targets.cpu().numpy()
    if hasattr(uncertainty_map, "cpu"):
        uncertainty_map = uncertainty_map.cpu().numpy()

    p_flat = np.asarray(probs, dtype=np.float32).ravel()
    y_flat = np.asarray(targets, dtype=np.int32).ravel()
    u_flat = np.asarray(uncertainty_map, dtype=np.float32).ravel()

    # Exclude ignore pixels and NaNs
    valid = (y_flat != ignore_index) & (~np.isnan(p_flat)) & (~np.isnan(u_flat))
    p = p_flat[valid]
    y = y_flat[valid]
    u = u_flat[valid]

    if len(p) == 0:
        return {
            "auroc": 0.5,
            "total_valid_pixels": 0,
            "error_pixels": 0,
            "error_rate": 0.0,
            "fpr": [],
            "tpr": [],
        }

    # Binary prediction at given threshold
    preds = (p >= threshold).astype(np.int32)
    # Ground truth error: 1 if prediction does not match label, 0 if correct
    errors = (preds != y).astype(np.int32)

    error_count = int(np.sum(errors))
    total_valid = len(y)
    error_rate = (error_count / total_valid * 100.0) if total_valid > 0 else 0.0

    # If no errors or all errors, AUROC is undefined/0.5
    if error_count == 0 or error_count == total_valid:
        return {
            "auroc": 0.5,
            "total_valid_pixels": total_valid,
            "error_pixels": error_count,
            "error_rate": error_rate,
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
        }

    auroc = float(roc_auc_score(errors, u))
    fpr, tpr, _ = roc_curve(errors, u)

    # Subsample curve points for efficient JSON serialization if very large
    if len(fpr) > 200:
        indices = np.linspace(0, len(fpr) - 1, 200, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]

    return {
        "auroc": auroc,
        "total_valid_pixels": total_valid,
        "error_pixels": error_count,
        "error_rate": error_rate,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def plot_error_detection_roc(
    error_results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Error Detection ROC Curve (Entropy vs Misclassification)",
):
    """
    Plots the ROC curve for identifying misclassified pixels using uncertainty.
    """
    output_path = Path(output_path)
    fpr = error_results["fpr"]
    tpr = error_results["tpr"]
    auroc = error_results["auroc"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color="#2ca25f", lw=2.5, label=f"ROC Curve (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUROC = 0.500)")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Alarm Rate (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Error Detection Rate (True Positive Rate)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SAVED] Error detection ROC plot saved to: {output_path}")
