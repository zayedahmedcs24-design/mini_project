"""
Probability calibration and reliability assessment for flood extent mapping.

Provides:
- Expected Calibration Error (ECE) computation with customizable binning
- Maximum Calibration Error (MCE)
- Publication-quality Reliability Diagram plotting
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import matplotlib.pyplot as plt


def compute_ece(
    probs: Union[np.ndarray, Any],
    targets: Union[np.ndarray, Any],
    num_bins: int = 15,
    ignore_index: int = -1,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) for binary segmentation.

    Args:
        probs: Array of predicted probabilities in [0.0, 1.0].
        targets: Array of binary ground truth labels (0, 1, or ignore_index).
        num_bins: Number of equal-width probability bins (default 15).
        ignore_index: Pixel value to exclude from calibration analysis (-1).

    Returns:
        Dict containing:
            - "ece": Expected Calibration Error in [0.0, 1.0]
            - "mce": Maximum Calibration Error
            - "bin_confidences": List of average confidence per bin
            - "bin_accuracies": List of empirical accuracy per bin
            - "bin_counts": List of pixel counts per bin
            - "total_evaluated_pixels": Total valid non-ignored pixels
    """
    if hasattr(probs, "cpu"):
        probs = probs.cpu().numpy()
    if hasattr(targets, "cpu"):
        targets = targets.cpu().numpy()

    probs_flat = np.asarray(probs, dtype=np.float32).ravel()
    targets_flat = np.asarray(targets, dtype=np.int32).ravel()

    # Filter out ignore pixels and NaNs
    valid_mask = (targets_flat != ignore_index) & (~np.isnan(probs_flat))
    p = probs_flat[valid_mask]
    y = targets_flat[valid_mask]

    n_total = len(p)
    if n_total == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "bin_confidences": [],
            "bin_accuracies": [],
            "bin_counts": [],
            "total_evaluated_pixels": 0,
        }

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for m in range(num_bins):
        low, high = bin_edges[m], bin_edges[m + 1]
        if m == num_bins - 1:
            in_bin = (p >= low) & (p <= high)
        else:
            in_bin = (p >= low) & (p < high)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            avg_conf = float(np.mean(p[in_bin]))
            avg_acc = float(np.mean(y[in_bin]))
            gap = abs(avg_acc - avg_conf)

            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)

            ece += (count / n_total) * gap
            mce = max(mce, gap)
        else:
            bin_confidences.append(float((low + high) / 2.0))
            bin_accuracies.append(0.0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
        "total_evaluated_pixels": n_total,
    }


def plot_reliability_diagram(
    calibration_results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Reliability Diagram (Calibration Curve)",
):
    """
    Plots and saves a publication-quality reliability diagram.
    """
    output_path = Path(output_path)
    confs = calibration_results["bin_confidences"]
    accs = calibration_results["bin_accuracies"]
    ece = calibration_results["ece"]
    mce = calibration_results["mce"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration reference diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Calibration bars
    bin_width = 1.0 / len(confs)
    centers = [(i + 0.5) * bin_width for i in range(len(confs))]
    ax.bar(
        centers,
        accs,
        width=bin_width * 0.9,
        alpha=0.6,
        color="#2b8cbe",
        edgecolor="#08589e",
        label="Observed Accuracy",
    )

    # Shaded calibration gap
    ax.bar(
        centers,
        [abs(a - c) for a, c in zip(accs, confs)],
        bottom=[min(a, c) for a, c in zip(accs, confs)],
        width=bin_width * 0.9,
        alpha=0.35,
        color="#e34a33",
        edgecolor="#b30000",
        hatch="//",
        label="Calibration Gap",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted Probability (Confidence)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy (Fraction of Water)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Metrics annotation box
    ax.text(
        0.05,
        0.88,
        f"ECE = {ece * 100:.2f}%\nMCE = {mce * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SAVED] Reliability diagram saved to: {output_path}")
