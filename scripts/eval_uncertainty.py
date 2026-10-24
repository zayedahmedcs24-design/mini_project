"""
Evaluate uncertainty quantification on held-out Bolivia chips:
- MC Dropout sampling (T=20 stochastic forward passes)
- Expected Calibration Error (ECE) and reliability diagram
- Predictive Entropy vs Misclassification Error AUROC detection
- Outputs JSON metrics and publication-ready diagnostic plots
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.uncertainty.mc_dropout import predict_mc_dropout
from src.floodmap.uncertainty.calibration import compute_ece, plot_reliability_diagram
from src.floodmap.uncertainty.error_analysis import compute_error_detection_auroc, plot_error_detection_roc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uncertainty Evaluation Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load Model
    model = FloodUNet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        decoder_dropout=0.2,
    ).to(device)

    checkpoint_path = Path("results/checkpoints/unet_resnet34_seed0_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Dataset: Bolivia held-out split
    bolivia_csv = Path("data/splits/flood_bolivia_data.csv")
    dataset = FloodDataset(split_csv=bolivia_csv, data_dir="data")
    print(f"Evaluating {len(dataset)} Bolivia chips with MC Dropout (T=20)...")

    all_means = []
    all_entropies = []
    all_targets = []

    for i in range(len(dataset)):
        s1, target, chip_id = dataset[i]
        # MC Dropout forward pass
        mc_res = predict_mc_dropout(model, s1, num_samples=20, device=device)

        mean_p = mc_res["mean"]
        entropy = mc_res["entropy"]
        t_np = target.numpy()

        all_means.append(mean_p.ravel())
        all_entropies.append(entropy.ravel())
        all_targets.append(t_np.ravel())

    # Concatenate all pixels
    means_cat = np.concatenate(all_means)
    entropies_cat = np.concatenate(all_entropies)
    targets_cat = np.concatenate(all_targets)

    # 1. Calibration Assessment (ECE)
    calib_res = compute_ece(means_cat, targets_cat, num_bins=15, ignore_index=-1)
    print(f"\n[CALIBRATION] Bolivia Expected Calibration Error (ECE): {calib_res['ece'] * 100:.2f}%")
    print(f"[CALIBRATION] Bolivia Maximum Calibration Error (MCE): {calib_res['mce'] * 100:.2f}%")

    calib_plot_path = Path("results/calibration_bolivia.png")
    plot_reliability_diagram(
        calib_res,
        output_path=calib_plot_path,
        title="Reliability Diagram — Bolivia Held-Out Split (U-Net Seed 0)",
    )

    # 2. Error Detection Assessment (Entropy vs Prediction Error)
    error_res = compute_error_detection_auroc(
        probs=means_cat,
        targets=targets_cat,
        uncertainty_map=entropies_cat,
        threshold=0.5,
        ignore_index=-1,
    )
    print(f"\n[ERROR DETECTION] Misclassified Pixels: {error_res['error_pixels']:,} / {error_res['total_valid_pixels']:,} ({error_res['error_rate']:.2f}%)")
    print(f"[ERROR DETECTION] Error-vs-Uncertainty AUROC: {error_res['auroc']:.4f}")

    roc_plot_path = Path("results/error_detection_bolivia.png")
    plot_error_detection_roc(
        error_res,
        output_path=roc_plot_path,
        title="Error Detection ROC Curve — Bolivia Held-Out (Entropy vs Misclassification)",
    )

    # Save summary JSON
    results_json = Path("results/uncertainty_bolivia.json")
    summary = {
        "split": "bolivia",
        "num_chips": len(dataset),
        "mc_samples": 20,
        "ece": calib_res["ece"],
        "mce": calib_res["mce"],
        "error_detection_auroc": error_res["auroc"],
        "error_rate": error_res["error_rate"],
        "total_valid_pixels": error_res["total_valid_pixels"],
        "error_pixels": error_res["error_pixels"],
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] Uncertainty evaluation results written to: {results_json}")


if __name__ == "__main__":
    main()
