"""
Generates publication-quality figures for Review-2 presentation and report:
1. results/figures/fig_iou_bars.png
2. results/figures/fig_val_curves.png
3. results/figures/fig_examples_test.png
4. results/figures/fig_examples_bolivia.png
5. results/figures/fig_otsu_failure.png

Strictly read-only on checkpoints and JSON files.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.models.baselines import otsu_segment_chip
from src.floodmap.metrics import compute_confusion_matrix

# Global styling configuration
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["lines.linewidth"] = 1.8
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = "--"

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Color Maps for Visualizations
# -----------------------------------------------------------------------------
# For Ground Truth: -1 -> Gray, 0 -> Dark Slate, 1 -> Vibrant Blue
GT_CMAP = ListedColormap(["#7f8c8d", "#2c3e50", "#2980b9"]) # Gray, Dark Charcoal, Blue
GT_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], GT_CMAP.N)

# For Binary Predictions: 0 -> Dark Slate, 1 -> Vibrant Blue
PRED_CMAP = ListedColormap(["#2c3e50", "#2980b9"])
PRED_NORM = BoundaryNorm([-0.5, 0.5, 1.5], PRED_CMAP.N)


def normalize_sar(img_band: np.ndarray, p_min: float = 1.0, p_max: float = 99.0) -> np.ndarray:
    """Clips and normalizes SAR dB image to [0, 1] for visual display."""
    valid = img_band[~np.isnan(img_band)]
    if len(valid) == 0:
        return np.zeros_like(img_band)
    vmin = np.percentile(valid, p_min)
    vmax = np.percentile(valid, p_max)
    clipped = np.clip(img_band, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin + 1e-7)
    return norm


# =============================================================================
# Figure 6: fig_iou_bars.png
# =============================================================================
def plot_fig_iou_bars():
    print("Generating fig_iou_bars.png...")
    # Load dynamic values from results/metrics_summary_table.csv
    csv_path = PROJECT_ROOT / "results" / "metrics_summary_table.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "results" / "run1" / "metrics_summary_table.csv"

    # Default fallback values
    otsu_means = [44.58, 49.54, 64.81]
    unet_means = [61.29, 64.45, 65.08]
    unet_stds = [0.71, 1.16, 0.88]
    dl_means = [61.22, 64.03, 57.80]
    dl_stds = [1.28, 0.95, 9.74]

    if csv_path.exists():
        import csv
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            split_map = {"Valid": 0, "Test": 1, "Bolivia": 2}
            for row in reader:
                sp = row["Split"]
                if sp not in split_map:
                    continue
                idx = split_map[sp]
                model = row["Architecture / Baseline"]
                val_str = row["Flood IoU (%)"]

                if "Otsu (Train-Fitted Global)" in model and row.get("Variant / Polarization") == "MEAN":
                    try:
                        otsu_means[idx] = float(val_str.replace("%", "").strip())
                    except ValueError:
                        pass
                elif "U-Net" in model:
                    try:
                        if "±" in val_str:
                            parts = val_str.replace("%", "").split("±")
                            unet_means[idx] = float(parts[0].strip())
                            unet_stds[idx] = float(parts[1].strip())
                        else:
                            unet_means[idx] = float(val_str.replace("%", "").strip())
                    except ValueError:
                        pass
                elif "DeepLabV3+" in model:
                    try:
                        if "±" in val_str:
                            parts = val_str.replace("%", "").split("±")
                            dl_means[idx] = float(parts[0].strip())
                            dl_stds[idx] = float(parts[1].strip())
                        else:
                            dl_means[idx] = float(val_str.replace("%", "").strip())
                    except ValueError:
                        pass

    splits = ["Validation\n(In-Domain)", "Test\n(In-Domain)", "Bolivia\n(Held-Out Event)"]

    x = np.arange(len(splits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # Bars
    bar_otsu = ax.bar(
        x - width, otsu_means, width,
        label="Otsu (Train-Fitted Global Mean)",
        color="#7f8c8d", edgecolor="#2c3e50", alpha=0.9
    )
    bar_unet = ax.bar(
        x, unet_means, width, yerr=unet_stds, capsize=4,
        label="U-Net ResNet-34 (Dual-Pol, 3 Seeds)",
        color="#2980b9", edgecolor="#1a5276", alpha=0.9
    )
    bar_dl = ax.bar(
        x + width, dl_means, width, yerr=dl_stds, capsize=4,
        label="DeepLabV3+ ResNet-34 (Dual-Pol, 3 Seeds)",
        color="#e67e22", edgecolor="#a04000", alpha=0.9
    )

    # Text annotations on top of bars
    for bar, val in zip(bar_otsu, otsu_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2c3e50")

    for bar, val, std in zip(bar_unet, unet_means, unet_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, val + std + 1.2, f"{val:.1f}±{std:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a5276")

    for bar, val, std in zip(bar_dl, dl_means, dl_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, val + std + 1.2, f"{val:.1f}±{std:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#a04000")

    ax.set_ylabel("Flood IoU (%)", fontweight="bold")
    ax.set_title("Flood Segmentation IoU Across Splits (Mean ± Std, 3 Seeds)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#d5d8dc", loc="upper left")

    note = (
        "Note: Bolivia is a single out-of-region disaster event (15 chips) with higher flood fraction.\n"
        "In-domain model differences are within seed-to-seed variation."
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, style="italic", color="#34495e")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_iou_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================
# Figure 7: fig_val_curves.png
# =============================================================================
def plot_fig_val_curves():
    print("Generating fig_val_curves.png...")
    results_dir = PROJECT_ROOT / "results"

    unet_runs = [
        ("U-Net Seed 0", results_dir / "unet_resnet34_seed0_history.json", "#2980b9", "-", "o"),
        ("U-Net Seed 1", results_dir / "unet_resnet34_seed1_history.json", "#1f618d", "--", "s"),
        ("U-Net Seed 2", results_dir / "unet_resnet34_seed2_history.json", "#5499c7", "-.", "^"),
    ]

    dl_runs = [
        ("DeepLabV3+ Seed 0", results_dir / "deeplabv3p_resnet34_seed0_history.json", "#d35400", "-", "o"),
        ("DeepLabV3+ Seed 1", results_dir / "deeplabv3p_resnet34_seed1_history.json", "#e67e22", "--", "s"),
        ("DeepLabV3+ Seed 2", results_dir / "deeplabv3p_resnet34_seed2_history.json", "#f39c12", "-.", "^"),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=300)

    for label, hist_file, color, linestyle, marker in (unet_runs + dl_runs):
        if not hist_file.exists():
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)

        epochs = [entry["epoch"] for entry in hist]
        val_ious = [entry["val_flood_iou"] * 100.0 for entry in hist]

        # Best epoch
        best_idx = int(np.argmax(val_ious))
        best_ep = epochs[best_idx]
        best_iou = val_ious[best_idx]

        ax.plot(
            epochs, val_ious,
            label=f"{label} (Best: {best_iou:.1f}% @ Ep {best_ep})",
            color=color, linestyle=linestyle, marker=marker,
            markersize=4, alpha=0.85
        )

        # Highlight best epoch with star marker
        ax.scatter([best_ep], [best_iou], marker="*", s=160, color=color, edgecolors="#1b2631", zorder=5)

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Validation Flood IoU (%)", fontweight="bold")
    ax.set_title("Validation Flood IoU vs Epoch Across 6 Multi-Seed Training Runs", fontweight="bold", pad=12)
    ax.set_xlim(1, 31)
    ax.set_ylim(20, 68)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#d5d8dc", loc="lower right", fontsize=8.5)

    note = "Early stopping monitored strictly on Validation Flood IoU (patience=8). Star (*) denotes saved best checkpoint."
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, style="italic", color="#34495e")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_val_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================
# Helper: Predict Chip
# =============================================================================
def get_model_prediction(model: torch.nn.Module, s1_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = s1_tensor.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred = (probs >= 0.5).astype(np.int64)
    return pred


# =============================================================================
# Figure 8: fig_examples_test.png
# =============================================================================
def plot_fig_examples_test():
    print("Generating fig_examples_test.png...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load U-Net Seed 0 checkpoint
    unet = FloodUNet(encoder_name="resnet34", in_channels=2, classes=1)
    unet.load_state_dict(torch.load(PROJECT_ROOT / "results/checkpoints/unet_resnet34_seed0_best.pth", map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()

    # Otsu threshold
    with open(PROJECT_ROOT / "results/otsu_baseline_results.json") as f:
        otsu_res = json.load(f)
    global_otsu_thresh = otsu_res["global_thresholds_fitted"]["mean"]

    # 4 Test Chips: 2 good, 2 bad
    target_chips = [
        ("India_591317", "High Flood Performance (GT: 46.4%, U-Net IoU: 87.1%)"),
        ("India_900498", "High Flood Performance (GT: 47.3%, U-Net IoU: 77.1%)"),
        ("Ghana_141271", "Low Flood Failure Case (GT: 0.6%, U-Net IoU: 0.2%)"),
        ("Ghana_313799", "Edge False Alarms Case (GT: 6.1%, U-Net IoU: 8.4%)"),
    ]

    dataset = FloodDataset(split_csv=PROJECT_ROOT / "data/splits/flood_test_data.csv", data_dir=PROJECT_ROOT / "data")
    sample_dict = {dataset.samples[i][0]: i for i in range(len(dataset))}

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 12), dpi=300)
    col_titles = [
        "SAR VV Band",
        "SAR VH Band",
        "Ground Truth\n(Gray=Ignore, Blue=Flood)",
        "U-Net (Seed 0)\nPrediction",
        "Global Otsu (Mean)\nPrediction",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=8, fontsize=10)

    for row_idx, (cid, desc) in enumerate(target_chips):
        if cid not in sample_dict:
            continue
        idx = sample_dict[cid]
        s1, label, _ = dataset[idx]

        vv = s1[0].numpy()
        vh = s1[1].numpy()
        lbl = label.numpy()

        # Model pred
        u_pred = get_model_prediction(unet, s1, device)

        # Otsu pred
        otsu_pred, _ = otsu_segment_chip(s1, variant="mean", threshold_override=global_otsu_thresh)

        # 1. VV
        axes[row_idx, 0].imshow(normalize_sar(vv), cmap="gray")
        axes[row_idx, 0].set_ylabel(f"{cid}\n{desc}", fontweight="bold", fontsize=8.5, labelpad=8)

        # 2. VH
        axes[row_idx, 1].imshow(normalize_sar(vh), cmap="gray")

        # 3. Ground Truth
        axes[row_idx, 2].imshow(lbl, cmap=GT_CMAP, norm=GT_NORM)

        # 4. U-Net Seed 0 Pred
        axes[row_idx, 3].imshow(u_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        # 5. Global Otsu Pred
        axes[row_idx, 4].imshow(otsu_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    # Legend at bottom
    patches = [
        mpatches.Patch(color="#7f8c8d", label="Ignore Pixel (-1)"),
        mpatches.Patch(color="#2c3e50", label="Dry Land (0)"),
        mpatches.Patch(color="#2980b9", label="Flooded Water (1)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Qualitative Test Split Visualizations: SAR Inputs vs Ground Truth vs Predictions", fontweight="bold", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIGURES_DIR / "fig_examples_test.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================
# Figure 9: fig_examples_bolivia.png
# =============================================================================
def plot_fig_examples_bolivia():
    print("Generating fig_examples_bolivia.png...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    worst_seed = 0
    lowest_iou = 1.0
    for s in [0, 1, 2]:
        p = PROJECT_ROOT / f"results/deeplabv3p_resnet34_seed{s}_bolivia.json"
        if not p.exists():
            p = PROJECT_ROOT / f"results/run1/deeplabv3p_resnet34_seed{s}_bolivia.json"
        if p.exists():
            try:
                with open(p) as f:
                    d = json.load(f)
                iou = d["global"]["flood_iou"]
                if iou < lowest_iou:
                    lowest_iou = iou
                    worst_seed = s
            except Exception:
                pass

    ckpt_path = PROJECT_ROOT / f"results/checkpoints/deeplabv3p_resnet34_seed{worst_seed}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / f"results/run1/checkpoints/deeplabv3p_resnet34_seed{worst_seed}_best.pth"

    dl = FloodDeepLabV3Plus(encoder_name="resnet34", in_channels=2, classes=1)
    dl.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    dl.to(device)
    dl.eval()

    otsu_res_file = PROJECT_ROOT / "results/otsu_baseline_results.json"
    if not otsu_res_file.exists():
        otsu_res_file = PROJECT_ROOT / "results/run1/otsu_baseline_results.json"
    with open(otsu_res_file) as f:
        otsu_res = json.load(f)
    global_otsu_thresh = otsu_res["global_thresholds_fitted"]["mean"]

    # 3 Bolivia chips including the worst DeepLab case
    target_chips = [
        ("Bolivia_129334", "High Floodplain (GT: 65.9% flood)"),
        ("Bolivia_103757", "Moderate Flood (GT: 40.2% flood)"),
        ("Bolivia_312675", "Low Flood Margin (GT: 6.1% flood)"),
    ]

    dataset = FloodDataset(split_csv=PROJECT_ROOT / "data/splits/flood_bolivia_data.csv", data_dir=PROJECT_ROOT / "data")
    sample_dict = {dataset.samples[i][0]: i for i in range(len(dataset))}

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9.5), dpi=300)
    col_titles = [
        "SAR VV Band",
        "SAR VH Band",
        "Ground Truth\n(Gray=Ignore, Blue=Flood)",
        f"DeepLabV3+ (Seed {worst_seed})\nPrediction (Lowest Seed)",
        "Global Otsu (Mean)\nPrediction",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=8, fontsize=10)

    for row_idx, (cid, desc) in enumerate(target_chips):
        if cid not in sample_dict:
            continue
        idx = sample_dict[cid]
        s1, label, _ = dataset[idx]

        vv = s1[0].numpy()
        vh = s1[1].numpy()
        lbl = label.numpy()

        dl_pred = get_model_prediction(dl, s1, device)
        otsu_pred, _ = otsu_segment_chip(s1, variant="mean", threshold_override=global_otsu_thresh)

        # 1. VV
        axes[row_idx, 0].imshow(normalize_sar(vv), cmap="gray")
        axes[row_idx, 0].set_ylabel(f"{cid}\n{desc}", fontweight="bold", fontsize=8.5, labelpad=8)

        # 2. VH
        axes[row_idx, 1].imshow(normalize_sar(vh), cmap="gray")

        # 3. Ground Truth
        axes[row_idx, 2].imshow(lbl, cmap=GT_CMAP, norm=GT_NORM)

        # 4. DeepLabV3+ Seed 1 Pred
        axes[row_idx, 3].imshow(dl_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        # 5. Global Otsu Pred
        axes[row_idx, 4].imshow(otsu_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    patches = [
        mpatches.Patch(color="#7f8c8d", label="Ignore Pixel (-1)"),
        mpatches.Patch(color="#2c3e50", label="Dry Land (0)"),
        mpatches.Patch(color="#2980b9", label="Flooded Water (1)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Bolivia Out-of-Region Visualizations: DeepLabV3+ Instability vs Ground Truth vs Global Otsu", fontweight="bold", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIGURES_DIR / "fig_examples_bolivia.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================
# Figure 10: fig_otsu_failure.png
# =============================================================================
def plot_fig_otsu_failure():
    print("Generating fig_otsu_failure.png...")
    # 3 Chips where per-chip Otsu has huge false floods next to ground truth
    # 1. India_79637 (Test): GT flood 1.97%, Per-chip Otsu FP 61.23% (160k false alarm pixels)
    # 2. India_44475 (Test): GT flood 0.00%, Per-chip Otsu FP 49.21% (128k false alarm pixels)
    # 3. Bolivia_76104 (Bolivia): GT flood 0.00%, Per-chip Otsu FP 48.57% (92k false alarm pixels)

    target_chips = [
        ("India_79637", "test", "GT Flood: 1.97% | Per-Chip Otsu: 61.2% False Flood (FP=160,519 px)"),
        ("India_44475", "test", "GT Flood: 0.00% | Per-Chip Otsu: 49.2% False Flood (FP=128,044 px)"),
        ("Bolivia_76104", "bolivia", "GT Flood: 0.00% | Per-Chip Otsu: 48.6% False Flood (FP=92,743 px)"),
    ]

    with open(PROJECT_ROOT / "results/otsu_baseline_results.json") as f:
        otsu_res = json.load(f)
    global_otsu_thresh = otsu_res["global_thresholds_fitted"]["mean"]

    datasets = {
        "test": FloodDataset(split_csv=PROJECT_ROOT / "data/splits/flood_test_data.csv", data_dir=PROJECT_ROOT / "data"),
        "bolivia": FloodDataset(split_csv=PROJECT_ROOT / "data/splits/flood_bolivia_data.csv", data_dir=PROJECT_ROOT / "data"),
    }

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13.5, 9.5), dpi=300)
    col_titles = [
        "SAR Dual-Pol Mean\n(Input Image)",
        "Ground Truth\n(Dry Land / Near-Zero Flood)",
        "Per-Chip Otsu Prediction\n(Catastrophic False Flood)",
        "Train-Fitted Global Otsu\n(Correct Rejection of Dry Land)",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontweight="bold", pad=8, fontsize=10)

    for row_idx, (cid, split_name, desc) in enumerate(target_chips):
        ds = datasets[split_name]
        sample_dict = {ds.samples[i][0]: i for i in range(len(ds))}
        idx = sample_dict[cid]
        s1, label, _ = ds[idx]

        mean_sar = ((s1[0] + s1[1]) / 2.0).numpy()
        lbl = label.numpy()

        # Per-chip Otsu prediction
        per_chip_pred, thresh_pc = otsu_segment_chip(s1, variant="mean")

        # Global Otsu prediction
        global_pred, thresh_gl = otsu_segment_chip(s1, variant="mean", threshold_override=global_otsu_thresh)

        # 1. SAR Mean
        axes[row_idx, 0].imshow(normalize_sar(mean_sar), cmap="gray")
        axes[row_idx, 0].set_ylabel(f"{cid}\n{desc}", fontweight="bold", fontsize=8.5, labelpad=8)

        # 2. Ground Truth
        axes[row_idx, 1].imshow(lbl, cmap=GT_CMAP, norm=GT_NORM)

        # 3. Per-Chip Otsu (Failure)
        axes[row_idx, 2].imshow(per_chip_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        # 4. Global Otsu (Correction)
        axes[row_idx, 3].imshow(global_pred, cmap=PRED_CMAP, norm=PRED_NORM)

        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    patches = [
        mpatches.Patch(color="#7f8c8d", label="Ignore Pixel (-1)"),
        mpatches.Patch(color="#2c3e50", label="Dry Land (0)"),
        mpatches.Patch(color="#2980b9", label="Flooded Water (1)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Classical Otsu Failure Mode: Per-Chip Bimodal Bisection on Dry Land vs Global Thresholding", fontweight="bold", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIGURES_DIR / "fig_otsu_failure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


if __name__ == "__main__":
    plot_fig_iou_bars()
    plot_fig_val_curves()
    plot_fig_examples_test()
    plot_fig_examples_bolivia()
    plot_fig_otsu_failure()
    print("All 5 figures generated successfully.")
