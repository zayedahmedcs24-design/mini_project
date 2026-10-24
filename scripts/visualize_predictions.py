"""
Generate publication-quality diagnostic prediction figures using real trained model checkpoints.
Produces multi-panel figures for Review-2 presentation and technical reports.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet


def generate_prediction_figure(
    model: torch.nn.Module,
    dataset: FloodDataset,
    sample_index: int,
    output_path: Path,
    device: torch.device,
    title_prefix: str = "Evaluation Sample",
):
    model.eval()
    s1_tensor, label_tensor, chip_id = dataset[sample_index]

    with torch.no_grad():
        input_batch = s1_tensor.unsqueeze(0).to(device)
        logits = model(input_batch)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    s1_np = s1_tensor.cpu().numpy()
    label_np = label_tensor.cpu().numpy()
    binary_pred = (probs >= 0.5).astype(np.int32)

    # 5-Panel Diagnostic Figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))

    # 1. SAR VV Band (Denormalized to dB: mean -11.5, std 4.5 approx)
    vv_db = s1_np[0] * 4.54 - 11.53
    im0 = axes[0].imshow(vv_db, cmap="gray", vmin=-25, vmax=-5)
    axes[0].set_title(f"SAR VV (dB)\n[{chip_id[:16]}...]", fontsize=10)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. SAR VH Band (Denormalized to dB: mean -18.7, std 5.2 approx)
    vh_db = s1_np[1] * 5.17 - 18.71
    im1 = axes[1].imshow(vh_db, cmap="gray", vmin=-32, vmax=-10)
    axes[1].set_title(f"SAR VH (dB)\n[Cross-Pol]", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Ground Truth Mask
    gt_vis = np.zeros((*label_np.shape, 3), dtype=np.float32)
    gt_vis[label_np == 0] = [0.15, 0.15, 0.15]  # Dark gray (Dry land)
    gt_vis[label_np == 1] = [0.0, 0.75, 1.0]   # Bright Cyan (Water)
    gt_vis[label_np == -1] = [1.0, 0.8, 0.0]   # Yellow (Ignore/Cloud)
    axes[2].imshow(gt_vis)
    axes[2].set_title("Ground Truth\n(Cyan=Water, Yellow=Ignore)", fontsize=10)
    axes[2].axis("off")

    # 4. Model Predicted Probability Surface
    im3 = axes[3].imshow(probs, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[3].set_title("U-Net Probability Map\n[P(Flood | SAR)]", fontsize=10)
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Error Confusion Map (TP, FP, FN, TN)
    # Ignore pixels where label == -1
    err_map = np.zeros((*label_np.shape, 3), dtype=np.float32)
    # True Negative (TN): Dry correctly predicted as dry -> Dark Gray
    err_map[(label_np == 0) & (binary_pred == 0)] = [0.15, 0.15, 0.15]
    # True Positive (TP): Flood correctly detected -> Bright Green
    err_map[(label_np == 1) & (binary_pred == 1)] = [0.0, 0.85, 0.2]
    # False Positive (FP): False alarm (Dry predicted as flood) -> Red
    err_map[(label_np == 0) & (binary_pred == 1)] = [0.9, 0.1, 0.1]
    # False Negative (FN): Missed flood -> Blue
    err_map[(label_np == 1) & (binary_pred == 0)] = [0.1, 0.3, 0.95]
    # Ignored pixels -> Yellow
    err_map[label_np == -1] = [1.0, 0.8, 0.0]

    axes[4].imshow(err_map)
    axes[4].set_title("Prediction Error Map\n(Green=TP, Red=FP, Blue=FN)", fontsize=10)
    axes[4].axis("off")

    plt.suptitle(f"{title_prefix} — Chip: {chip_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Generated diagnostic figure: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualization device: {device}")

    # Load Model
    model = FloodUNet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    ).to(device)

    checkpoint_path = Path("results/checkpoints/unet_resnet34_seed0_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 1. Validation Split Sample (Find one with significant flood)
    valid_csv = Path("data/splits/flood_valid_data.csv")
    valid_ds = FloodDataset(split_csv=valid_csv, data_dir="data")

    # Pick a chip with good water coverage
    val_idx = 0
    for i in range(len(valid_ds)):
        _, lbl, _ = valid_ds[i]
        if (lbl == 1).sum() > 5000:
            val_idx = i
            break

    generate_prediction_figure(
        model=model,
        dataset=valid_ds,
        sample_index=val_idx,
        output_path=Path("results/real_prediction_valid.png"),
        device=device,
        title_prefix="Review-2 Real U-Net Validation Prediction",
    )

    # 2. Bolivia Split Sample
    bolivia_csv = Path("data/splits/flood_bolivia_data.csv")
    bolivia_ds = FloodDataset(split_csv=bolivia_csv, data_dir="data")

    # Pick a prominent Bolivia chip
    bol_idx = 0
    for i in range(len(bolivia_ds)):
        _, lbl, _ = bolivia_ds[i]
        if (lbl == 1).sum() > 20000:
            bol_idx = i
            break

    generate_prediction_figure(
        model=model,
        dataset=bolivia_ds,
        sample_index=bol_idx,
        output_path=Path("results/real_prediction_bolivia.png"),
        device=device,
        title_prefix="Review-2 Real U-Net Bolivia Generalization Prediction",
    )


if __name__ == "__main__":
    main()
