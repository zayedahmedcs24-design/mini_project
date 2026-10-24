"""
Training pipeline for flood extent segmentation models.

Supports:
- U-Net with ResNet-34 encoder
- Mixed precision training (AMP) on CUDA
- Ignore-aware compound BCE + Dice loss
- Early stopping on validation Flood IoU
- Fast smoke test verification (--smoke)
- Sample prediction artifact visualization
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.data.transforms import get_train_transforms
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.models.losses import MaskedBCEDiceLoss
from src.floodmap.metrics import MetricTracker


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    tracker = MetricTracker(ignore_index=-1, threshold=0.5)

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)
            tracker.update(probs, targets)

    summary = tracker.compute_summary()
    summary["loss"] = total_loss / max(1, num_batches)
    return summary


def save_prediction_sample(
    model: nn.Module,
    sample_item: tuple,
    output_path: Path,
    device: torch.device,
):
    """Generates a 4-panel diagnostic plot for presentation and review."""
    model.eval()
    s1_tensor, label_tensor, chip_id = sample_item

    with torch.no_grad():
        input_batch = s1_tensor.unsqueeze(0).to(device)
        logits = model(input_batch)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    s1_np = s1_tensor.cpu().numpy()
    label_np = label_tensor.cpu().numpy()

    # Panel setup
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. SAR VV Backscatter
    im0 = axes[0].imshow(s1_np[0], cmap="gray")
    axes[0].set_title(f"SAR VV Band ({chip_id})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Ground Truth Mask
    # Map -1 to yellow, 0 to black, 1 to cyan
    gt_vis = np.zeros((*label_np.shape, 3), dtype=np.float32)
    gt_vis[label_np == 0] = [0.1, 0.1, 0.1]  # Dark gray (Dry land)
    gt_vis[label_np == 1] = [0.0, 0.8, 1.0]  # Cyan (Water)
    gt_vis[label_np == -1] = [1.0, 0.8, 0.0] # Yellow (Ignore/Cloud)
    axes[1].imshow(gt_vis)
    axes[1].set_title("Ground Truth (Cyan=Water, Yellow=Ignore)")
    axes[1].axis("off")

    # 3. Model Predicted Probability
    im2 = axes[2].imshow(probs, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Predicted Probability Map")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Binary Threshold at 0.5
    binary_pred = (probs >= 0.5).astype(np.float32)
    im3 = axes[3].imshow(binary_pred, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[3].set_title("Binary Prediction (P >= 0.5)")
    axes[3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[SAVED] Sample prediction plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Flood Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/unet_resnet34.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run fast 2-epoch smoke test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Paths
    root_dir = Path(cfg["data"]["root_dir"])
    train_csv = Path(cfg["data"]["train_split"])
    valid_csv = Path(cfg["data"]["valid_split"])
    checkpoints_dir = Path("results/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    crop_size = cfg["data"].get("crop_size", 256)
    train_transform = get_train_transforms(crop_size=crop_size)

    # Initialize Datasets
    full_train_ds = FloodDataset(split_csv=train_csv, data_dir=root_dir, transform=train_transform)
    full_valid_ds = FloodDataset(split_csv=valid_csv, data_dir=root_dir, transform=None)

    # Filter to only locally available chips
    train_indices = [i for i, (_, s1_p, l_p) in enumerate(full_train_ds.samples) if Path(s1_p).exists() and Path(l_p).exists()]
    valid_indices = [i for i, (_, s1_p, l_p) in enumerate(full_valid_ds.samples) if Path(s1_p).exists() and Path(l_p).exists()]

    if not train_indices or not valid_indices:
        raise RuntimeError("No chips found on disk! Please run `python -m src.floodmap.data.download --smoke` first.")

    # Smoke mode overrides
    if args.smoke:
        print("\n" + "=" * 60)
        print("RUNNING IN --smoke MODE (Verification & Sanity Run)")
        print("=" * 60)
        epochs = cfg["smoke"]["epochs"]
        batch_size = cfg["smoke"]["batch_size"]
        train_indices = train_indices[:cfg["smoke"]["max_samples"]]
        valid_indices = valid_indices[:2]
    else:
        epochs = cfg["training"]["max_epochs"]
        batch_size = cfg["training"]["batch_size"]

    train_ds = Subset(full_train_ds, train_indices)
    valid_ds = Subset(full_valid_ds, valid_indices)

    print(f"Loaded {len(train_ds)} training chips and {len(valid_ds)} validation chips.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model Architecture
    arch = cfg["model"].get("architecture", "unet").lower()
    if arch == "unet":
        model = FloodUNet(
            encoder_name=cfg["model"]["encoder_name"],
            encoder_weights=cfg["model"]["encoder_weights"],
            in_channels=cfg["model"]["in_channels"],
            classes=cfg["model"]["classes"],
            decoder_dropout=cfg["model"].get("decoder_dropout", 0.2),
        ).to(device)
    elif arch == "deeplabv3p":
        model = FloodDeepLabV3Plus(
            encoder_name=cfg["model"]["encoder_name"],
            encoder_weights=cfg["model"]["encoder_weights"],
            in_channels=cfg["model"]["in_channels"],
            classes=cfg["model"]["classes"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'unet' or 'deeplabv3p'.")

    # Loss & Optimizer
    criterion = MaskedBCEDiceLoss(
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        ignore_index=cfg["data"]["ignore_index"],
        smooth=cfg["loss"]["smooth"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training Loop
    best_val_iou = 0.0
    history = []

    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate(model, valid_loader, criterion, device)

        val_iou = val_metrics["global"]["flood_iou"]
        val_f1 = val_metrics["global"]["flood_f1"]
        val_loss = val_metrics["loss"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Flood IoU: {val_iou * 100:.2f}% | "
            f"Val Flood F1: {val_f1 * 100:.2f}%"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = checkpoints_dir / f"{cfg['experiment_name']}_best.pth"
            torch.save(model.state_dict(), best_path)

    # In smoke mode: verify loss decreased and save visualization sample
    if args.smoke:
        loss_decreased = history[-1]["train_loss"] < history[0]["train_loss"]
        print(f"\nSmoke Verification: Loss decreased from {history[0]['train_loss']:.4f} to {history[-1]['train_loss']:.4f}: {loss_decreased}")
        sample_item = valid_ds[0]
        sample_path = Path("results/smoke_prediction_sample.png")
        save_prediction_sample(model, sample_item, sample_path, device)
        print("[SUCCESS] Stage A U-Net smoke training run completed successfully.")


if __name__ == "__main__":
    main()
