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
import time
import json
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
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--no-save", action="store_true", help="Do not save checkpoints (benchmark only)")
    parser.add_argument("--smoke", action="store_true", help="Run fast 2-epoch smoke test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Seed configuration
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Architecture & Run Naming
    arch = cfg["model"].get("architecture", "unet").lower()
    encoder = cfg["model"].get("encoder_name", "resnet34")
    model_name = f"{arch}_{encoder}"

    # Paths: Smoke runs isolate all outputs to results/smoke/ and never use real run names
    root_dir = Path(cfg["data"]["root_dir"])
    train_csv = Path(cfg["data"]["train_split"])
    valid_csv = Path(cfg["data"]["valid_split"])

    if args.smoke:
        run_name = f"{model_name}_smoke"
        smoke_dir = Path("results/smoke")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = smoke_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoints_dir / f"{run_name}_best.pth"
        history_path = smoke_dir / f"{run_name}_history.json"
        sample_path = smoke_dir / "smoke_prediction_sample.png"
    else:
        run_name = f"{model_name}_seed{seed}"
        checkpoints_dir = Path("results/checkpoints")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoints_dir / f"{run_name}_best.pth"
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        history_path = results_dir / f"{run_name}_history.json"
        sample_path = None

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

    # Smoke mode overrides vs Real Run
    if args.smoke:
        print("\n" + "=" * 60)
        print("RUNNING IN --smoke MODE (Verification & Sanity Run)")
        print(f"Outputs will be written to results/smoke/")
        print("=" * 60)
        epochs = cfg["smoke"]["epochs"] if args.epochs is None else args.epochs
        batch_size = cfg["smoke"]["batch_size"] if args.batch_size is None else args.batch_size
        train_indices = train_indices[:cfg["smoke"]["max_samples"]]
        valid_indices = valid_indices[:2]
    else:
        epochs = cfg["training"]["max_epochs"] if args.epochs is None else args.epochs
        batch_size = cfg["training"]["batch_size"] if args.batch_size is None else args.batch_size

    train_ds = Subset(full_train_ds, train_indices)
    valid_ds = Subset(full_valid_ds, valid_indices)

    print(f"Loaded {len(train_ds)} training chips and {len(valid_ds)} validation chips (Batch Size: {batch_size}, Epochs: {epochs}).")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model Architecture
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

    # Reset max memory tracker
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Training Loop with Early Stopping monitored on Validation Flood IoU only
    early_stopping_patience = cfg.get("training", {}).get("early_stopping_patience", 8)
    best_val_iou = -1.0
    patience_counter = 0
    history = []

    print(f"\nStarting Training: {run_name} (Early Stopping Patience: {early_stopping_patience} on Val Flood IoU)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate(model, valid_loader, criterion, device)
        epoch_seconds = time.time() - t0

        val_iou = val_metrics["global"]["flood_iou"]
        val_f1 = val_metrics["global"]["flood_f1"]
        val_loss = val_metrics["loss"]

        # VRAM monitoring
        if device.type == "cuda":
            vram_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            vram_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            vram_str = f" | Peak VRAM: {vram_alloc_mb:.1f} MB (reserved: {vram_res_mb:.1f} MB)"
        else:
            vram_str = ""

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_flood_iou": float(val_iou),
            "val_iou": float(val_iou),
            "epoch_seconds": float(round(epoch_seconds, 2)),
        })

        # Persist per-epoch history
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_seconds:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Flood IoU: {val_iou * 100:.2f}% | "
            f"Val Flood F1: {val_f1 * 100:.2f}%"
            f"{vram_str}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            if not args.no_save:
                torch.save(model.state_dict(), best_path)
                print(f"  [SAVED] New best checkpoint saved to {best_path} (Val Flood IoU: {best_val_iou * 100:.2f}%)")
            else:
                print(f"  [NO-SAVE] Best Val Flood IoU reached: {best_val_iou * 100:.2f}% (saving skipped)")
        else:
            patience_counter += 1
            print(f"  [PATIENCE] No improvement in validation flood IoU for {patience_counter}/{early_stopping_patience} epoch(s)")
            if patience_counter >= early_stopping_patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. Stopped training because validation flood IoU did not improve for {early_stopping_patience} epochs.")
                break

    # In smoke mode: verify loss decreased and save visualization sample
    if args.smoke:
        loss_decreased = history[-1]["train_loss"] < history[0]["train_loss"]
        print(f"\nSmoke Verification: Loss decreased from {history[0]['train_loss']:.4f} to {history[-1]['train_loss']:.4f}: {loss_decreased}")
        sample_item = valid_ds[0]
        save_prediction_sample(model, sample_item, sample_path, device)
        print(f"[SUCCESS] Smoke training run completed successfully. Outputs in {smoke_dir}/")


if __name__ == "__main__":
    main()
