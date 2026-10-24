"""
Evaluation pipeline for flood segmentation models on Sen1Floods11 splits.

Computes:
- Ignore-aware Flood IoU, F1-Score, Precision, Recall, Mean IoU
- Aggregated global confusion matrix metrics
- Per-chip distribution statistics (median and IQR)
- Outputs structured JSON to results/<model>_<split>.json
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.metrics import MetricTracker


def evaluate_model_on_split(
    model: torch.nn.Module,
    split_csv: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 4,
    threshold: float = 0.5,
    ignore_index: int = -1,
) -> Dict[str, Any]:
    """
    Evaluates a PyTorch segmentation model on a given split.
    """
    model.eval()
    dataset = FloodDataset(split_csv=split_csv, data_dir=data_dir)

    # Filter to chips present on disk
    available_indices = [
        i for i, (_, s1_p, l_p) in enumerate(dataset.samples)
        if Path(s1_p).exists() and Path(l_p).exists()
    ]

    if not available_indices:
        raise FileNotFoundError(f"No available GeoTIFF chips found on disk for split: {split_csv}")

    eval_ds = Subset(dataset, available_indices)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tracker = MetricTracker(ignore_index=ignore_index, threshold=threshold)

    chip_results = []

    with torch.no_grad():
        for images, targets, chip_ids in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Update tracker per chip to ensure 1-to-1 metric logging
            for i, c_id in enumerate(chip_ids):
                p_chip = probs[i].squeeze(0)
                t_chip = targets[i]
                tracker.update(p_chip, t_chip)
                chip_m = tracker.per_chip_metrics[-1]
                chip_results.append({
                    "chip_id": c_id,
                    "flood_iou": chip_m["flood_iou"],
                    "flood_f1": chip_m["flood_f1"],
                    "precision": chip_m["flood_precision"],
                    "recall": chip_m["flood_recall"],
                })

    summary = tracker.compute_summary()
    summary["evaluated_chips"] = len(available_indices)
    summary["total_split_chips"] = len(dataset)
    summary["per_chip_details"] = chip_results

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model on Sen1Floods11")
    parser.add_argument("--model", type=str, default="unet_resnet34", help="Model identifier")
    parser.add_argument("--config", type=str, default="configs/unet_resnet34.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        choices=["valid", "test", "bolivia"],
        default="bolivia",
        help="Split to evaluate on",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Map split to CSV path
    split_key = f"{args.split}_split"
    split_csv = Path(cfg["data"][split_key])

    print(f"Loading split: {args.split} ({split_csv})...")

    # Load Model Architecture
    arch = cfg["model"].get("architecture", "unet").lower()
    if arch == "unet":
        model = FloodUNet(
            encoder_name=cfg["model"]["encoder_name"],
            encoder_weights=None,  # Weights loaded from checkpoint
            in_channels=cfg["model"]["in_channels"],
            classes=cfg["model"]["classes"],
            decoder_dropout=cfg["model"].get("decoder_dropout", 0.2),
        ).to(device)
    elif arch == "deeplabv3p":
        model = FloodDeepLabV3Plus(
            encoder_name=cfg["model"]["encoder_name"],
            encoder_weights=None,  # Weights loaded from checkpoint
            in_channels=cfg["model"]["in_channels"],
            classes=cfg["model"]["classes"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'unet' or 'deeplabv3p'.")

    # Load Weights
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded checkpoint: {checkpoint_path}")

    # Run Evaluation
    results = evaluate_model_on_split(
        model=model,
        split_csv=split_csv,
        data_dir=data_dir,
        device=device,
        batch_size=cfg["training"].get("batch_size", 4),
        threshold=args.threshold,
        ignore_index=cfg["data"].get("ignore_index", -1),
    )

    g = results["global"]
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {args.model.upper()} on {args.split.upper()} SPLIT")
    print("=" * 65)
    print(f"  • Evaluated Chips:     {results['evaluated_chips']} / {results['total_split_chips']}")
    print(f"  • Flood IoU:           {g['flood_iou'] * 100:.2f}%")
    print(f"  • Flood F1-Score:      {g['flood_f1'] * 100:.2f}%")
    print(f"  • Precision:           {g['flood_precision'] * 100:.2f}%")
    print(f"  • Recall:              {g['flood_recall'] * 100:.2f}%")
    print(f"  • Mean IoU:            {g['mean_iou'] * 100:.2f}%")
    print(f"  • Total Valid Pixels:  {g['total_valid_pixels']:,}")
    print("=" * 65)

    # Save to JSON
    output_json = results_dir / f"{args.model}_{args.split}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] Evaluation summary written to {output_json}\n")


if __name__ == "__main__":
    main()
