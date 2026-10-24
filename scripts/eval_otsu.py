"""
CLI evaluation runner for classical Otsu thresholding baselines.

Evaluates:
- Otsu on VV band
- Otsu on VH band
- Otsu on Dual-Pol Mean: (VV + VH) / 2

Evaluates on:
- Validation split (in-domain)
- Bolivia split (held-out out-of-region disaster event)
- Standard test split

Outputs structured metrics to results/otsu_baseline_results.json.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.baselines import otsu_segment_chip
from src.floodmap.metrics import MetricTracker


def evaluate_otsu_on_split(
    split_csv: Path,
    data_dir: Path,
    variant: str = "vh",
) -> Dict[str, Any]:
    """Runs Otsu thresholding across all chips in a split and aggregates metrics."""
    dataset = FloodDataset(split_csv=split_csv, data_dir=data_dir)
    tracker = MetricTracker(ignore_index=-1, threshold=0.5)

    per_chip_thresholds = []

    for idx in range(len(dataset)):
        # Check if the underlying files exist before attempting to read
        chip_id, s1_path, label_path = dataset.samples[idx]
        if not Path(s1_path).exists() or not Path(label_path).exists():
            continue

        s1, label, chip_id = dataset[idx]
        pred_mask, thresh = otsu_segment_chip(s1, variant=variant)
        tracker.update(pred_mask, label)
        per_chip_thresholds.append(thresh)

    summary = tracker.compute_summary()
    summary["mean_threshold"] = float(np.mean(per_chip_thresholds)) if per_chip_thresholds else 0.0
    summary["evaluated_chips"] = len(per_chip_thresholds)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Otsu Baselines on Sen1Floods11 splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.data_dir / "splits"

    target_splits = {
        "valid": splits_dir / "flood_valid_data.csv",
        "test": splits_dir / "flood_test_data.csv",
        "bolivia": splits_dir / "flood_bolivia_data.csv",
    }

    all_results = {}
    print("\n" + "=" * 75)
    print("EVALUATING CLASSICAL OTSU BASELINES ON SEN1FLOODS11")
    print("=" * 75)
    print(f"{'Split':10s} | {'Variant':7s} | {'Flood IoU (%)':13s} | {'Flood F1 (%)':12s} | {'Precision (%)':13s} | {'Recall (%)':10s}")
    print("-" * 75)

    for split_name, split_csv in target_splits.items():
        if not split_csv.exists():
            continue
        all_results[split_name] = {}

        for variant in ["vv", "vh", "mean"]:
            res = evaluate_otsu_on_split(split_csv, args.data_dir, variant=variant)
            all_results[split_name][variant] = res

            g = res["global"]
            iou_pct = g["flood_iou"] * 100.0
            f1_pct = g["flood_f1"] * 100.0
            prec_pct = g["flood_precision"] * 100.0
            rec_pct = g["flood_recall"] * 100.0

            print(f"{split_name:10s} | {variant:7s} | {iou_pct:13.2f} | {f1_pct:12.2f} | {prec_pct:13.2f} | {rec_pct:10.2f}")

    print("=" * 75)

    # Save to results JSON
    out_json = args.results_dir / "otsu_baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[SAVED] Otsu baseline results saved to {out_json}")


if __name__ == "__main__":
    main()
