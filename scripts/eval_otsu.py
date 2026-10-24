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
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.baselines import otsu_segment_chip, GlobalOtsuBaseline
from src.floodmap.metrics import MetricTracker


def evaluate_otsu_on_split(
    split_csv: Path,
    data_dir: Path,
    variant: str = "vh",
    threshold_override: Optional[float] = None,
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
        pred_mask, thresh = otsu_segment_chip(
            s1,
            variant=variant,
            threshold_override=threshold_override,
        )
        tracker.update(pred_mask, label)
        per_chip_thresholds.append(thresh)

    summary = tracker.compute_summary()
    summary["mean_threshold"] = float(np.mean(per_chip_thresholds)) if per_chip_thresholds else 0.0
    summary["evaluated_chips"] = len(per_chip_thresholds)
    if threshold_override is not None:
        summary["global_threshold"] = float(threshold_override)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Otsu Baselines on Sen1Floods11 splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.data_dir / "splits"

    train_csv = splits_dir / "flood_train_data.csv"
    target_splits = {
        "valid": splits_dir / "flood_valid_data.csv",
        "test": splits_dir / "flood_test_data.csv",
        "bolivia": splits_dir / "flood_bolivia_data.csv",
    }

    print("\n" + "=" * 80)
    print("FITTING GLOBAL OTSU THRESHOLDS ON TRAIN SPLIT (NON-IGNORED, CLIPPED PIXELS)")
    print("=" * 80)
    train_dataset = FloodDataset(split_csv=train_csv, data_dir=args.data_dir)
    global_thresholds = {}
    for variant in ["vv", "vh", "mean"]:
        gob = GlobalOtsuBaseline(variant=variant)
        thresh = gob.fit(train_dataset)
        global_thresholds[variant] = thresh
        print(f"  • Global Otsu Threshold [{variant.upper():4s}]: {thresh:.4f}")

    all_results = {
        "global_thresholds_fitted": global_thresholds,
        "per_chip_otsu": {},
        "global_otsu": {},
    }

    print("\n" + "=" * 85)
    print("EVALUATING CLASSICAL OTSU BASELINES ON SEN1FLOODS11 (PER-CHIP & GLOBAL)")
    print("=" * 85)
    print(f"{'Split':10s} | {'Method':12s} | {'Variant':7s} | {'Flood IoU (%)':13s} | {'Flood F1 (%)':12s} | {'Precision (%)':13s} | {'Recall (%)':10s}")
    print("-" * 85)

    for split_name, split_csv in target_splits.items():
        if not split_csv.exists():
            continue
        all_results[split_name] = {}
        all_results["per_chip_otsu"][split_name] = {}
        all_results["global_otsu"][split_name] = {}

        for variant in ["vv", "vh", "mean"]:
            # 1. Per-chip Otsu evaluation (preserved)
            res_chip = evaluate_otsu_on_split(split_csv, args.data_dir, variant=variant)
            all_results[split_name][variant] = res_chip
            all_results["per_chip_otsu"][split_name][variant] = res_chip

            g_c = res_chip["global"]
            print(f"{split_name:10s} | {'Per-Chip':12s} | {variant:7s} | {g_c['flood_iou']*100:13.2f} | {g_c['flood_f1']*100:12.2f} | {g_c['flood_precision']*100:13.2f} | {g_c['flood_recall']*100:10.2f}")

            # 2. Global-threshold Otsu evaluation
            res_glob = evaluate_otsu_on_split(
                split_csv,
                args.data_dir,
                variant=variant,
                threshold_override=global_thresholds[variant],
            )
            all_results[split_name][f"{variant}_global"] = res_glob
            all_results["global_otsu"][split_name][variant] = res_glob

            g_g = res_glob["global"]
            print(f"{split_name:10s} | {'Global':12s} | {variant:7s} | {g_g['flood_iou']*100:13.2f} | {g_g['flood_f1']*100:12.2f} | {g_g['flood_precision']*100:13.2f} | {g_g['flood_recall']*100:10.2f}")

    print("=" * 85)

    # Save to results JSON (alongside per-chip results, without deleting per-chip results)
    out_json = args.results_dir / "otsu_baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[SAVED] Combined Otsu baseline results saved to {out_json}")


if __name__ == "__main__":
    main()
