"""
Aggregates evaluation results across seeds and classical baselines into a unified CSV summary table.

Computes mean ± std across seeds (0, 1, 2) for:
- U-Net ResNet-34
- DeepLabV3+ ResNet-34
And reports deterministic benchmark metrics for:
- Otsu (Per-Chip Dynamic: VV, VH, Mean)
- Otsu (Train-Fitted Global: VV, VH, Mean)

Across splits:
- Validation (in-domain)
- Test (in-domain)
- Bolivia (held-out out-of-region disaster event)

Outputs to: results/metrics_summary_table.csv
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Error reading {filepath}: {e}")
        return None


def format_mean_std(values: List[float], as_percentage: bool = True) -> str:
    if not values:
        return "N/A"
    arr = np.array(values, dtype=np.float64)
    if as_percentage:
        arr = arr * 100.0
    mean_val = float(np.mean(arr))
    if len(arr) > 1:
        std_val = float(np.std(arr, ddof=1))
        return f"{mean_val:.2f} ± {std_val:.2f}%"
    else:
        return f"{mean_val:.2f}%"


def generate_metrics_summary_table(
    results_dir: Path = Path("results"),
    output_csv: Path = Path("results/metrics_summary_table.csv"),
    output_md: Path = Path("results/metrics_summary_table.md"),
    seeds: List[int] = [0, 1, 2],
    splits: List[str] = ["valid", "test", "bolivia"],
):
    print("=" * 80)
    print("COMPILING REVIEW-2 METRICS SUMMARY TABLE ACROSS SEEDS & BASELINES")
    print("=" * 80)

    rows = []
    # Header
    header = [
        "Split",
        "Architecture / Baseline",
        "Variant / Polarization",
        "Seeds Evaluated",
        "Flood IoU (%)",
        "Flood F1 (%)",
        "Precision (%)",
        "Recall (%)",
    ]

    # 1. Classical Otsu Baselines (from results/otsu_baseline_results.json)
    otsu_path = results_dir / "otsu_baseline_results.json"
    otsu_data = load_json(otsu_path)

    if otsu_data:
        for split in splits:
            for variant in ["vv", "vh", "mean"]:
                # Per-chip Otsu
                if "per_chip_otsu" in otsu_data and split in otsu_data["per_chip_otsu"]:
                    pc_glob = otsu_data["per_chip_otsu"][split][variant]["global"]
                elif split in otsu_data and variant in otsu_data[split]:
                    pc_glob = otsu_data[split][variant]["global"]
                else:
                    pc_glob = None

                if pc_glob:
                    rows.append({
                        "Split": split.capitalize(),
                        "Architecture / Baseline": "Otsu (Per-Chip Dynamic)",
                        "Variant / Polarization": variant.upper(),
                        "Seeds Evaluated": "Deterministic (N/A)",
                        "Flood IoU (%)": f"{pc_glob['flood_iou'] * 100:.2f}%",
                        "Flood F1 (%)": f"{pc_glob['flood_f1'] * 100:.2f}%",
                        "Precision (%)": f"{pc_glob['flood_precision'] * 100:.2f}%",
                        "Recall (%)": f"{pc_glob['flood_recall'] * 100:.2f}%",
                    })

                # Global Otsu
                if "global_otsu" in otsu_data and split in otsu_data["global_otsu"]:
                    gl_glob = otsu_data["global_otsu"][split][variant]["global"]
                elif split in otsu_data and f"{variant}_global" in otsu_data[split]:
                    gl_glob = otsu_data[split][f"{variant}_global"]["global"]
                else:
                    gl_glob = None

                if gl_glob:
                    rows.append({
                        "Split": split.capitalize(),
                        "Architecture / Baseline": "Otsu (Train-Fitted Global)",
                        "Variant / Polarization": variant.upper(),
                        "Seeds Evaluated": "Deterministic (N/A)",
                        "Flood IoU (%)": f"{gl_glob['flood_iou'] * 100:.2f}%",
                        "Flood F1 (%)": f"{gl_glob['flood_f1'] * 100:.2f}%",
                        "Precision (%)": f"{gl_glob['flood_precision'] * 100:.2f}%",
                        "Recall (%)": f"{gl_glob['flood_recall'] * 100:.2f}%",
                    })

    # 2. Deep Learning Models across seeds
    models = [
        ("U-Net ResNet-34", "unet_resnet34", "Dual-Pol (VV + VH)"),
        ("DeepLabV3+ ResNet-34", "deeplabv3p_resnet34", "Dual-Pol (VV + VH)"),
    ]

    for model_label, model_prefix, variant_label in models:
        for split in splits:
            ious = []
            f1s = []
            precisions = []
            recalls = []
            seeds_found = []

            for s in seeds:
                res_path = results_dir / f"{model_prefix}_seed{s}_{split}.json"
                data = load_json(res_path)
                if data and "global" in data:
                    g = data["global"]
                    ious.append(g["flood_iou"])
                    f1s.append(g["flood_f1"])
                    precisions.append(g["flood_precision"])
                    recalls.append(g["flood_recall"])
                    seeds_found.append(str(s))

            if seeds_found:
                seed_str = f"{len(seeds_found)}/3 ({', '.join(seeds_found)})"
                rows.append({
                    "Split": split.capitalize(),
                    "Architecture / Baseline": model_label,
                    "Variant / Polarization": variant_label,
                    "Seeds Evaluated": seed_str,
                    "Flood IoU (%)": format_mean_std(ious),
                    "Flood F1 (%)": format_mean_std(f1s),
                    "Precision (%)": format_mean_std(precisions),
                    "Recall (%)": format_mean_std(recalls),
                })
            else:
                rows.append({
                    "Split": split.capitalize(),
                    "Architecture / Baseline": model_label,
                    "Variant / Polarization": variant_label,
                    "Seeds Evaluated": "0/3 (Pending)",
                    "Flood IoU (%)": "Pending",
                    "Flood F1 (%)": "Pending",
                    "Precision (%)": "Pending",
                    "Recall (%)": "Pending",
                })

    # Write CSV with utf-8-sig encoding for Windows / Excel compatibility
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] Metrics summary CSV successfully written to {output_csv}")

    # Write Markdown version
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Sen1Floods11 Metrics Summary Table\n\n")
        f.write(f"Generated automatically from evaluation JSON files across 3 random seeds (0, 1, 2) and deterministic baselines.\n\n")
        # Markdown table header
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in rows:
            line = "| " + " | ".join(str(r[h]) for h in header) + " |\n"
            f.write(line)

    print(f"[SAVED] Metrics summary Markdown table successfully written to {output_md}\n")

    # Display preview table in console
    print(f"{'Split':10s} | {'Architecture / Baseline':26s} | {'Variant':18s} | {'Flood IoU (%)':18s} | {'Flood F1 (%)':18s}")
    print("-" * 98)
    for r in rows:
        print(f"{r['Split']:10s} | {r['Architecture / Baseline']:26s} | {r['Variant / Polarization']:18s} | {r['Flood IoU (%)']:18s} | {r['Flood F1 (%)']:18s}")
    print("=" * 98)


if __name__ == "__main__":
    generate_metrics_summary_table()
