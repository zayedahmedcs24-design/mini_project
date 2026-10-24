"""
Generates detailed tabular analysis files strictly from evaluations:
1. results/per_seed_table.csv
2. results/per_chip_bolivia.csv
3. results/per_chip_test.csv

All CSVs are written with utf-8-sig encoding for Windows/Excel compatibility.
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.models.baselines import otsu_segment_chip
from src.floodmap.metrics import compute_confusion_matrix


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_per_seed_table(results_dir: Path = Path("results"), max_cap: int = 60):
    output_csv = results_dir / "per_seed_table.csv"
    print(f"Generating {output_csv}...")

    # Both models now use batch size 8 in run2
    models = [
        ("U-Net ResNet-34", "unet_resnet34", 8),
        ("DeepLabV3+ ResNet-34", "deeplabv3p_resnet34", 8),
    ]
    seeds = [0, 1, 2]
    splits = ["valid", "test", "bolivia"]

    rows = []
    header = [
        "Model",
        "Seed",
        "Split",
        "Flood IoU (%)",
        "Flood F1 (%)",
        "Precision (%)",
        "Recall (%)",
        "Best Epoch",
        "Epochs Trained",
        "Hit Epoch Cap",
        "Batch Size",
        "Notes",
    ]

    for model_name, model_prefix, batch_size in models:
        for seed in seeds:
            hist_path = results_dir / f"{model_prefix}_seed{seed}_history.json"
            best_epoch = "N/A"
            epochs_trained = "N/A"
            hit_cap = "N/A"
            if hist_path.exists():
                hist = load_json(hist_path)
                epochs_trained = len(hist)
                best_idx = max(range(len(hist)), key=lambda i: hist[i]["val_flood_iou"])
                best_epoch = best_idx + 1
                hit_cap = "Yes" if epochs_trained >= max_cap else f"No (Stopped at Ep {epochs_trained})"

            for split in splits:
                res_path = results_dir / f"{model_prefix}_seed{seed}_{split}.json"
                if not res_path.exists():
                    continue
                data = load_json(res_path)
                g = data["global"]

                notes = ""
                rows.append({
                    "Model": model_name,
                    "Seed": seed,
                    "Split": split.capitalize(),
                    "Flood IoU (%)": f"{g['flood_iou'] * 100:.2f}%",
                    "Flood F1 (%)": f"{g['flood_f1'] * 100:.2f}%",
                    "Precision (%)": f"{g['flood_precision'] * 100:.2f}%",
                    "Recall (%)": f"{g['flood_recall'] * 100:.2f}%",
                    "Best Epoch": best_epoch,
                    "Epochs Trained": epochs_trained,
                    "Hit Epoch Cap": hit_cap,
                    "Batch Size": batch_size,
                    "Notes": notes,
                })

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [SAVED] {output_csv} ({len(rows)} rows)")


def generate_per_chip_tables(
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
):
    otsu_res_path = results_dir / "otsu_baseline_results.json"
    if not otsu_res_path.exists():
        # Fallback to run1 if not yet in results/
        otsu_res_path = results_dir / "run1" / "otsu_baseline_results.json"
    otsu_res = load_json(otsu_res_path)
    global_otsu_mean_thresh = otsu_res["global_thresholds_fitted"]["mean"]
    print(f"Global Otsu Mean Threshold: {global_otsu_mean_thresh:.6f}")

    models = ["unet_resnet34", "deeplabv3p_resnet34"]
    seeds = [0, 1, 2]

    # Pre-load U-Net seed 0 model to compute exact FP pixel counts per chip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_ckpt = results_dir / "checkpoints" / "unet_resnet34_seed0_best.pth"
    unet_model = None
    if unet_ckpt.exists():
        try:
            unet_model = FloodUNet(encoder_name="resnet34", in_channels=2, classes=1)
            unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=True))
            unet_model.to(device)
            unet_model.eval()
        except Exception as e:
            print(f"[WARN] Could not load U-Net checkpoint for FP counting: {e}")

    for split in ["bolivia", "test"]:
        split_csv = data_dir / "splits" / f"flood_{split}_data.csv"
        dataset = FloodDataset(split_csv=split_csv, data_dir=data_dir)

        # Collect per-chip metrics from JSON evaluations
        dl_chip_ious = defaultdict(lambda: defaultdict(list))
        for m in models:
            for s in seeds:
                p = results_dir / f"{m}_seed{s}_{split}.json"
                if p.exists():
                    d = load_json(p)
                    for item in d.get("per_chip_details", []):
                        dl_chip_ious[m][item["chip_id"]].append(item["flood_iou"] * 100.0)

        rows = []
        for i in range(len(dataset)):
            chip_id, s1_path, label_path = dataset.samples[i]
            if not Path(s1_path).exists() or not Path(label_path).exists():
                continue

            s1, label, cid = dataset[i]
            valid_mask = (label != -1) & (~torch.isnan(label))
            valid_count = int(valid_mask.sum())
            total_count = label.numel()
            flood_count = int((label == 1).sum())

            flood_pct = (flood_count / valid_count * 100.0) if valid_count > 0 else 0.0
            ignore_pct = ((total_count - valid_count) / total_count * 100.0)

            # Global Otsu Mean prediction
            pred_mask, _ = otsu_segment_chip(
                s1,
                variant="mean",
                threshold_override=global_otsu_mean_thresh,
            )
            cm_otsu = compute_confusion_matrix(pred_mask, label.numpy(), threshold=0.5, ignore_index=-1)
            f_denom = cm_otsu["tp"] + cm_otsu["fp"] + cm_otsu["fn"]
            otsu_iou = (cm_otsu["tp"] / f_denom * 100.0) if f_denom > 0 else 0.0

            # U-Net mean IoU across seeds
            unet_vals = dl_chip_ious["unet_resnet34"].get(cid, [])
            unet_mean_iou = float(np.mean(unet_vals)) if unet_vals else 0.0

            # DeepLabV3+ mean IoU across seeds
            dl_vals = dl_chip_ious["deeplabv3p_resnet34"].get(cid, [])
            dl_mean_iou = float(np.mean(dl_vals)) if dl_vals else 0.0

            # Compute U-Net FP pixels
            u_fp = 0
            if unet_model is not None:
                with torch.no_grad():
                    inp = s1.unsqueeze(0).to(device)
                    probs = torch.sigmoid(unet_model(inp)).squeeze().cpu().numpy()
                    u_pred = (probs >= 0.5).astype(np.int64)
                    cm_u = compute_confusion_matrix(u_pred, label.numpy(), threshold=0.5, ignore_index=-1)
                    u_fp = cm_u["fp"]

            # Criteria for meaningful IoU: >= 1% flood AND < 100% ignore
            is_meaningful = (flood_pct >= 1.0) and (ignore_pct < 99.99) and (valid_count > 0)
            status = "Valid Evaluation" if is_meaningful else "IoU not meaningful"

            note = ""
            if not is_meaningful:
                if ignore_pct >= 99.99:
                    note = "IoU not meaningful (100% ignore pixels)"
                else:
                    note = f"IoU not meaningful (<1% flood: {flood_pct:.2f}%)"

            rows.append({
                "Chip ID": cid,
                "Flood % (Valid Pixels)": f"{flood_pct:.2f}%",
                "Ignore %": f"{ignore_pct:.2f}%",
                "U-Net Flood IoU (%) [Mean]": f"{unet_mean_iou:.2f}%",
                "DeepLabV3+ Flood IoU (%) [Mean]": f"{dl_mean_iou:.2f}%",
                "Global Otsu Mean Flood IoU (%)": f"{otsu_iou:.2f}%",
                "Evaluation Status": status,
                "U-Net False Positive Pixels": u_fp,
                "Notes": note,
                "_is_meaningful": is_meaningful,
                "_raw_unet_iou": unet_mean_iou,
                "_raw_flood_pct": flood_pct,
                "_u_fp": u_fp,
            })

        if split == "bolivia":
            out_file = results_dir / "per_chip_bolivia.csv"
            header = [
                "Chip ID",
                "Flood % (Valid Pixels)",
                "Ignore %",
                "U-Net Flood IoU (%) [Mean]",
                "DeepLabV3+ Flood IoU (%) [Mean]",
                "Global Otsu Mean Flood IoU (%)",
                "Evaluation Status",
                "Notes",
            ]
            with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [SAVED] {out_file} ({len(rows)} chips)")

        elif split == "test":
            out_file = results_dir / "per_chip_test.csv"

            # 1. Rank 10 worst test chips by IoU strictly AMONG chips with >= 1% flood
            meaningful_indices = [idx for idx, r in enumerate(rows) if r["_is_meaningful"]]
            sorted_meaningful = sorted(meaningful_indices, key=lambda k: rows[k]["_raw_unet_iou"])
            worst_10_rank_map = {}
            for rank, idx in enumerate(sorted_meaningful[:10], start=1):
                worst_10_rank_map[idx] = rank

            # 2. Rank 10 chips with the most false-positive pixels
            sorted_fp_indices = sorted(range(len(rows)), key=lambda k: rows[k]["_u_fp"], reverse=True)
            top_10_fp_map = {}
            for rank, idx in enumerate(sorted_fp_indices[:10], start=1):
                top_10_fp_map[idx] = rank

            print("\n" + "=" * 80)
            print("10 WORST TEST CHIPS BY U-NET IOU (AMONG CHIPS WITH >= 1% FLOOD):")
            print("=" * 80)
            print(f"{'Rank':4s} | {'Chip ID':18s} | {'U-Net Mean IoU':16s} | {'Flood % of Valid':18s} | {'FP Pixels':10s}")
            print("-" * 80)
            for idx in sorted_meaningful[:10]:
                r = rows[idx]
                rank = worst_10_rank_map[idx]
                print(f"#{rank:<3d} | {r['Chip ID']:18s} | {r['U-Net Flood IoU (%) [Mean]']:16s} | {r['Flood % (Valid Pixels)']:18s} | {r['_u_fp']:<10d}")

            print("\n" + "=" * 80)
            print("10 TEST CHIPS WITH THE MOST FALSE-POSITIVE PIXELS (U-NET SEED 0):")
            print("=" * 80)
            print(f"{'Rank':4s} | {'Chip ID':18s} | {'FP Pixels':12s} | {'Flood % of Valid':18s} | {'U-Net Mean IoU':16s}")
            print("-" * 80)
            for idx in sorted_fp_indices[:10]:
                r = rows[idx]
                rank = top_10_fp_map[idx]
                print(f"#{rank:<3d} | {r['Chip ID']:18s} | {r['_u_fp']:<12d} | {r['Flood % (Valid Pixels)']:18s} | {r['U-Net Flood IoU (%) [Mean]']:16s}")

            for i, r in enumerate(rows):
                # Assign worst unet rank
                if i in worst_10_rank_map:
                    r["Worst U-Net Rank (>=1% Flood)"] = f"#{worst_10_rank_map[i]}"
                    if r["Notes"]:
                        r["Notes"] += f"; Ranked #{worst_10_rank_map[i]} worst IoU"
                    else:
                        r["Notes"] = f"Ranked #{worst_10_rank_map[i]} worst IoU"
                else:
                    r["Worst U-Net Rank (>=1% Flood)"] = ""

                # Assign FP rank
                if i in top_10_fp_map:
                    r["FP Rank (Most False Positives)"] = f"#{top_10_fp_map[i]}"
                    if r["Notes"]:
                        r["Notes"] += f"; Ranked #{top_10_fp_map[i]} highest FP ({r['_u_fp']} px)"
                    else:
                        r["Notes"] = f"Ranked #{top_10_fp_map[i]} highest FP ({r['_u_fp']} px)"
                else:
                    r["FP Rank (Most False Positives)"] = ""

            header = [
                "Chip ID",
                "Flood % (Valid Pixels)",
                "Ignore %",
                "U-Net Flood IoU (%) [Mean]",
                "DeepLabV3+ Flood IoU (%) [Mean]",
                "Global Otsu Mean Flood IoU (%)",
                "Evaluation Status",
                "Worst U-Net Rank (>=1% Flood)",
                "FP Rank (Most False Positives)",
                "U-Net False Positive Pixels",
                "Notes",
            ]
            with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"\n  [SAVED] {out_file} ({len(rows)} chips)\n")


if __name__ == "__main__":
    generate_per_seed_table()
    generate_per_chip_tables()
