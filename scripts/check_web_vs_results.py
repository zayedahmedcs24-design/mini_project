"""
Consistency check script: scripts/check_web_vs_results.py

Verifies:
1. For all 24 chips that have exposure results (15 Bolivia + 9 Sri-Lanka):
   The number of predicted-flood pixels in web/data/<chip>/prediction*.png (valid pixels)
   must equal the expected-threshold (p >= 0.50) flooded pixel count behind results/exposure_*.json.
2. For all 30 chips:
   The meta.json IoU must equal the canonical ensemble IoU within 0.01 percentage points.
Fails loudly on any mismatch.
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.metrics import compute_confusion_matrix, compute_metrics_from_counts


def check_web_vs_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CHECK] Consistency validation running on device: {device}")

    # Load 6 ensemble checkpoints
    cp_dir = PROJECT_ROOT / "results" / "checkpoints"
    unet_models = []
    deeplab_models = []
    for s in [0, 1, 2]:
        m_u = FloodUNet(encoder_name="resnet34", in_channels=2, classes=1).to(device)
        m_u.load_state_dict(torch.load(cp_dir / f"unet_resnet34_seed{s}_best.pth", map_location=device, weights_only=False))
        m_u.eval()
        unet_models.append(m_u)

        m_d = FloodDeepLabV3Plus(encoder_name="resnet34", in_channels=2, classes=1).to(device)
        m_d.load_state_dict(torch.load(cp_dir / f"deeplabv3p_resnet34_seed{s}_best.pth", map_location=device, weights_only=False))
        m_d.eval()
        deeplab_models.append(m_d)

    # Load FloodDatasets
    bolivia_ds = FloodDataset(split_csv=PROJECT_ROOT / "data" / "splits" / "flood_bolivia_data.csv", data_dir=PROJECT_ROOT / "data")
    test_ds = FloodDataset(split_csv=PROJECT_ROOT / "data" / "splits" / "flood_test_data.csv", data_dir=PROJECT_ROOT / "data")

    dataset_map = {}
    for i, (cid, _, _) in enumerate(bolivia_ds.samples):
        dataset_map[cid] = (bolivia_ds, i)
    for i, (cid, _, _) in enumerate(test_ds.samples):
        dataset_map[cid] = (test_ds, i)

    # Load Exposure JSONs
    with open(PROJECT_ROOT / "results" / "exposure_bolivia.json", "r", encoding="utf-8") as f:
        exp_bol = json.load(f)["per_chip"]
    with open(PROJECT_ROOT / "results" / "exposure_sri-lanka.json", "r", encoding="utf-8") as f:
        exp_sl = json.load(f)["per_chip"]

    all_exposure_chips = {**exp_bol, **exp_sl}
    print(f"[CHECK] Total exposure chips to verify: {len(all_exposure_chips)} (expected: 24)")
    assert len(all_exposure_chips) == 24, f"Expected 24 exposure chips, found {len(all_exposure_chips)}"

    web_data_dir = PROJECT_ROOT / "web" / "data"
    chip_dirs = sorted([d.name for d in web_data_dir.iterdir() if d.is_dir()])
    print(f"[CHECK] Total web data chips to verify: {len(chip_dirs)} (expected: 30)")
    assert len(chip_dirs) == 30, f"Expected 30 chips in web/data, found {len(chip_dirs)}"

    mismatches = []

    # 1. Verify Exposure Pixel Counts for the 24 exposure chips
    print("\n--- 1. Checking Prediction PNG Pixel Counts vs Exposure JSONs (24 Chips) ---")
    for cid in sorted(all_exposure_chips.keys()):
        chip_dir = web_data_dir / cid
        exp_entry = all_exposure_chips[cid]

        # Ground truth label to filter ignore pixels
        ds, idx = dataset_map[cid]
        _, label_tensor, _ = ds[idx]
        label = label_tensor.numpy()
        valid_mask = (label != -1)

        # U-Net PNG check
        u_png_path = chip_dir / "prediction.png"
        u_png_arr = np.array(Image.open(u_png_path))
        # Valid predicted flood pixels: alpha > 0 and label != -1
        u_png_flood_px = int(np.sum((u_png_arr[..., 3] > 0) & valid_mask))
        u_exp_flood_px = int(exp_entry["unet_resnet34"]["expected"]["flooded_pixels"])

        if u_png_flood_px != u_exp_flood_px:
            err = f"[MISMATCH] {cid} U-Net: PNG flood pixels ({u_png_flood_px}) != Exposure expected flooded_pixels ({u_exp_flood_px})"
            print(f"  FAIL: {err}")
            mismatches.append(err)
        else:
            print(f"  PASS: {cid:<20} U-Net PNG ({u_png_flood_px:,} px) == Exposure JSON ({u_exp_flood_px:,} px)")

        # DeepLab PNG check
        d_png_path = chip_dir / "prediction_deeplab.png"
        d_png_arr = np.array(Image.open(d_png_path))
        d_png_flood_px = int(np.sum((d_png_arr[..., 3] > 0) & valid_mask))
        d_exp_flood_px = int(exp_entry["deeplabv3p_resnet34"]["expected"]["flooded_pixels"])

        if d_png_flood_px != d_exp_flood_px:
            err = f"[MISMATCH] {cid} DeepLab: PNG flood pixels ({d_png_flood_px}) != Exposure expected flooded_pixels ({d_exp_flood_px})"
            print(f"  FAIL: {err}")
            mismatches.append(err)
        else:
            print(f"  PASS: {cid:<20} DeepLab PNG ({d_png_flood_px:,} px) == Exposure JSON ({d_exp_flood_px:,} px)")

    # 2. Verify meta.json IoU vs Canonical Ensemble IoU for all 30 chips
    print("\n--- 2. Checking meta.json IoU vs Canonical Ensemble IoU within 0.01 pp (30 Chips) ---")
    for cid in chip_dirs:
        meta_path = web_data_dir / cid / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ds, idx = dataset_map[cid]
        s1_tensor, label_tensor, _ = ds[idx]
        label = label_tensor.numpy()

        x = s1_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            u_probs = [torch.sigmoid(m(x)).squeeze().cpu().numpy() for m in unet_models]
            d_probs = [torch.sigmoid(m(x)).squeeze().cpu().numpy() for m in deeplab_models]

        u_mean = np.mean(np.stack(u_probs, axis=0), axis=0)
        d_mean = np.mean(np.stack(d_probs, axis=0), axis=0)

        cm_u = compute_confusion_matrix(u_mean, label, threshold=0.5, ignore_index=-1)
        m_u = compute_metrics_from_counts(cm_u["tp"], cm_u["fp"], cm_u["fn"], cm_u["tn"])
        canon_u_iou = m_u["flood_iou"] * 100.0

        cm_d = compute_confusion_matrix(d_mean, label, threshold=0.5, ignore_index=-1)
        m_d = compute_metrics_from_counts(cm_d["tp"], cm_d["fp"], cm_d["fn"], cm_d["tn"])
        canon_d_iou = m_d["flood_iou"] * 100.0

        meta_u_iou = meta["metrics"]["unet_resnet34"]["iou"]
        meta_d_iou = meta["metrics"]["deeplabv3p_resnet34"]["iou"]

        delta_u = abs(meta_u_iou - canon_u_iou)
        delta_d = abs(meta_d_iou - canon_d_iou)

        if delta_u > 0.01:
            err = f"[MISMATCH] {cid} U-Net IoU: meta.json ({meta_u_iou:.2f}%) vs Canon ({canon_u_iou:.2f}%), delta={delta_u:.4f} pp"
            print(f"  FAIL: {err}")
            mismatches.append(err)
        else:
            print(f"  PASS: {cid:<20} U-Net IoU: meta={meta_u_iou:5.2f}% vs canon={canon_u_iou:5.2f}% (delta: {delta_u:.4f} pp)")

        if delta_d > 0.01:
            err = f"[MISMATCH] {cid} DeepLab IoU: meta.json ({meta_d_iou:.2f}%) vs Canon ({canon_d_iou:.2f}%), delta={delta_d:.4f} pp"
            print(f"  FAIL: {err}")
            mismatches.append(err)
        else:
            print(f"  PASS: {cid:<20} DeepLab IoU: meta={meta_d_iou:5.2f}% vs canon={canon_d_iou:5.2f}% (delta: {delta_d:.4f} pp)")

    print("\n" + "="*80)
    if mismatches:
        print(f"CONSISTENCY CHECK FAILED WITH {len(mismatches)} MISMATCHES:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("CONSISTENCY CHECK PASSED: 100% MATCH ACROSS ALL 24 EXPOSURE PIXEL COUNTS AND ALL 30 CHIP IOUS!")
        print("="*80)


if __name__ == "__main__":
    check_web_vs_results()
