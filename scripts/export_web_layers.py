"""
Export Precomputed Web Layer Overlays and Metadata for Static Interactive Viewer.

Exports for:
1. 15 Bolivia chips (Sen1Floods11 Hand-Labeled Bolivia Split)
2. 9 Sri-Lanka chips (Sen1Floods11 Official TEST Split)
3. 6 other TEST chips:
   - 3 highest flood IoU chips (>= 1% flood): USA_905409, Nigeria_812045, Paraguay_790830
   - 3 lowest flood IoU chips (>= 1% flood): Somalia_699062, Paraguay_271769, Pakistan_70625

Outputs per chip in web/data/<chip>/:
- sar_vv.png (2-98 percentile stretch, RGBA)
- ground_truth.png (flood blue, dry transparent, ignore gray)
- prediction.png (U-Net 3-seed ensemble mean, p > 0.50)
- prediction_deeplab.png (DeepLabV3+ 3-seed ensemble mean, p > 0.50)
- entropy.png (ensemble predictive entropy, plasma colormap with uncertainty alpha scaling)
- error.png (FP red, FN orange, ignore gray, correct transparent)
- meta.json (EPSG:4326 bounds, centroid, metrics, split info, and full exposure data)

Also writes web/data/index.json.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import rasterio
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.uncertainty.ensemble import compute_uncertainty_maps

# Targeted chips to export
BOLIVIA_CHIPS = [
    "Bolivia_103757", "Bolivia_129334", "Bolivia_195474", "Bolivia_23014",
    "Bolivia_233925", "Bolivia_242570", "Bolivia_290290", "Bolivia_294583",
    "Bolivia_312675", "Bolivia_314919", "Bolivia_360519", "Bolivia_379434",
    "Bolivia_432776", "Bolivia_60373", "Bolivia_76104"
]

SRILANKA_CHIPS = [
    "Sri-Lanka_117737", "Sri-Lanka_1049830", "Sri-Lanka_534068", "Sri-Lanka_922192",
    "Sri-Lanka_377277", "Sri-Lanka_649970", "Sri-Lanka_450918", "Sri-Lanka_849649",
    "Sri-Lanka_713926"
]

# 3 highest and 3 lowest IoU test chips (>= 1% flood)
OTHER_TEST_CHIPS_HIGH = [
    "USA_905409",       # 16.97% flood, U-Net IoU 91.53%, DeepLab IoU 89.65%
    "Nigeria_812045",   # 50.65% flood, U-Net IoU 91.00%, DeepLab IoU 90.80%
    "Paraguay_790830"   # 12.23% flood, U-Net IoU 90.91%, DeepLab IoU 89.51%
]

OTHER_TEST_CHIPS_LOW = [
    "Somalia_699062",   # 1.62% flood, U-Net IoU 3.15%, DeepLab IoU 5.31%
    "Paraguay_271769",  # 1.00% flood, U-Net IoU 4.37%, DeepLab IoU 14.00%
    "Pakistan_70625"    # 1.78% flood, U-Net IoU 5.26%, DeepLab IoU 5.25%
]


def load_all_models(device: torch.device):
    cp_dir = PROJECT_ROOT / "results" / "checkpoints"
    unet_models = []
    deeplab_models = []
    for s in [0, 1, 2]:
        cp_u = cp_dir / f"unet_resnet34_seed{s}_best.pth"
        m_u = FloodUNet(encoder_name="resnet34", in_channels=2, classes=1).to(device)
        m_u.load_state_dict(torch.load(cp_u, map_location=device, weights_only=False))
        m_u.eval()
        unet_models.append(m_u)

        cp_d = cp_dir / f"deeplabv3p_resnet34_seed{s}_best.pth"
        m_d = FloodDeepLabV3Plus(encoder_name="resnet34", in_channels=2, classes=1).to(device)
        m_d.load_state_dict(torch.load(cp_d, map_location=device, weights_only=False))
        m_d.eval()
        deeplab_models.append(m_d)
    return unet_models, deeplab_models


def export_layers():
    out_dir = PROJECT_ROOT / "web" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("Loading 6 ensemble checkpoints...")
    unet_models, deeplab_models = load_all_models(device)

    # Load precomputed exposure jsons if present
    exposure_data = {}
    bol_json_p = PROJECT_ROOT / "results" / "exposure_bolivia.json"
    if bol_json_p.exists():
        with open(bol_json_p, "r", encoding="utf-8") as f:
            exposure_data["bolivia"] = json.load(f)

    sl_json_p = PROJECT_ROOT / "results" / "exposure_sri-lanka.json"
    if sl_json_p.exists():
        with open(sl_json_p, "r", encoding="utf-8") as f:
            exposure_data["sri-lanka"] = json.load(f)

    # Build work items: (chip_id, group_name, split_name, has_exposure, event_key)
    work_items = []
    for cid in BOLIVIA_CHIPS:
        work_items.append((cid, "Bolivia (Held-out Event)", "Sen1Floods11 Hand-Labeled Bolivia Split", True, "bolivia"))
    for cid in SRILANKA_CHIPS:
        work_items.append((cid, "Sri-Lanka (Test Split)", "Sen1Floods11 Official TEST Split", True, "sri-lanka"))
    for cid in OTHER_TEST_CHIPS_HIGH:
        work_items.append((cid, "Test Split (High IoU >=90%)", "Sen1Floods11 Official TEST Split", False, None))
    for cid in OTHER_TEST_CHIPS_LOW:
        work_items.append((cid, "Test Split (Low IoU <=5%)", "Sen1Floods11 Official TEST Split", False, None))

    print(f"\nExporting {len(work_items)} chips to {out_dir}...")
    index_entries = []

    for i, (cid, group_name, split_name, has_exp, ev_key) in enumerate(work_items):
        t0 = time.time()
        chip_dir = out_dir / cid
        chip_dir.mkdir(parents=True, exist_ok=True)

        # 1. Read S1 and Label
        s1_path = PROJECT_ROOT / "data" / "raw" / "S1Hand" / f"{cid}_S1Hand.tif"
        lbl_path = PROJECT_ROOT / "data" / "raw" / "LabelHand" / f"{cid}_LabelHand.tif"

        with rasterio.open(s1_path) as s1_src:
            s1_arr = s1_src.read()  # (2, 512, 512)
            bounds = s1_src.bounds
            crs = s1_src.crs

        with rasterio.open(lbl_path) as lbl_src:
            label = lbl_src.read(1)  # (512, 512)

        # Clean NaNs in SAR if any
        s1_clean = np.nan_to_num(s1_arr, nan=-25.0)

        # Centroid & Bounds in EPSG:4326
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        c_lat = (south + north) / 2.0
        c_lon = (west + east) / 2.0

        # Dynamic planar UTM CRS
        chip_box = box(west, south, east, north)
        chip_gdf = gpd.GeoDataFrame(geometry=[chip_box], crs=crs)
        utm_crs = chip_gdf.estimate_utm_crs()
        chip_utm = chip_gdf.to_crs(utm_crs)
        chip_area_m2 = float(chip_utm.geometry[0].area)
        chip_area_km2 = chip_area_m2 / 1e6
        pixel_area_m2 = chip_area_m2 / (512 * 512)

        # 2. sar_vv.png (2-98 percentile stretch)
        vv = s1_clean[0]
        valid_vv = vv[np.isfinite(vv)]
        p2 = np.percentile(valid_vv, 2)
        p98 = np.percentile(valid_vv, 98)
        stretched_vv = np.clip((vv - p2) / (p98 - p2 + 1e-7), 0.0, 1.0) * 255.0
        stretched_vv = stretched_vv.astype(np.uint8)

        sar_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        sar_rgba[..., 0] = stretched_vv
        sar_rgba[..., 1] = stretched_vv
        sar_rgba[..., 2] = stretched_vv
        sar_rgba[..., 3] = 255  # fully opaque
        Image.fromarray(sar_rgba).save(chip_dir / "sar_vv.png", optimize=True)

        # 3. ground_truth.png
        # Flood (1) = Blue, Dry (0) = Transparent, Ignore (-1) = Gray
        gt_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        # Flood
        gt_rgba[label == 1] = [41, 128, 185, 200]
        # Ignore
        gt_rgba[label == -1] = [127, 140, 141, 140]
        # Dry remains [0, 0, 0, 0]
        Image.fromarray(gt_rgba).save(chip_dir / "ground_truth.png", optimize=True)

        # 4. Model Inference
        x_tensor = torch.from_numpy(s1_clean).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u_probs = [torch.sigmoid(m(x_tensor)).squeeze().cpu().numpy() for m in unet_models]
            d_probs = [torch.sigmoid(m(x_tensor)).squeeze().cpu().numpy() for m in deeplab_models]

        u_mean = compute_uncertainty_maps(u_probs)["mean"]
        d_mean = compute_uncertainty_maps(d_probs)["mean"]

        # 5. prediction.png (U-Net, p > 0.50)
        u_pred = (u_mean >= 0.50) & (label != -1)
        pred_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        pred_rgba[u_pred] = [0, 206, 209, 200]  # bright cyan / turquoise
        Image.fromarray(pred_rgba).save(chip_dir / "prediction.png", optimize=True)

        # 6. prediction_deeplab.png (DeepLabV3+, p > 0.50)
        d_pred = (d_mean >= 0.50) & (label != -1)
        pred_dl_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        pred_dl_rgba[d_pred] = [46, 204, 113, 200]  # emerald green
        Image.fromarray(pred_dl_rgba).save(chip_dir / "prediction_deeplab.png", optimize=True)

        # 7. entropy.png (Predictive Entropy)
        # H = -p*log2(p) - (1-p)*log2(1-p)
        p_safe = np.clip(u_mean, 1e-7, 1.0 - 1e-7)
        entropy = -p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe)
        entropy_norm = np.clip(entropy, 0.0, 1.0)

        # Map to plasma colormap
        cmap = plt.get_cmap("plasma")
        colored = (cmap(entropy_norm)[..., :3] * 255).astype(np.uint8)

        ent_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        ent_rgba[..., :3] = colored
        # Alpha scaling: transparent for low entropy, opaque for high boundary uncertainty
        alpha_ent = np.clip(np.power(entropy_norm, 1.6) * 255.0, 0, 240).astype(np.uint8)
        alpha_ent[label == -1] = 0  # mask ignore pixels
        alpha_ent[entropy_norm < 0.08] = 0  # clear confident regions
        ent_rgba[..., 3] = alpha_ent
        Image.fromarray(ent_rgba).save(chip_dir / "entropy.png", optimize=True)

        # 8. error.png (FP red, FN orange, ignore gray)
        err_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        fp_mask = (u_mean >= 0.50) & (label == 0)
        fn_mask = (u_mean < 0.50) & (label == 1)
        ign_mask = (label == -1)

        err_rgba[fp_mask] = [231, 76, 60, 220]   # Red (False Positive)
        err_rgba[fn_mask] = [230, 126, 34, 220]  # Orange (False Negative)
        err_rgba[ign_mask] = [149, 165, 166, 120] # Gray (Ignore)
        Image.fromarray(err_rgba).save(chip_dir / "error.png", optimize=True)

        # 9. Compute Per-Chip Classification Metrics
        valid_mask = (label == 0) | (label == 1)
        gt_binary = (label == 1) & valid_mask
        n_valid = int(valid_mask.sum())
        n_gt_flood = int(gt_binary.sum())
        flood_frac_valid = (n_gt_flood / n_valid * 100.0) if n_valid > 0 else 0.0
        ignore_frac = float((label == -1).sum() / label.size * 100.0)

        def calc_metrics(pred_bin):
            tp = int(((pred_bin == 1) & (gt_binary == 1)).sum())
            fp = int(((pred_bin == 1) & (gt_binary == 0)).sum())
            fn = int(((pred_bin == 0) & (gt_binary == 1)).sum())
            tn = int(((pred_bin == 0) & (gt_binary == 0)).sum())

            denom_iou = tp + fp + fn
            iou = (tp / denom_iou * 100.0) if denom_iou > 0 else (100.0 if (tp == 0 and fn == 0 and fp == 0) else 0.0)
            prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            return {
                "iou": round(float(iou), 2),
                "f1": round(float(f1), 2),
                "precision": round(float(prec), 2),
                "recall": round(float(rec), 2),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
            }

        metrics_unet = calc_metrics((u_mean >= 0.50) & valid_mask)
        metrics_dl = calc_metrics((d_mean >= 0.50) & valid_mask)

        # 10. Pull Exposure Data (if present)
        chip_exposure = None
        if has_exp and ev_key in exposure_data:
            ev_chips = exposure_data[ev_key].get("per_chip", {})
            if cid in ev_chips:
                raw_exp = ev_chips[cid]
                chip_exposure = {
                    "available": True,
                    "event_key": ev_key,
                    "total_footprint_population": raw_exp.get("total_footprint_population"),
                    "implied_density_per_km2": raw_exp.get("implied_population_density_per_km2"),
                    "ground_truth": raw_exp.get("ground_truth"),
                    "unet_resnet34": raw_exp.get("unet_resnet34"),
                    "deeplabv3p_resnet34": raw_exp.get("deeplabv3p_resnet34"),
                    "caveats": (
                        "Sensitivity range, not a statistical confidence interval; "
                        "WorldPop year and product differ per event; OSM extract is from 2026; "
                        "roads 'no roads in the extract' is not 0 km."
                    )
                }

        if chip_exposure is None:
            chip_exposure = {
                "available": False,
                "note": "Exposure analysis not computed for this scene (WorldPop raster not present in benchmark dataset)."
            }

        # 11. Write meta.json
        meta_dict = {
            "chip_id": cid,
            "group": group_name,
            "split_name": split_name,
            "banner_text": "Bolivia = held-out event, 15 chips" if "Bolivia" in cid else "TEST split (not used in training)",
            "bounds": {
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north)
            },
            "leaflet_bounds": [
                [float(south), float(west)],
                [float(north), float(east)]
            ],
            "centroid": {"lat": round(float(c_lat), 5), "lon": round(float(c_lon), 5)},
            "utm_crs": str(utm_crs),
            "chip_area_km2": round(chip_area_km2, 2),
            "pixel_area_m2": round(pixel_area_m2, 2),
            "valid_pixels": n_valid,
            "flood_fraction_valid_pct": round(flood_frac_valid, 2),
            "ignore_fraction_pct": round(ignore_frac, 2),
            "metrics": {
                "unet_resnet34": metrics_unet,
                "deeplabv3p_resnet34": metrics_dl
            },
            "exposure": chip_exposure
        }

        with open(chip_dir / "meta.json", "w", encoding="utf-8") as f_meta:
            json.dump(meta_dict, f_meta, indent=2)

        index_entries.append({
            "chip_id": cid,
            "group": group_name,
            "split_name": split_name,
            "centroid": meta_dict["centroid"],
            "leaflet_bounds": meta_dict["leaflet_bounds"],
            "flood_fraction_valid_pct": round(flood_frac_valid, 2),
            "unet_iou": metrics_unet["iou"],
            "deeplab_iou": metrics_dl["iou"],
            "has_exposure": meta_dict["exposure"]["available"],
            "banner_text": meta_dict["banner_text"]
        })

        t1 = time.time()
        print(f"  [{i+1:2d}/{len(work_items)}] {cid:<20} | IoU U-Net: {metrics_unet['iou']:5.2f}% | IoU DL: {metrics_dl['iou']:5.2f}% | Exp: {'YES' if meta_dict['exposure']['available'] else 'NO '} ({t1 - t0:.2f}s)")

    # 12. Write web/data/index.json
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f_idx:
        json.dump({
            "total_chips": len(index_entries),
            "groups": [
                "Bolivia (Held-out Event)",
                "Sri-Lanka (Test Split)",
                "Test Split (High IoU >=90%)",
                "Test Split (Low IoU <=5%)"
            ],
            "chips": index_entries
        }, f_idx, indent=2)

    print(f"\n[SAVED] {index_path} ({len(index_entries)} chips indexed)")


if __name__ == "__main__":
    export_layers()
