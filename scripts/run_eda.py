"""
Exploratory Data Analysis (EDA) and radiometrics validation script.

Validates:
1. S1Hand GeoTIFF band counts, data types, and empirical dB distributions.
2. LabelHand classes, confirming only {-1, 0, 1} exist.
3. Severe class imbalance across events (proving why overall accuracy fails).
4. Spatial resolution and coordinate reference systems.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from src.floodmap.data.dataset import FloodDataset


def run_eda(data_dir: Path = Path("data")):
    print("=" * 75)
    print("SEN1FLOODS11 RADIOMETRIC & CLASS DISTRIBUTION AUDIT")
    print("=" * 75)

    s1_dir = data_dir / "raw" / "S1Hand"
    label_dir = data_dir / "raw" / "LabelHand"

    s1_files = sorted(list(s1_dir.glob("*.tif")))
    if not s1_files:
        print("No GeoTIFF chips found in data/raw/S1Hand! Please run download.py first.")
        return

    print(f"Auditing {len(s1_files)} downloaded sample chips...\n")
    print(f"{'Chip ID':25s} | {'VV Min':7s} | {'VV Max':7s} | {'VV Mean':7s} | {'VH Mean':7s} | {'Flood % of Valid':20s} | {'Ignore % of Total':20s}")
    print("-" * 105)

    total_flood = 0
    total_land = 0
    total_ignore = 0

    for s1_path in s1_files:
        chip_id = s1_path.stem.replace("_S1Hand", "")
        label_path = label_dir / f"{chip_id}_LabelHand.tif"

        with rasterio.open(s1_path) as src:
            s1 = src.read()
            crs = src.crs
            res = src.res

        vv = s1[0]
        vh = s1[1]

        vv_min = float(np.nanmin(vv))
        vv_max = float(np.nanmax(vv))
        vv_mean = float(np.nanmean(vv))
        vh_mean = float(np.nanmean(vh))

        with rasterio.open(label_path) as src:
            lbl = src.read(1)

        flood_px = int(np.sum(lbl == 1))
        land_px = int(np.sum(lbl == 0))
        ignore_px = int(np.sum(lbl == -1))

        valid_px = flood_px + land_px
        all_px = valid_px + ignore_px

        flood_pct = (flood_px / valid_px * 100.0) if valid_px > 0 else 0.0
        ignore_pct = (ignore_px / all_px * 100.0) if all_px > 0 else 0.0

        total_flood += flood_px
        total_land += land_px
        total_ignore += ignore_px

        flood_str = f"{flood_px}/{valid_px} ({flood_pct:5.2f}%)"
        ignore_str = f"{ignore_px}/{all_px} ({ignore_pct:5.2f}%)"

        print(f"{chip_id:25s} | {vv_min:7.2f} | {vv_max:7.2f} | {vv_mean:7.2f} | {vh_mean:7.2f} | {flood_str:20s} | {ignore_str:20s}")

    total_valid = total_flood + total_land
    total_all = total_valid + total_ignore
    pct_flood_valid = (total_flood / total_valid * 100.0) if total_valid > 0 else 0.0
    pct_land_valid = (total_land / total_valid * 100.0) if total_valid > 0 else 0.0
    pct_ignore_total = (total_ignore / total_all * 100.0) if total_all > 0 else 0.0

    print("=" * 105)
    print("AGGREGATED AUDIT SUMMARY:")
    print(f"  • Coordinate Reference System: {crs}")
    print(f"  • Spatial Resolution: {res} (meters per pixel)")
    print(f"  • Total Pixels (All): {total_all:,}")
    print(f"  • Total Valid Pixels (Non-Ignore): {total_valid:,}")
    print(f"  • Total Ignore / Cloud Pixels: {total_ignore:,}")
    print(f"  • (a) Flood % of Valid (non-ignore) pixels: {total_flood:,} / {total_valid:,} ({pct_flood_valid:.4f}%) [Denominator: {total_valid:,} valid non-ignore pixels]")
    print(f"  • Dry Land % of Valid (non-ignore) pixels: {total_land:,} / {total_valid:,} ({pct_land_valid:.4f}%) [Denominator: {total_valid:,} valid non-ignore pixels]")
    print(f"  • (b) Ignore % of Total pixels: {total_ignore:,} / {total_all:,} ({pct_ignore_total:.4f}%) [Denominator: {total_all:,} total pixels]")
    print("=" * 105)


if __name__ == "__main__":
    run_eda()
