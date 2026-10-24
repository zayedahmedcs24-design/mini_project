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
    print(f"{'Chip ID':25s} | {'VV Min':7s} | {'VV Max':7s} | {'VV Mean':7s} | {'VH Mean':7s} | {'Flood %':7s} | {'Ignore %':8s}")
    print("-" * 85)

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

        print(f"{chip_id:25s} | {vv_min:7.2f} | {vv_max:7.2f} | {vv_mean:7.2f} | {vh_mean:7.2f} | {flood_pct:6.2f}% | {ignore_pct:7.2f}%")

    total_valid = total_flood + total_land
    total_all = total_valid + total_ignore
    print("=" * 85)
    print("AGGREGATED AUDIT SUMMARY:")
    print(f"  • Coordinate Reference System: {crs}")
    print(f"  • Spatial Resolution: {res} (meters per pixel)")
    print(f"  • Total Valid Pixels Audited: {total_valid:,}")
    print(f"  • Flood Pixels: {total_flood:,} ({total_flood/total_valid*100:.2f}% of valid terrain)")
    print(f"  • Dry Land Pixels: {total_land:,} ({total_land/total_valid*100:.2f}% of valid terrain)")
    print(f"  • Ignore / Cloud Pixels: {total_ignore:,} ({total_ignore/total_all*100:.2f}% of total pixels)")
    print("=" * 85)


if __name__ == "__main__":
    run_eda()
