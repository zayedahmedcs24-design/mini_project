"""
Profile TEST split chips by country:
1. Count of chips per country.
2. Mean flood fraction of valid pixels (excluding ignore label -1) per country.
3. Check if WorldPop raster is present in data/worldpop/ for that country.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.floodmap.data.dataset import FloodDataset

def profile_test_split():
    split_csv = PROJECT_ROOT / "data" / "splits" / "flood_test_data.csv"
    data_dir = PROJECT_ROOT / "data"
    dataset = FloodDataset(split_csv=split_csv, data_dir=data_dir)

    print(f"Total TEST chips loaded: {len(dataset)}\n")

    country_chips = defaultdict(list)
    country_flood_fracs = defaultdict(list)
    country_flood_pixels = defaultdict(int)
    country_valid_pixels = defaultdict(int)

    worldpop_dir = PROJECT_ROOT / "data" / "worldpop"
    wp_files = os.listdir(worldpop_dir) if worldpop_dir.exists() else []

    # ISO codes / country mapping
    country_iso = {
        "Bolivia": "bol",
        "Ghana": "gha",
        "India": "ind",
        "Mekong": "vnm_khm_lao",
        "Nigeria": "nga",
        "Pakistan": "pak",
        "Paraguay": "pry",
        "Somalia": "som",
        "Spain": "esp",
        "Sri-Lanka": "lka",
        "USA": "usa",
    }

    for i in range(len(dataset)):
        s1, label_tensor, chip_id = dataset[i]
        label = label_tensor.numpy()

        country = chip_id.split("_")[0]
        country_chips[country].append(chip_id)

        valid_mask = (label == 0) | (label == 1)
        flood_mask = (label == 1)

        n_valid = int(valid_mask.sum())
        n_flood = int(flood_mask.sum())

        frac = (n_flood / n_valid) if n_valid > 0 else 0.0
        country_flood_fracs[country].append(frac)
        country_valid_pixels[country] += n_valid
        country_flood_pixels[country] += n_flood

    print("=" * 95)
    print(f"{'Country':<15} | {'Chips':<6} | {'Mean Flood % (per chip)':<25} | {'Agg Flood %':<13} | {'WorldPop in data/worldpop/'}")
    print("=" * 95)

    for country in sorted(country_chips.keys()):
        chips = country_chips[country]
        fracs = country_flood_fracs[country]
        mean_pct = np.mean(fracs) * 100.0
        tot_valid = country_valid_pixels[country]
        tot_flood = country_flood_pixels[country]
        agg_pct = (tot_flood / tot_valid * 100.0) if tot_valid > 0 else 0.0

        iso = country_iso.get(country, country.lower()[:3])
        # Check if any wp file matches country code or name
        has_wp = any(iso in f.lower() or country.lower() in f.lower() for f in wp_files)
        wp_status = "PRESENT (" + ", ".join([f for f in wp_files if iso in f.lower()]) + ")" if has_wp else "NOT PRESENT"

        print(f"{country:<15} | {len(chips):<6} | {mean_pct:<25.2f}% | {agg_pct:<13.2f}% | {wp_status}")

    print("=" * 95)

    # Detailed per-chip statistics
    print("\nDetailed breakdown of top candidates:")
    for country in ["India", "Spain", "Sri-Lanka", "Paraguay", "Ghana"]:
        if country in country_chips:
            chips = country_chips[country]
            fracs = [f * 100.0 for f in country_flood_fracs[country]]
            print(f"\n--- {country} ({len(chips)} chips) ---")
            for cid, f in zip(chips, fracs):
                print(f"  {cid:<25}: {f:5.2f}% flood")

if __name__ == "__main__":
    profile_test_split()
