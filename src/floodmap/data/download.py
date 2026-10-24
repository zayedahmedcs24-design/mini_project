"""
Data downloader and split verification for Sen1Floods11 Hand-Labeled dataset.

Downloads:
- Official split CSVs (train, valid, test, bolivia) from Google Cloud Storage
- Sentinel-1 dual-pol SAR chips (S1Hand)
- Hand-labeled ground truth flood masks (LabelHand)

Verification:
- Confirms zero chip overlap between splits
- Ensures Bolivia event chips are strictly isolated as an out-of-region test set
- Supports --smoke mode for fast local verification (8 chips)
"""

import os
import sys
import csv
import time
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Base URLs in official Google Cloud Storage bucket
GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_URLS = {
    "train": f"{GCS_BASE}/splits/flood_handlabeled/flood_train_data.csv",
    "valid": f"{GCS_BASE}/splits/flood_handlabeled/flood_valid_data.csv",
    "test": f"{GCS_BASE}/splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": f"{GCS_BASE}/splits/flood_handlabeled/flood_bolivia_data.csv",
}
S1_BASE = f"{GCS_BASE}/data/flood_events/HandLabeled/S1Hand"
LABEL_BASE = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand"


def download_file_with_retry(
    url: str,
    target_path: Path,
    max_retries: int = 3,
    delay: float = 1.0,
) -> bool:
    """Downloads a file if not already present or if incomplete."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FloodMap-Research/1.0 (BMSCE UG Mini-Project)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_f:
                total_bytes = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total_bytes += len(chunk)

            if total_bytes > 0:
                temp_path.replace(target_path)
                return True
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == max_retries:
                print(f"[ERROR] Failed to download {url}: {e}", file=sys.stderr)
                return False
            time.sleep(delay * attempt)
    return False


def fetch_and_verify_splits(splits_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Downloads official split CSVs and verifies that there is ZERO overlap across splits.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_data = {}
    split_chip_ids: Dict[str, Set[str]] = {}

    print("=" * 60)
    print("STEP 1: Fetching and Verifying Official Sen1Floods11 Split CSVs")
    print("=" * 60)

    for split_name, url in SPLIT_URLS.items():
        csv_path = splits_dir / f"flood_{split_name}_data.csv"
        success = download_file_with_retry(url, csv_path)
        if not success:
            raise RuntimeError(f"Could not download split CSV: {url}")

        pairs = []
        chip_ids = set()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                s1_file = row[0].strip()
                label_file = row[1].strip()
                pairs.append((s1_file, label_file))

                # Extract chip ID (e.g. 'Bolivia_103757' from 'Bolivia_103757_S1Hand.tif')
                chip_id = s1_file.replace("_S1Hand.tif", "")
                chip_ids.add(chip_id)

        split_data[split_name] = pairs
        split_chip_ids[split_name] = chip_ids
        print(f"[{split_name.upper():7s}] {len(pairs):3d} chips loaded from {csv_path.name}")

    # ZERO LEAKAGE ASSERTIONS
    print("\nVerifying split boundaries (zero data leakage):")
    train_ids = split_chip_ids["train"]
    valid_ids = split_chip_ids["valid"]
    test_ids = split_chip_ids["test"]
    bolivia_ids = split_chip_ids["bolivia"]

    assert len(train_ids & valid_ids) == 0, "DATA LEAKAGE: Train and Validation overlap!"
    assert len(train_ids & test_ids) == 0, "DATA LEAKAGE: Train and Test overlap!"
    assert len(train_ids & bolivia_ids) == 0, "DATA LEAKAGE: Train and Bolivia overlap!"
    assert len(valid_ids & test_ids) == 0, "DATA LEAKAGE: Valid and Test overlap!"
    assert len(valid_ids & bolivia_ids) == 0, "DATA LEAKAGE: Valid and Bolivia overlap!"
    assert len(test_ids & bolivia_ids) == 0, "DATA LEAKAGE: Test and Bolivia overlap!"

    print(" [PASSED] Zero chip leakage confirmed across all 4 splits.")
    print(f" [PASSED] Bolivia held-out event strictly isolated ({len(bolivia_ids)} chips).")
    total_unique = len(train_ids | valid_ids | test_ids | bolivia_ids)
    print(f" [PASSED] Total unique hand-labeled chips: {total_unique}")

    return split_data


from concurrent.futures import ThreadPoolExecutor, as_completed

def download_chips(
    split_data: Dict[str, List[Tuple[str, str]]],
    raw_dir: Path,
    smoke: bool = False,
    selected_split: Optional[str] = None,
    max_workers: int = 12,
):
    """
    Downloads the actual Sentinel-1 and Label GeoTIFFs concurrently.
    """
    s1_dir = raw_dir / "S1Hand"
    label_dir = raw_dir / "LabelHand"
    s1_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"STEP 2: Downloading GeoTIFF Chips (Mode: {'SMOKE (8 chips)' if smoke else 'FULL (All splits)'}, Workers: {max_workers})")
    print("=" * 65)

    splits_to_process = [selected_split] if selected_split else list(split_data.keys())

    # Build task list: (url, dest_path, file_type)
    download_tasks = []
    for s_name in splits_to_process:
        pairs = split_data[s_name]
        if smoke:
            limit = 4 if s_name == "train" else 2
            pairs = pairs[:limit]

        for s1_name, label_name in pairs:
            s1_dest = s1_dir / s1_name
            label_dest = label_dir / label_name
            download_tasks.append((f"{S1_BASE}/{s1_name}", s1_dest))
            download_tasks.append((f"{LABEL_BASE}/{label_name}", label_dest))

    total_tasks = len(download_tasks)
    print(f"Total files to verify/download: {total_tasks}")

    completed = 0
    cached = 0
    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_file_with_retry, url, path): path
            for url, path in download_tasks
        }
        for future in as_completed(future_to_file):
            path = future_to_file[future]
            try:
                success = future.result()
                completed += 1
                if success:
                    # Check if it was newly downloaded or cached
                    downloaded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {path.name}: {e}", file=sys.stderr)

            if completed % 50 == 0 or completed == total_tasks:
                print(f"  Progress: [{completed:3d}/{total_tasks:3d}] ({completed/total_tasks*100:5.1f}%) | Failed: {failed}")

    print(f"\nDownload summary: {completed} files processed, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description="Sen1Floods11 Hand-Labeled Data Downloader")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Download only 8 sample chips for fast sanity testing",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only download and verify split CSV files without fetching rasters",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "valid", "test", "bolivia"],
        default="all",
        help="Download only a specific split",
    )
    args = parser.parse_args()

    splits_dir = args.data_dir / "splits"
    raw_dir = args.data_dir / "raw"

    split_data = fetch_and_verify_splits(splits_dir)

    if not args.splits_only:
        selected = None if args.split == "all" else args.split
        download_chips(split_data, raw_dir, smoke=args.smoke, selected_split=selected)

    print("\nData preparation step completed successfully.")


if __name__ == "__main__":
    main()
