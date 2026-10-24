"""
Unit tests for FloodDataset.
Verifies raster loading, tensor shapes, dtypes, normalization, and class balance calculation.
"""

from pathlib import Path
import pytest
import torch
import numpy as np

from src.floodmap.data.dataset import FloodDataset


def test_flood_dataset_loading():
    """Verify that FloodDataset correctly loads chips with appropriate shapes and dtypes."""
    split_csv = Path("data/splits/flood_bolivia_data.csv")
    if not split_csv.exists():
        pytest.skip("Bolivia split CSV not found.")

    ds = FloodDataset(split_csv=split_csv, data_dir="data")
    assert len(ds) == 15, f"Expected 15 Bolivia chips, got {len(ds)}"

    s1, label, chip_id = ds[0]

    # Verify tensor shapes
    assert s1.shape == (2, 512, 512), f"Expected (2, 512, 512), got {s1.shape}"
    assert label.shape == (512, 512), f"Expected (512, 512), got {label.shape}"

    # Verify dtypes
    assert s1.dtype == torch.float32
    assert label.dtype == torch.int64

    # Verify label values are restricted to {-1, 0, 1}
    unique_vals = torch.unique(label).tolist()
    for val in unique_vals:
        assert val in [-1, 0, 1], f"Unexpected label value {val} found in chip {chip_id}"

    # Verify S1 normalized values are finite
    assert not torch.isnan(s1).any(), "Found NaNs in normalized S1 tensor"
    assert not torch.isinf(s1).any(), "Found Infs in normalized S1 tensor"


def test_class_balance_calculation():
    """Verify get_class_balance() computes valid distribution metrics."""
    split_csv = Path("data/splits/flood_bolivia_data.csv")
    if not split_csv.exists():
        pytest.skip("Bolivia split CSV not found.")

    ds = FloodDataset(split_csv=split_csv, data_dir="data")
    stats = ds.get_class_balance()

    assert "flood_pixels" in stats
    assert "dry_land_pixels" in stats
    assert "ignore_pixels" in stats
    assert "flood_pct_of_valid" in stats
    assert 0.0 <= stats["flood_pct_of_valid"] <= 100.0
