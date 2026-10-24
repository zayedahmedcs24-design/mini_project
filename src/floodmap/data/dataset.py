"""
PyTorch Dataset for Sentinel-1 dual-pol SAR flood chips (Sen1Floods11).

Key Design Principles:
1. S1Hand GeoTIFF chips are ALREADY in decibels (dB), with values typically
   ranging from -45 dB to +5 dB. NO log10 is applied.
2. Robust clipping and channel-wise normalization:
   VV: clipped to [-35.0, 5.0] dB
   VH: clipped to [-45.0, 0.0] dB
3. Ground truth labels:
    1 = Flooded / Water
    0 = Non-Flooded / Dry Land
   -1 = Ignore / Cloud / Invalid
4. NaNs or infinities in raw SAR are safely imputed with floor backscatter values.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Default empirical normalization parameters from Sen1Floods11 training split
DEFAULT_VV_MEAN = -13.0
DEFAULT_VV_STD = 6.0
DEFAULT_VH_MEAN = -19.5
DEFAULT_VH_STD = 6.0

DEFAULT_VV_CLIP = (-35.0, 5.0)
DEFAULT_VH_CLIP = (-45.0, 0.0)


class FloodDataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 Hand-Labeled flood segmentation.
    """

    def __init__(
        self,
        split_csv: Union[str, Path],
        data_dir: Union[str, Path] = "data",
        transform: Optional[Callable] = None,
        norm_mean: Tuple[float, float] = (DEFAULT_VV_MEAN, DEFAULT_VH_MEAN),
        norm_std: Tuple[float, float] = (DEFAULT_VV_STD, DEFAULT_VH_STD),
        clip_vv: Tuple[float, float] = DEFAULT_VV_CLIP,
        clip_vh: Tuple[float, float] = DEFAULT_VH_CLIP,
        cache_in_memory: bool = False,
    ):
        self.split_csv = Path(split_csv)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.norm_mean = np.array(norm_mean, dtype=np.float32)
        self.norm_std = np.array(norm_std, dtype=np.float32)
        self.clip_vv = clip_vv
        self.clip_vh = clip_vh
        self.cache_in_memory = cache_in_memory

        self.s1_dir = self.data_dir / "raw" / "S1Hand"
        self.label_dir = self.data_dir / "raw" / "LabelHand"

        self.samples: List[Tuple[str, str, str]] = []  # (chip_id, s1_path, label_path)
        self._load_split_csv()

        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str]] = {}

    def _load_split_csv(self):
        """Loads and parses the split CSV file."""
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        with open(self.split_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                parts = line.split(",")
                s1_name = parts[0].strip()
                label_name = parts[1].strip()

                chip_id = s1_name.replace("_S1Hand.tif", "")
                s1_path = self.s1_dir / s1_name
                label_path = self.label_dir / label_name

                self.samples.append((chip_id, str(s1_path), str(label_path)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid chip pairs found in {self.split_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_raster(self, s1_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Reads and cleans raw GeoTIFF rasters."""
        # Read S1 (2 bands: 0 = VV, 1 = VH)
        with rasterio.open(s1_path) as src:
            s1 = src.read()  # Shape: (2, 512, 512), float32

        # Read Label (1 band: 512, 512, int16)
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Shape: (512, 512), int16

        # Robust handling of NaNs / Infs in S1
        nan_vv = np.isnan(s1[0]) | np.isinf(s1[0])
        nan_vh = np.isnan(s1[1]) | np.isinf(s1[1])
        s1[0, nan_vv] = self.clip_vv[0]
        s1[1, nan_vh] = self.clip_vh[0]

        # Clip extreme radar outliers
        s1[0] = np.clip(s1[0], self.clip_vv[0], self.clip_vv[1])
        s1[1] = np.clip(s1[1], self.clip_vh[0], self.clip_vh[1])

        # Channel-wise Z-score Normalization: (x - mean) / std
        s1[0] = (s1[0] - self.norm_mean[0]) / (self.norm_std[0] + 1e-7)
        s1[1] = (s1[1] - self.norm_mean[1]) / (self.norm_std[1] + 1e-7)

        return s1, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        chip_id, s1_path, label_path = self.samples[idx]

        s1, label = self._read_raster(s1_path, label_path)

        # Apply Albumentations augmentations if provided
        if self.transform is not None:
            # Albumentations expects (H, W, C) for image and (H, W) for mask
            s1_hwc = np.transpose(s1, (1, 2, 0))  # (512, 512, 2)
            augmented = self.transform(image=s1_hwc, mask=label)
            s1_hwc = augmented["image"]
            label = augmented["mask"]
            s1 = np.transpose(s1_hwc, (2, 0, 1))  # (2, H, W)

        # Convert to PyTorch tensors
        s1_tensor = torch.from_numpy(s1).float()
        label_tensor = torch.from_numpy(label).long()

        item = (s1_tensor, label_tensor, chip_id)

        if self.cache_in_memory:
            self._cache[idx] = item

        return item

    def get_class_balance(self, verbose: bool = False) -> Dict[str, Union[int, float]]:
        """
        Computes the empirical class balance across all chips in the dataset.
        Reports both:
          (a) flood % of valid (non-ignore) pixels, with valid_pixels denominator
          (b) ignore % of total pixels, with total_pixels denominator
        """
        total_flood = 0
        total_land = 0
        total_ignore = 0

        for _, _, label_path in self.samples:
            if not Path(label_path).exists():
                continue
            with rasterio.open(label_path) as src:
                lbl = src.read(1)
            total_flood += int(np.sum(lbl == 1))
            total_land += int(np.sum(lbl == 0))
            total_ignore += int(np.sum(lbl == -1))

        total_valid = total_flood + total_land
        total_all = total_valid + total_ignore

        pct_flood_valid = (total_flood / total_valid * 100.0) if total_valid > 0 else 0.0
        pct_land_valid = (total_land / total_valid * 100.0) if total_valid > 0 else 0.0
        pct_ignore_total = (total_ignore / total_all * 100.0) if total_all > 0 else 0.0

        if verbose:
            print(f"(a) Flood % of valid pixels: {total_flood:,} / {total_valid:,} ({pct_flood_valid:.2f}%) [Denominator: {total_valid:,} valid non-ignore pixels]")
            print(f"(b) Ignore % of total pixels: {total_ignore:,} / {total_all:,} ({pct_ignore_total:.2f}%) [Denominator: {total_all:,} total pixels]")

        return {
            "total_chips": len(self.samples),
            "flood_pixels": total_flood,
            "dry_land_pixels": total_land,
            "ignore_pixels": total_ignore,
            "valid_pixels": total_valid,
            "total_pixels": total_all,
            "denominator_valid_pixels": total_valid,
            "denominator_total_pixels": total_all,
            "flood_pct_of_valid": pct_flood_valid,
            "dry_land_pct_of_valid": pct_land_valid,
            "ignore_pct_of_total": pct_ignore_total,
        }
