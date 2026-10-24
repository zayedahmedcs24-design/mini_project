"""
Data augmentation pipelines using Albumentations for SAR flood segmentation.

Features:
- Random spatial cropping (e.g. 256x256 from 512x512)
- Flips and rotations (dihedral group D4)
- Zero RGB-specific augmentations (ColorJitter is strictly prohibited for SAR)
- Preserves integer label values {-1, 0, 1} with nearest-neighbor interpolation
"""

import albumentations as A
import cv2


def get_train_transforms(crop_size: int = 256) -> A.Compose:
    """
    Augmentation pipeline for training chips.
    """
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def get_valid_transforms() -> None:
    """
    Validation uses full 512x512 chips without geometric deformation.
    """
    return None
