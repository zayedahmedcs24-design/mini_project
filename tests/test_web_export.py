"""
Regression tests for web layer export consistency.

Verifies that for 3 representative chips:
1. One Bolivia chip: Bolivia_129334
2. One Sri-Lanka chip: Sri-Lanka_713926
3. One TEST non-exposure chip: USA_905409

The exporter's ensemble prediction mask at p >= 0.50 on valid pixels
equals the evaluation pipeline's mask pixel-for-pixel for both U-Net and DeepLabV3+,
and the meta.json IoU/TP/FP/FN/TN metrics equal the evaluation pipeline values.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.floodmap.data.dataset import FloodDataset
from src.floodmap.models.unet import FloodUNet
from src.floodmap.models.deeplab import FloodDeepLabV3Plus
from src.floodmap.metrics import compute_confusion_matrix, compute_metrics_from_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CHIPS = [
    ("Bolivia_129334", "data/splits/flood_bolivia_data.csv"),
    ("Sri-Lanka_713926", "data/splits/flood_test_data.csv"),
    ("USA_905409", "data/splits/flood_test_data.csv"),
]


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def models(device):
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
    return {"unet": unet_models, "deeplab": deeplab_models}


@pytest.mark.parametrize("chip_id, split_rel_path", TEST_CHIPS)
def test_web_export_mask_and_metrics_consistency(chip_id, split_rel_path, device, models):
    """
    Asserts pixel-for-pixel mask equality and exact metric equality between
    web/data/<chip>/ assets and evaluation pipeline.
    """
    web_dir = PROJECT_ROOT / "web" / "data" / chip_id
    assert web_dir.exists(), f"Web data directory missing for {chip_id}"

    # 1. Load canonical dataset tensor and label
    split_csv = PROJECT_ROOT / split_rel_path
    ds = FloodDataset(split_csv=split_csv, data_dir=PROJECT_ROOT / "data")
    idx = None
    for i, (cid, _, _) in enumerate(ds.samples):
        if cid == chip_id:
            idx = i
            break
    assert idx is not None, f"Chip {chip_id} not found in {split_rel_path}"

    s1_tensor, label_tensor, _ = ds[idx]
    label = label_tensor.numpy()
    valid_mask = (label != -1)

    # 2. Run evaluation pipeline inference
    x = s1_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        u_probs = [torch.sigmoid(m(x)).squeeze().cpu().numpy() for m in models["unet"]]
        d_probs = [torch.sigmoid(m(x)).squeeze().cpu().numpy() for m in models["deeplab"]]

    u_mean = np.mean(np.stack(u_probs, axis=0), axis=0)
    d_mean = np.mean(np.stack(d_probs, axis=0), axis=0)

    pipe_mask_u = (u_mean >= 0.50) & valid_mask
    pipe_mask_d = (d_mean >= 0.50) & valid_mask

    # 3. Load Exporter PNG prediction masks
    png_u_path = web_dir / "prediction.png"
    png_d_path = web_dir / "prediction_deeplab.png"
    assert png_u_path.exists(), f"Missing prediction.png for {chip_id}"
    assert png_d_path.exists(), f"Missing prediction_deeplab.png for {chip_id}"

    png_u_arr = np.array(Image.open(png_u_path))
    png_d_arr = np.array(Image.open(png_d_path))

    exp_mask_u = (png_u_arr[..., 3] > 0) & valid_mask
    exp_mask_d = (png_d_arr[..., 3] > 0) & valid_mask

    # ASSERT PIXEL-FOR-PIXEL EQUALITY
    assert np.array_equal(exp_mask_u, pipe_mask_u), f"Pixel mismatch in U-Net prediction mask for {chip_id}"
    assert np.array_equal(exp_mask_d, pipe_mask_d), f"Pixel mismatch in DeepLabV3+ prediction mask for {chip_id}"

    # 4. Verify meta.json metrics equality
    meta_path = web_dir / "meta.json"
    assert meta_path.exists(), f"Missing meta.json for {chip_id}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Compute expected confusion matrix and metrics from evaluation pipeline
    cm_u = compute_confusion_matrix(u_mean, label, threshold=0.5, ignore_index=-1)
    m_u = compute_metrics_from_counts(cm_u["tp"], cm_u["fp"], cm_u["fn"], cm_u["tn"])

    cm_d = compute_confusion_matrix(d_mean, label, threshold=0.5, ignore_index=-1)
    m_d = compute_metrics_from_counts(cm_d["tp"], cm_d["fp"], cm_d["fn"], cm_d["tn"])

    meta_u = meta["metrics"]["unet_resnet34"]
    meta_d = meta["metrics"]["deeplabv3p_resnet34"]

    # Verify counts
    assert meta_u["tp_pixels"] == cm_u["tp"]
    assert meta_u["fp_pixels"] == cm_u["fp"]
    assert meta_u["fn_pixels"] == cm_u["fn"]
    assert meta_u["tn_pixels"] == cm_u["tn"]

    assert meta_d["tp_pixels"] == cm_d["tp"]
    assert meta_d["fp_pixels"] == cm_d["fp"]
    assert meta_d["fn_pixels"] == cm_d["fn"]
    assert meta_d["tn_pixels"] == cm_d["tn"]

    # Verify IoU rounded to 2 decimals
    expected_u_iou = round(float(m_u["flood_iou"] * 100.0), 2)
    expected_d_iou = round(float(m_d["flood_iou"] * 100.0), 2)

    assert abs(meta_u["iou"] - expected_u_iou) < 1e-4, f"U-Net IoU mismatch in meta.json for {chip_id}"
    assert abs(meta_d["iou"] - expected_d_iou) < 1e-4, f"DeepLab IoU mismatch in meta.json for {chip_id}"
