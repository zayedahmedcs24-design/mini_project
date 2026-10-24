# Uncertainty-Aware Flood Extent Mapping from Sentinel-1 SAR

**Department of Computer Science & Engineering | B.M.S. College of Engineering, Bengaluru**  
*UG 5th-Semester Mini-Project (Academic Year 2026–2027)*  
*Anchor Milestone: Review-2 (27–31 October 2026)*

---

## 1. Project Overview
This repository implements an end-to-end, uncertainty-calibrated deep learning system for mapping floods from Sentinel-1 Synthetic Aperture Radar (SAR) dual-polarization imagery ($\text{VV} + \text{VH}$). 

The system:
1. Segments flooded pixels through deep convolutional neural networks (U-Net ResNet-34, DeepLabV3+) and transformers (SegFormer).
2. Evaluates out-of-region generalization on the geographically held-out Bolivia flood event in Sen1Floods11.
3. Produces per-pixel predictive uncertainty via Monte Carlo Dropout and Deep Ensembles, validating calibration (ECE, Reliability Diagrams, and Error-vs-Uncertainty AUROC).
4. Translates segmentation uncertainty into actionable exposure intervals (affected population via WorldPop and submerged road networks via OpenStreetMap).
5. Serves precomputed results and diagnostics via a local interactive web application.

---

## 2. Quickstart & Environment Setup

### Prerequisites
* Python 3.11
* CUDA-capable GPU (recommended; tested on NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.1)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Team/floodmap.git
cd floodmap

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install geospatial and ML dependencies
pip install -r requirements.txt
```

---

## 3. Stage A (Review-2) Execution Runbook

### Step 1: Run Automated Unit Tests
Verifies ignore-aware metrics, confusion matrix math, label masking (`-1`), and dataset loaders:
```bash
pytest tests/ -v
```
*(All 10 unit tests pass in < 3 seconds).*

### Step 2: Download & Verify Sen1Floods11 Hand-Labeled Splits
Fetches the official split CSVs from Google Cloud Storage and downloads verification chips in smoke mode:
```bash
# Smoke mode (downloads split CSVs + 8 sample chips for instant verification):
python -m src.floodmap.data.download --smoke

# Full mode (downloads all 446 hand-labeled chips ~450 MB):
python -m src.floodmap.data.download
```

### Step 3: Run Exploratory Data Analysis (EDA) & Radiometric Audit
Validates that SAR inputs are already in decibels ($\text{dB}$) and prints exact class balance percentages:
```bash
python scripts/run_eda.py
```

### Step 4: Run Classical Baseline Evaluation (Otsu Thresholding)
Evaluates Otsu thresholding across $\text{VV}$, $\text{VH}$, and Dual-Pol $\text{Mean}$ on validation and Bolivia splits:
```bash
python scripts/eval_otsu.py
```
*(Results are saved to `results/otsu_baseline_results.json`).*

### Step 5: Execute U-Net Smoke Training Run
Verifies model instantiation, 2-channel adaptation via SMP, loss backpropagation, and checkpointing:
```bash
python train.py --config configs/unet_resnet34.yaml --smoke
```
*(Outputs decreasing loss, checkpoint `results/checkpoints/unet_resnet34_best.pth`, and prediction visualization `results/smoke_prediction_sample.png`).*

---

## 4. Review-2 Deliverables Checklist (Ready for 27–31 Oct 2026)

| Deliverable | Location in Repository | Status |
| :--- | :--- | :---: |
| **System Architecture Diagram & Slides** | `docs/slides/review_2_presentation.md` | **READY** |
| **Data Ingestion & Split Isolation (0 Leakage)** | `src/floodmap/data/download.py` | **VERIFIED** |
| **Empirical Radiometric & Class Balance Audit** | `scripts/run_eda.py` | **VERIFIED** |
| **Ignore-Aware Evaluation Metrics Suite** | `src/floodmap/metrics.py` | **VERIFIED (10/10 Tests Passed)** |
| **Classical Otsu Baseline Benchmarks** | `results/otsu_baseline_results.json` | **GENERATED** |
| **U-Net Architecture & Converging Smoke Run** | `train.py` & `results/smoke_prediction_sample.png`| **GENERATED** |

---

## 5. Team Roles & Accountability
* **Member A (Data & Metrics):** Ingestion, radiometric normalization, split isolation, ignore-aware metrics, Otsu baseline.
* **Member B (Models & Training):** Deep learning architectures (U-Net, DeepLabV3+, SegFormer), compound loss, GPU training runs.
* **Member C (Uncertainty & Exposure):** MC Dropout, Deep Ensembles, ECE calibration, fractional WorldPop population and OSM road exposure.
* **Member D (Web & Systems):** FastAPI backend, Leaflet interactive map viewer, Turnitin screening (< 25%), report compilation.
# mini_project
