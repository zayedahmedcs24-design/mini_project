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
*(All 29 unit tests pass in ~7.5 seconds).*

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
Evaluates dynamic per-chip and train-fitted global Otsu thresholding across $\text{VV}$, $\text{VH}$, and Dual-Pol $\text{Mean}$ on validation, test, and Bolivia splits:
```bash
python scripts/eval_otsu.py
```
*(Results are saved to `results/otsu_baseline_results.json`).*

### Step 5: Execute U-Net Smoke Training Run
Verifies model instantiation, 2-channel adaptation via SMP, loss backpropagation, early stopping, and checkpointing:
```bash
python train.py --config configs/unet_resnet34.yaml --smoke
```
*(Outputs decreasing loss, checkpoint `results/smoke/checkpoints/unet_resnet34_seed0_best.pth`, per-epoch history `results/smoke/unet_resnet34_seed0_smoke_history.json`, and prediction visualization `results/smoke/smoke_prediction_sample.png`).*

### Step 6: Multi-Seed Training & Benchmark Generation
Executes full training across seeds 0, 1, 2 for U-Net ResNet-34 and DeepLabV3+ ResNet-34 on GPU with early stopping, followed by evaluation on `valid`, `test`, and `bolivia`:
```powershell
# On Windows PowerShell:
.\scripts\run_all.ps1

# On Linux / macOS:
bash scripts/run_all.sh
```
*(Generates all checkpoints in `results/checkpoints/`, evaluations `results/*_<split>.json`, summary table `results/metrics_summary_table.csv`, and figures in `results/figures/`).*

---

## 4. Multi-Seed Benchmark Results (Mean ± Std Across 3 Seeds)

Summary compiled from `results/metrics_summary_table.csv` and `results/metrics_summary_table.md` (Run 2 matched recipe):

| Split | Architecture / Baseline | Setup / Polarization | Seeds Evaluated | Flood IoU (%) | Flood F1 (%) | Precision (%) | Recall (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Valid** (89 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Deterministic | 21.52% | 35.42% | 23.25% | 74.33% |
| **Valid** (89 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Deterministic | 44.58% | 61.67% | 53.49% | 72.81% |
| **Valid** (89 chips) | **U-Net ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **60.60 ± 0.26%** | 75.47 ± 0.20% | 75.88 ± 4.37% | 75.38 ± 4.23% |
| **Valid** (89 chips) | **DeepLabV3+ ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **61.62 ± 0.35%** | 76.25 ± 0.27% | 78.08 ± 3.93% | 74.71 ± 2.99% |
| **Test** (90 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Deterministic | 24.73% | 39.65% | 27.04% | 74.32% |
| **Test** (90 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Deterministic | 49.54% | 66.26% | 58.60% | 76.22% |
| **Test** (90 chips) | **U-Net ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **63.81 ± 0.15%** | 77.91 ± 0.11% | 78.27 ± 4.01% | 77.82 ± 4.06% |
| **Test** (90 chips) | **DeepLabV3+ ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **64.27 ± 0.52%** | 78.25 ± 0.39% | 79.72 ± 2.92% | 76.99 ± 3.37% |
| **Bolivia** (15 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Deterministic | 34.73% | 51.55% | 44.20% | 61.84% |
| **Bolivia** (15 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Deterministic | 64.81% | 78.65% | 74.89% | 82.80% |
| **Bolivia** (15 chips) | **U-Net ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **60.30 ± 7.59%** | 75.04 ± 6.08% | 78.40 ± 1.17% | 72.66 ± 11.45% |
| **Bolivia** (15 chips) | **DeepLabV3+ ResNet-34** | Dual-Pol (BS=8) | 3/3 (0, 1, 2) | **61.89 ± 5.45%** | 76.37 ± 4.18% | 80.13 ± 3.49% | 73.76 ± 10.56% |

### Key Technical Findings:
1. **Bolivia Event Scope:** The Bolivia split contains 15 hand-labeled chips representing a single regional flood event.
2. **Matched Data Recipe (Run 2):** Both U-Net and DeepLabV3+ were trained with batch size 8 on random $256 \times 256$ crops with `max_epochs = 60` and early stopping patience 8 on validation flood IoU. DeepLabV3+ operated at 519.4 MB peak VRAM without OOM.
3. **In-Domain Comparisons:** Differences are **within seed-to-seed variation** on in-domain splits (Validation: $60.60\%$ vs $61.62\%$, a $+1.02$ percentage point difference; Test: $63.81\%$ vs $64.27\%$, a $+0.46$ percentage point difference).
4. **Bolivia Regional Performance:** On the held-out Bolivia event, DeepLabV3+ achieves $61.89 \pm 5.45\%$ and U-Net achieves $60.30 \pm 7.59\%$ Flood IoU, while train-fitted Global Otsu Mean scores $64.81\%$.
5. **Inter-Seed Variation on Bolivia:** On Bolivia, both models varied substantially between seeds (U-Net ±7.6, DeepLabV3+ ±5.5 IoU points; DeepLabV3+ seeds 56.3%, 62.3%, 67.1%). In run 1 the outlier DeepLabV3+ seed hit the 30-epoch cap; run 2 also changed the data recipe, so we cannot attribute the change to a single cause.
6. **Regional Comparison on Bolivia:** Bolivia means are 2–3.5 percentage points below test means for both models (U-Net: -3.51 percentage points, from 63.81% to 60.30%; DeepLabV3+: -2.38 percentage points, from 64.27% to 61.89%), within the large seed-to-seed spread; with 15 chips from one event we cannot conclude whether there is a regional drop. Flood fraction of valid pixels is 15.86% on Bolivia (454,864 / 2,867,815) vs. 12.51% on test (2,566,101 / 20,517,367); we did not investigate why Otsu performs well there.
7. **Test Suite Status:** **29 passed** out of 29 tests (`pytest tests/ -v`).

---

## 5. Review-2 Deliverables Checklist (Ready for 27–31 Oct 2026)

| Deliverable | Location in Repository | Status |
| :--- | :--- | :---: |
| **System Architecture Diagram & Slides** | `docs/slides/review_2_presentation.md` | **READY** |
| **Executive Findings & Limitations (10 Bullets)** | `docs/FINDINGS.md` | **READY** |
| **Data Ingestion & Split Isolation (0 Leakage)** | `src/floodmap/data/download.py` | **VERIFIED** |
| **Empirical Radiometric & Class Balance Audit** | `scripts/run_eda.py` & `results/sar_stats.json` | **VERIFIED** |
| **Ignore-Aware Evaluation Metrics Suite** | `src/floodmap/metrics.py` | **VERIFIED (29/29 Tests Passed)** |
| **Classical Otsu Baseline Benchmarks** | `results/otsu_baseline_results.json` | **GENERATED** |
| **Multi-Seed GPU Models & Checkpoints** | `results/checkpoints/*.pth` (Seeds 0, 1, 2) | **GENERATED** |
| **Per-Seed & Per-Chip Detailed Analysis CSVs** | `results/per_seed_table.csv`, `per_chip_*.csv` | **GENERATED** |
| **Summary Metrics Tables (CSV & Markdown)** | `results/metrics_summary_table.{csv,md}` | **GENERATED** |
| **Diagnostic Publication Figures** | `results/figures/fig_*.png` | **GENERATED (5 Figures)** |

---

## 6. Team Roles & Accountability
* **Member A (Data & Metrics):** Ingestion, radiometric normalization, split isolation, ignore-aware metrics, Otsu baseline.
* **Member B (Models & Training):** Deep learning architectures (U-Net, DeepLabV3+, SegFormer), compound loss, GPU training runs.
* **Member C (Uncertainty & Exposure):** MC Dropout, Deep Ensembles, ECE calibration, fractional WorldPop population and OSM road exposure.
* **Member D (Web & Systems):** FastAPI backend, Leaflet interactive map viewer, Turnitin screening (< 25%), report compilation.
