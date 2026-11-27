# Uncertainty-Aware Flood Extent Mapping from Sentinel-1 SAR

**Department of Computer Science & Engineering | B.M.S. College of Engineering, Bengaluru**  
*UG 5th-Semester Mini-Project (Academic Year 2026–2027)*  
*Anchor Milestone: Review-3 (23–30 November 2026)*

---

## 1. Project Overview
This repository implements an end-to-end, uncertainty-calibrated deep learning system for mapping floods from Sentinel-1 Synthetic Aperture Radar (SAR) dual-polarization imagery ($\text{VV} + \text{VH}$). 

The system:
1. Segments flooded pixels through deep convolutional neural networks (U-Net ResNet-34 and DeepLabV3+ ResNet-34).
2. Evaluates out-of-region generalization on the geographically held-out Bolivia flood event in Sen1Floods11 (15 chips).
3. Produces per-pixel predictive uncertainty via 3-seed deep ensembles (MC dropout skipped because trained models have no active dropout layers), assessing calibration (ECE, Reliability Diagrams, and Error-vs-Uncertainty AUROC).
4. Translates segmentation uncertainty into actionable exposure intervals (affected population via WorldPop 100m fractional subpixel aggregation and road networks via local Geofabrik shapefile extracts).
5. Serves precomputed results, overlays, and diagnostics via a static, offline, Leaflet-vendored web viewer.

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

## 3. How to Run the Web Demo

The interactive map viewer runs locally without external network dependencies:

1. **Start the local HTTP server** from the repository root:
   ```bash
   python -m http.server 8000 --directory web
   ```
2. **Open your web browser** and navigate to:
   ```text
   http://localhost:8000/index.html
   ```

> [!IMPORTANT]
> Do **not** open `index.html` by double-clicking it from File Explorer. Modern web browsers block local file-origin (`file://`) asynchronous requests to JSON metadata and PNG overlays due to CORS security rules. It must be accessed via `http://localhost:8000/index.html`.

---

## 4. Execution Runbook

### Step 1: Run Automated Unit Tests
Verifies ignore-aware metrics, confusion matrix math, label masking (`-1`), dataset loader, exposure aggregation, and web export consistency:
```bash
pytest tests/ -v
```
*(All 44 automated tests pass in ~57 seconds).*

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

### Step 6: Multi-Seed Training & Benchmark Generation
Executes full training across seeds 0, 1, 2 for U-Net ResNet-34 and DeepLabV3+ ResNet-34 on GPU with early stopping, followed by evaluation on `valid`, `test`, and `bolivia`:
```powershell
# On Windows PowerShell:
.\scripts\run_all.ps1

# On Linux / macOS:
bash scripts/run_all.sh
```
*(Generates all checkpoints in `results/checkpoints/`, evaluations `results/*_<split>.json`, summary table `results/metrics_summary_table.csv`, and figures in `results/figures/`).*

### Step 7: Evaluate Uncertainty Pipeline (Deep Ensembles & Calibration)
Runs probability averaging across seeds 0, 1, 2, fits temperature scaling strictly on valid split, computes error-detection AUROC/AUPRC, boundary proximity, accuracy-rejection curves, and saves diagnostic figures:
```bash
python scripts/evaluate_uncertainty_pipeline.py
```
*(Generates `results/uncertainty_results.json`, 1,552 float16 uncertainty maps in `results/uncertainty/`, and publication figures `reliability_diagrams.png`, `rejection_curves.png`, `example_chips.png`).*

### Step 8: Compute Multi-Risk Disaster Exposure
Quantifies population and road network inundation brackets (0.70 conservative, 0.50 expected, 0.30 worst-case) using WorldPop rasters and local Geofabrik OpenStreetMap shapefiles:
```bash
python scripts/run_exposure.py --event bolivia
python scripts/run_exposure.py --event sri-lanka
```
*(Outputs saved to `results/exposure_bolivia.json`, `results/exposure_sri-lanka.json`, and summary tables).*

### Step 9: Export Web Viewer Layers & Run Consistency Check
Exports precomputed layer overlays into `web/data/` using canonical `FloodDataset` preprocessing and verifies exact pixel consistency against exposure records:
```bash
python scripts/export_web_layers.py
python scripts/check_web_vs_results.py
```

---

## 5. Multi-Seed Benchmark Results (Mean ± Std Across 3 Seeds)

Summary compiled from `results/metrics_summary_table.csv` and `results/metrics_summary_table.md`:

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
6. **Regional Comparison on Bolivia:** Bolivia single-seed means are 2–3.5 percentage points below test means for both models (U-Net: -3.51 percentage points, from 63.81% to 60.30%; DeepLabV3+: -2.38 percentage points, from 64.27% to 61.89%), within the large seed-to-seed spread; with 15 chips from one event we cannot conclude whether there is a regional drop. Flood fraction of valid pixels is 15.86% on Bolivia (454,864 / 2,867,815) vs. 12.51% on test (2,566,101 / 20,517,367); we did not investigate why Otsu performs well there.
7. **Test Suite Status:** **44 passed** out of 44 tests (`pytest tests/ -v`).

---

## 6. Uncertainty Quantification

Compiled strictly from `results/uncertainty_results.json` using 3-seed Deep Ensembles (seeds 0, 1, 2) on U-Net ResNet-34 and DeepLabV3+ ResNet-34 (*Note: MC dropout was not used because the trained models have no active dropout layers*):

### 6.1 Deep Ensemble vs. Single-Seed Mean Flood IoU
| Model | Split | 3-Seed Mean IoU (%) | Ensemble IoU (%) | Gain ($\Delta$) | Seed Spread |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **U-Net ResNet-34** | Validation (In-Domain) | 60.60% | **62.31%** | +1.71 pp | 0.46 pp |
| **U-Net ResNet-34** | Test (In-Domain) | 63.81% | **65.33%** | +1.52 pp | 0.26 pp |
| **U-Net ResNet-34** | Bolivia (Domain Shift) | 60.30% | **60.91%** | +0.61 pp | 13.24 pp |
| **DeepLabV3+ ResNet-34** | Validation (In-Domain) | 61.62% | **63.18%** | +1.56 pp | 0.67 pp |
| **DeepLabV3+ ResNet-34** | Test (In-Domain) | 64.27% | **66.43%** | +2.16 pp | 1.02 pp |
| **DeepLabV3+ ResNet-34** | Bolivia (Domain Shift) | 61.89% | **63.32%** | +1.43 pp | 10.88 pp |

### 6.2 Error Detection & Calibration Metrics
* **Error Detection (Predictive Entropy vs. Misclassification):** Evaluated across 500,000 pixels per split (random seed 42):
  * **U-Net:** Valid AUROC **0.8686** (AUPRC 0.3186 vs baseline 0.050); Test AUROC **0.8140** (AUPRC 0.2934 vs baseline 0.050); Bolivia AUROC **0.8643** (AUPRC 0.3529 vs baseline 0.072).
  * **DeepLabV3+:** Valid AUROC **0.9204** (AUPRC 0.3524 vs baseline 0.049); Test AUROC **0.8967** (AUPRC 0.3177 vs baseline 0.049); Bolivia AUROC **0.8941** (AUPRC 0.3601 vs baseline 0.068).
* **Temperature Scaling Calibration (10 Bins):** Optimal temperature $T^*$ fitted **strictly on validation split** ($T^*=0.7289$ for U-Net, $T^*=1.2720$ for DeepLabV3+):
  * **U-Net:** Valid ECE dropped from **4.44% $\to$ 1.73%**; Test ECE dropped from **4.40% $\to$ 1.67%**; Bolivia ECE dropped from **4.74% $\to$ 3.03%**.
  * **DeepLabV3+:** Valid ECE dropped from **1.55% $\to$ 0.71%**; Test ECE dropped from **1.86% $\to$ 0.82%**; Bolivia ECE dropped from **3.15% $\to$ 2.71%**.
* **Boundary Proximity Analysis:** Within $\le 3\text{ px}$ of ground truth flood boundaries:
  * For U-Net, 37.79% (valid), 38.37% (test), and 44.33% (bolivia) of top-10% uncertain pixels concentrate along boundaries (**3.65x to 3.95x concentration ratio** over spatial base rate).
  * For DeepLabV3+, 40.14% (valid), 41.01% (test), and 43.24% (bolivia) of top-10% uncertain pixels concentrate near boundaries (**3.56x to 4.20x concentration ratio**).

### 6.3 Limitations
1. Errors strongly concentrate at water-land transition boundaries, meaning high error-detection AUROC may largely reflect boundary ambiguity.
2. Bolivia represents a single geographic disaster event (15 chips); ensemble scores on Bolivia fall within the range of single-seed scores.
3. MC dropout was skipped because the trained models contain no active dropout layers.

---

## 7. Review-3 Deliverables Checklist

| Deliverable Artifact | File Path | Status |
| :--- | :--- | :---: |
| **Ignore-Aware Evaluation Metrics Suite** | `src/floodmap/metrics.py` | **VERIFIED (44/44 Tests Passed)** |
| **Classical Otsu Baseline Benchmarks** | `results/otsu_baseline_results.json` | **GENERATED** |
| **Multi-Seed GPU Models & Checkpoints** | `results/checkpoints/*.pth` (Seeds 0, 1, 2) | **GENERATED** |
| **Per-Seed & Per-Chip Detailed Analysis CSVs** | `results/per_seed_table.csv`, `per_chip_*.csv` | **GENERATED** |
| **Summary Metrics Tables (CSV & Markdown)** | `results/metrics_summary_table.{csv,md}` | **GENERATED** |
| **Diagnostic Publication Figures** | `results/figures/*.png` (8 figures) | **GENERATED** |
| **Uncertainty Pipeline Results & Maps** | `results/uncertainty_results.json`, `results/uncertainty/` | **GENERATED** |
| **Multi-Risk Exposure Estimates (JSON/CSV)** | `results/exposure_*.json`, `results/exposure_*_summary.csv` | **GENERATED** |
| **Static Offline Interactive Web Viewer** | `web/index.html`, `web/js/app.js`, `web/data/` | **GENERATED & VERIFIED** |
| **Web Export Consistency Regression Suite** | `tests/test_web_export.py`, `scripts/check_web_vs_results.py` | **VERIFIED (100% Match)** |

---

## 8. Repository Structure

```text
Mini_Project/
├── configs/                          # Experiment configuration YAMLs
│   ├── unet_resnet34.yaml            # U-Net training & model hyperparams
│   └── deeplabv3p_resnet34.yaml      # DeepLabV3+ training hyperparams
├── data/                             # Dataset storage (splits committed, raw ignored)
│   ├── splits/                       # Official Sen1Floods11 train, valid, test, bolivia CSVs
│   ├── worldpop/                     # WorldPop population count GeoTIFF rasters
│   ├── osm/                          # Local Geofabrik OpenStreetMap shapefiles
│   └── raw/                          # S1Hand & LabelHand GeoTIFF chips (446 chips each)
├── docs/                             # Presentation, architecture, guides, and audit reports
│   ├── slides/
│   │   ├── review_2_presentation.md  # 11-slide presentation deck for Review-2
│   │   └── review_3_presentation.md  # 14-slide presentation deck for Review-3
│   ├── screenshots/                  # 9 high-resolution automated verification screenshots
│   ├── DEMO_WEB.md                   # 3-minute live demonstration script with exact numbers
│   ├── VIVA_QA.md                    # Grounded viva examination questions & answers
│   ├── RESULTS_FACTS.md              # Fact sheet of all quantitative project numbers
│   ├── FINDINGS.md                   # Core empirical observations & limitations
│   └── architecture.md               # Detailed system design & Mermaid source code
├── results/                          # Checkpoints, benchmark JSONs, tables, and figures
│   ├── checkpoints/                  # Trained best weights (*.pth, seeds 0, 1, 2)
│   ├── figures/                      # 8 publication-grade diagnostic PNG figures
│   ├── uncertainty/                  # Precomputed float16 uncertainty .npy maps
│   ├── metrics_summary_table.csv     # Master benchmark summary table across all splits
│   ├── otsu_baseline_results.json    # Classical baseline benchmark records
│   ├── uncertainty_results.json      # Complete ensemble, calibration & error-detection stats
│   ├── exposure_bolivia.json         # Bolivia multi-threshold exposure assessment
│   └── exposure_sri-lanka.json       # Sri-Lanka multi-threshold exposure assessment
├── scripts/                          # Pipeline execution scripts
│   ├── run_eda.py                    # Radiometric & class balance audit
│   ├── eval_otsu.py                  # Dynamic & global Otsu evaluation
│   ├── evaluate_uncertainty_pipeline.py # Deep ensemble, calibration & rejection pipeline
│   ├── run_exposure.py               # WorldPop population & OSM road length exposure analysis
│   ├── export_web_layers.py          # Precomputed web layer raster & metadata exporter
│   ├── check_web_vs_results.py       # Consistency audit script
│   ├── run_browser_verification.py   # Automated CDP headless browser verification harness
│   └── run_all.ps1 / run_all.sh      # Full multi-seed GPU training & evaluation runner
├── src/floodmap/                     # Modular core package
│   ├── data/                         # GeoTIFF loading, imputation & FloodDataset
│   ├── models/                       # 2-channel U-Net, DeepLabV3+, compound loss
│   ├── uncertainty/                  # Deep ensembles, temperature scaling, ECE math
│   ├── exposure/                     # WorldPop population & OSM road length analysis
│   └── metrics.py                    # Ignore-aware Flood IoU, F1, precision, recall
├── tests/                            # Automated pytest unit & regression test suite (44 tests)
└── web/                              # Static, offline interactive map viewer (Leaflet vendored)
    ├── css/style.css                 # Clean application styling
    ├── js/app.js                     # Offline Leaflet map controller & pane management
    ├── vendor/leaflet/               # 100% vendored Leaflet v1.9.4 CSS, JS, and marker assets
    ├── data/                         # 30 precomputed chip folders with PNG overlays & meta.json
    ├── index.html                    # Interactive map viewer
    └── results.html                  # Master benchmark tables & publication figures viewer
```

---

## 9. Team Roles & Accountability
* **Member A (Data & Metrics):** Ingestion, radiometric normalization, split isolation, ignore-aware metrics, Otsu baseline.
* **Member B (Models & Training):** Deep learning architectures (U-Net ResNet-34, DeepLabV3+ ResNet-34), compound loss, GPU training runs.
* **Member C (Uncertainty & Exposure):** 3-seed deep ensemble, temperature scaling calibration, fractional WorldPop population and local OSM road exposure analysis.
* **Member D (Web & Systems):** Offline Leaflet interactive map viewer, web layer precomputation, Turnitin screening (< 25%), report compilation.
