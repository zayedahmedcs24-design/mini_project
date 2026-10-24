# B.M.S. COLLEGE OF ENGINEERING, BENGALURU-19
### Department of Computer Science & Engineering
**UG 5th-Semester Mini-Project | Review-2 Presentation**  
*Anchor Date: 27–31 October 2026*

---

## SLIDE 1: Title & Team Metadata
* **Project Title:** Uncertainty-Aware Flood Extent Mapping from Sentinel-1 SAR with Cross-Region Evaluation and Exposure Estimation
* **Academic Year:** 2026–2027 (5th Semester)
* **Team Members:**
  * Member A: Data Engineering, Radiometrics & Metric Integrity
  * Member B: Deep Learning Architectures, Optimization & Bolivia Evaluation
  * Member C: Uncertainty Quantification, Calibration & Geospatial Exposure
  * Member D: API Services, Frontend Deployment, Documentation & Integrity
* **Project Guide:** [Guide Name, Designation]
* **Department:** Department of Computer Science & Engineering, B.M.S. College of Engineering

---

## SLIDE 2: Problem Statement & Operational Relevance
* **The Disaster Dilemma:** Flooding causes massive economic and human loss. Emergency teams require immediate, accurate water maps.
* **The Optical Barrier:** Heavy cloud cover and rain blind optical satellites (Sentinel-2, Landsat) throughout active flood events.
* **SAR Advantage:** Sentinel-1 Synthetic Aperture Radar (C-band, 5.405 GHz) pierces clouds, providing day/night observation.
* **SAR Interpretation Challenges:**
  * *Specular Reflection:* Flat surfaces (smooth asphalt, runways, calm soil) reflect radar pulses away, mimicking water.
  * *Terrain Shadows:* Mountain ridges cast dark radar shadows, creating false-positive water predictions.
  * *Domain Shift:* Models trained on one region can degrade on unseen regions (literature to be cited in the survey table); on the held-out Bolivia event our models' scores varied widely between seeds.
  * *Silent Overconfidence:* Standard deep networks output uncalibrated, overconfident predictions without flagging uncertain pixels.

---

## SLIDE 3: UN Sustainable Development Goals (SDG) Alignment
* **Primary: SDG 13 (Climate Action) — Target 13.1**
  * *Focus:* Strengthen resilience and adaptive capacity to climate-related hazards and natural disasters.
  * *Role:* Delivers rapid, cloud-independent flood boundary delineations to guide civil protection and disaster response.
* **Secondary: SDG 11 (Sustainable Cities and Communities) — Target 11.5**
  * *Focus:* Significantly reduce deaths, affected people, and direct economic losses caused by disasters.
  * *Role:* Combines water masks with WorldPop gridded population data and OpenStreetMap roads to provide impact ranges (people at risk, submerged road length).
* **Domain:** Machine Learning / Computer Vision for Remote Sensing (Geospatial AI).

---

## SLIDE 4: Peer-Reviewed Literature Matrix & Research Gap
* **Journal Audit Standard:** Restricted strictly to 2024–2026 peer-reviewed journals (IEEE, Elsevier, Springer, ACM, MDPI). Conference papers excluded from survey matrix.
* **Audited Findings from Literature:**
  * Most studies report in-distribution metrics on random splits; very few test on geographically held-out disaster events.
  * While uncertainty methods (MC Dropout, Ensembles) are occasionally implemented, few papers validate whether high-uncertainty pixels actually correspond to misclassifications.
  * Inundation maps are rarely integrated into downstream human exposure ranges (affected population and cut-off road networks).
* **Candidate gap — to be confirmed by the literature table:**
  > "Few studies systematically evaluate whether predictive uncertainty remains well-calibrated under regional domain shift (such as the held-out Bolivia event in Sen1Floods11), nor do they propagate per-pixel uncertainty into bounded exposure intervals for disaster response."

---

## SLIDE 5: Scope Boundary & MoSCoW Prioritization
* **MUST HAVE (Review-3 Baseline):**
  * Data pipeline for Sen1Floods11 hand-labeled chips (512x512) and official splits.
  * Ignore-aware metrics (`Flood IoU`, `F1`, `Precision`, `Recall`, `Mean IoU`).
  * Otsu thresholding baselines on VV, VH, and Mean backscatter.
  * U-Net (ResNet-34) and DeepLabV3+ trained on GPU.
  * Evaluation on the held-out Bolivia event.
  * Deep Ensemble uncertainty with error-detection validation.
  * Exposure estimation (WorldPop population and OSM road length) on Bolivia.
  * Interactive local Leaflet map viewer displaying SAR, Prediction, Ground Truth, Uncertainty & Exposure.
* **SHOULD HAVE:** SegFormer (MiT-B0), MC Dropout comparison, ECE calibration plots, 3-seed variance reporting.
* **COULD HAVE:** Single-chip GeoTIFF upload endpoint, building footprint exposure.
* *System Context:* Benchmark study on curated labeled datasets, not a live planetary satellite ingestion daemon.

---

## SLIDE 6: System Architecture & Data Flow

```
  [Sentinel-1 SAR VV/VH Chips] ───> [Radiometric Ingestion & NaN Imputation]
                 │
                 ├───> [Baseline: Otsu Thresholding (VV, VH, Mean)]
                 │
                 └───> [Deep Segmentation: U-Net (ResNet-34) with MC Dropout]
                             │
                             ▼
                 [In-Domain Test Split vs Held-Out Bolivia Event]
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
      [Uncertainty Engine]      [Geospatial Exposure Engine]
      • Predictive Mean         • Dynamic UTM Projection (Per-Chip)
      • Epistemic Entropy/Var   • Fractional WorldPop Population Overlay
      • ECE Calibration & AUROC • OpenStreetMap Flooded Road Length (km)
                 │                       │
                 └───────────┬───────────┘
                             ▼
      [Interactive Local Dashboard: FastAPI + Leaflet Map Explorer]
```

---

## SLIDE 7: Data Engineering & Radiometric Verification
* **Radiometric Check & Distribution Audit (from `results/sar_stats.json` on Train Split):**
  * Inspection of `S1Hand` chips confirms raw values are **already in decibels (dB)**:
    * **1. Raw Backscatter (All Pixels):**
      * VV: `[-65.45, +29.93] dB` (Mean: `-10.38 dB`, Std: `4.02 dB`)
      * VH: `[-79.16, +17.38] dB` (Mean: `-17.22 dB`, Std: `4.73 dB`)
    * **2. 1st–99th Percentile (Core Empirical Distribution):**
      * VV: `[-23.80, -2.43] dB` (Median: `-9.83 dB`)
      * VH: `[-34.08, -9.54] dB` (Median: `-16.34 dB`)
    * **3. Post-Clipping (Implemented in `dataset.py`):**
      * VV: `[-35.00, +5.00] dB` (Mean: `-10.89 dB`, Std: `5.27 dB`)
      * VH: `[-45.00, 0.00] dB` (Mean: `-17.79 dB`, Std: `6.09 dB`)
  * *Critical Safeguard:* Applying $10 \cdot \log_{10}$ would be mathematically invalid on negative numbers. Values are normalized via Z-score scaling.
* **NaN Pixel Analysis & Safeguard Handling:**
  * Across all 252 train chips (66,060,288 pixels), `1,325,136` pixels contain NaNs (`2.01%`).
  * **98.18% of NaNs** (`1,300,986` pixels) fall **strictly inside ignore (`-1`) regions** (cloud/no-data borders). Only `24,150` NaNs (`1.82%`) occur in labeled terrain.
  * *Handling in `dataset.py`:* Imputed safely with floor backscatter (`-35.0 dB` for VV, `-45.0 dB` for VH) before clipping and standardization, ensuring zero NaNs leak to model tensors.
* **Label Integrity & Class Imbalance:**
  * Label values are strictly verified: `1` (Water), `0` (Dry land), `-1` (Ignore/Cloud).
  * Empirical class balance shows severe imbalance: flood terrain is only **10.60%** of valid pixels across the dataset (`10,705,605 / 100,983,314`), while ignore pixels account for **13.63%** of all pixels (`15,932,910 / 116,916,224`).
  * *Why Overall Accuracy Fails:* A trivial model predicting all dry land achieves $> 89\%$ accuracy, but **0.00% Flood IoU**.

---

## SLIDE 8: Comprehensive Baseline & Multi-Seed Benchmark Evaluation
* **Empirical Multi-Seed Benchmark Results (Mean ± Std Across Seeds 0, 1, 2 from `results/*.json` and `results/metrics_summary_table.csv` — Run 2 Matched Recipe):**

| Split Evaluated | Method / Architecture | Variant / Setup | Seeds | Flood IoU (%) | Flood F1 (%) | Precision (%) | Recall (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation** (89 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Det. | 21.52% | 35.42% | 23.25% | 74.33% |
| **Validation** (89 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Det. | 44.58% | 61.67% | 53.49% | 72.81% |
| **Validation** (89 chips) | **U-Net (ResNet-34)** | Dual-Pol (BS=8) | 3 | **60.60 ± 0.26%** | 75.47 ± 0.20% | 75.88 ± 4.37% | 75.38 ± 4.23% |
| **Validation** (89 chips) | **DeepLabV3+ (ResNet-34)** | Dual-Pol (BS=8) | 3 | **61.62 ± 0.35%** | 76.25 ± 0.27% | 78.08 ± 3.93% | 74.71 ± 2.99% |
| **Test** (90 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Det. | 24.73% | 39.65% | 27.04% | 74.32% |
| **Test** (90 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Det. | 49.54% | 66.26% | 58.60% | 76.22% |
| **Test** (90 chips) | **U-Net (ResNet-34)** | Dual-Pol (BS=8) | 3 | **63.81 ± 0.15%** | 77.91 ± 0.11% | 78.27 ± 4.01% | 77.82 ± 4.06% |
| **Test** (90 chips) | **DeepLabV3+ (ResNet-34)** | Dual-Pol (BS=8) | 3 | **64.27 ± 0.52%** | 78.25 ± 0.39% | 79.72 ± 2.92% | 76.99 ± 3.37% |
| **Bolivia** (15 chips) | Otsu (Per-Chip Dynamic) | Dual-Pol Mean | Det. | 34.73% | 51.55% | 44.20% | 61.84% |
| **Bolivia** (15 chips) | Otsu (Train-Fitted Global) | Dual-Pol Mean | Det. | 64.81% | 78.65% | 74.89% | 82.80% |
| **Bolivia** (15 chips) | **U-Net (ResNet-34)** | Dual-Pol (BS=8) | 3 | **60.30 ± 7.59%** | 75.04 ± 6.08% | 78.40 ± 1.17% | 72.66 ± 11.45% |
| **Bolivia** (15 chips) | **DeepLabV3+ (ResNet-34)** | Dual-Pol (BS=8) | 3 | **61.89 ± 5.45%** | 76.37 ± 4.18% | 80.13 ± 3.49% | 73.76 ± 10.56% |

* **Rigorous Empirical Findings & Key Distinctions:**
  1. **Evaluation Scope on Bolivia:** Bolivia contains exactly 15 hand-labeled chips representing a single geographic disaster event.
  2. **Matched Run 2 Recipe:** Both models trained with batch size 8 on $256 \times 256$ random crops with `max_epochs = 60` and patience 8 early stopping. DeepLabV3+ peak VRAM is 519.4 MB (no OOM).
  3. **In-Domain Comparison:** U-Net and DeepLabV3+ show **differences within seed-to-seed variation** on in-domain splits (Validation: $60.60\%$ vs $61.62\%$, a $+1.02$ percentage points difference; Test: $63.81\%$ vs $64.27\%$, a $+0.46$ percentage points difference).
  4. **Regional Performance on Bolivia:** DeepLab achieves $61.89 \pm 5.45\%$ and U-Net achieves $60.30 \pm 7.59\%$ Flood IoU, while train-fitted Global Otsu Mean scores $64.81\%$.
  5. **Inter-Seed Variation on Bolivia:** On Bolivia, both models varied substantially between seeds (U-Net ±7.6, DeepLabV3+ ±5.5 IoU points; DeepLabV3+ seeds 56.3%, 62.3%, 67.1%). In run 1 the outlier DeepLabV3+ seed hit the 30-epoch cap; run 2 also changed the data recipe, so we cannot attribute the change to a single cause.
  6. **Regional Comparison on Bolivia:** Bolivia means are 2–3.5 percentage points below test means for both models (U-Net: -3.51 pp; DeepLabV3+: -2.38 pp), within the large seed-to-seed spread; with 15 chips from one event we cannot conclude whether there is a regional drop. Flood fraction of valid pixels is 15.86% on Bolivia (454,864 / 2,867,815) vs. 12.51% on test (2,566,101 / 20,517,367); we did not investigate why Otsu performs well there.

---

## SLIDE 9: Deep Learning Architecture, Failure Diagnostics & Verification
* **Model Implementations & Safeguards:**
  * Both architectures employ an ImageNet pre-trained ResNet-34 backbone adapted to 2-channel Sentinel-1 SAR inputs with scale preservation.
  * Monitored with early stopping strictly on Validation Flood IoU (patience=8); best checkpoints saved as `<model>_seed<k>_best.pth`.
  * Compound loss: $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE, masked}} + 1.0 \cdot \mathcal{L}_{\text{Dice, masked}}$ (excluding `-1` ignore pixels).
* **Diagnostic Visualizations Generated in `results/figures/`:**
  * `fig_iou_bars.png`: Flood IoU across splits (mean ± std) comparing Global Otsu Mean, U-Net, and DeepLabV3+.
  * `fig_val_curves.png`: Validation Flood IoU vs epoch trajectories across all 6 runs with saved best checkpoints indicated.
  * `fig_examples_test.png`: 4 test chips (2 good floodplains, 2 edge-case failures) displaying SAR VV, SAR VH, Ground Truth (gray ignore), U-Net prediction, and Global Otsu prediction.
  * `fig_examples_bolivia.png`: 3 Bolivia chips highlighting DeepLabV3+ predictions vs Ground Truth and Global Otsu.
  * `fig_otsu_failure.png`: 3 dry-land chips demonstrating the classical per-chip Otsu failure mode (bisection of unimodal dry land resulting in $> 48\%$ false alarm flood) vs correct rejection by train-fitted Global Otsu.
* **Near-Zero Flood Trap in Per-Chip Analysis:**
  * Chips with $<1\%$ flood or $100\%$ ignore are marked *"IoU not meaningful"* and excluded from worst-chip ranking. Ranking the 10 worst test chips strictly among chips with $\ge 1\%$ flood identifies genuine structural errors (e.g., `Somalia_699062` at $3.15\%$ IoU), while separate ranking highlights chips with high false positives (e.g., `Paraguay_271769` and `Pakistan_70625`).
* **Automated Test Suite Status:**
  * **29 / 29 unit tests passing** via `pytest tests/ -v` (covering datasets, ignore masking, metrics, architectures, Otsu baselines, exposure math, and uncertainty calibration).

---

## SLIDE 10: Roadmap to Review-3 (Stages B through E)
* **Stage B (27 Oct – 10 Nov):** Remote/local GPU training of U-Net, DeepLabV3+, and SegFormer across 3 seeds; MC Dropout ($T=20$) and Deep Ensembles; compute ECE and error-detection AUROC on Bolivia.
* **Stage C (10 Nov – 18 Nov):** Implement fractional WorldPop population aggregation and dynamic UTM OSM road clipping; freeze numbers by 18 Nov.
* **Stage D (18 Nov – 22 Nov):** Complete Leaflet web app and dry-run 10-minute demo; prepare answers for viva questions.
* **Stage E (15 Nov – 15 Dec):** Complete 50-page report; run Turnitin screening by 05 Dec ($< 25\%$ similarity & AI score); Guide signature by 10 Dec; submit hard copy by 15 Dec.

