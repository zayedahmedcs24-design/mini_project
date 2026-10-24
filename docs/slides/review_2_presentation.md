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
  * *Domain Shift:* Models trained on one region degrade severely when tested on unseen topographies (e.g., Bolivia).
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
* **Formal Project Research Gap:**
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
* **Radiometric Check (Empirical Ground Truth):**
  * Inspection of `S1Hand` chips confirms raw values are **already in decibels (dB)**:
    * VV Range: `[-40.82, -0.69] dB`, Mean: `-13.01 dB`
    * VH Range: `[-48.25, -6.99] dB`, Mean: `-19.29 dB`
  * *Critical Safeguard:* Applying $10 \cdot \log_{10}$ would be mathematically invalid on negative numbers. Values are normalized via Z-score scaling.
* **Label Integrity & Class Imbalance:**
  * Label values are strictly verified: `1` (Water), `0` (Dry land), `-1` (Ignore/Cloud).
  * Empirical class balance on sample chips shows extreme imbalance (e.g., Ghana flood pixels are only **0.6%** of valid pixels; Bolivia water is **13.5%**).
  * *Why Overall Accuracy Fails:* A trivial model predicting all dry land achieves $> 86\text{--}99\%$ accuracy, but **0.00% Flood IoU**.

---

## SLIDE 8: Baseline Module Implementation (Otsu Thresholding)
* **Full Dataset Empirical Results from `results/otsu_baseline_results.json` (All 446 Chips):**

| Split Evaluated | Chips | Variant | Flood IoU (%) | Flood F1 (%) | Precision (%) | Recall (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Validation** | 89 / 89 | VV | 18.72% | 31.54% | 20.37% | 69.82% |
| **Validation** | 89 / 89 | VH | 22.00% | 36.06% | 23.77% | 74.67% |
| **Validation** | 89 / 89 | Dual-Pol Mean | **21.52%** | **35.42%** | 23.25% | 74.33% |
| **Test** | 90 / 90 | VV | 21.87% | 35.89% | 23.77% | 73.22% |
| **Test** | 90 / 90 | VH | 23.98% | 38.68% | 26.24% | 73.52% |
| **Test** | 90 / 90 | Dual-Pol Mean | **24.73%** | **39.65%** | 27.04% | 74.32% |
| **Bolivia (Held-Out)** | 15 / 15 | VV | 31.11% | 47.46% | 39.68% | 59.03% |
| **Bolivia (Held-Out)** | 15 / 15 | VH | 33.34% | 50.01% | 41.66% | 62.54% |
| **Bolivia (Held-Out)** | 15 / 15 | Dual-Pol Mean | **34.73%** | **51.55%** | 44.20% | 61.84% |

* **The Otsu Anomaly Resolved:**
  * In general scenes (Valid & Test), scenes are 89.4% dry land; unimodal histograms force Otsu to split terrestrial modes, causing low precision (~23–27%) and over 5 million false positive pixels.
  * In Bolivia, massive flooding (up to 60% water) creates a truly bimodal histogram, raising Otsu IoU to 34.73%.
  * VH consistently outperforms VV because volume scattering over water is minimal in cross-polarization.

---

## SLIDE 9: Deep Learning Architecture & Real U-Net Performance
* **Model:** U-Net (ResNet-34 encoder pre-trained on ImageNet, 2-channel adapted with $\times 1.5$ scale preservation).
* **Training Setup:** Local NVIDIA RTX 4050 GPU (6 GB VRAM), Mixed Precision (AMP), $256 \times 256$ random crops, batch size 8, AdamW ($\text{lr}=3\times 10^{-4}$), 25 epochs in ~4 minutes.
* **Compound Masked Loss:** $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE, masked}} + 1.0 \cdot \mathcal{L}_{\text{Dice, masked}}$ (excluding `-1` ignore pixels).
* **Empirical Validation vs Otsu Baseline:**

| Evaluation Split | Model Architecture | Flood IoU (%) | Flood F1 (%) | Precision (%) | Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation** (89 chips) | Otsu (Dual-Pol Mean) | 21.52% | 35.42% | 23.25% | 74.33% |
| **Validation** (89 chips) | **U-Net ResNet-34 (Seed 0)** | **61.98%** (+40.5%) | **76.52%** | **75.27%** | **77.82%** |
| **Test** (90 chips) | Otsu (Dual-Pol Mean) | 24.73% | 39.65% | 27.04% | 74.32% |
| **Test** (90 chips) | **U-Net ResNet-34 (Seed 0)** | **65.01%** (+40.3%) | **78.80%** | **78.27%** | **79.33%** |
| **Bolivia** (15 chips) | Otsu (Dual-Pol Mean) | 34.73% | 51.55% | 44.20% | 61.84% |
| **Bolivia** (15 chips) | **U-Net ResNet-34 (Seed 0)** | **62.76%** (+28.0%) | **77.12%** | **77.46%** | **76.78%** |

* **Real Visual Predictions Generated:** 5-panel diagnostic maps (`results/real_prediction_valid.png` and `results/real_prediction_bolivia.png`) demonstrating precise water delineation, low false alarms, and minimal cross-region degradation (-2.25% IoU drop on Bolivia).
* **Test Suite Status:** 22/22 unit tests passing in `tests/` across dataset, metrics, models, losses, otsu, and exposure modules.

---

## SLIDE 10: Roadmap to Review-3 (Stages B through E)
* **Stage B (27 Oct – 10 Nov):** Remote/local GPU training of U-Net, DeepLabV3+, and SegFormer across 3 seeds; MC Dropout ($T=20$) and Deep Ensembles; compute ECE and error-detection AUROC on Bolivia.
* **Stage C (10 Nov – 18 Nov):** Implement fractional WorldPop population aggregation and dynamic UTM OSM road clipping; freeze numbers by 18 Nov.
* **Stage D (18 Nov – 22 Nov):** Complete Leaflet web app and dry-run 10-minute demo; prepare answers for viva questions.
* **Stage E (15 Nov – 15 Dec):** Complete 50-page report; run Turnitin screening by 05 Dec ($< 25\%$ similarity & AI score); Guide signature by 10 Dec; submit hard copy by 15 Dec.

