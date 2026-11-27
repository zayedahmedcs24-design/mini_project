# Review-3 Presentation: Flood Mapping and Geospatial Exposure Analysis

---

## Slide 1: Title
### Automated Flood Inundation Mapping and Exposure Analysis via Sentinel-1 SAR and Deep Learning

- **Project Phase:** Review-3 Implementation and Verification
- **Authors:** [NAMES]
- **Project Guide:** [GUIDE]
- **Academic Year:** 2026–2027

---

## Slide 2: Problem Definition and Why Synthetic Aperture Radar (SAR)
### The Operational Challenge in Cloud-Covered Flood Disasters

- **Severe Cloud Penetration Limitation in Optical Sensors:**
  - Optical sensors (e.g., Sentinel-2, Landsat) fail during monsoonal storm systems, heavy cloud cover, precipitation, and nighttime conditions.
  - Floods inherently coincide with severe meteorological cloud cover, rendering optical imagery unavailable when rapid mapping is needed.
- **Physical Rationale for Spaceborne C-Band SAR (Sentinel-1):**
  - Sentinel-1 operates at C-band microwave frequency (~5.405 GHz, wavelength ~5.6 cm), penetrating cloud cover, precipitation, haze, and smoke.
  - Active microwave imaging operates independently of solar illumination (day and night acquisition).
  - Open water acts as a specular reflector: smooth water surfaces scatter incident radar energy away from the sensor, yielding low backscatter (dark pixels).
  - Terrestrial terrain produces diffuse backscatter (brighter radar returns), establishing a physical backscatter contrast.
- **Physical Challenges in SAR Processing:**
  - Rough water surfaces caused by surface wind induce Bragg scattering, increasing backscatter.
  - Radar shadow in high-relief terrain mimics smooth water backscatter.
  - Specular reflection occurs over dry flat surfaces (paved runways, asphalt, smooth sand).

---

## Slide 3: Sustainable Development Goals (SDG) Alignment
### Mapping Engineering Implementation to UN Target Indicators

- **UN SDG 13: Climate Action — Target 13.1:**
  - *"Strengthen resilience and adaptive capacity to climate-related hazards and natural disasters in all countries."*
  - Implementation: Quantitative inundation extent mapping at 10-meter spatial resolution provides rapid assessment for flood hazard response.
- **UN SDG 11: Sustainable Cities and Communities — Target 11.5:**
  - *"By 2030, significantly reduce the number of deaths and the number of people affected and substantially decrease the direct economic losses relative to global gross domestic product caused by disasters..."*
  - Implementation: Downstream coupling of predicted flood boundaries with high-resolution gridded population (WorldPop) and transport infrastructure (OpenStreetMap / Geofabrik) directly estimates impacted people and road disruption.

---

## Slide 4: Literature Survey
### Comparative Analysis of SAR Flood Segmentation Methodologies

TO BE FILLED BY TEAM FROM VERIFIED 2024-2026 JOURNAL PAPERS

| Author(s) & Year | Title | Journal / Venue | Polarization / Data | Model / Approach | Key Metrics | Limitations | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

---

## Slide 5: Research Gap
### Systematic Limitations in Current SAR Inundation Mapping

*(Candidate gap, to be confirmed by the literature table)*

1. **Deterministic Predictions Without Uncertainty Quantification:**
   - Standard binary segmentation masks produce single-threshold decisions without spatial confidence estimates or error indicators.
   - Decision-makers receive uncalibrated masks where borderline shoreline errors cannot be distinguished from reliable open-water detections.
2. **Decoupling of Segmentation from Exposure Estimation:**
   - Most published benchmarks report computer vision overlap metrics (IoU, F1) on curated image chips without translating spatial masks into operational risk exposure metrics (impacted population counts, disrupted transport infrastructure).
3. **Rigid Single-Threshold Baselines Across Heterogeneous Geographies:**
   - Standard pipelines frequently rely on global or local Otsu thresholding without systematic evaluation of failure modes across differing land cover and dry chips.

---

## Slide 6: System Architecture and Workflow
### End-to-End Modular Pipeline: Implemented vs. Future Work

```
+---------------------------------------------------------------------------------------------+
|                                    IMPLEMENTED PIPELINE                                     |
+---------------------------------------------------------------------------------------------+
|  [Sentinel-1 Dual-Pol SAR (VV, VH)]                                                         |
|         |                                                                                   |
|         v                                                                                   |
|  [Standardized Preprocessing: dB Conversion, NaN Fill, [-50, 1] Clip, Min-Max Normalization]|
|         |                                                                                   |
|         +---------------------------------------+                                           |
|         |                                       |                                           |
|         v                                       v                                           |
|  [U-Net ResNet-34 (3 Seeds)]             [DeepLabV3+ ResNet-34 (3 Seeds)]                   |
|         |                                       |                                           |
|         +-------------------+-------------------+                                           |
|                             |                                                               |
|                             v                                                               |
|         [Probability Averaging (3-Seed Ensemble)]                                           |
|                             |                                                               |
|         +-------------------+-------------------+                                           |
|         |                                       |                                           |
|         v                                       v                                           |
|  [Uncertainty & Calibration Engine]      [Operational Risk Thresholds]                     |
|   - Temperature Scaling (T* on valid)     - Conservative (p > 0.70)                         |
|   - Predictive Entropy                     - Expected (p > 0.50)                             |
|   - Error Detection (AUROC/AUPRC)          - Worst-Case (p > 0.30)                           |
|   - Boundary Proximity Analysis                 |                                           |
|   - Rejection Curves                            v                                           |
|                                          [Geospatial Exposure Integration]                  |
|                                           - WorldPop 100m Gridded Population Counts         |
|                                           - Local Geofabrik OpenStreetMap Road Shapefiles   |
|                                           - Local UTM Planar Metric Projections             |
|                                                 |                                           |
|                                                 v                                           |
|                                          [Static Offline Leaflet Web Viewer]                |
+---------------------------------------------------------------------------------------------+
|                                         FUTURE WORK                                         |
+---------------------------------------------------------------------------------------------+
|  - Vision Transformer Backbone Integration (SegFormer MiT-B0)                              |
|  - Active Monte Carlo Dropout (training with persistent dropout layers for inference sampling)|
|  - Automated Ingestion Pipeline via Copernicus Data Space API                               |
|  - Dynamic Global Multi-Event Ingestion and Live OSM API Integration                        |
+---------------------------------------------------------------------------------------------+
```

---

## Slide 7: Dataset Properties, Label Structure, and Class Imbalance
### Sen1Floods11 Hand-Labeled Ground Truth Benchmark

- **Dataset Characteristics:**
  - 446 hand-labeled chips ($512 \times 512$ pixels, 10m spatial resolution) across 11 global flood events.
  - Sensor: Sentinel-1 C-band Synthetic Aperture Radar (Interferometric Wide swath mode, Dual-Pol VV + VH).
- **Split Definitions:**
  - Train: 252 chips ($66,060,288$ total pixels).
  - Valid: 89 chips ($23,330,816$ total pixels).
  - Test: 90 chips ($23,592,960$ total pixels).
  - Bolivia Split: 15 hand-labeled chips ($3,932,160$ total pixels) representing one specific regional flood event.
- **Handling of Special Label Values:**
  - Ground truth contains three values: `0` (non-flood), `1` (flood), and `-1` (ignore / nodata).
  - Ignore pixels (`-1`) are masked out during training (loss computation ignores `-1`) and during evaluation.
  - Across the dataset, ignore pixels constitute between $10\%$ and $27\%$ of chip areas due to sensor edge limits and invalid values.
- **Severe Class Imbalance and the Failure of Pixel Accuracy:**
  - Flood pixels constitute only $4.87\%$ of all valid pixels across the dataset (valid: $8.40\%$, test: $4.94\%$, Bolivia: $6.97\%$).
  - A naive majority-class classifier predicting non-flood everywhere achieves $>95\%$ pixel accuracy while detecting zero flood extent.
  - Primary evaluation metric is therefore Intersection over Union (IoU) of the flood class (Jaccard Index), accompanied by F1-Score (Dice Coefficient), Precision, and Recall.

---

## Slide 8: Verification, Test Coverage, and the 27 November Exporter Finding
### Software Quality Assurance and Regression Prevention

- **Comprehensive Test Suite:**
  - All 44 automated unit and integration tests passing (`venv\Scripts\python.exe -m pytest tests -q`).
  - Suite covers dataset loading, NaN handling, tensor shapes, loss functions, metrics, temperature scaling, ensemble inference, exposure estimation, and web export consistency.
- **The 27 November Web-Export Finding:**
  - *Symptom:* Initial inspection of web viewer overlay generation revealed false-positive over-segmentation across multiple chips.
  - *Root Cause Analysis:* `scripts/export_web_layers.py` had reimplemented preprocessing independently and evaluated raw decibel values directly without applying the standard clipping to $[-50, 1]$ dB and min-max normalization used by `FloodDataset`.
  - *Detection:* Caught through systematic end-to-end consistency verification comparing web export prediction pixel counts against canonical checkpoint metrics.
  - *Resolution:* Exporter refactored to directly reuse `FloodDataset.__getitem__` in evaluation mode, eliminating duplicated preprocessing logic.
  - *Regression Test:* Created `tests/test_web_export.py` asserting pixel-for-pixel identity between web exported masks and model evaluation outputs across multiple chips (`Bolivia_129334`, `Sri-Lanka_713926`, `USA_905409`).

---

## Slide 9: Classical SAR Baseline Evaluation: Otsu Thresholding
### Comparison of Dynamic Per-Chip vs. Global Train-Fitted Otsu

- **Otsu Algorithm Mechanism:**
  - Computes a threshold minimizing intraclass variance of pixel intensity histograms.
- **Why Dynamic Per-Chip Otsu Fails on Dry and Near-Dry Chips:**
  - Otsu fundamentally assumes a bimodal intensity distribution (foreground and background).
  - On dry or low-flood chips, the distribution is unimodal (land only). Per-chip Otsu forces a partition through dry land, generating massive false-positive masks.
- **Fair Baseline: Global Train-Fitted Threshold:**
  - Threshold fitted once across all valid pixels in the training split:
    - VV threshold: $0.070313$ dB
    - VH threshold: $-0.192383$ dB
    - Dual-Pol Mean threshold: $-0.153809$ dB

| Thresholding Method | Polarization / Channel | Valid Flood IoU | Test Flood IoU | Bolivia Flood IoU |
| :--- | :---: | :---: | :---: | :---: |
| Per-Chip Dynamic Otsu | VV | 18.72% | 21.87% | 31.11% |
| Train-Fitted Global Otsu | VV | 31.47% | 33.88% | 56.38% |
| Per-Chip Dynamic Otsu | VH | 22.00% | 23.98% | 33.34% |
| Train-Fitted Global Otsu | VH | 42.62% | 48.12% | 62.13% |
| Per-Chip Dynamic Otsu | Dual-Pol Mean | 21.52% | 24.73% | 34.73% |
| Train-Fitted Global Otsu | Dual-Pol Mean | 44.58% | 49.54% | 64.81% |
<!-- Source: results/otsu_baseline_results.json and results/metrics_summary_table.csv -->

---

## Slide 10: Deep Learning Segmentation Master Results
### Single-Seed Means vs. 3-Seed Probability Averaging Ensembles

| Split | Architecture | Single-Seed Mean ± Std (Flood IoU) | 3-Seed Ensemble Flood IoU | Single-Seed Mean Flood F1 | 3-Seed Ensemble Flood F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Valid** | U-Net ResNet-34 | 60.60 ± 0.26% | 62.31% | 75.47 ± 0.20% | 76.78% |
| **Valid** | DeepLabV3+ ResNet-34 | 61.62 ± 0.35% | 63.18% | 76.25 ± 0.27% | 77.44% |
| **Test** | U-Net ResNet-34 | 63.81 ± 0.15% | 65.33% | 77.91 ± 0.11% | 79.03% |
| **Test** | DeepLabV3+ ResNet-34 | 64.27 ± 0.52% | 66.43% | 78.25 ± 0.39% | 79.83% |
| **Bolivia** | U-Net ResNet-34 | 60.30 ± 7.59% | 60.91% | 75.04 ± 6.08% | 75.71% |
| **Bolivia** | DeepLabV3+ ResNet-34 | 61.89 ± 5.45% | 63.32% | 76.37 ± 4.18% | 77.54% |
<!-- Source: results/metrics_summary_table.csv and results/uncertainty_results.json -->

- **Key Observations:**
  - Deep learning architectures improve Flood IoU over the global Otsu Dual-Pol Mean baseline ($49.54\%$ on test).
  - 3-seed probability averaging ensembles show higher IoU than single-seed means across splits ($+1.52$ pp on test U-Net, $+2.16$ pp on test DeepLabV3+).

---

## Slide 11: Quantitative Findings on Held-Out Bolivia Split
### 15 Chips from One Regional Event: Exact Differences and Dispersion

- **Dataset Scope:**
  - 15 chips from the February 2018 flood event in Beni Department, Bolivia.
- **Exact Differences from the Test Split:**
  - Single-seed means:
    - U-Net: $60.30\%$ on Bolivia vs. $63.81\%$ on Test ($3.51$ percentage points lower).
    - DeepLabV3+: $61.89\%$ on Bolivia vs. $64.27\%$ on Test ($2.38$ percentage points lower).
  - 3-seed ensembles:
    - U-Net: $60.91\%$ on Bolivia vs. $65.33\%$ on Test ($4.42$ percentage points lower).
    - DeepLabV3+: $63.32\%$ on Bolivia vs. $66.43\%$ on Test ($3.11$ percentage points lower).
- **Dispersion and Seed Sensitivity:**
  - Standard deviation across seeds is substantially larger on Bolivia than on Test:
    - U-Net seed standard deviation is $\pm 7.59\%$ on Bolivia compared to $\pm 0.15\%$ on Test.
    - DeepLabV3+ seed standard deviation is $\pm 5.45\%$ on Bolivia compared to $\pm 0.52\%$ on Test.
  - Ensemble scores on Bolivia fall within the range of single-seed scores.
- **Per-Chip Spread on Bolivia:**
  - Per-chip IoU spans from $0.00\%$ (`Bolivia_76104`, `Bolivia_233925`) and $0.16\%$ (`Bolivia_195474`) up to $84.46\%$ (U-Net) / $84.89\%$ (DeepLabV3+) on `Bolivia_129334`.
  - Because Bolivia consists of 15 chips from one single event, we report the numerical differences without asserting causal explanations.
<!-- Source: results/metrics_summary_table.csv, results/uncertainty_results.json, and results/per_chip_bolivia.csv -->

---

## Slide 12: Uncertainty Quantification, Calibration, and Error Detection
### Statistical Behavior of Predictive Entropy and Temperature Scaling

| Evaluation Dimension | Metric / Parameter | U-Net ResNet-34 | DeepLabV3+ ResNet-34 |
| :--- | :--- | :---: | :---: |
| **Optimal Temperature** | $T^*$ fitted on validation split | 0.7289 | 1.2720 |
| **Expected Calibration Error (ECE)** | Valid (Raw $\to$ Scaled) | 4.44% $\to$ 1.73% | 1.55% $\to$ 0.71% |
| | Test (Raw $\to$ Scaled) | 4.40% $\to$ 1.67% | 1.86% $\to$ 0.82% |
| | Bolivia (Raw $\to$ Scaled) | 4.74% $\to$ 3.03% | 3.15% $\to$ 2.71% |
| **Error Detection (AUROC)** | Valid (500k sample, seed 42) | 0.8686 | 0.9204 |
| | Test (500k sample, seed 42) | 0.8140 | 0.8967 |
| | Bolivia (500k sample, seed 42) | 0.8643 | 0.8941 |
| **Error Detection (AUPRC)** | Valid (Baseline: 0.0504 / 0.0487) | 0.3186 | 0.3524 |
| | Test (Baseline: 0.0500 / 0.0494) | 0.2934 | 0.3177 |
| | Bolivia (Baseline: 0.0722 / 0.0676) | 0.3529 | 0.3601 |
| **Boundary Proximity Ratio** | Test (Top-10% uncertain $\le 3\text{ px}$) | 3.65x (38.37% near edge) | 4.20x (41.01% near edge) |
<!-- Source: results/uncertainty_results.json -->

- **Retained-Pixel Rejection Curve (Test Split):**
  - U-Net: $k=0\%$ removed: $65.33\%$ IoU $\to k=10\%$ removed: $76.96\%$ IoU $\to k=30\%$ removed: $80.59\%$ IoU $\to k=50\%$ removed: $82.98\%$ IoU.
  - DeepLabV3+: $k=0\%$ removed: $66.43\%$ IoU $\to k=10\%$ removed: $76.65\%$ IoU $\to k=30\%$ removed: $21.05\%$ IoU $\to k=50\%$ removed: $4.00\%$ IoU.
- **Uncertainty Limitations:**
  - Uncertain pixels concentrate heavily along water-land boundaries (boundary proximity ratios $3.56\text{x}$ to $4.20\text{x}$).
  - Monte Carlo dropout was skipped because trained model checkpoints contain no active dropout layers during evaluation/inference.

---

## Slide 13: Geospatial Exposure Assessment: Bolivia and Sri Lanka
### Coupling Segmented Inundation with Gridded Population and Road Infrastructure

> totals compare area and people exposed, not the location of every flooded pixel; e.g. Bolivia_290290 predicted area 1.65 vs 1.91 km2 but IoU 53%

### Bolivia Exposure Summary (15 Chips, $379.7\text{ km}^2$, Pop: $262.6$, Density: $0.7\text{ ppl/km}^2$)
| Model / Tier | Threshold | Flooded Area ($\text{km}^2$) | Exposed Population | Flooded Roads (km) | Area Rel Error | Pop Rel Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (Observed)** | Label == 1 | 43.86 | 19.2 | 1.66 | 0.00% (Ref) | 0.00% (Ref) |
| U-Net Ensemble (Conservative) | $p > 0.70$ | 29.81 | 14.0 | 1.90 | -32.03% | -26.95% |
| U-Net Ensemble (Expected) | $p > 0.50$ | 39.67 | 19.1 | 2.12 | -9.56% | -0.27% |
| U-Net Ensemble (Worst-Case) | $p > 0.30$ | 48.37 | 24.3 | 2.44 | +10.27% | +26.73% |
| DeepLabV3+ Ensemble (Conservative) | $p > 0.70$ | 30.51 | 12.7 | 1.42 | -30.45% | -33.62% |
| DeepLabV3+ Ensemble (Expected) | $p > 0.50$ | 41.23 | 19.6 | 2.34 | -6.00% | +2.13% |
| DeepLabV3+ Ensemble (Worst-Case) | $p > 0.30$ | 48.96 | 24.3 | 2.49 | +11.62% | +26.56% |
<!-- Source: results/exposure_bolivia.json and results/exposure_summary.md -->

### Sri-Lanka Exposure Summary (9 Chips, $232.5\text{ km}^2$, Pop: $65,032.7$, Density: $279.7\text{ ppl/km}^2$)
| Model / Tier | Threshold | Flooded Area ($\text{km}^2$) | Exposed Population | Flooded Roads (km) | Area Rel Error | Pop Rel Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (Observed)** | Label == 1 | 37.40 | 1,648.9 | 4.40 | 0.00% (Ref) | 0.00% (Ref) |
| U-Net Ensemble (Conservative) | $p > 0.70$ | 31.44 | 586.1 | 1.70 | -15.95% | -64.45% |
| U-Net Ensemble (Expected) | $p > 0.50$ | 35.98 | 1,240.7 | 5.79 | -3.79% | -24.76% |
| U-Net Ensemble (Worst-Case) | $p > 0.30$ | 39.50 | 2,069.5 | 11.39 | +5.60% | +25.51% |
| DeepLabV3+ Ensemble (Conservative) | $p > 0.70$ | 31.63 | 642.3 | 1.62 | -15.44% | -61.04% |
| DeepLabV3+ Ensemble (Expected) | $p > 0.50$ | 36.71 | 1,307.3 | 5.51 | -1.86% | -20.72% |
| DeepLabV3+ Ensemble (Worst-Case) | $p > 0.30$ | 39.68 | 2,189.9 | 12.98 | +6.09% | +32.81% |
<!-- Source: results/exposure_sri-lanka.json and results/exposure_summary.md -->

- **Essential Methodological Caveats:**
  - Operational sensitivity range: Fixed thresholds ($0.70, 0.50, 0.30$) define a sensitivity range, not statistical confidence bounds.
  - WorldPop products and years differ per event (Bolivia: unconstrained 2018; Sri-Lanka: constrained 2020); people counts must not be compared across events.
  - OpenStreetMap road extracts are dated 2026 (local Geofabrik).
  - Status "no roads in the extract" is recorded as null, not $0\text{ km}$.
  - In Bolivia, all $1.66\text{ km}$ of ground-truth flooded roads occur on a single chip (`Bolivia_290290`); 9 of 15 chips contain no road features in the extract.
  - Sri-Lanka event year is not verified locally in dataset files.

---

## Slide 14: Interactive Web Demonstration, Limitations, and Future Work
### Verified Offline Interface and Roadmap to Final Review

- **Interactive Static Web Demonstration:**
  - Client-side Leaflet viewer running fully offline with vendored assets (`web/vendor/leaflet/`).
  - Served via local HTTP server: `python -m http.server 8000 --directory web`.
  - Displays dual-pol SAR (VV), ground truth, model predictions (U-Net and DeepLabV3+ ensembles), predictive entropy, and error decomposition overlays across 30 chips.
- **Automated Interface Screenshots (Neutral Verification):**
  - `01_viewer_bolivia_default.png`: View of `Bolivia_290290` (Area: $1.91\text{ km}^2$, Pop: $2.3$, Road: $1.66\text{ km}$).
  - `02_viewer_bolivia_roads_null.png`: View of `Bolivia_103757` with null road status ("no roads in the extract").
  - `03_viewer_sri_lanka_largest_pop.png`: View of `Sri-Lanka_713926` (Area: $7.92\text{ km}^2$, Pop: $542.2$, Road: $0.24\text{ km}$).
  - `04_viewer_sri_lanka_largest_area.png`: View of `Sri-Lanka_534068` ($25.30\text{ km}^2$ inundated area).
  - `05_viewer_entropy_uncertainty.png`: Overlay of predictive entropy.
  - `06_viewer_error_analysis.png`: Overlay showing false positives (red) and false negatives (orange).
  - `07_viewer_deeplab_prediction.png`: Prediction overlay for DeepLabV3+ ensemble.
  - `08_results_master_table.png`: Master quantitative benchmark table in browser.
  - `09_results_exposure_benchmark.png`: Multi-event exposure comparison table in browser.
- **Identified Limitations:**
  - Evaluated on a limited sample of chips (15 for Bolivia, 9 for Sri-Lanka).
  - High model uncertainty and segmentation errors concentrate around complex water-land boundaries.
- **Future Work Roadmap:**
  - SegFormer (MiT-B0 transformer backbone) implementation for multi-scale attention.
  - Active Monte Carlo Dropout implementation during training to enable inference-time dropout sampling.
  - Multi-event ingestion and live streaming Sentinel-1 ingestion pipelines.
- **Conclusion:**
  - Complete, reproducible deep learning flood segmentation and geospatial exposure assessment pipeline with 44 passing unit tests and verified quantitative benchmarks.
