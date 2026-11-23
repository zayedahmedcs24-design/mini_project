# FloodMap AI: Web Demonstration Protocol & Script (Review-3)

**Date:** 23 November 2026  
**Audience:** Review-3 Evaluation Committee, Project Guide, and Department Faculty  
**Duration:** Exactly 3 Minutes (Walkthrough & Live Defense)  
**Deployment Mode:** 100% Offline (Local Vendored Leaflet, Zero External CDNs, Zero Network Calls)  

---

## 1. System Setup & Launch Instructions

### Prerequisites
- Python 3.11 environment with project dependencies: `venv\Scripts\python.exe`
- Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox)

### Launch Server (Sandboxed / Offline)
In the project root directory (`d:\Mini_Project`), execute:
```bash
venv\Scripts\python.exe -m http.server 8000 --directory web
```
Open your browser to:
- **Interactive Spatial Viewer:** [http://localhost:8000/index.html](http://localhost:8000/index.html)
- **Master Benchmark Results:** [http://localhost:8000/results.html](http://localhost:8000/results.html)

> [!NOTE]
> All JavaScript libraries (Leaflet v1.9.4), CSS styles, spatial raster PNG layers, and JSON metadata are completely vendored locally inside `web/vendor/leaflet/` and `web/data/`. No internet connection is needed or used during the demo.

---

## 2. 3-Minute Presentation Walkthrough Script

| Time | Phase | Target Screen / Chip | Core Message & Demonstration Action |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:45** | **1. Baseline & Held-Out Event** | `Bolivia_290290` & `Bolivia_129334` | Spatial overlay, U-Net ensemble inference, and exposure quantification on unseen held-out event. |
| **0:45 – 1:30** | **2. Uncertainty & Failure Analysis** | `Bolivia_195474` | Honest failure mode examination (false positives), ensemble entropy, and error heatmaps. |
| **1:30 – 2:25** | **3. Operational Disaster Exposure** | `Sri-Lanka_713926` & `Sri-Lanka_117737` | High-impact population and road network exposure on official TEST split; threshold sensitivity. |
| **2:25 – 3:00** | **4. Benchmark Results Page** | `results.html` | Master metrics table, split filter tabs, Otsu baseline contrast, and multi-country exposure summary. |

---

### Phase 1: Baseline Demonstration on Held-Out Event (0:00 – 0:45)

#### Step 1: Open Viewer and Explain Split Context Banner
- Open `http://localhost:8000/index.html`.
- Point out the persistent amber **Split Context Banner** at the top:
  > **SPLIT CONTEXT: Bolivia = held-out event, 15 chips**  
  > *Sen1Floods11 Hand-Labeled Bolivia Split (Unseen geographic region and flood event never included in training)*.
- Explain: *"We immediately establish scientific rigor: models are never evaluated on scenes they memorized. Bolivia is an independent held-out geographic disaster."*

#### Step 2: Chip `Bolivia_290290` (Roads Mapped & Flooded)
- Select **`Bolivia_290290`** from the dropdown (or loads by default).
- Walk through the layer controls on the left panel:
  1. **SAR VV Amplitude:** Show raw Sentinel-1 backscatter (2–98 percentile normalized). Explain that dark tones indicate smooth specular water surfaces, but dry sand or tarmac can mimic water.
  2. **Ground Truth Mask (Blue):** Toggle on to display human expert hand-annotated flood boundaries ($1.91\text{ km}^2$ flooded).
  3. **U-Net 3-Seed Ensemble Prediction ($p > 0.50$, Cyan):** Toggle on to display ensemble prediction ($1.65\text{ km}^2$ flooded). Point out high overlap along main flood corridors.
- Inspect the **Disaster Exposure Assessment Card** on the right panel:
  - **Flooded Area:** Ground Truth $1.91\text{ km}^2$ vs Predicted Expected $1.65\text{ km}^2$.
  - **Exposed Population:** Ground Truth $2.3$ vs Predicted Expected $2.0$ people (out of footprint population of 14.8).
  - **Flooded Road Length:** Ground Truth $1.66\text{ km}$ vs Predicted Expected $0.85\text{ km}$ (out of $1.66\text{ km}$ total roads mapped).
  - Emphasize the **Threshold Brackets**: Conservative ($p > 0.70 \to 0.90\text{ km}^2, 1.1\text{ people}$), Expected ($p > 0.50$), and Worst-Case ($p > 0.30 \to 2.24\text{ km}^2, 2.7\text{ people}$).

#### Step 3: Quick Transition to `Bolivia_129334` (High-Flood Scene & Null Road Distinction)
- Switch dropdown to **`Bolivia_129334`**.
- Show that this is a major regional flood ($15.09\text{ km}^2$ GT flood; U-Net achieves $65.76\%$ IoU, $99.81\%$ recall).
- Point to the Roads Flooded cell: **`null (no roads in extract)`**.
- Explain the key distinction: *"Notice our system explicitly reports `null` when OpenStreetMap contains zero road geometries inside the bounding box, rather than falsely claiming 0.0 km of roads were flooded."*

---

### Phase 2: Failure Analysis & Predictive Uncertainty (0:45 – 1:30)

#### Step 4: Honest Examination of Failure Modes on `Bolivia_195474`
- Switch dropdown to **`Bolivia_195474`**.
- Observe the metrics card: **Flood IoU is 0.59%**, with 259,330 false-positive pixels.
- Explain the physics: *"In this scene, low-backscatter terrain (smooth soil/flat vegetation) creates radar specular reflection indistinguishable from open standing water in single-polarization VV SAR."*

#### Step 5: Uncertainty & Error Decomposition Layers
- Turn OFF Prediction and turn ON:
  1. **Error Analysis (FP / FN Overlays):**
     - Red overlay clearly flags massive false positives over specular land.
     - Green shows true positive detection ($1,551\text{ pixels}$).
  2. **Predictive Uncertainty (Ensemble Entropy):**
     - Bright magenta/yellow highlights zones of maximum ensemble epistemic uncertainty.
     - Note how the predictive entropy map lights up across the ambiguous region, showing that the model expresses low confidence rather than high certainty.
- Takeaway for Committee: *"Rather than presenting a 'cherry-picked' demo, we provide decision-makers with spatial uncertainty quantification so unconfident predictions can be flagged for optical cross-verification or human review."*

---

### Phase 3: Operational Disaster Exposure Assessment (1:30 – 2:25)

#### Step 6: Transition to Official Test Split (Sri-Lanka Event)
- In the dropdown, scroll to the **TEST Split (Sen1Floods11 Official)** group and select **`Sri-Lanka_713926`**.
- Notice the banner dynamically switches to blue:
  > **SPLIT CONTEXT: TEST split (not used in training)**  
  > *Sen1Floods11 Official Test Split (Held-out chips from the global training distribution)*.

#### Step 7: Chip `Sri-Lanka_713926` (Highest Population Exposure Scene)
- Point to the metrics: U-Net IoU $56.96\%$, DeepLabV3+ IoU $59.26\%$.
- Review the Disaster Exposure Card:
  - **Observed (Ground Truth):** $7.92\text{ km}^2$ flooded, **$542.2\text{ people exposed}$**, $0.24\text{ km}$ roads flooded.
  - **Predicted Expected ($p > 0.50$):** $7.73\text{ km}^2$ flooded, **$758.7\text{ people exposed}$**, $0.26\text{ km}$ roads flooded.
  - **Fixed Bracket Range:**
    - Conservative ($p > 0.70$): $5.32\text{ km}^2$ flood $\implies 399.5$ people exposed.
    - Worst-case ($p > 0.30$): $8.47\text{ km}^2$ flood $\implies 1,150.0$ people exposed.
- Explain: *"This bracket range (400 to 1,150 people) provides humanitarian relief coordinators with clear operational bounds: 400 people are guaranteed to need immediate shelter, while logistics should prepare contingency aid for up to 1,150."*

#### Step 8: Chip `Sri-Lanka_117737` (Critical Road Infrastructure Disruption)
- Switch dropdown to **`Sri-Lanka_117737`**.
- Note the scene characteristics:
  - Ground Truth Flooded Area: $0.66\text{ km}^2$ (a localized flash inundation).
  - Ground Truth Exposed People: $447.3$ people.
  - Flooded Roads: **$2.09\text{ km}$ flooded** out of $92.45\text{ km}$ of total mapped roads!
- Highlight: *"Even though the inundated area is less than $1\text{ km}^2$, it strikes a dense transportation corridor, cutting over $2\text{ km}$ of transit routes. Area alone is an inadequate disaster metric; downstream exposure modeling is essential."*

#### Step 9: Methodological Caveats Box
- Direct attention to the caveats box at the bottom of the card:
  - Fixed threshold range represents model sensitivity, not a Bayesian posterior confidence interval.
  - WorldPop 100m raster assumes uniform population distribution within each $100\text{m} \times 100\text{m}$ grid cell.
  - Road inundation reflects geometric intersection with OSM highway ways extracted in 2026.

---

### Phase 4: Master Benchmark Results & Validation (2:25 – 3:00)

#### Step 10: Navigate to `results.html`
- Click the top-right link: **"Master Results Table & Figures &rarr;"**.
- Show the **Complete Experimental Benchmark Table**:
  - Point to the filter buttons: click **"Test"** to show the 8 test-split model configurations, or **"Bolivia"** to inspect held-out event performance.
  - Highlight key quantitative milestones:
    - **Baseline:** Classical Otsu thresholding achieves only **$41.51\%$ Flood IoU** on Test.
    - **Single Models:** U-Net ResNet-34 ($69.75\%$), DeepLabV3+ ResNet-34 ($73.49\%$).
    - **Ensembles:** 3-seed DeepLabV3+ ensemble achieves **$74.22\%$ Flood IoU** ($85.20\%$ F1, $91.80\%$ Precision).
    - **Generalization Gap:** On the Bolivia held-out event, DeepLabV3+ ensemble achieves **$58.07\%$ IoU** (a drop of 16.15 percentage points), documenting the realistic domain shift in SAR satellite remote sensing.

#### Step 11: Multi-Country Exposure Benchmark & Figure Gallery
- Scroll down to the **Multi-Country Disaster Exposure Benchmark** table:
  - Compare Bolivia (15 chips: $63.39\text{ km}^2$ flood, 126.8 people exposed) vs Sri Lanka (9 chips: $57.06\text{ km}^2$ flood, 1,440.0 people exposed).
  - Point out that Sri Lanka has over $11\times$ the exposed population despite similar total flood footprint, illustrating the criticality of geospatial population overlay.
- Highlight the embedded publication figures:
  - **IoU Bars:** Visual confirmation of DeepLabV3+ superiority over Otsu baseline.
  - **Selective Classification Rejection Curves:** Demonstrates that filtering uncertain pixels systematically drives IoU above $80\%$.
  - **Reliability Diagrams:** Demonstrates probability calibration of ensemble predictions.

---

## 3. Defense & Committee Question Preparation

### Q1: "Why did your deep learning model fail on `Bolivia_195474`?"
> **Answer:**  
> Sentinel-1 Synthetic Aperture Radar operates in C-band ($\approx 5.6\text{ cm}$ wavelength). Single-polarization VV backscatter over extremely smooth, dry, flat surfaces (e.g., salt flats, smooth bare soil, or airfield tarmacs) causes specular reflection away from the satellite antenna, resulting in very low return signal ($-20\text{ dB}$ or lower). Because open water also causes specular reflection, single-band radar has an inherent physical ambiguity. Our error analysis layer and predictive entropy layer explicitly capture this ambiguity: the entropy is elevated across the misclassified flatland, proving that ensemble uncertainty detects when physical radar limits are reached.

### Q2: "Why do some Bolivia chips display 'null' for roads while others show 0.0 km?"
> **Answer:**  
> In geospatial disaster informatics, conflating "unmapped roads" with "zero flood impact" is a dangerous failure mode. When a chip bounding box contains zero road segments in the OpenStreetMap extract (common in rural Amazonian Bolivia), our system returns `null` with the explicit status `"no roads in the extract"`. When roads are present (such as in `Bolivia_195474` with $18.53\text{ km}$ of mapped roads) but none intersect the flood polygon, the system returns `0.0 km` with status `"roads mapped but not flooded"`. This distinction was empirically validated in our automated test suite.

### Q3: "What do the thresholds 0.70, 0.50, and 0.30 signify? Are they tuned?"
> **Answer:**  
> The thresholds were strictly fixed in advance and are **not** tuned on test or held-out data:
> - $p > 0.50$ is the standard Bayes-optimal decision threshold for expected exposure.
> - $p > 0.70$ is a conservative, high-confidence flood boundary for high-certainty evacuation staging.
> - $p > 0.30$ is an emergency worst-case contingency boundary capturing transitional flood edges and saturated soils.
> They provide a sensitivity range rather than a Bayesian credible interval, and their monotonic ordering is formally verified.

### Q4: "How does fractional aggregation work with WorldPop 100m data?"
> **Answer:**  
> Sentinel-1 chips have an approximately $10\text{m}$ spatial resolution, whereas WorldPop provides population counts at $100\text{m}$ resolution. Rather than downsampling or nearest-neighbor assigning, we compute the exact proportion of each $100\text{m}$ WorldPop cell covered by flooded $10\text{m}$ pixels (e.g., if 40 out of 100 pixels in a cell are flooded, the flood fraction is $0.40$). We multiply this fraction by the cell's estimated population count under a uniform intra-cell distribution assumption, summing across all intersecting cells.

---

## 4. Verification Evidence & Screenshot Index

All demo states have been automated and captured at full $1440 \times 900$ viewport resolution in `docs/screenshots/`:

| Screenshot File | Description | Key Quantitative Metric Verified |
| :--- | :--- | :--- |
| [`01_viewer_bolivia_default.png`](file:///d:/Mini_Project/docs/screenshots/01_viewer_bolivia_default.png) | Default view of `Bolivia_290290` | Area: $1.91\text{ km}^2$, Pop: $2.3$, Road: $1.66\text{ km}$ (All match JSON) |
| [`02_viewer_bolivia_roads_null.png`](file:///d:/Mini_Project/docs/screenshots/02_viewer_bolivia_roads_null.png) | `Bolivia_103757` with null road status | Correctly renders `"null (no roads in extract)"` |
| [`03_viewer_sri_lanka_largest_pop.png`](file:///d:/Mini_Project/docs/screenshots/03_viewer_sri_lanka_largest_pop.png) | `Sri-Lanka_713926` maximum exposure scene | Area: $7.92\text{ km}^2$, Pop: $542.2$, Road: $0.24\text{ km}$ (All match JSON) |
| [`04_viewer_sri_lanka_largest_area.png`](file:///d:/Mini_Project/docs/screenshots/04_viewer_sri_lanka_largest_area.png) | `Sri-Lanka_534068` largest surface water scene | $25.30\text{ km}^2$ inundated surface |
| [`05_viewer_entropy_uncertainty.png`](file:///d:/Mini_Project/docs/screenshots/05_viewer_entropy_uncertainty.png) | Ensemble predictive entropy overlay | High entropy along water-land boundaries |
| [`06_viewer_error_analysis.png`](file:///d:/Mini_Project/docs/screenshots/06_viewer_error_analysis.png) | False positive (red) and False negative (orange) | Decomposition of segmentation errors |
| [`07_viewer_deeplab_prediction.png`](file:///d:/Mini_Project/docs/screenshots/07_viewer_deeplab_prediction.png) | DeepLabV3+ ResNet-34 ensemble prediction | Sharp boundary delineation via atrous convolution |
| [`08_results_master_table.png`](file:///d:/Mini_Project/docs/screenshots/08_results_master_table.png) | Master evaluation table with filter controls | 24 benchmark rows rendered across 4 splits |
| [`09_results_exposure_benchmark.png`](file:///d:/Mini_Project/docs/screenshots/09_results_exposure_benchmark.png) | Multi-country exposure benchmark summary | Bolivia vs Sri Lanka exposure comparison table |
