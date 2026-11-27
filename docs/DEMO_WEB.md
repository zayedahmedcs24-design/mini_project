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
| **0:00 – 0:45** | **1. Baseline & Held-Out Event** | `Bolivia_290290` & `Bolivia_129334` | Spatial overlay, U-Net ensemble inference, and exposure quantification on unseen held-out event *(chips selected by IoU after seeing results, not typical cases)*. |
| **0:45 – 1:30** | **2. Uncertainty & Metric Examination** | `Bolivia_195474` | Metric examination on low-flood scene *(chip selected by IoU after seeing results, not a typical case)*, ensemble entropy, and error heatmaps. |
| **1:30 – 2:25** | **3. Operational Disaster Exposure** | `Sri-Lanka_713926` & `Sri-Lanka_117737` | High-impact population and road network exposure on official TEST split; threshold sensitivity. |
| **2:25 – 3:00** | **4. Benchmark Results Page** | `results.html` | Master metrics table from `metrics_summary_table.csv` and `uncertainty_results.json`, split filter tabs, Otsu baseline, and exposure summary. |

---

### Phase 1: Baseline Demonstration on Held-Out Event (0:00 – 0:45)

#### Step 1: Open Viewer and Explain Split Context Banner
- Open `http://localhost:8000/index.html`.
- Point out the persistent amber **Split Context Banner** at the top:
  > **SPLIT CONTEXT: Bolivia = held-out event, 15 chips**  
  > *Sen1Floods11 Hand-Labeled Bolivia Split (Unseen geographic region and flood event never included in training)*.
- Explain: *"We immediately establish scientific rigor: models are never evaluated on scenes they memorized. Bolivia is an independent held-out geographic disaster."*

#### Step 2: Chip `Bolivia_290290` (Roads Mapped & Flooded)
*(Note: Chip `Bolivia_290290` was chosen specifically because roads are mapped and flooded there; not a typical case)*
- Select **`Bolivia_290290`** from the dropdown (or loads by default).
- Ensemble segmentation metrics:
  - **U-Net 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$52.95\%$** (Precision = $74.71\%$, Recall = $64.51\%$, $\text{TP} = 12,708$, $\text{FP} = 4,302$, $\text{FN} = 6,991$).
  - **DeepLabV3+ 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$56.81\%$** (Precision = $72.13\%$, Recall = $72.79\%$, $\text{TP} = 14,339$, $\text{FP} = 5,540$, $\text{FN} = 5,360$).
- Walk through the layer controls on the left panel:
  1. **SAR VV Amplitude:** Show raw Sentinel-1 backscatter (2–98 percentile normalized).
  2. **Ground Truth Mask (Blue):** Toggle on to display human expert hand-annotated flood boundaries ($1.91\text{ km}^2$ flooded).
  3. **U-Net 3-Seed Ensemble Prediction ($p \ge 0.50$, Cyan):** Toggle on to display ensemble prediction ($1.65\text{ km}^2$ flooded). Point out overlap along main flood corridors.
- Inspect the **Disaster Exposure Assessment Card** on the right panel:
  - **Flooded Area:** Ground Truth $1.91\text{ km}^2$ vs Predicted Expected $1.65\text{ km}^2$.
  - **Exposed Population:** Ground Truth $2.3$ vs Predicted Expected $2.0$ people (out of footprint population of 33.6).
  - **Flooded Road Length:** Ground Truth $1.66\text{ km}$ vs Predicted Expected $0.85\text{ km}$ (out of $5.52\text{ km}$ total roads mapped in this chip).
  - **Critical Road Context:** Note that Bolivia's total ground-truth flooded road length ($1.66\text{ km}$) comes entirely from this single chip (`Bolivia_290290`), and 9 of the 15 Bolivia chips have no road features in the OpenStreetMap extract.
  - Emphasize the **Threshold Brackets**: Conservative ($p > 0.70 \to 0.90\text{ km}^2, 1.1\text{ people}$), Expected ($p > 0.50 \to 1.65\text{ km}^2, 2.0\text{ people}$), and Worst-Case ($p > 0.30 \to 2.24\text{ km}^2, 2.7\text{ people}$) — thresholds fixed in advance as a sensitivity range; the labels are our naming, not an operational standard.

#### Step 3: Transition to `Bolivia_129334` (High-Flood Scene & Null Road Distinction)
*(Note: chips selected by IoU after seeing results, not typical cases)*
- Switch dropdown to **`Bolivia_129334`**.
- Detailed chip metrics from `results/per_chip_bolivia.csv` and `web/data/Bolivia_129334/meta.json`:
  - **Flood fraction of valid pixels:** **$65.87\%$** ($156,643$ flooded pixels out of $237,814$ valid pixels; ignore fraction: $9.28\%$; ground-truth flooded area: $15.09\text{ km}^2$).
  - **U-Net 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$84.46\%$**, Precision = **$94.59\%$**, Recall = **$88.75\%$**, $\text{TP} = 139,016$, $\text{FP} = 7,946$, $\text{FN} = 17,627$, $\text{TN} = 73,225$.
  - **DeepLabV3+ 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$84.89\%$**, Precision = **$94.66\%$**, Recall = **$89.16\%$**, $\text{TP} = 139,667$, $\text{FP} = 7,886$, $\text{FN} = 16,976$, $\text{TN} = 73,285$.
  - *(For reference, per-chip single-seed means from `results/per_chip_bolivia.csv` are U-Net mean IoU $83.35\%$, DeepLabV3+ mean IoU $84.10\%$, Global Otsu Mean IoU $86.48\%$. Do not confuse flood fraction $65.87\%$ with ensemble IoU $84.46\%$/$84.89\%$ or single-seed mean IoU).*
- Point to the Roads Flooded cell: **`no roads in the extract`**.
- Explain the key distinction: *"Notice our system explicitly reports `no roads in the extract` when OpenStreetMap contains zero road geometries inside the bounding box, rather than falsely claiming 0.0 km of roads were flooded."*

---

### Phase 2: Uncertainty & Metric Examination (0:45 – 1:30)

#### Step 4: Metric Examination on `Bolivia_195474`
*(Note: chips selected by IoU after seeing results, not typical cases)*
- Switch dropdown to **`Bolivia_195474`**.
- Detailed chip metrics from `results/per_chip_bolivia.csv` and `web/data/Bolivia_195474/meta.json`:
  - **Flood fraction of valid pixels:** **$0.59\%$** of valid pixels ($1,551$ flooded pixels out of $261,335$ valid pixels; ignore fraction: $0.31\%$; ground-truth flooded area: $0.15\text{ km}^2$). Status in CSV: `IoU not meaningful (<1% flood: 0.59%)`.
  - **Ensemble performance:** The ensembles find 3 (U-Net) and 7 (DeepLabV3+) of 1,551 flood pixels, i.e. recall below 0.5%, with about 360 and 232 false positives; per-chip global Otsu also scores low. No causal explanation.
  - **Exact canonical counts:**
    - **U-Net 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$0.16\%$**, Precision = **$0.83\%$**, Recall = **$0.19\%$**, $\text{TP} = 3$, $\text{FP} = 360$, $\text{FN} = 1,548$, $\text{TN} = 259,424$.
    - **DeepLabV3+ 3-seed ensemble ($p \ge 0.50$):** Flood IoU = **$0.39\%$**, Precision = **$2.93\%$**, Recall = **$0.45\%$**, $\text{TP} = 7$, $\text{FP} = 232$, $\text{FN} = 1,544$, $\text{TN} = 259,552$.
    - *(Single-seed means from `results/per_chip_bolivia.csv`: U-Net mean IoU $0.21\%$, DeepLabV3+ mean IoU $0.63\%$, Global Otsu Mean IoU $3.65\%$).*

#### Step 5: Uncertainty & Error Decomposition Layers
- Turn OFF Prediction and turn ON:
  1. **Predictive Uncertainty (Ensemble Entropy):**
     - Open the entropy map and observe its behavior: the map shows near-zero entropy across almost all valid dry land (where all ensemble members confidently agree on dry land), with elevated entropy (plasma colormap) appearing only in localized, narrow linear fringes along subtle radar backscatter boundaries.
  2. **Error Analysis (FP / FN Overlays):**
     - Orange flags false negatives ($\text{FN} = 1,548\text{ pixels}$ for U-Net ensemble, where the 0.59% diffuse water was missed).
     - Red flags false positives ($\text{FP} = 360\text{ pixels}$ for U-Net ensemble), showing they are small isolated speckles rather than widespread false alarms.
- Takeaway for Committee: *"Rather than presenting a 'cherry-picked' demo, we provide decision-makers with spatial uncertainty quantification so unconfident predictions can be flagged for optical cross-verification or human review."*

---

### Phase 3: Operational Disaster Exposure Assessment (1:30 – 2:25)

#### Step 6: Transition to Official Test Split (Sri-Lanka Event)
- In the dropdown, scroll to the **TEST Split (Sen1Floods11 Official)** group and select **`Sri-Lanka_713926`**.
- Notice the banner dynamically switches to blue:
  > **SPLIT CONTEXT: TEST split (not used in training)**  
  > *Sen1Floods11 Official Test Split (Held-out chips from the global training distribution)*.

#### Step 7: Chip `Sri-Lanka_713926` (Highest Population Exposure Scene)
- Point to the metrics: U-Net IoU **$92.25\%$**, DeepLabV3+ IoU **$94.14\%$** (Flood fraction: $30.72\%$).
- Review the Disaster Exposure Card:
  - **Observed (Ground Truth):** $7.92\text{ km}^2$ flooded, **$542.2\text{ people exposed}$**, $0.24\text{ km}$ roads flooded.
  - **Predicted Expected ($p > 0.50$):** $7.73\text{ km}^2$ flooded, **$758.7\text{ people exposed}$**, $0.26\text{ km}$ roads flooded.
  - **Fixed Bracket Range:**
    - Conservative ($p > 0.70$): $5.32\text{ km}^2$ flood $\implies 399.5$ people exposed.
    - Worst-case ($p > 0.30$): $8.47\text{ km}^2$ flood $\implies 1,150.0$ people exposed.
- Explain: *"This bracket range (400 to 1,150 people) provides humanitarian relief coordinators with clear sensitivity bounds: 400 people are identified under conservative thresholding, while contingency planning can account for up to 1,150 under worst-case thresholding."*

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
  - Highlight exact quantitative milestones taken directly from `results/metrics_summary_table.csv` and `results/uncertainty_results.json`:
    - **Global Otsu Baseline (Mean variant, Train-Fitted):**
      - Test Flood IoU: **$49.54\%$** (F1 = $66.26\%$)
      - Bolivia Flood IoU: **$64.81\%$** (F1 = $78.65\%$)
    - **U-Net ResNet-34 (Dual-Pol):**
      - Test Flood IoU (single-seed mean $\pm$ std over seeds): **$63.81 \pm 0.15\%$** (F1 = $77.91 \pm 0.11\%$)
      - Bolivia Flood IoU (single-seed mean $\pm$ std over seeds): **$60.30 \pm 7.59\%$** (F1 = $75.04 \pm 6.08\%$)
      - Exact single-seed difference: Bolivia is **$3.51$ percentage points lower** than test ($63.81\% - 60.30\%$), which is within the $\pm 7.59\%$ Bolivia seed spread.
      - 3-Seed Ensemble Test Flood IoU: **$65.33\%$** (F1 = $79.03\%$)
      - 3-Seed Ensemble Bolivia Flood IoU: **$60.91\%$** (F1 = $75.71\%$)
      - Exact ensemble difference: Bolivia is **$4.42$ percentage points lower** than test ($65.33\% - 60.91\%$).
    - **DeepLabV3+ ResNet-34 (Dual-Pol):**
      - Test Flood IoU (single-seed mean $\pm$ std over seeds): **$64.27 \pm 0.52\%$** (F1 = $78.25 \pm 0.39\%$)
      - Bolivia Flood IoU (single-seed mean $\pm$ std over seeds): **$61.89 \pm 5.45\%$** (F1 = $76.37 \pm 4.18\%$)
      - Exact single-seed difference: Bolivia is **$2.38$ percentage points lower** than test ($64.27\% - 61.89\%$), which is within the $\pm 5.45\%$ Bolivia seed spread.
      - 3-Seed Ensemble Test Flood IoU: **$66.43\%$** (F1 = $79.83\%$)
      - 3-Seed Ensemble Bolivia Flood IoU: **$63.32\%$** (F1 = $77.54\%$)
      - Exact ensemble difference: Bolivia is **$3.11$ percentage points lower** than test ($66.43\% - 63.32\%$).
    - **Held-Out Event Evaluation:**
      - 15 chips from one event, so we cannot conclude whether there is a regional drop.

#### Step 11: Multi-Country Exposure Benchmark & Figure Gallery
- Scroll down to the **Multi-Country Disaster Exposure Benchmark** table:
  - Compare Bolivia (15 chips: $43.86\text{ km}^2$ ground-truth flood, $19.2$ people exposed, $1.66\text{ km}$ flooded roads from single chip `Bolivia_290290` while 9/15 chips have no roads in extract) vs Sri Lanka (9 chips: $37.40\text{ km}^2$ ground-truth flood, $1,648.9$ people exposed, $4.40\text{ km}$ flooded roads across all 9 chips).
  - **Important Methodological Caution:** WorldPop products and years differ between the two events; do not compare people counts across events.
- Highlight the embedded publication figures:
  - **IoU Bars:** Visual confirmation of DeepLabV3+ and U-Net against classical baselines.
  - **Selective Classification Rejection Curves:** IoU is computed strictly on the retained pixels after discarding the fraction $k$ of pixels with the highest predictive entropy. From `results/uncertainty_results.json`, the fraction of pixels removed at each step is $k \in \{0\%, 5\%, 10\%, 15\%, 20\%, 25\%, 30\%, 35\%, 40\%, 45\%, 50\%\}$. For U-Net on Test, retained-pixel Flood IoU rises from $65.33\%$ ($k=0\%$, 0% removed) to $72.64\%$ ($k=5\%$, 5% removed), $76.96\%$ ($k=10\%$, 10% removed), $80.02\%$ ($k=25\%$, 25% removed), and $82.98\%$ ($k=50\%$, 50% removed). For DeepLabV3+ on Test, retained-pixel IoU rises from $66.43\%$ at $k=0\%$ to $76.65\%$ at $k=10\%$ ($10\%$ removed).
  - **Reliability Diagrams:** Demonstrates calibration of ensemble probabilities.

---

## 3. Defense & Committee Question Preparation

### Q1: "What does the performance on `Bolivia_195474` indicate?"
> **Answer:**  
> Flood fraction 0.59% of valid pixels, IoU not meaningful; the ensembles find 3 (U-Net) and 7 (DeepLabV3+) of 1,551 flood pixels, i.e. recall below 0.5%, with about 360 and 232 false positives; per-chip global Otsu also scores low. No causal explanation. Opening the predictive entropy map shows low entropy across almost all valid dry land, with elevated uncertainty confined to narrow fringes along subtle radar backscatter boundaries.

### Q2: "Why do some Bolivia chips display 'no roads in the extract' while others show 0.0 km?"
> **Answer:**  
> In geospatial disaster informatics, conflating "unmapped roads" with "zero flood impact" is a dangerous failure mode. When a chip bounding box contains zero road segments in the OpenStreetMap extract (which is the case for 9 of the 15 Bolivia chips), our system explicitly reports `"no roads in the extract"`. When roads are present (such as in `Bolivia_195474` with $18.53\text{ km}$ of mapped roads) but none intersect the flood polygon, the system reports `0.00 km` with status `"roads mapped but not flooded"`. Furthermore, all $1.66\text{ km}$ of ground-truth flooded roads in Bolivia originate from a single chip (`Bolivia_290290`).

### Q3: "What do the thresholds 0.70, 0.50, and 0.30 signify? Are they tuned?"
> **Answer:**  
> Thresholds fixed in advance as a sensitivity range; the labels are our naming, not an operational standard. They are **not** tuned on test or held-out data:
> - $p > 0.50$ is labeled "expected".
> - $p > 0.70$ is labeled "conservative".
> - $p > 0.30$ is labeled "worst-case".
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
| [`02_viewer_bolivia_roads_null.png`](file:///d:/Mini_Project/docs/screenshots/02_viewer_bolivia_roads_null.png) | `Bolivia_103757` with null road status | Correctly renders `"no roads in the extract"` (not 0 km) |
| [`03_viewer_sri_lanka_largest_pop.png`](file:///d:/Mini_Project/docs/screenshots/03_viewer_sri_lanka_largest_pop.png) | `Sri-Lanka_713926` maximum exposure scene | Area: $7.92\text{ km}^2$, Pop: $542.2$, Road: $0.24\text{ km}$ (All match JSON) |
| [`04_viewer_sri_lanka_largest_area.png`](file:///d:/Mini_Project/docs/screenshots/04_viewer_sri_lanka_largest_area.png) | `Sri-Lanka_534068` largest surface water scene | $25.30\text{ km}^2$ inundated surface |
| [`05_viewer_entropy_uncertainty.png`](file:///d:/Mini_Project/docs/screenshots/05_viewer_entropy_uncertainty.png) | Ensemble predictive entropy overlay | High entropy along water-land boundaries |
| [`06_viewer_error_analysis.png`](file:///d:/Mini_Project/docs/screenshots/06_viewer_error_analysis.png) | False positive (red) and False negative (orange) | Decomposition of segmentation errors |
| [`07_viewer_deeplab_prediction.png`](file:///d:/Mini_Project/docs/screenshots/07_viewer_deeplab_prediction.png) | DeepLabV3+ ResNet-34 ensemble prediction | Sharp boundary delineation via atrous convolution |
| [`08_results_master_table.png`](file:///d:/Mini_Project/docs/screenshots/08_results_master_table.png) | Master evaluation table with filter controls | 24 benchmark rows rendered across 4 splits |
| [`09_results_exposure_benchmark.png`](file:///d:/Mini_Project/docs/screenshots/09_results_exposure_benchmark.png) | Multi-country exposure benchmark summary | Bolivia vs Sri Lanka exposure comparison table |
