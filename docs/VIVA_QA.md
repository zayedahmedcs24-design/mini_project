# VIVA EXAMINATION Q&A STUDY GUIDE: REVIEW-3

**Project:** Automated Flood Inundation Mapping and Geospatial Exposure Analysis via Sentinel-1 SAR  
**Course:** UG 5th-Semester Mini-Project (Academic Year 2026–2027)  
**Milestone:** Review-3 (Complete Implementation, Verification, and Web Demonstration)  
**Nature:** 25 concise, high-impact answers strictly grounded in repository code, raw metrics files, and verified empirical results.

---

### Q1: Why is overall pixel accuracy an invalid evaluation metric for flood extent mapping?
**Answer:** Because of severe class imbalance in satellite imagery. Across the 446 hand-labeled chips in Sen1Floods11, flood pixels account for only $4.87\%$ of all valid pixels (valid split: $8.40\%$, test split: $4.94\%$, Bolivia split: $6.97\%$). A trivial classifier predicting dry land everywhere achieves $>95\%$ overall pixel accuracy but produces $0.00\%$ Flood IoU. Intersection over Union on the flood class (Flood IoU / Jaccard Index) and Flood F1 (Dice Coefficient) are the appropriate evaluation metrics.

---

### Q2: What is the ignore label (`-1`) in Sen1Floods11 and how does the pipeline handle it?
**Answer:** The label `-1` designates nodata areas, such as non-overlapping Sentinel-1 orbit margins or persistent cloud/shadow in the optical reference imagery. In `src/floodmap/models/losses.py`, loss functions (`MaskedBCEDiceLoss`) compute gradients strictly on pixels where `target != -1`. In `src/floodmap/metrics.py`, evaluation metrics filter out `-1` pixels. In geospatial exposure calculations (`src/floodmap/exposure/`), ignore pixels are treated as not-flood.

---

### Q3: Why is Synthetic Aperture Radar (SAR) Sentinel-1 utilized instead of optical sensors like Sentinel-2?
**Answer:** Flood disasters inherently coincide with extreme meteorological events involving heavy cloud cover, storm systems, and precipitation. Optical sensors cannot penetrate cloud decks or operate without sunlight. Sentinel-1 operates an active C-band microwave radar (~5.405 GHz, wavelength ~5.6 cm) that penetrates cloud cover and precipitation day or night. Smooth open water acts as a specular reflector, scattering radar energy away and yielding low backscatter (dark pixels), in contrast to diffuse terrestrial backscatter.

---

### Q4: What is the fair classical baseline for SAR flood mapping and why does per-chip Otsu fail?
**Answer:** The fair baseline is train-fitted Global Otsu, which fits fixed intensity thresholds once across all valid pixels in the training split (Dual-Pol Mean threshold: $-0.153809$ dB), achieving $49.54\%$ test Flood IoU. Dynamic per-chip Otsu fails because it assumes a bimodal intensity histogram; on dry chips with unimodal land distributions, it forces a threshold through dry terrain, falsely classifying large areas as flooded and resulting in only $24.73\%$ test Flood IoU.

---

### Q5: Which polarization channel yields the best segmentation performance in Otsu thresholding?
**Answer:** The Dual-Pol Mean ($\frac{\text{VV} + \text{VH}}{2}$) yields the best performance across all splits:
- Valid Split: VV $31.47\%$, VH $42.62\%$, Dual-Pol Mean **$44.58\%$** Flood IoU.
- Test Split: VV $33.88\%$, VH $48.12\%$, Dual-Pol Mean **$49.54\%$** Flood IoU.
- Bolivia Split: VV $56.38\%$, VH $62.13\%$, Dual-Pol Mean **$64.81\%$** Flood IoU.  
*(Source: `results/otsu_baseline_results.json` and `results/metrics_summary_table.csv`)*

---

### Q6: What deep learning architectures were evaluated and what are their test benchmark scores?
**Answer:** Two architectures with ResNet-34 backbones adapted for dual-polarization inputs were evaluated over 3 random seeds (0, 1, 2):
- **U-Net ResNet-34:** Single-seed mean of $63.81 \pm 0.15\%$ Flood IoU ($77.91 \pm 0.11\%$ F1); 3-seed ensemble achieves **$65.33\%$** Flood IoU ($79.03\%$ F1).
- **DeepLabV3+ ResNet-34:** Single-seed mean of $64.27 \pm 0.52\%$ Flood IoU ($78.25 \pm 0.39\%$ F1); 3-seed ensemble achieves **$66.43\%$** Flood IoU ($79.83\%$ F1).  
*(Source: `results/metrics_summary_table.csv` and `results/uncertainty_results.json`)*

---

### Q7: How does probability averaging across ensemble seeds affect segmentation metrics?
**Answer:** Ensembling 3 model seeds via mean predicted probability consistently improves Flood IoU over single-seed means:
- Valid Split: U-Net increases from $60.60\%$ to $62.31\%$ ($+1.71$ pp); DeepLabV3+ increases from $61.62\%$ to $63.18\%$ ($+1.56$ pp).
- Test Split: U-Net increases from $63.81\%$ to $65.33\%$ ($+1.52$ pp); DeepLabV3+ increases from $64.27\%$ to $66.43\%$ ($+2.16$ pp).
- Bolivia Split: U-Net increases from $60.30\%$ to $60.91\%$ ($+0.61$ pp); DeepLabV3+ increases from $61.89\%$ to $63.32\%$ ($+1.43$ pp).  
*(Source: `results/metrics_summary_table.csv` and `results/uncertainty_results.json`)*

---

### Q8: Why was Monte Carlo (MC) Dropout skipped in the uncertainty pipeline?
**Answer:** Our diagnostic verification (`check_model_has_active_dropout` in `src/floodmap/uncertainty/ensemble.py`) checked forward passes with BatchNorm frozen. The difference across stochastic passes was exactly $0.000000$ because standard `smp.Unet` and `smp.DeepLabV3Plus` architectures lack active dropout layers in their evaluation mode. Rather than reporting invalid stochastic sampling, we implemented 3-seed Deep Ensembles as our epistemic uncertainty mechanism.

---

### Q9: How is temperature scaling calibrated and what are the fitted parameters?
**Answer:** Post-hoc temperature scaling optimizes a single temperature parameter $T^*$ on validation set log-loss to minimize Expected Calibration Error (ECE) without altering classification accuracy:
- Optimal temperature: $T^* = 0.7289$ (U-Net) and $T^* = 1.2720$ (DeepLabV3+).
- Valid ECE: U-Net decreases from $4.44\%$ to $1.73\%$; DeepLabV3+ decreases from $1.55\%$ to $0.71\%$.
- Test ECE: U-Net decreases from $4.40\%$ to $1.67\%$; DeepLabV3+ decreases from $1.86\%$ to $0.82\%$.
- Bolivia ECE: U-Net decreases from $4.74\%$ to $3.03\%$; DeepLabV3+ decreases from $3.15\%$ to $2.71\%$.  
*(Source: `results/uncertainty_results.json`)*

---

### Q10: What does the predictive entropy error detection analysis demonstrate?
**Answer:** Evaluated across a 500,000-pixel sample (seed 42), predictive entropy discriminates misclassified pixels from correct predictions:
- Valid Split: U-Net AUROC $0.8686$ (AUPRC $0.3186$ vs $0.0504$ baseline); DeepLabV3+ AUROC $0.9204$ (AUPRC $0.3524$ vs $0.0487$ baseline).
- Test Split: U-Net AUROC $0.8140$ (AUPRC $0.2934$ vs $0.0500$ baseline); DeepLabV3+ AUROC $0.8967$ (AUPRC $0.3177$ vs $0.0494$ baseline).
- Bolivia Split: U-Net AUROC $0.8643$ (AUPRC $0.3529$ vs $0.0722$ baseline); DeepLabV3+ AUROC $0.8941$ (AUPRC $0.3601$ vs $0.0676$ baseline).  
*(Source: `results/uncertainty_results.json`)*

---

### Q11: Where do model uncertainty and segmentation errors spatially concentrate?
**Answer:** Spatial proximity analysis confirms that high predictive entropy concentrates along water-land boundaries. Defining boundary proximity as within 3 pixels of a true flood edge:
- Top-10% most uncertain pixels concentrate near boundaries at $3.95\text{x}$ (Valid), $3.65\text{x}$ (Test), and $3.86\text{x}$ (Bolivia) for U-Net.
- For DeepLabV3+, concentration is $3.56\text{x}$ (Valid), $4.20\text{x}$ (Test), and $3.76\text{x}$ (Bolivia).  
On the test split, $38.37\%$ (U-Net) and $41.01\%$ (DeepLabV3+) of all uncertain pixels fall within 3 pixels of a flood edge.  
*(Source: `results/uncertainty_results.json`)*

---

### Q12: How do accuracy-rejection curves operate on the test split?
**Answer:** In an accuracy-rejection curve, pixels are discarded in descending order of predictive entropy, and Flood IoU is recomputed on the retained pixels:
- U-Net: $k=0\%$ removed: $65.33\%$ IoU $\to k=5\%$ removed: $72.64\% \to k=10\%$ removed: $76.96\% \to k=20\%$ removed: $79.37\% \to k=30\%$ removed: $80.59\% \to k=50\%$ removed: $82.98\%$.
- DeepLabV3+: $k=0\%$ removed: $66.43\%$ IoU $\to k=5\%$ removed: $73.24\% \to k=10\%$ removed: $76.65\% \to k=20\%$ removed: $69.26\% \to k=30\%$ removed: $21.05\% \to k=50\%$ removed: $4.00\%$.  
For DeepLabV3+, rejecting above $15\%$ of pixels rapidly depletes true flood boundary pixels, lowering IoU.  
*(Source: `results/uncertainty_results.json`)*

---

### Q13: What were the quantitative findings on the held-out Bolivia split?
**Answer:** Bolivia comprises 15 chips from a single February 2018 event in Beni Department:
- Single-seed means are $3.51$ pp lower than test for U-Net ($60.30\%$ vs $63.81\%$) and $2.38$ pp lower for DeepLabV3+ ($61.89\%$ vs $64.27\%$).
- 3-seed ensemble scores are $4.42$ pp lower than test for U-Net ($60.91\%$ vs $65.33\%$) and $3.11$ pp lower for DeepLabV3+ ($63.32\%$ vs $66.43\%$).
- Seed-to-seed standard deviation on Bolivia is large: $\pm 7.59\%$ for U-Net and $\pm 5.45\%$ for DeepLabV3+ (compared to $\pm 0.15\%$ and $\pm 0.52\%$ on test). Ensemble scores fall within the single-seed range.
- Per-chip IoU ranges from $0.00\%$ (`Bolivia_76104`, `Bolivia_233925`) to $84.89\%$ (`Bolivia_129334`). Because all 15 chips come from one event, we do not draw causal conclusions.  
*(Source: `results/metrics_summary_table.csv`, `results/uncertainty_results.json`, `results/per_chip_bolivia.csv`)*

---

### Q14: Why is Flood IoU not meaningful for chip `Bolivia_195474`?
**Answer:** Chip `Bolivia_195474` contains $261,335$ valid pixels, of which only $1,551$ are ground-truth flood ($0.59\%$ flood fraction).
- U-Net ensemble detects 3 True Positives and 360 False Positives (IoU $0.16\%$, recall $0.19\%$).
- DeepLabV3+ ensemble detects 7 True Positives and 232 False Positives (IoU $0.39\%$, recall $0.45\%$).  
When the ground truth flood count is extremely small, a modest count of false-positive pixels heavily inflates the denominator ($\text{TP} + \text{FP} + \text{FN}$), causing IoU to approach zero despite high overall pixel agreement.  
*(Source: `web/data/Bolivia_195474/meta.json` and `results/per_chip_bolivia.csv`)*

---

### Q15: Why was Sri Lanka selected as a second disaster event for exposure analysis?
**Answer:** Sri Lanka chips belong to the official unseen TEST split of Sen1Floods11. Evaluating Sri Lanka alongside Bolivia provides a contrast between two distinct geographical settings:
- Bolivia: Rural Amazonian floodplain, 15 chips, footprint $379.7\text{ km}^2$, population $262.6$, implied density $0.7\text{ ppl/km}^2$.
- Sri Lanka: Densely populated island setting, 9 chips, footprint $232.5\text{ km}^2$, population $65,032.7$, implied density $279.7\text{ ppl/km}^2$.  
*(Source: `results/exposure_summary.md`)*

---

### Q16: What is the WorldPop year mismatch between events and why must population counts not be compared across events?
**Answer:** The Bolivia analysis uses `bol_ppp_2018.tif` (2018 unconstrained count) matching the 2018 flood event year. Sri Lanka uses `lka_ppp_2020_constrained.tif` (2020 building-constrained count) because the local event year was not recorded in dataset files. Because the WorldPop products (unconstrained vs building-constrained) and reference years (2018 vs 2020) differ, population exposure counts must not be compared directly between events.  
*(Source: `results/exposure_summary.md`)*

---

### Q17: What was the 27 November web-export bug, how was it detected, and how was it resolved?
**Answer:**
- *Bug:* `scripts/export_web_layers.py` implemented independent preprocessing without clipping to $[-50, 1]$ dB and min-max normalization, feeding raw decibel values into inference and producing false-positive saturation on 29 of 30 chips.
- *Detection:* A consistency check script compared predicted pixel counts in exported PNG overlays against canonical checkpoint metrics.
- *Fix:* Refactored `scripts/export_web_layers.py` to import and reuse `FloodDataset.__getitem__` directly in evaluation mode, guaranteeing identical preprocessing.
- *Regression Test:* Created `tests/test_web_export.py`, asserting pixel-for-pixel mask equality (`p >= 0.50`) and metric equality across multiple test chips.

---

### Q18: What do the three exposure probability thresholds (`0.70`, `0.50`, `0.30`) represent?
**Answer:** The thresholds define an a priori sensitivity range for operational planning:
- **Conservative ($p > 0.70$):** High-confidence inundation core.
- **Expected ($p > 0.50$):** Standard model decision threshold.
- **Worst-Case ($p > 0.30$):** Inundation boundary including marginal and saturated soils.  
These thresholds represent an operational sensitivity range, not statistical confidence intervals.  
*(Source: `results/exposure_summary.md`)*

---

### Q19: What is the critical distinction between exposure totals and pixel-level segmentation?
**Answer:** Exposure totals compare aggregated area and population exposed, not the exact location of every flooded pixel. For example, on chip `Bolivia_290290`, predicted flooded area is $1.65\text{ km}^2$ compared to ground truth $1.91\text{ km}^2$ (a $13.6\%$ aggregate difference), yet pixel Flood IoU is $52.95\%$. Aggregated exposure metrics can align closely even when spatial overlap is moderate.  
*(Source: `results/exposure_summary.md` and `web/data/Bolivia_290290/meta.json`)*

---

### Q20: What road data source was used and how are missing roads handled in results?
**Answer:** Road infrastructure was extracted from official local Geofabrik OpenStreetMap shapefiles (`data/osm/<country>/gis_osm_roads_free_1.shp`, extract dated 2026). When a chip contains no road features in the extract, the status is explicitly recorded as `null` ("no roads in the extract"), strictly distinguishing absence of mapped data from $0.00\text{ km}$ ("roads mapped but not flooded"). In Bolivia, 9 of 15 chips contain no road features in the extract.  
*(Source: `results/exposure_summary.md` and `results/exposure_bolivia.json`)*

---

### Q21: Where do flooded roads occur in the Bolivia evaluation?
**Answer:** Across the 15 Bolivia chips ($379.7\text{ km}^2$), the ground-truth flooded road total is $1.66\text{ km}$, and the entire $1.66\text{ km}$ occurs on a single chip: `Bolivia_290290` (which has $4.59\text{ km}$ of total mapped roads). The remaining 14 chips contain either unflooded roads (5 chips) or no mapped roads in the extract (9 chips).  
*(Source: `results/exposure_bolivia.json`)*

---

### Q22: How is population exposure quantitatively calculated?
**Answer:** Using fractional aggregation:
1. Binary flood masks at 10-meter resolution are projected into the local metric UTM CRS.
2. The 10m flood mask is overlaid onto the 100m WorldPop population count raster.
3. For each 100m grid cell, the fraction of flooded 10m pixels is multiplied by the cell's population count under the standard assumption of uniform population distribution within each cell.
4. Cell-level exposed populations are summed across the chip.  
*(Source: `src/floodmap/exposure/worldpop.py`)*

---

### Q23: How is road inundation quantitatively calculated?
**Answer:** Using planar metric vector intersection:
1. Contiguous flooded pixel regions from the binary mask are converted into polygon geometries in local metric UTM projection.
2. Road linestrings from the local Geofabrik shapefile are clipped to the chip bounding box and reprojected to UTM.
3. Geometric intersection (`geopandas.overlay(roads, flood_polygons, how='intersection')`) isolates flooded road segments.
4. Lengths are summed in meters and converted to kilometers.  
*(Source: `src/floodmap/exposure/roads.py`)*

---

### Q24: How is the web demonstration structured and executed?
**Answer:** The web demo is an offline, static browser application located in `web/` using vendored Leaflet JS/CSS (`web/vendor/leaflet/`) without external CDN dependencies.
- To run: start a local HTTP server using `python -m http.server 8000 --directory web` and navigate to `http://localhost:8000/index.html`.
- Double-clicking `index.html` fails because modern browsers enforce CORS restrictions on `file://` URIs when fetching local JSON and PNG assets.  
*(Source: `README.md` and `docs/DEMO_WEB.md`)*

---

### Q25: What is the current test suite status and what are the primary avenues for future work?
**Answer:**
- **Current Status:** All 44 automated unit and integration tests in `pytest tests -q` pass in under 60 seconds, covering the data loader, losses, metrics, calibration, ensembles, exposure calculation, and web export consistency.
- **Future Work:**
  1. Integrating transformer backbones (SegFormer MiT-B0) to assess multi-scale self-attention.
  2. Training models with persistent active dropout layers to enable Monte Carlo dropout inference.
  3. Expanding multi-event ingestion beyond Bolivia and Sri Lanka and building automated ingestion pipelines for Copernicus Sentinel-1 data.
