# Geospatial Disaster Exposure Summary Report

**Generated:** 2026-11-22  
**Pipeline:** Multi-Seed Ensemble Inference (U-Net & DeepLabV3+, Seeds 0, 1, 2)  
**Risk Tiers:** Conservative ($p > 0.70$), Expected ($p > 0.50$), Worst-Case ($p > 0.30$)  

---

## 1. Executive Summary Table

| Event / Country | Split Type | Chips | Total Footprint Area | Total Footprint Pop | Implied Density | GT Flooded Area | GT Exposed Pop | GT Flooded Roads |
|---|---|---|---|---|---|---|---|---|
| **Bolivia** | Sen1Floods11 Hand-Labeled Bolivia Split | 15 | 379.7 km² | 262.6 | 0.7 ppl/km² | 43.86 km² | 19.2 | 1.66 km |
| **Sri-Lanka** | Sen1Floods11 Official TEST Split (Unseen During Training) | 9 | 232.5 km² | 65,032.7 | 279.7 ppl/km² | 37.40 km² | 1,648.9 | 4.40 km |

---

## 2. Multi-Risk Exposure Comparison by Model and Operational Tier

### Bolivia (Sen1Floods11 Hand-Labeled Bolivia Split)

- **Event Date/Year:** February 2018 (Beni Department Flood Event) (Year: 2018)
- **WorldPop Dataset:** `data/worldpop/bol_ppp_2018.tif` (Year: 2018)
- **Road Network Source:** `data/osm/bolivia/gis_osm_roads_free_1.shp` (2026-11-22 (Geofabrik South America / Bolivia latest extract))
- **Road Coverage:** Mapped across 6 of 15 chips.

| Model Architecture | Risk Tier / Threshold | Flooded Area (km²) | Exposed Population | Flooded Roads (km) | Area Rel Error (%) | Pop Rel Error (%) |
|---|---|---|---|---|---|---|
| **Ground Truth (Observed)** | Label == 1 | 43.86 | 19.2 | 1.66 | 0.00% (Ref) | 0.00% (Ref) |
| U-Net ResNet-34 Ensemble | Conservative ($p > 0.70$) | 29.81 | 14.0 | 1.90 | -32.03% | -26.95% |
| U-Net ResNet-34 Ensemble | Expected ($p > 0.50$) | 39.67 | 19.1 | 2.12 | -9.56% | -0.27% |
| U-Net ResNet-34 Ensemble | Worst-Case ($p > 0.30$) | 48.37 | 24.3 | 2.44 | +10.27% | +26.73% |
| DeepLabV3+ ResNet-34 Ensemble | Conservative ($p > 0.70$) | 30.51 | 12.7 | 1.42 | -30.45% | -33.62% |
| DeepLabV3+ ResNet-34 Ensemble | Expected ($p > 0.50$) | 41.23 | 19.6 | 2.34 | -6.00% | +2.13% |
| DeepLabV3+ ResNet-34 Ensemble | Worst-Case ($p > 0.30$) | 48.96 | 24.3 | 2.49 | +11.62% | +26.56% |

### Sri-Lanka (Sen1Floods11 Official TEST Split (Unseen During Training))

- **Event Date/Year:** Event year not verified locally in dataset files (Year: Event year not verified locally)
- **WorldPop Dataset:** `data/worldpop/lka_ppp_2020_constrained.tif` (Year: 2020)
- **Road Network Source:** `data/osm/sri-lanka/gis_osm_roads_free_1.shp` (2026-11-22 (Geofabrik Asia / Sri-Lanka latest extract))
- **Road Coverage:** Mapped across 9 of 9 chips.

| Model Architecture | Risk Tier / Threshold | Flooded Area (km²) | Exposed Population | Flooded Roads (km) | Area Rel Error (%) | Pop Rel Error (%) |
|---|---|---|---|---|---|---|
| **Ground Truth (Observed)** | Label == 1 | 37.40 | 1,648.9 | 4.40 | 0.00% (Ref) | 0.00% (Ref) |
| U-Net ResNet-34 Ensemble | Conservative ($p > 0.70$) | 31.44 | 586.1 | 1.70 | -15.95% | -64.45% |
| U-Net ResNet-34 Ensemble | Expected ($p > 0.50$) | 35.98 | 1,240.7 | 5.79 | -3.79% | -24.76% |
| U-Net ResNet-34 Ensemble | Worst-Case ($p > 0.30$) | 39.50 | 2,069.5 | 11.39 | +5.60% | +25.51% |
| DeepLabV3+ ResNet-34 Ensemble | Conservative ($p > 0.70$) | 31.63 | 642.3 | 1.62 | -15.44% | -61.04% |
| DeepLabV3+ ResNet-34 Ensemble | Expected ($p > 0.50$) | 36.71 | 1,307.3 | 5.51 | -1.86% | -20.72% |
| DeepLabV3+ ResNet-34 Ensemble | Worst-Case ($p > 0.30$) | 39.68 | 2,189.9 | 12.98 | +6.09% | +32.81% |

---

## 3. Methodological Notes & Limitations

1. **Operational Sensitivity vs Statistical Confidence:** Thresholds $p > 0.70$ (conservative), $p > 0.50$ (expected), and $p > 0.30$ (worst-case) represent disaster risk management sensitivity bounds, not Bayesian credibility or frequentist confidence intervals.
2. **Population Aggregation:** Based on the standard uniform-distribution-within-cell assumption (10m flood fraction inside each 100m WorldPop cell multiplied by cell count).
3. **Temporal Mismatch:** WorldPop population rasters (Bolivia: 2018; Sri-Lanka: 2020) reflect annualized gridded counts, while flood inundation is an acute episodic event.
4. **Road Network Completeness:** Evaluated strictly with local Geofabrik shapefile extracts. In rural Bolivia (Beni), 9 chips contain no road features in the extract (`null`), while 6 chips have mapped roads totaling 28 segments. In Sri-Lanka, road networks are densely mapped across all 9 chips in the TEST split.
5. **Sample Size:** Metrics reflect 15 chips for Bolivia and 9 unseen chips for Sri-Lanka. Performance should be interpreted within this geographic footprint.
