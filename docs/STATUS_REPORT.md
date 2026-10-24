# STATUS REPORT - Sen1Floods11 Flood Mapping Mini-Project
## Stage A / Review-2 Audit
**Audited:** 2026-10-24T19:01 IST
**Auditor:** Automated audit (read-only, no files modified)

---

## 1. REPO AND ENVIRONMENT

| Item | Status | Evidence |
|:---|:---:|:---|
| Folder tree (depth 3) | DONE | All expected dirs present: `src/floodmap/{data,models,uncertainty,exposure}`, `tests/`, `docs/slides/`, `results/checkpoints/`, `configs/`, `scripts/`, `data/raw/{S1Hand,LabelHand}`, `data/splits/` |
| Git status | NOT VERIFIED | `git` is not installed / not on PATH. `.git/` directory does **not** exist. This is not a git repo. |
| .gitignore excludes `data/` | PARTIAL | `data/raw/`, `data/worldpop/`, `data/osm/`, `data/precomputed_layers/`, `data/processed/` are listed, plus `*.tif`. But bare `data/` is **not** listed. Split CSVs could be committed (acceptable). |
| .gitignore excludes `*.pth` | DONE | `*.pth`, `*.pt`, `*.ckpt` all listed. |
| .gitignore excludes `results/checkpoints/` | **NOT DONE** | `results/checkpoints/` is **not** in `.gitignore`. Only `results/*.png`, `results/*.jpg`, `results/*.pdf`. The `*.pth` glob covers the files themselves. |
| `.agent/rules/project.md` exists | DONE | 31 lines, 5 rules: env/stack, data conventions and label masking, cross-region integrity (Bolivia held-out), CRS/spatial ops, execution safety and reproducibility. |
| pytest tests/ -v | DONE | **28 passed, 0 failed, 5 warnings** (rasterio PendingDeprecationWarning). Tests: dataset(2), exposure(5), metrics(5), models(4), otsu(3), uncertainty(5). |
| GPU confirmed | DONE | `torch.cuda.is_available() = True`, `NVIDIA GeForce RTX 4050 Laptop GPU`, `torch 2.5.1+cu121`, `CUDA 12.1` |

> [!WARNING]
> **README says "All 10 unit tests pass"** (line 54). Actual count is **28 tests**. The README is stale.

---

## 2. DATA

| Item | Status | Evidence |
|:---|:---:|:---|
| S1Hand chip count on disk | DONE | **446 files** in `data/raw/S1Hand/` |
| LabelHand chip count on disk | DONE | **446 files** in `data/raw/LabelHand/` |
| Full 446 hand-labeled set | DONE | 446 S1 + 446 Label = full dataset present |
| Split CSV counts | DONE | train: 252, valid: 89, test: 90, bolivia: 15. Sum = **446** |
| Chips on disk per split | DONE | All splits: 100% present (252/252, 89/89, 90/90, 15/15) |
| Zero overlap across splits | DONE | **True** - 446 unique IDs, pairwise intersection empty for all 6 pairs |
| Label values only {-1, 0, 1} | DONE | Verified across all 446 chips: `{-1, 0, 1}` confirmed |

---

## 3. EDA NUMBERS (recomputed from disk)

### 3.1 SAR Statistics (all 446 chips, finite pixels only)

| Band | Min | Max | Mean | NaN Count |
|:---|:---:|:---:|:---:|:---:|
| **VV** | -89.7858 | **36.8320** | -10.4345 | 2,511,705 |
| **VH** | -103.8760 | **17.3764** | -17.3080 | 2,511,705 |

- Total SAR pixels: 116,916,224 (= 446 x 512 x 512)
- NaN fraction: 2,511,705 / 116,916,224 = **2.15%**
- **Are they in dB?** The means are negative and typical for dB, but the extreme min/max values (VV min = -89.8, VV max = +36.8) are unphysical for SAR dB backscatter. These are likely NoData/corrupt pixels. The dataset clipping in `dataset.py` clips VV to [-35, 5] and VH to [-45, 0], which handles this.

> [!IMPORTANT]
> **Slide 7 SAR stats are WRONG.** Slide claims VV range `[-40.82, -0.69]`, mean `-13.01`; VH range `[-48.25, -6.99]`, mean `-19.29`. Actual raw ranges and means differ significantly. The slide numbers may come from an older subset analysis or a different computation.

### 3.2 Class Balance (per split)

| Split | Chips | Flood % | Non-flood % | Ignore % |
|:---|:---:|:---:|:---:|:---:|
| **Train** | 252 | 8.25% | 78.50% | 13.26% |
| **Valid** | 89 | 9.59% | 77.40% | 13.01% |
| **Test** | 90 | 10.88% | 76.09% | 13.04% |
| **Bolivia** | 15 | 11.57% | 61.36% | 27.07% |
| **Overall** | **446** | **9.16%** | **77.22%** | **13.63%** |

> [!WARNING]
> **The slides/README do not state the numbers 10.58% or 18.71% anywhere in tracked files.** The user-requested comparison against "flood 10.58%, ignore 18.71%" cannot be matched to any slide or README content. The closest is the test split flood percentage (10.88%). The overall computed flood is **9.16%** and ignore is **13.63%** - both differ substantially from the questioned figures.

### 3.3 Low-Flood Chips

| Split | Chips with less than 1% flood | Total |
|:---|:---:|:---:|
| Valid | 31 | 89 |
| Test | 29 | 90 |
| Bolivia | 3 | 15 |

---

## 4. METRICS CODE

| Item | Status | Evidence |
|:---|:---:|:---|
| -1 pixels excluded | DONE | `metrics.py` L58-61: `valid_mask = (y_true_int != ignore_index) & (~np.isnan(y_true))`, all counts computed on `pred_v = y_pred_bin[valid_mask]` |
| Confusion matrix summed over whole split | DONE | `MetricTracker` class (L161-221) accumulates `total_tp/fp/fn/tn` across all chips, then calls `compute_metrics_from_counts()` on the totals. Not averaged per-chip. |
| Epsilon handling | DONE | `eps=1e-7` in `compute_metrics_from_counts()` (L86). Each metric checks `denom > 0` before applying eps: e.g., `(tp / (flood_denom + eps)) if flood_denom > 0 else 0.0` |
| No plain accuracy in eval output | DONE | `eval.py` prints only Flood IoU, F1, Precision, Recall, Mean IoU. `accuracy_valid` is computed internally but **never printed** to console. However, it **IS stored in every results JSON file**. |
| Hand-check matches formulas | DONE | Tiny 3x3 example: TP=3, FP=1, FN=1, TN=2. IoU=0.60, F1=0.75, Prec=0.75, Rec=0.75 - all match to 6 decimal places. |

> [!NOTE]
> **`accuracy_valid` leaks into JSON output.** While `eval.py` does not print it, it appears in every `results/*.json` file because `compute_metrics_from_counts()` always returns it. Project rule #2 says "NEVER report plain overall accuracy." The JSON files technically report it. Low severity since it is not highlighted or used for decisions.

---

## 5. OTSU BASELINE

### 5.1 How the Threshold Is Computed

- **Per-chip Otsu**, not global. `eval_otsu.py` calls `otsu_segment_chip()` without `threshold_override`, so `skimage.filters.threshold_otsu()` runs on each chip independently.
- `GlobalOtsuBaseline` class exists in `baselines.py` (L76-121) but **is never used** by `eval_otsu.py`.
- **NaN handling:** `valid_pixels = band[np.isfinite(band)]` filters NaN/Inf before Otsu (L57).
- **No clipping of extreme dB outliers** before Otsu - the raw band values (including extreme outliers like +36 dB VV) are passed to Otsu.
- **No filtering of ignore pixels** from the SAR band - Otsu thresholds on all finite SAR pixels regardless of whether the ground truth is -1.
- **Histogram range/bins:** Uses `skimage.threshold_otsu()` defaults (256 bins, data range auto-detected).

### 5.2 Otsu Anomaly Analysis

The user's initial description claimed "VV alone and VH alone ~10% IoU, VV+VH mean ~70%." This does **not** match the actual results JSON:

| Split | VV IoU | VH IoU | Mean IoU | VV Prec | VH Prec | Mean Prec | VV Rec | VH Rec | Mean Rec |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Valid | 18.72% | 22.00% | 21.52% | 20.37% | 23.77% | 23.25% | 69.82% | 74.67% | 74.33% |
| Test | 21.87% | 23.98% | 24.73% | 23.77% | 26.24% | 27.04% | 73.22% | 73.52% | 74.32% |
| Bolivia | 31.11% | 33.34% | 34.73% | 39.68% | 41.66% | 44.20% | 59.03% | 62.54% | 61.84% |

**Finding:** All three variants (VV, VH, Mean) give **similar** IoU in the 18-25% range on valid/test. There is **no** anomaly of VV/VH at ~10% and Mean at ~70%. The user's description was incorrect or referred to an older buggy run.

### 5.3 Why Otsu IoU Is Low (~20%)

- **Predicted flood fraction** on valid: VV=37.78%, VH=34.63%, Mean=35.26%. True flood fraction: 11.03%.
- Otsu massively **over-predicts** flood (3x more than actual), causing ~5-6M false positive pixels.
- Root cause: Most chips are unimodal (dominated by dry land). Otsu tries to find a bimodal split and ends up splitting within the dry-land distribution, predicting everything below the midpoint as "water."
- Bolivia does better (31-35% IoU) because flooding is genuinely higher (15.9%) and creates a more bimodal histogram.

### 5.4 Per-Chip Otsu on Low-Flood Chips

31/89 valid chips and 29/90 test chips have less than 1% flood. Per-chip Otsu is still applied on these - it will find a threshold in a unimodal distribution and predict ~50% as flood, yielding near-zero precision and IoU. This is expected behavior for Otsu, not a bug.

---

## 6. DEEP MODELS AND TRAINING

### 6.1 Code Review

| Item | Status | Evidence |
|:---|:---:|:---|
| 2-channel input | DONE | `in_channels: 2` in config, `FloodUNet(in_channels=2)` and `FloodDeepLabV3Plus(in_channels=2)` |
| Ignore-masked BCE + Dice loss | DONE | `MaskedBCEDiceLoss` (losses.py L13-64): valid_mask excludes -1, BCE masked via element-wise multiply, Dice excludes invalid pixels |
| AdamW settings | DONE | `train.py` L257-261: `AdamW(lr=3e-4, weight_decay=0.01)` from config |
| Seed control | DONE | `seed_everything()` (L40-47): sets random, numpy, torch, cuda, cudnn deterministic seeds |
| `--smoke` flag | DONE | L175-176, L209-216: 2 epochs, batch_size=2, max_samples=8, valid=2 chips |
| Bolivia never in training | DONE | `train.py` does **not contain** the word "bolivia" at all. Only `train_split` and `valid_split` are loaded. Bolivia appears only in `eval.py`. |
| Early stopping | **NOT DONE** | Config has `early_stopping_patience: 8` but `train.py` has **no early stopping logic**. It runs all `max_epochs` epochs unconditionally. The config key is dead/unused. |

### 6.2 Checkpoints

| File | Size | Date | Origin |
|:---|:---:|:---:|:---|
| `unet_resnet34_best.pth` | 93.38 MB | 10/23 23:15 | **Smoke run** - associated `unet_resnet34_valid.json` has `evaluated_chips: 2`, `total_split_chips: 89` |
| `deeplabv3p_resnet34_best.pth` | 85.76 MB | 10/23 23:43 | **Unknown (likely smoke)** - no DeepLab eval results JSON exists. Cannot verify. |
| `unet_resnet34_seed0_best.pth` | 93.38 MB | 10/24 00:04 | **Real training run** - associated `unet_resnet34_seed0_valid.json` has `evaluated_chips: 89/89` |

> [!IMPORTANT]
> `deeplabv3p_resnet34_best.pth` exists (85.76 MB) but there are **zero DeepLab evaluation result JSON files**. This checkpoint has not been evaluated on any split.

### 6.3 Smoke Run Loss Verification

The smoke run loss values are **not logged in any results JSON file**. They are only printed to console during execution. The `train.py` smoke block (L300-306) checks `loss_decreased = history[-1]["train_loss"] < history[0]["train_loss"]` and prints the result, but does **not** save the loss history to a file.

**Verdict:** NOT VERIFIED - no saved loss history to inspect. The code structurally verifies loss decrease, but I have no logged output to confirm actual values.

### 6.4 UNet Seed0 Real Training Results

From the `unet_resnet34_seed0_*.json` files:

| Split | Chips | Flood IoU | Flood F1 | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| Valid | 89/89 | 61.98% | 76.52% | 75.27% | 77.82% |
| Test | 90/90 | 65.01% | 78.80% | 78.27% | 79.33% |
| Bolivia | 15/15 | 62.76% | 77.12% | 77.46% | 76.78% |

---

## 7. REVIEW-2 DELIVERABLES

### 7.1 Slide-by-Slide Audit

| Slide | Content | Backed by File/Result? | Issues |
|:---|:---|:---:|:---|
| **1: Title** | Team metadata, project title, guide placeholder | N/A | `[Guide Name, Designation]` is a placeholder - needs filling |
| **2: Problem Statement** | SAR advantage, interpretation challenges | N/A (conceptual) | None |
| **3: SDG Alignment** | SDG 13 / SDG 11 mapping | N/A (conceptual) | None |
| **4: Literature Matrix** | Claims "2024-2026 peer-reviewed journals" | **NO PAPERS CITED** | Slide says "Restricted strictly to 2024-2026 peer-reviewed journals" but **names zero specific papers, authors, journals, or scores**. This is a claim without evidence. Cannot verify. |
| **5: Scope Boundary** | MoSCoW prioritization | N/A (planning) | None |
| **6: Architecture** | ASCII system diagram | `docs/architecture.md` has full Mermaid version | OK |
| **7: Data Engineering** | VV/VH ranges, label integrity, class imbalance | **PARTIAL** | VV range `[-40.82, -0.69]`, mean `-13.01` and VH range `[-48.25, -6.99]`, mean `-19.29` **do not match** actual raw stats. See section 3.1. |
| **8: Otsu Baseline** | Full table of 9 Otsu results | `results/otsu_baseline_results.json` | All 36 numbers verified to match JSON exactly |
| **9: U-Net Performance** | Full table of 6 result rows | `results/unet_resnet34_seed0_*.json` | All 12 U-Net numbers match JSON exactly. "22/22 unit tests" claim is **wrong** - actual count is **28/28**. |
| **10: Roadmap** | Timeline for Stages B-E | N/A (planning) | None |

### 7.2 Literature Survey Flag

> [!CAUTION]
> **Slide 4 contains NO specific paper citations.** It makes general claims about research gaps ("Few studies systematically evaluate...") without naming a single paper, journal, or numeric score. The team must populate this with actual 2024-2026 papers before the review presentation.

### 7.3 Number Verification Summary

- **Slide 8 (Otsu):** All 36 metric numbers match `results/otsu_baseline_results.json` PASS
- **Slide 9 (U-Net):** All 12 metric numbers match `results/unet_resnet34_seed0_*.json` PASS
- **Slide 7 (SAR stats):** VV/VH min, max, mean values **DO NOT MATCH** recomputed raw stats FAIL
- **Slide 9 test count:** Claims "22/22 unit tests" - actual is **28/28** FAIL

---

## 8. SUMMARY TABLE

| # | Item | Status | Evidence |
|:---|:---|:---:|:---|
| 1.1 | Folder tree | DONE | All expected dirs present |
| 1.2 | Git status | NOT VERIFIED | No `.git/`, git not on PATH |
| 1.3 | .gitignore coverage | PARTIAL | `data/raw/`, `*.pth` covered; `results/checkpoints/` not explicitly listed |
| 1.4 | Project rules | DONE | 5 rules in `.agent/rules/project.md` |
| 1.5 | pytest | DONE | 28/28 passed, 5 warnings |
| 1.6 | GPU confirmed | DONE | RTX 4050, torch 2.5.1+cu121, CUDA 12.1 |
| 2.1 | Full 446 chips | DONE | 446 S1Hand + 446 LabelHand |
| 2.2 | Split counts and disk presence | DONE | 252+89+90+15 = 446, all on disk |
| 2.3 | Zero split overlap | DONE | True |
| 2.4 | Label values {-1,0,1} | DONE | Verified all 446 chips |
| 3.1 | SAR stats recomputed | DONE | VV: [-89.79, 36.83], mean -10.43; VH: [-103.88, 17.38], mean -17.31 |
| 3.2 | Class balance per split | DONE | Overall: 9.16% flood, 77.22% non-flood, 13.63% ignore |
| 3.3 | Slide SAR numbers match | **MISMATCH** | Slide 7 ranges and means differ from raw data |
| 4.1 | -1 excluded in metrics | DONE | Code verified + hand-check passed |
| 4.2 | Global confusion matrix | DONE | MetricTracker accumulates totals |
| 4.3 | Epsilon handling | DONE | eps=1e-7 with guard |
| 4.4 | No accuracy in eval prints | DONE | `eval.py` does not print it; JSON stores `accuracy_valid` (low concern) |
| 4.5 | Hand-check | DONE | IoU, F1, Prec, Rec all match to 6 decimals |
| 5.1 | Otsu threshold method | DONE | Per-chip Otsu, no global fitting |
| 5.2 | Otsu anomaly explained | DONE | No anomaly exists - all variants give similar ~20% IoU |
| 5.3 | Low-flood chip Otsu | DONE | 31/89 valid, 29/90 test have less than 1% flood; Otsu runs on them |
| 6.1 | 2-channel, loss, AdamW, seed | DONE | All verified in code |
| 6.2 | Bolivia not in training | DONE | grep returns nothing |
| 6.3 | Early stopping | **NOT DONE** | Config has `patience: 8` but code never implements it |
| 6.4 | Checkpoint provenance | PARTIAL | 1 real, 1 smoke, 1 unknown (DeepLab unverified) |
| 6.5 | Smoke loss decreasing | NOT VERIFIED | Loss history not saved to file |
| 7.1 | Slide numbers match JSON | DONE | Slides 8 and 9 match; Slide 7 SAR stats mismatch |
| 7.2 | Literature citations | **NOT DONE** | Zero papers cited in Slide 4 |
| 7.3 | Test count on slides | **MISMATCH** | Slide says 22, actual is 28 |

---

## DISCREPANCIES AND SUSPECTED BUGS (ordered by severity)

### HIGH SEVERITY

1. **No git repository initialized.** `.git/` does not exist and `git` is not on PATH. Version control is non-functional. No commit history, no branch management, no ability to track changes. For a review submission this is a serious gap.

2. **Early stopping is configured but never implemented.** `configs/unet_resnet34.yaml` specifies `early_stopping_patience: 8`, but `train.py` has zero early-stopping logic. It always runs all `max_epochs`. The config is misleading and training may overfit on longer runs.

3. **DeepLabV3+ checkpoint exists (85.76 MB) with NO evaluation results.** `deeplabv3p_resnet34_best.pth` was saved but has no corresponding `deeplabv3p_*.json` results. Unknown whether this came from a smoke or real run, or what its performance is.

4. **Slide 7 SAR statistics are wrong.** The claimed VV range `[-40.82, -0.69]` and VH range `[-48.25, -6.99]` with means of -13.01 / -19.29 do not match the actual raw data (VV: [-89.79, 36.83], mean -10.43; VH: [-103.88, 17.38], mean -17.31). Presenting wrong numbers at review is problematic.

### MEDIUM SEVERITY

5. **Slide 4 (Literature Survey) contains zero paper citations.** Claims "Restricted strictly to 2024-2026 peer-reviewed journals" but does not name a single paper, author, or journal. This will not survive a review viva.

6. **README test count is stale.** README says "All 10 unit tests pass" - actual is 28. Slide 9 says "22/22" - also wrong.

7. **`accuracy_valid` stored in all results JSON files.** Project rule says "NEVER report plain overall accuracy." While `eval.py` does not print it, it is serialized in every JSON output. The metric function computes and returns it unconditionally.

8. **Smoke run loss history is not persisted.** `train.py` checks loss decrease in smoke mode but only prints to console - no JSON or log file stores the epoch-by-epoch loss. Cannot retroactively verify.

### LOW SEVERITY

9. **`results/checkpoints/` not in .gitignore.** The `*.pth` glob covers checkpoint files, but the directory itself is not excluded. Not an issue if git is not initialized, but will matter once it is.

10. **Otsu baseline uses per-chip threshold but `GlobalOtsuBaseline` class is dead code.** The `GlobalOtsuBaseline` class in `baselines.py` is never called by any script. This is unused code.

11. **`get_class_balance()` in `dataset.py` reports flood % of valid pixels, but EDA script output reports flood % of total pixels.** This inconsistency could cause confusion when comparing numbers across different parts of the codebase.

---

## NEXT 5 ACTIONS

1. **Initialize git and make a clean first commit.** Run `git init`, add `.gitignore`, and commit the current working state. This is blocking for any collaborative work and submission audit trail.

2. **Fix Slide 7 SAR statistics.** Either recompute after clipping (e.g., report the 1st-99th percentile range instead of raw min/max) or clearly label them as "post-clipping" values. Currently they are presented as raw empirical values but are neither.

3. **Evaluate the DeepLabV3+ checkpoint.** Run `python eval.py --model deeplabv3p_resnet34 --config configs/deeplabv3p_resnet34.yaml --checkpoint results/checkpoints/deeplabv3p_resnet34_best.pth --split valid` (and test/bolivia) to generate evaluation JSONs. If it was a smoke run, retrain with full data.

4. **Implement early stopping in `train.py`** or remove `early_stopping_patience` from the config. Currently the config promises something the code does not deliver.

5. **Populate Slide 4 with real paper citations.** Identify 5-8 actual 2024-2026 peer-reviewed papers on SAR flood mapping, uncertainty quantification, and cross-region evaluation. Include authors, journal/conference, year, and key results.
