# STATUS REPORT 2: COMPREHENSIVE AUDIT & REVIEW-2 READINESS

**Audit Date:** 30 October 2026 (Local Time: 19:30 IST)  
**Milestone:** Review-2 (Design + Basic Implementation)  
**Target Review Date:** 31 October 2026  
**Auditor Mode:** Read-Only Verification Engine (Strict Empirical Validation)  
**Python Environment:** `venv\Scripts\python.exe` (Python 3.11.9)  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121)  

---

## SECTION A: MASTER VERIFICATION MATRIX

Status Legend:
- **DONE**: Fully implemented, empirical evidence verified on disk / in memory.
- **PARTIAL**: Partially completed or operational, but contains omissions, outdated metrics, or uncommitted files.
- **NOT DONE**: Missing entirely.
- **NOT VERIFIED**: Cannot be confirmed with available empirical evidence.

| ID | Item / Subsystem | Status | Empirical Evidence & Verification Source |
| :--- | :--- | :---: | :--- |
| **A1** | **Git Working Tree Hygiene** | **PARTIAL** | `git status --short`: 1 modified file (`src/floodmap/uncertainty/__init__.py`), 8 untracked items (`results/figures/{example_chips,rejection_curves,reliability_diagrams}.png`, `results/uncertainty/`, `results/uncertainty_results.json`, `scripts/evaluate_uncertainty_pipeline.py`, `src/floodmap/uncertainty/ensemble.py`, `tests/test_uncertainty_metrics.py`). Zero committed changes from latest uncertainty run. |
| **A2** | **Forbidden Tracked Artifacts** | **DONE** | `git ls-files \| grep`: 0 `.pth`, 0 `.npy`, 0 `data/raw/`, 0 `results/uncertainty/` files tracked. (Note: `results/uncertainty_bolivia.json` is tracked, 272 bytes smoke record from 24 Oct). `.git` folder size: 30.21 MB (31,673,338 bytes), 333 files. `results/uncertainty/` on disk: 1,164.19 MB (1,220,741,120 bytes), 1,552 `.npy` files. |
| **A3** | **Git Author Accountability** | **PARTIAL** | All 5 commits in `git log` are authored by `zayedakhan2006-dev <zayedakhan2006@gmail.com>`. No commits exist from Members A, B, C, D despite slide attributions. |
| **A4** | **Test Suite Execution** | **DONE** | `venv\Scripts\python.exe -m pytest tests -q`: **40 passed, 5 warnings in 37.69s**. Exactly 40 tests collected across 7 files; 0 failed. |
| **A5** | **GPU & CUDA Environment** | **DONE** | `torch.__version__ = 2.5.1+cu121`, `torch.cuda.is_available() = True`, `torch.cuda.get_device_name(0) = 'NVIDIA GeForce RTX 4050 Laptop GPU'`. |
| **A6** | **Dataset & Split Counts** | **DONE** | Exactly 446 `.tif` chips on disk in `data/raw/S1Hand` and 446 in `data/raw/LabelHand`. Split counts verified from headerless CSVs: Train=252, Valid=89, Test=90, Bolivia=15. Total = 446 chips (zero missing, zero overlap). |
| **A7** | **Trained Model Checkpoints** | **DONE** | Exactly 6 best checkpoints in `results/checkpoints/`: 3 U-Net (`unet_resnet34_seed{0,1,2}_best.pth`, 97,914,112 bytes each, 93.38 MB) and 3 DeepLabV3+ (`deeplabv3p_resnet34_seed{0,1,2}_best.pth`, 89,927,495 bytes each, 85.76 MB). Modified dates: 2026-10-24 22:04 to 22:32. |
| **A8** | **Metrics Summary Consistency** | **DONE** | `results/metrics_summary_table.csv` perfectly matches recomputed mean ± std (ddof=1) from per-seed JSON files across all splits (e.g., U-Net Valid `60.60 ± 0.26%`, DeepLabV3+ Test `64.27 ± 0.52%`, Bolivia U-Net `60.30 ± 7.59%`). |
| **A9** | **Training History & Epoch Caps** | **DONE** | Extracted from `results/*_history.json`: All 6 runs terminated via early stopping (patience=8) on validation flood IoU. None hit the 60-epoch cap: U-Net (s0: 16 ep, best 8; s1: 12 ep, best 4; s2: 17 ep, best 9); DeepLabV3+ (s0: 21 ep, best 13; s1: 40 ep, best 32; s2: 17 ep, best 9). |
| **A10** | **Uncertainty Pipeline Artifacts** | **DONE** | `results/uncertainty_results.json` exists (1,291 lines, generated 28 Oct 23:39). Contains ensemble metrics, calibration (10 bins), AUROC/AUPRC (500k sample size), 11-point accuracy-rejection curves, and boundary-proximity analysis for both models across all splits. |
| **A11** | **MC Dropout Architecture Check** | **DONE** | `check_model_has_active_dropout` confirms `FloodUNet` using `smp.Unet(..., decoder_dropout=0.2)` contains 0 active `nn.Dropout` modules in its decoder; empirical difference `max_diff = 0.000000`. MC Dropout was legitimately skipped; Deep Ensembles (3 seeds) was executed instead. |
| **A12** | **Bolivia Data Isolation Rule** | **DONE** | Grep of `src/` and `scripts/` reveals all "bolivia" hits are strictly evaluation or leakage assertions (`assert len(train_ids & bolivia_ids) == 0`). In `scripts/evaluate_uncertainty_pipeline.py:611-619`, temperature scaling is fitted STRICTLY on `valid` split ($T=0.7289$ for U-Net, $T=1.2720$ for DeepLabV3+). Bolivia data NEVER influences any fitted parameter. |
| **A13** | **Uncertainty Unit Tests** | **DONE** | 17 unit tests in `tests/test_uncertainty.py` (6 tests) and `tests/test_uncertainty_metrics.py` (11 tests). 100% pass (`17 passed in 6.39s`). |
| **A14** | **Publication Figures** | **DONE** | 8 figures exist in `results/figures/` (5 core Stage A figures + 3 new uncertainty figures: `example_chips.png`, `rejection_curves.png`, `reliability_diagrams.png`). |
| **A15** | **Slide 4: Literature Matrix** | **NOT DONE** | `docs/slides/review_2_presentation.md` Slide 4 has **ZERO cited papers**, zero author names, zero journal names, and zero quantitative literature benchmarks. |
| **A16** | **Review-2 Presentation Placeholders** | **PARTIAL** | Slide 1 contains placeholders: `Member A`, `Member B`, `Member C`, `Member D`, and `[Guide Name, Designation]`. Slide 9 cites outdated test count ("29 / 29 unit tests passing" instead of 40). |
| **A17** | **Review-2 Uncertainty Slide** | **NOT DONE** | No dedicated slide exists in `docs/slides/review_2_presentation.md` presenting the new uncertainty results (ensemble gains, ECE calibration, AUROC/AUPRC, boundary proximity, rejection curves). |
| **A18** | **Architecture Diagram Rendering** | **PARTIAL** | Mermaid diagrams in `docs/architecture.md` are valid markdown text, but are **NOT rendered as image files (PNG/SVG)** anywhere in the repository. |
| **A19** | **Scientific Language Integrity** | **DONE** | Audit of `README.md`, `docs/FINDINGS.md`, and slides confirms removal of unsubstantiated causal claims (commit `8a68153`). Bolivia variations are strictly reported as empirical observations with explicit caveats regarding small sample size (15 chips). |
| **A20** | **README Documentation Accuracy** | **PARTIAL** | `README.md` states 29 tests (actual: 40), states 5 figures (actual: 8), lacks folder structure overview, and omits documentation for `scripts/evaluate_uncertainty_pipeline.py`. |

---

## SECTION B: PER-AUTHOR CONTRIBUTION SUMMARY (GIT AUDIT)

### 1. Git Repository State
* **Current Branch:** `main` tracking `origin/main` (`https://github.com/zayedahmedcs24-design/mini_project.git`).
* **Commit Count:** 5 commits.
* **Commit History:**
  1. `8a68153` (2026-10-24 22:51:36 +0530) — *Reword Bolivia findings: report seed variation, no causal claims*
  2. `51793a6` (2026-10-24 22:43:20 +0530) — *Stage A: data, metrics, Otsu, U-Net and DeepLabV3+ (3 seeds), figures, slides*
  3. `cafc750` (2026-10-24 19:47:07 +0530) — *1st review*
  4. `aeea4c6` (2026-10-24 19:44:37 +0530) — *first commit*
  5. `b217670` (2026-10-24 19:43:40 +0530) — *first commit*

### 2. Contributor Attribution Analysis
* **Sole Author in Git Log:** `zayedakhan2006-dev` (`zayedakhan2006@gmail.com`).
  * **Top-Level Folders Touched:** `.agent`, `.agents`, `.gemini`, `.gsd`, `adapters`, `configs`, `data`, `docs`, `notebooks`, `results`, `scripts`, `src`, `tests`, and root configuration files.
  * **Commits:** 5 / 5 (100%).
* **Team Members A, B, C, D:**
  * While `README.md` and Slide 1 designate specific roles (Member A: Data, Member B: Deep Learning, Member C: Uncertainty, Member D: Web/Deployment), **none of these members have distinct git commits, authorships, or branches**.
  * **Viva Risk:** The review panel may cross-check GitHub commit history. If individual accountability is questioned, the team must explain that work was merged/pushed through a single repository maintainer account (`zayedakhan2006-dev`).

### 3. Uncommitted Working Tree State (Latest Run: 28 Oct 2026)
The following files are modified/untracked on disk and represent the complete uncertainty quantification milestone:
* `M src/floodmap/uncertainty/__init__.py` (exposes ensemble and calibration functions)
* `?? src/floodmap/uncertainty/ensemble.py` (core ensemble, ECE, temperature scaling, boundary analysis)
* `?? scripts/evaluate_uncertainty_pipeline.py` (execution pipeline for uncertainty evaluation)
* `?? tests/test_uncertainty_metrics.py` (11 unit tests for uncertainty math)
* `?? results/uncertainty_results.json` (complete structured uncertainty benchmark)
* `?? results/figures/example_chips.png` (4-panel spatial uncertainty maps)
* `?? results/figures/rejection_curves.png` (accuracy vs rejection curves)
* `?? results/figures/reliability_diagrams.png` (10-bin calibration diagrams before/after scaling)
* `?? results/uncertainty/` (1,552 float16 uncertainty prediction maps, 1.14 GB)

---

## SECTION C: DISCREPANCIES, RISKS, AND BUGS (ORDERED BY SEVERITY)

### [CRITICAL] 1. Slide 4 Has Zero Literature Citations (Academic Defense Hazard)
* **Location:** `docs/slides/review_2_presentation.md:44-53`
* **Evidence:** Slide 4 proclaims: *"Journal Audit Standard: Restricted strictly to 2024–2026 peer-reviewed journals (IEEE, Elsevier, Springer, ACM, MDPI)"*, but **does not cite a single paper, author, publication year, journal, or quantitative baseline score**.
* **Impact:** In an academic project review, presenting an empty literature survey slide while claiming adherence to strict journal standards will trigger immediate faculty reprimand and marks deduction.
* **Remedy:** Populate Slide 4 with actual 2024–2026 peer-reviewed papers by the student team.

### [CRITICAL] 2. Missing Uncertainty Slide in Review-2 Presentation
* **Location:** `docs/slides/review_2_presentation.md`
* **Evidence:** The presentation has 10 slides, ending with Slide 10 (Roadmap). The extensive uncertainty work completed on 28 October (Ensemble IoU gains, 500k-pixel AUROC, temperature scaling ECE reductions, boundary proximity ratio >3.5x, and rejection curves) is **not represented in any slide**.
* **Impact:** The strongest new empirical contribution of the team will be completely invisible during the presentation.
* **Remedy:** Insert a new slide (Slide 9A or update Slide 9) dedicated to Uncertainty Quantification & Calibration before the review.

### [HIGH] 3. Unrendered Mermaid Diagrams for Presentation
* **Location:** `docs/architecture.md:11-57, 63-98, 105-140`
* **Evidence:** Architecture, sequence data-flow, and class diagrams exist only as Mermaid markdown text. No PNG/SVG images exist in `results/figures/` or `docs/`.
* **Impact:** Slide 6 uses a simplified ASCII flowchart. If evaluators ask to see the full system architecture, data flow sequence, or module class diagram, raw Mermaid markdown in a text editor appears unprofessional.
* **Remedy:** Render the three Mermaid diagrams to high-resolution PNGs or SVGs for presentation slides.

### [HIGH] 4. Placeholders Remaining in Presentation
* **Location:** `docs/slides/review_2_presentation.md:12-16`
* **Evidence:**
  * Line 12: `Member A: Data Engineering, Radiometrics & Metric Integrity`
  * Line 13: `Member B: Deep Learning Architectures, Optimization & Bolivia Evaluation`
  * Line 14: `Member C: Uncertainty Quantification, Calibration & Geospatial Exposure`
  * Line 15: `Member D: API Services, Frontend Deployment, Documentation & Integrity`
  * Line 16: `Project Guide: [Guide Name, Designation]`
* **Impact:** Displaying literal brackets `[Guide Name, Designation]` and generic `Member A/B/C/D` during Review-2 reflects poorly on preparation.
* **Remedy:** Replace with actual student names, USNs, and Guide name/designation.

### [MEDIUM] 5. Outdated Test Count in README and Slide 9
* **Location:** `README.md:54, 125, 137` and `docs/slides/review_2_presentation.md:163`
* **Evidence:** Documents claim `29 / 29 unit tests passing`. In reality, `pytest tests -q` executes **40 passed tests** (11 new tests added in `tests/test_uncertainty_metrics.py`).
* **Impact:** Minor discrepancy between printed documentation and live demo command output (`40 passed in 37.69s`).

### [MEDIUM] 6. Vacuous Dropout Test in `tests/test_models.py`
* **Location:** `tests/test_models.py:28-38`
* **Evidence:** `test_unet_mc_dropout_activation` loops over `model.modules()` checking `if isinstance(m, (nn.Dropout, nn.Dropout2d))`. Because `smp.Unet(..., decoder_dropout=0.2)` does not insert dropout layers into `UnetDecoderBlock`, the loop body never executes for dropout, vacuously passing.
* **Note:** This was caught and handled correctly by `scripts/evaluate_uncertainty_pipeline.py`, which verified `max_diff == 0.0` and appropriately switched to Deep Ensembles.

---

## SECTION D: EMPIRICAL BENCHMARKS & UNCERTAINTY AUDIT

### 1. Multi-Seed Model Performance (Run 2 Matched Recipe)
Compiled from `results/metrics_summary_table.csv` and re-verified against per-seed JSON files:

| Split | Architecture / Baseline | Seeds / Setup | Flood IoU (%) | Flood F1 (%) | Precision (%) | Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Valid** (89 chips) | Otsu (Per-Chip Dynamic) | Deterministic (Mean) | 21.52% | 35.42% | 23.25% | 74.33% |
| **Valid** (89 chips) | Otsu (Train-Fitted Global) | Deterministic (Mean) | 44.58% | 61.67% | 53.49% | 72.81% |
| **Valid** (89 chips) | **U-Net ResNet-34** | 3 Seeds (0, 1, 2) | **60.60 ± 0.26%** | 75.47 ± 0.20% | 75.88 ± 4.37% | 75.38 ± 4.23% |
| **Valid** (89 chips) | **DeepLabV3+ ResNet-34** | 3 Seeds (0, 1, 2) | **61.62 ± 0.35%** | 76.25 ± 0.27% | 78.08 ± 3.93% | 74.71 ± 2.99% |
| **Test** (90 chips) | Otsu (Per-Chip Dynamic) | Deterministic (Mean) | 24.73% | 39.65% | 27.04% | 74.32% |
| **Test** (90 chips) | Otsu (Train-Fitted Global) | Deterministic (Mean) | 49.54% | 66.26% | 58.60% | 76.22% |
| **Test** (90 chips) | **U-Net ResNet-34** | 3 Seeds (0, 1, 2) | **63.81 ± 0.15%** | 77.91 ± 0.11% | 78.27 ± 4.01% | 77.82 ± 4.06% |
| **Test** (90 chips) | **DeepLabV3+ ResNet-34** | 3 Seeds (0, 1, 2) | **64.27 ± 0.52%** | 78.25 ± 0.39% | 79.72 ± 2.92% | 76.99 ± 3.37% |
| **Bolivia** (15 chips) | Otsu (Per-Chip Dynamic) | Deterministic (Mean) | 34.73% | 51.55% | 44.20% | 61.84% |
| **Bolivia** (15 chips) | Otsu (Train-Fitted Global) | Deterministic (Mean) | 64.81% | 78.65% | 74.89% | 82.80% |
| **Bolivia** (15 chips) | **U-Net ResNet-34** | 3 Seeds (0, 1, 2) | **60.30 ± 7.59%** | 75.04 ± 6.08% | 78.40 ± 1.17% | 72.66 ± 11.45% |
| **Bolivia** (15 chips) | **DeepLabV3+ ResNet-34** | 3 Seeds (0, 1, 2) | **61.89 ± 5.45%** | 76.37 ± 4.18% | 80.13 ± 3.49% | 73.76 ± 10.56% |

### 2. Deep Ensemble vs. Single-Seed Mean Comparison
Extracted directly from `results/uncertainty_results.json`:

| Model Architecture | Evaluation Split | Individual Seed IoUs (%) | Seed Spread | 3-Seed Mean IoU | Deep Ensemble IoU | Ensemble Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net ResNet-34** | Validation (In-Domain) | [60.90, 60.45, 60.44]% | 0.46 pp | 60.60% | **62.31%** | **+1.71 pp** |
| **U-Net ResNet-34** | Test (In-Domain) | [63.89, 63.90, 63.64]% | 0.26 pp | 63.81% | **65.33%** | **+1.52 pp** |
| **U-Net ResNet-34** | Bolivia (Domain Shift) | [64.78, 64.58, 51.54]% | **13.24 pp** | 60.30% | **60.91%** | **+0.61 pp** |
| **DeepLabV3+ ResNet-34** | Validation (In-Domain) | [62.03, 61.36, 61.48]% | 0.67 pp | 61.62% | **63.18%** | **+1.56 pp** |
| **DeepLabV3+ ResNet-34** | Test (In-Domain) | [63.70, 64.39, 64.72]% | 1.02 pp | 64.27% | **66.43%** | **+2.16 pp** |
| **DeepLabV3+ ResNet-34** | Bolivia (Domain Shift) | [56.26, 62.27, 67.14]% | **10.88 pp** | 61.89% | **63.32%** | **+1.43 pp** |

*Key Takeaway:* Ensembling consistently improves Flood IoU across all splits and architectures. On Bolivia, while individual seeds fluctuate across a ~11–13 percentage point spread, ensembling yields predictions that exceed the single-model mean by +0.61 pp (U-Net) and +1.43 pp (DeepLabV3+).

### 3. Calibration & Error Detection Metrics (Uncertainty Audit)
Extracted from `results/uncertainty_results.json` (subsample size: 500,000 pixels, random seed: 42):

| Model Architecture | Split | Raw ECE (10 Bins) | Scaled ECE ($T^*_{\text{valid}}$) | Optimal Temperature ($T^*$) | Error AUROC | Error AUPRC | Baseline Error Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net ResNet-34** | Valid | 4.44% | **1.73%** | $T^* = 0.7289$ (fitted on Valid) | **0.8686** | 0.3186 | 5.01% |
| **U-Net ResNet-34** | Test | 4.40% | **1.67%** | $T^* = 0.7289$ (applied) | **0.8140** | 0.2934 | 4.99% |
| **U-Net ResNet-34** | Bolivia | 4.74% | **3.03%** | $T^* = 0.7289$ (applied) | **0.8643** | 0.3529 | 7.23% |
| **DeepLabV3+ ResNet-34**| Valid | 1.55% | **0.71%** | $T^* = 1.2720$ (fitted on Valid) | **0.9204** | 0.3524 | 4.88% |
| **DeepLabV3+ ResNet-34**| Test | 1.86% | **0.82%** | $T^* = 1.2720$ (applied) | **0.8967** | 0.3177 | 4.85% |
| **DeepLabV3+ ResNet-34**| Bolivia | 3.15% | **2.71%** | $T^* = 1.2720$ (applied) | **0.8941** | 0.3601 | 6.84% |

*Key Takeaway:* Temperature scaling strictly fitted on the validation set reduces ECE by over 50% across in-domain test splits and cuts ECE on Bolivia from 4.74% down to 3.03% (U-Net). Error-detection AUROCs of 0.81–0.92 confirm that high predictive entropy strongly correlates with misclassified pixels.

### 4. Flood Boundary Proximity Analysis
Extracted from `results/uncertainty_results.json` (`boundary_proximity_analysis`, distance $\le 3\text{ px}$ from GT flood boundary):

| Model Architecture | Split | Top-10% Uncertain Pixels Near Boundary (%) | All Valid Pixels Near Boundary (%) | Boundary Proximity Concentration Ratio |
| :--- | :--- | :---: | :---: | :---: |
| **U-Net ResNet-34** | Valid | 37.79% | 9.56% | **3.95x** |
| **U-Net ResNet-34** | Test | 38.37% | 10.09% | **3.80x** |
| **U-Net ResNet-34** | Bolivia | 44.33% | 12.15% | **3.65x** |
| **DeepLabV3+ ResNet-34**| Valid | 40.14% | 9.56% | **4.20x** |
| **DeepLabV3+ ResNet-34**| Test | 41.01% | 10.09% | **4.06x** |
| **DeepLabV3+ ResNet-34**| Bolivia | 43.24% | 12.15% | **3.56x** |

*Key Takeaway:* Between 37% and 44% of the most uncertain pixels fall within 3 pixels of the flood boundary—representing an empirical concentration ratio of 3.56x to 4.20x above spatial base rate. This shows uncertainty is localized along complex shorelines and transition zones rather than randomly scattered across flat terrain.

---

## SECTION E: REVIEW-2 READINESS CHECKLIST

| Component | Status | Verification & Readiness Notes |
| :--- | :---: | :--- |
| **1. Architecture / Data-Flow Diagram** | **PARTIAL** | Complete in `docs/architecture.md` as Mermaid text. Needs rendering to PNG/SVG for presentation slides. |
| **2. Problem Statement & SDG Alignment** | **READY** | Slides 2 & 3 cover radar physics, optical obstruction, domain shift, and UN SDGs 13.1 & 11.5 with full domain rationale. |
| **3. Literature Table with Real Papers** | **NOT READY** | Slide 4 is empty of specific papers. Must insert 4–6 real papers before Review-2 tomorrow. |
| **4. Data Engineering & EDA Slide** | **READY** | Slide 7 contains exact dB ranges, percentiles, NaN statistics, and class imbalance numbers matching `results/sar_stats.json`. |
| **5. Evaluation Metrics & Unit Tests** | **READY** | Ignore-aware metrics (`-1` masking), confusion matrices, and 40 passing pytest unit tests (`pytest tests -q`). |
| **6. Classical Baselines (Otsu)** | **READY** | Dynamic per-chip and train-fitted global Otsu evaluated across VV, VH, and Mean; diagnostic failure mode documented in `results/figures/fig_otsu_failure.png`. |
| **7. Multi-Seed Deep Models** | **READY** | U-Net and DeepLabV3+ trained across 3 random seeds on GPU; all 6 `.pth` checkpoints verified on disk. |
| **8. Uncertainty Quantification** | **READY (Code/Data) / MISSING (Slide)** | JSON results, figures, and tests complete. Needs dedicated presentation slide to showcase ensemble gains and calibration. |
| **9. Metadata & Placeholders** | **NOT READY** | Team names (`Member A/B/C/D`) and `[Guide Name, Designation]` need human write-in. |
| **10. 3-Minute Live Demo Path** | **READY** | Commands tested and verified (see Section F). |
| **11. Backup Diagnostic Figures** | **READY** | 8 high-resolution figures in `results/figures/` ready for offline slide insertion or screen share. |

---

## SECTION F: 3-MINUTE LIVE DEMO PATH & EXACT COMMANDS

For Review-2, the team can demonstrate end-to-end functionality live in 3 minutes without touching training or downloading large files:

```powershell
# STEP 1: Verify test suite and mathematical integrity (Time: ~35 seconds)
venv\Scripts\python.exe -m pytest tests -q
# Expected Output: 40 passed, 5 warnings in ~37s

# STEP 2: Execute Exploratory Data Analysis & Radiometric Verification (Time: ~5 seconds)
venv\Scripts\python.exe scripts/run_eda.py
# Expected Output: Prints backscatter dB stats, NaN handling (98.18% inside ignore), and 10.6% class imbalance

# STEP 3: Run Classical Otsu Baseline Evaluation (Time: ~15 seconds)
venv\Scripts\python.exe scripts/eval_otsu.py
# Expected Output: Prints dynamic vs global Otsu results on valid, test, and bolivia splits

# STEP 4: Inspect Generated Publication Figures (Time: ~60 seconds)
# Open and present key figures from results/figures/:
# 1. results/figures/fig_val_curves.png (6-run training trajectories & early stopping)
# 2. results/figures/fig_otsu_failure.png (Otsu bisection failure on dry land)
# 3. results/figures/reliability_diagrams.png (ECE before and after temperature scaling)
# 4. results/figures/rejection_curves.png (Monotonic IoU improvement under uncertainty rejection)
# 5. results/figures/example_chips.png (Ensemble prediction & entropy maps on sample chips)
```

---

## SECTION G: PRIORITIZED 24-HOUR ACTION PLAN BEFORE REVIEW-2

| Priority | Task Description | Est. Time | Required Action & Role |
| :---: | :--- | :---: | :--- |
| **P1** | **Fill Slide 4 Literature Survey Matrix** | **45 min** | **HUMAN REQUIRED**: Replace the empty text in `docs/slides/review_2_presentation.md` Slide 4 with actual 2024–2026 journal papers. |
| **P2** | **Add Slide 9A: Uncertainty Quantification & Calibration** | **30 min** | Insert a new slide in `docs/slides/review_2_presentation.md` featuring the table from Section D: Ensemble IoU improvements (+1.5–2.1 pp), Temperature scaling ECE drop (4.4% $\to$ 1.7%), AUROC (0.81–0.92), and the 3.8x boundary proximity ratio. |
| **P3** | **Replace Member & Guide Placeholders** | **10 min** | **HUMAN REQUIRED**: In `docs/slides/review_2_presentation.md` Slide 1, replace `Member A, B, C, D` with student names and USNs; replace `[Guide Name, Designation]` with actual project guide details. |
| **P4** | **Render Mermaid Architecture Diagrams to PNG** | **20 min** | Use Mermaid CLI (`mmdc`), VS Code Mermaid Preview export, or [mermaid.live](https://mermaid.live) to render the 3 diagrams in `docs/architecture.md` (`system_architecture.png`, `data_flow.png`, `class_diagram.png`) and save into `results/figures/`. |
| **P5** | **Update Outdated Numbers in Docs & README** | **15 min** | In `README.md` and Slide 9, update test count from `29` to `40 passed` and figures count from `5` to `8 figures`. Document `scripts/evaluate_uncertainty_pipeline.py`. |
| **P6** | **Commit Working Tree to Git** | **10 min** | Once files are polished, commit the uncommitted uncertainty code and figures (`ensemble.py`, `evaluate_uncertainty_pipeline.py`, `test_uncertainty_metrics.py`, `STATUS_REPORT_2.md`) to maintain a clean git working tree for the review. |
