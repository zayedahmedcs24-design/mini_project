# Empirical Findings & Limitations: Sentinel-1 SAR Flood Mapping

This document synthesizes key empirical observations, training configurations, and limitations derived strictly from multi-seed deep learning runs (seeds 0, 1, 2) and deterministic classical baselines on the Sen1Floods11 dataset.

---

## 1. Experimental Training Recipes

### Run 1 Recipe (Archived in `results/run1/`)
* **U-Net (ResNet-34):** Random $256 \times 256$ crops for training, batch size 8, full $512 \times 512$ chips for validation/test; `max_epochs = 25`, early stopping patience 8 on validation flood IoU.
* **DeepLabV3+ (ResNet-34):** Full $512 \times 512$ chips for training and validation/test, batch size 4 due to initial ASPP memory assumptions; `max_epochs = 30`, early stopping patience 8.

### Run 2 Recipe ("run2 recipe" — Matched Configuration)
Applied identically to **both** U-Net and DeepLabV3+:
* **Max Epochs:** `max_epochs = 60` for both U-Net and DeepLabV3+.
* **Early Stopping:** Monitored strictly on **VALIDATION flood IoU** with patience 8 (unchanged). Bolivia split never influenced training or stopping.
* **Matched Data Recipe:** Both models trained with random $256 \times 256$ crops, batch size 8. Validation and test evaluated on full $512 \times 512$ chips.
* **Optimization & Loss:** AdamW optimizer, learning rate $3 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, AMP (FP16 mixed precision), compound loss $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE, masked}} + 1.0 \cdot \mathcal{L}_{\text{Dice, masked}}$ (excluding `-1` ignore pixels).
* **Seeds:** $\{0, 1, 2\}$ for both models.
* **Architectures, Loss, and Splits:** Untouched. Bolivia did not influence any design choice.
* **VRAM / Out-of-Memory Report:** DeepLabV3+ at batch size 8 with $256 \times 256$ random crops did **not** hit out-of-memory (OOM). Peak VRAM usage during training steps was **519.4 MB** on an NVIDIA RTX 4050 GPU (6 GB VRAM). No batch size reduction was required.

---

## 2. Run Epochs & Early Stopping Cap Status

| Model | Seed | Best Epoch | Epochs Trained | Hit Epoch Cap (60)? | Stopping Condition |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **U-Net ResNet-34** | Seed 0 | 8 | 16 | **No** | Early stopping (patience 8 after Ep 8) |
| **U-Net ResNet-34** | Seed 1 | 4 | 12 | **No** | Early stopping (patience 8 after Ep 4) |
| **U-Net ResNet-34** | Seed 2 | 9 | 17 | **No** | Early stopping (patience 8 after Ep 9) |
| **DeepLabV3+ ResNet-34** | Seed 0 | 13 | 21 | **No** | Early stopping (patience 8 after Ep 13) |
| **DeepLabV3+ ResNet-34** | Seed 1 | 32 | 40 | **No** | Early stopping (patience 8 after Ep 32) |
| **DeepLabV3+ ResNet-34** | Seed 2 | 9 | 17 | **No** | Early stopping (patience 8 after Ep 9) |

* **Epoch Cap Summary:** **0 out of 6 runs** hit the 60-epoch cap. All 6 runs converged and stopped via validation flood IoU early stopping.

---

## 3. Run1 vs. Run2 Comparative Benchmark Table

Format: `run1 = max 25/30 epochs, different data recipes; run2 = matched recipe, max 60 epochs`.
Metrics show Mean ± Std across seeds 0, 1, 2.

| Split | Architecture / Baseline | Run1: Flood IoU (%) | Run2: Flood IoU (%) | Run1 $\to$ Run2 $\Delta$ (pp) | Run1: Flood F1 (%) | Run2: Flood F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Valid** | Otsu (Per-Chip Dynamic, Mean) | 21.52% | 21.52% | 0.00 pp | 35.42% | 35.42% |
| **Valid** | Otsu (Train-Fitted Global, Mean) | 44.58% | 44.58% | 0.00 pp | 61.67% | 61.67% |
| **Valid** | **U-Net ResNet-34** | 61.29 ± 0.71% | 60.60 ± 0.26% | -0.69 pp | 76.00 ± 0.54% | 75.47 ± 0.20% |
| **Valid** | **DeepLabV3+ ResNet-34** | 61.22 ± 1.28% | 61.62 ± 0.35% | **+0.40 pp** | 75.94 ± 0.99% | 76.25 ± 0.27% |
| **Test** | Otsu (Per-Chip Dynamic, Mean) | 24.73% | 24.73% | 0.00 pp | 39.65% | 39.65% |
| **Test** | Otsu (Train-Fitted Global, Mean) | 49.54% | 49.54% | 0.00 pp | 66.26% | 66.26% |
| **Test** | **U-Net ResNet-34** | 64.45 ± 1.16% | 63.81 ± 0.15% | -0.64 pp | 78.38 ± 0.86% | 77.91 ± 0.11% |
| **Test** | **DeepLabV3+ ResNet-34** | 64.03 ± 0.95% | 64.27 ± 0.52% | **+0.24 pp** | 78.07 ± 0.70% | 78.25 ± 0.39% |
| **Bolivia** | Otsu (Per-Chip Dynamic, Mean) | 34.73% | 34.73% | 0.00 pp | 51.55% | 51.55% |
| **Bolivia** | Otsu (Train-Fitted Global, Mean) | 64.81% | 64.81% | 0.00 pp | 78.65% | 78.65% |
| **Bolivia** | **U-Net ResNet-34** | 65.08 ± 0.88% | 60.30 ± 7.59% | -4.78 pp | 78.85 ± 0.65% | 75.04 ± 6.08% |
| **Bolivia** | **DeepLabV3+ ResNet-34** | 57.80 ± 9.74% | 61.89 ± 5.45% | **+4.09 pp** | 72.92 ± 8.07% | 76.37 ± 4.18% |

---

## 4. Key Empirical Findings (10 Core Insights)

1. **Input Backscatter is Pre-Calibrated in Decibels (dB):**
   Radiometric inspection confirms raw values in Sentinel-1 `S1Hand` chips are natively stored in decibels (ranging from $-65.45$ to $+29.93$ dB on VV, and $-79.16$ to $+17.38$ dB on VH). Applying $\log_{10}$ transformations on these negative values is mathematically invalid. Clamping inputs to $[-35.0, +5.0]$ dB (VV) and $[-45.0, 0.0]$ dB (VH) covers the 1st–99th percentiles while suppressing extreme sensor noise without distorting terrain distributions.

2. **NaN Noise is Concentrated in Ignored Cloud Margins:**
   Across all 252 training chips (66.06M pixels), exactly 1,325,136 pixels contain NaNs ($2.01\%$). Of these, $98.18\%$ fall strictly within ignored boundary/cloud regions labeled as $-1$. Imputing NaNs to floor backscatter before Z-score normalization prevents numerical instability in neural network layers without altering valid ground truth annotations.

3. **Severe Class Imbalance Invalidates Overall Accuracy:**
   Flooded water accounts for only $10.60\%$ of valid labeled pixels, while ignore pixels (`-1`) comprise $13.63\%$ of the total imagery. A trivial baseline predicting all dry land scores $> 89\%$ overall accuracy but achieves $0.00\%$ Flood IoU. Consequently, ignore-aware Flood IoU and Flood F1 serve as the sole reliable evaluation metrics.

4. **Dynamic Per-Chip Otsu Suffers from Bimodal Bisection Failures:**
   Dynamic per-chip Otsu thresholding performs poorly on standard validation ($21.52\%$ IoU) and test ($24.73\%$ IoU) splits. Because Otsu assumes a bimodal intensity distribution, it bisects predominantly dry scenes, falsely classifying $45\%$ to $61\%$ of dry land as floodwater on flood-sparse chips (e.g., `India_79637` and `India_44475`).

5. **Train-Fitted Global Otsu Drastically Suppresses False Positives:**
   Fitting Otsu once across all valid, clipped pixels of the training split yields fixed global thresholds ($-0.1538$ for Dual-Pol Mean). This global threshold improves test Flood IoU from $24.73\%$ to $49.54\%$ (+24.81 percentage points gain) by establishing a realistic water-versus-land boundary that correctly rejects unimodal dry terrain.

6. **In-Domain Comparisons: Differences Within Seed-to-Seed Variation:**
   Across 3 random seeds, U-Net ResNet-34 ($63.81 \pm 0.15\%$ IoU) and DeepLabV3+ ResNet-34 ($64.27 \pm 0.52\%$ IoU) show **differences within seed-to-seed variation** on the 90 test chips (a difference of $+0.46$ percentage points in favor of DeepLabV3+, against standard deviations of $\pm 0.15\%$ and $\pm 0.52\%$). On validation, DeepLabV3+ achieves $61.62 \pm 0.35\%$ vs. U-Net $60.60 \pm 0.26\%$ ($+1.02$ percentage points difference). Multi-scale Atrous Spatial Pyramid Pooling (ASPP) provides no substantial in-domain advantage over standard U-Net skip connections on $512 \times 512$ dual-polarization SAR chips.

7. **Run2 Matched Training Recipe Eliminates Memory Bottleneck:**
   By matching the training recipe (batch size 8 with $256 \times 256$ random crops), DeepLabV3+ achieved full parity in training conditions with U-Net, consuming only **519.4 MB** peak VRAM with zero OOM issues. All runs converged comfortably under the 60-epoch cap (U-Net in 12–17 epochs; DeepLabV3+ in 17–40 epochs).

8. **Regional Comparison on Bolivia:**
   Bolivia means are 2–3.5 percentage points below test means for both models (U-Net: -3.51 percentage points, from 63.81% to 60.30%; DeepLabV3+: -2.38 percentage points, from 64.27% to 61.89%), within the large seed-to-seed spread; with 15 chips from one event we cannot conclude whether there is a regional drop. Flood fraction of valid pixels is 15.86% on Bolivia (454,864 / 2,867,815 valid pixels) vs. 12.51% on test (2,566,101 / 20,517,367 valid pixels); we did not investigate why Otsu performs well there.

9. **Inter-Seed Variation on Bolivia:**
   On Bolivia, both models varied substantially between seeds (U-Net ±7.6, DeepLabV3+ ±5.5 IoU points; DeepLabV3+ seeds 56.3%, 62.3%, 67.1%). In run 1 the outlier DeepLabV3+ seed hit the 30-epoch cap; run 2 also changed the data recipe, so we cannot attribute the change to a single cause.

10. **Near-Zero Flood Trap in Per-Chip Metric Distributions:**
    Per-chip evaluation indicates that chips with $< 1\%$ flood or $100\%$ ignore have mathematically unrepresentative IoU scores (marked as *"IoU not meaningful"*). Ranking the 10 worst test chips strictly among chips with $\ge 1\%$ flood isolates genuine model errors: missed narrow flood features (e.g., `Somalia_699062` at $3.15\%$ IoU) or heavy false positive alarms (e.g., `Pakistan_70625` with 44,647 false positive pixels). Dataset-level confusion matrix accumulation is mathematically required to eliminate small-sample divide-by-zero bias.
