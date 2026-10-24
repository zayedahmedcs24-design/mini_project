"""
Monte Carlo Dropout for epistemic uncertainty quantification in flood segmentation.

Features:
- Executes T stochastic forward passes with dropout active at test-time
- Freezes BatchNorm running stats in eval mode to prevent distribution drift
- Computes Predictive Mean, Predictive Variance, Predictive Entropy, and Mutual Information
"""

from typing import Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn


def predict_mc_dropout(
    model: nn.Module,
    x: Union[torch.Tensor, np.ndarray],
    num_samples: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Performs Monte Carlo Dropout sampling on input image(s).

    Args:
        model: PyTorch segmentation model with dropout layers.
        x: Input tensor or ndarray of shape (2, H, W) or (B, 2, H, W).
        num_samples: Number of stochastic forward passes (T, default 20).
        device: Torch device (defaults to model device or cuda/cpu).

    Returns:
        Dict with numpy arrays:
            - "mean": (H, W) or (B, H, W) predictive mean probability in [0, 1]
            - "variance": (H, W) or (B, H, W) predictive variance
            - "entropy": (H, W) or (B, H, W) predictive Shannon entropy in [0, 1]
            - "mutual_info": (H, W) or (B, H, W) epistemic mutual information
            - "samples": (T, H, W) or (T, B, H, W) raw sampled probabilities
    """
    if device is None:
        device = next(model.parameters()).device

    # Convert numpy to torch if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    squeeze_batch = False
    if x.ndim == 3:  # (2, H, W)
        x = x.unsqueeze(0)  # (1, 2, H, W)
        squeeze_batch = True

    x = x.to(device)

    # Activate MC Dropout: keep BatchNorm in eval, Dropout in train
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    sample_probs = []

    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            p = torch.sigmoid(logits)  # (B, 1, H, W)
            sample_probs.append(p.squeeze(1).cpu().numpy())  # (B, H, W)

    # Shape: (T, B, H, W)
    all_samples = np.stack(sample_probs, axis=0)

    # 1. Predictive Mean: (B, H, W)
    mean_p = np.mean(all_samples, axis=0)

    # 2. Predictive Variance: (B, H, W)
    variance = np.var(all_samples, axis=0)

    # Numerical epsilon to prevent log2(0)
    eps = 1e-7

    # 3. Total Predictive Entropy: H(p_mean)
    p_clamped = np.clip(mean_p, eps, 1.0 - eps)
    entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))

    # 4. Expected Entropy over samples: E_t[H(p_t)]
    samples_clamped = np.clip(all_samples, eps, 1.0 - eps)
    sample_entropies = -(samples_clamped * np.log2(samples_clamped) + (1.0 - samples_clamped) * np.log2(1.0 - samples_clamped))
    expected_entropy = np.mean(sample_entropies, axis=0)

    # 5. Mutual Information (Epistemic Uncertainty): I(y, W | x) = H(p_mean) - E_t[H(p_t)]
    mutual_info = np.maximum(0.0, entropy - expected_entropy)

    if squeeze_batch:
        mean_p = mean_p[0]
        variance = variance[0]
        entropy = entropy[0]
        mutual_info = mutual_info[0]
        all_samples = all_samples[:, 0]

    return {
        "mean": mean_p,
        "variance": variance,
        "entropy": entropy,
        "mutual_info": mutual_info,
        "samples": all_samples,
    }
