"""
Uncertainty quantification package:
- Monte Carlo Dropout inference
- Expected Calibration Error (ECE) and reliability diagrams
- Error-vs-Uncertainty AUROC detection analysis
"""

from src.floodmap.uncertainty.mc_dropout import predict_mc_dropout
from src.floodmap.uncertainty.calibration import compute_ece, plot_reliability_diagram
from src.floodmap.uncertainty.error_analysis import compute_error_detection_auroc, plot_error_detection_roc

__all__ = [
    "predict_mc_dropout",
    "compute_ece",
    "plot_reliability_diagram",
    "compute_error_detection_auroc",
    "plot_error_detection_roc",
]
