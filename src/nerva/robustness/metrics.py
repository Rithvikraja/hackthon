"""
NERVA-ML Robustness Metrics
Computes skill retention R_p and overall robustness score R.
"""

import numpy as np

def calculate_skill_retention(clean_auroc: float, perturbed_auroc: float) -> float:
    """
    Computes above-chance skill retention:
        R_p = clip((AUROC_p - 0.5) / (AUROC_clean - 0.5), 0, 1)
    Returns 0.0 if clean_auroc <= 0.55.
    """
    if clean_auroc <= 0.55:
        return 0.0
    denom = clean_auroc - 0.50
    retention = (perturbed_auroc - 0.50) / denom
    return float(np.clip(retention, 0.0, 1.0))

def calculate_robustness_score(retention_scores: list) -> float:
    """Computes mean skill retention across all perturbations."""
    if not retention_scores:
        return 0.0
    return float(np.mean(retention_scores))
