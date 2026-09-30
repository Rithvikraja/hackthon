"""
Unit test: Robustness metrics and skill retention
Validates Section 9.3 toy example calculations, clipping bounds, and no-skill handling.
"""

import pytest
from nerva.robustness.metrics import calculate_skill_retention, calculate_robustness_score

def test_toy_robustness_calculation():
    """
    Direct test of Section 9.3 toy example:
    Clean: 0.90
    Noise alpha 0.10: 0.86 -> R_p = (0.86 - 0.50) / 0.40 = 0.900
    Dropout q 0.10: 0.80 -> R_p = (0.80 - 0.50) / 0.40 = 0.750
    Batch shift L2: 0.75 -> R_p = (0.75 - 0.50) / 0.40 = 0.625
    Overall R = 0.75833...
    """
    clean_auc = 0.90

    r_noise = calculate_skill_retention(clean_auc, 0.86)
    r_drop = calculate_skill_retention(clean_auc, 0.80)
    r_shift = calculate_skill_retention(clean_auc, 0.75)

    assert pytest.approx(r_noise, abs=1e-3) == 0.900
    assert pytest.approx(r_drop, abs=1e-3) == 0.750
    assert pytest.approx(r_shift, abs=1e-3) == 0.625

    overall_r = calculate_robustness_score([r_noise, r_drop, r_shift])
    assert pytest.approx(overall_r, abs=1e-3) == 0.758

def test_skill_retention_edge_cases():
    # AUROC <= 0.55 should return 0.0
    assert calculate_skill_retention(0.54, 0.52) == 0.0
    assert calculate_skill_retention(0.50, 0.50) == 0.0

    # Perturbed AUROC below chance (0.50) clipped to 0.0
    assert calculate_skill_retention(0.80, 0.45) == 0.0

    # Perturbed AUROC higher than clean (lucky draw) clipped to 1.0
    assert calculate_skill_retention(0.80, 0.85) == 1.0
