"""
NERVA-ML Pareto Solution Selection
Extracts Knee point (trade-off) and 1-SE point (most compact within tolerance).
"""

from typing import List, Tuple
import numpy as np

def select_knee_and_1se_solutions(
    pareto_aurocs: np.ndarray,
    pareto_robustness: np.ndarray,
    pareto_genes: np.ndarray
) -> Tuple[int, int]:
    """
    Identifies knee index and 1-SE compact index on the Pareto front.
    Returns:
        (knee_idx, one_se_idx)
    """
    n_points = len(pareto_aurocs)
    if n_points <= 2:
        return 0, 0

    # 1. Knee point calculation (distance to diagonal chord in normalized objective space)
    norm_auc = (pareto_aurocs - np.min(pareto_aurocs)) / (np.ptp(pareto_aurocs) + 1e-6)
    norm_rob = (pareto_robustness - np.min(pareto_robustness)) / (np.ptp(pareto_robustness) + 1e-6)

    # Line from (0, 0) to (1, 1) in normalized space
    distances = np.abs(norm_auc + norm_rob - 1.0) / np.sqrt(2.0)
    knee_idx = int(np.argmax(distances))

    # 2. 1-SE compact point: smallest gene count with AUROC >= max_auroc - 0.03
    max_auc = np.max(pareto_aurocs)
    eligible = np.where(pareto_aurocs >= (max_auc - 0.03))[0]
    if len(eligible) > 0:
        one_se_idx = int(eligible[np.argmin(pareto_genes[eligible])])
    else:
        one_se_idx = knee_idx

    return knee_idx, one_se_idx
