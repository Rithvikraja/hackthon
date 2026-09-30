"""
NERVA-ML Chromosome Representation & Decoder
Mixed binary (gene mask) + continuous (hyperparameter) chromosome.
Fully reproducible with explicit seeded Generator.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

PARAM_RANGES = {
    "learning_rate": (0.01, 0.30, "log"),
    "n_estimators": (50, 400, "int"),
    "max_depth": (2, 6, "int"),
    "subsample": (0.5, 1.0, "linear"),
    "colsample_bytree": (0.4, 1.0, "linear"),
    "min_child_weight": (1, 10, "int"),
    "reg_alpha": (0.001, 10.0, "log"),
    "reg_lambda": (0.1, 20.0, "log"),
    "gamma": (0.0, 5.0, "linear"),
    "aug_sigma": (0.0, 0.30, "linear")
}

def decode_chromosome(chromosome: np.ndarray, pool_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decodes flat chromosome vector into binary gene mask and XGBoost hyperparameter dict.
    """
    mask = chromosome[:pool_size] > 0.5
    raw_params = chromosome[pool_size:]

    decoded_params = {}
    param_keys = list(PARAM_RANGES.keys())

    for idx, key in enumerate(param_keys):
        val = raw_params[idx] if idx < len(raw_params) else 0.5
        val = float(np.clip(val, 0.0, 1.0))
        low, high, scale = PARAM_RANGES[key]

        if scale == "log":
            real_val = float(np.exp(np.log(low) + val * (np.log(high) - np.log(low))))
        elif scale == "int":
            real_val = int(round(low + val * (high - low)))
        else:
            real_val = float(low + val * (high - low))

        decoded_params[key] = real_val

    return mask, decoded_params

def repair_mask(
    mask: np.ndarray,
    k_min: int = 5,
    k_max: int = 60,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Enforces sparsity bounds k_min <= sum(mask) <= k_max reproducibly."""
    if rng is None:
        rng = np.random.default_rng(42)

    repaired = mask.copy().astype(bool)
    k = int(np.sum(repaired))

    if k < k_min:
        unselected = np.where(~repaired)[0]
        needed = min(k_min - k, len(unselected))
        if needed > 0:
            chosen = rng.choice(unselected, size=needed, replace=False)
            repaired[chosen] = True
    elif k > k_max:
        selected = np.where(repaired)[0]
        excess = k - k_max
        chosen = rng.choice(selected, size=excess, replace=False)
        repaired[chosen] = False

    return repaired
