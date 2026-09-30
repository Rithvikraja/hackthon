"""
NERVA-ML Perturbation Functions
Implements Gaussian noise, feature dropout, and affine batch-like shifts.
"""

import numpy as np

def apply_gaussian_noise(X: np.ndarray, alpha: float, train_sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x'_ig = x_ig + e_ig, e_ig ~ N(0, (alpha * s_g)^2)"""
    noise = rng.normal(0, alpha * train_sd, size=X.shape)
    return X + noise

def apply_feature_dropout(X: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly zeros out max(1, round(q * k)) features across all rows."""
    X_pert = X.copy()
    k = X.shape[1]
    n_drop = max(1, int(round(q * k)))
    drop_cols = rng.choice(k, size=n_drop, replace=False)
    X_pert[:, drop_cols] = 0.0
    return X_pert

def apply_batch_shift(X: np.ndarray, a: float, tau: float, train_sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Affine shift: x'_ig = a * x_ig + b_g, b_g ~ N(0, tau^2 * s_g^2) drawn once per gene."""
    b_g = rng.normal(0, tau * train_sd, size=(1, X.shape[1]))
    return a * X + b_g
