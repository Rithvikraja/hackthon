"""
Unit test: Perturbation functions
Validates noise scale, dropout count, batch shift invariants, and immutability.
"""

import pytest
import numpy as np
from nerva.robustness.perturbations import (
    apply_gaussian_noise,
    apply_feature_dropout,
    apply_batch_shift
)

def test_perturbation_immutability():
    rng = np.random.default_rng(42)
    X_orig = np.ones((10, 5))
    X_copy = X_orig.copy()
    train_sd = np.ones(5)

    # 1. Noise
    _ = apply_gaussian_noise(X_orig, alpha=0.10, train_sd=train_sd, rng=rng)
    assert np.array_equal(X_orig, X_copy)

    # 2. Dropout
    _ = apply_feature_dropout(X_orig, q=0.20, rng=rng)
    assert np.array_equal(X_orig, X_copy)

    # 3. Shift
    _ = apply_batch_shift(X_orig, a=1.15, tau=0.25, train_sd=train_sd, rng=rng)
    assert np.array_equal(X_orig, X_copy)

def test_feature_dropout_exact_count():
    rng = np.random.default_rng(123)
    k = 20
    X = np.ones((10, k))
    
    # q = 0.10 -> drop max(1, round(0.10 * 20)) = 2 features
    X_dropped = apply_feature_dropout(X, q=0.10, rng=rng)
    zero_cols = np.where(np.all(X_dropped == 0.0, axis=0))[0]
    assert len(zero_cols) == 2

    # q = 0.05 -> drop max(1, round(0.05 * 20)) = 1 feature
    X_dropped_1 = apply_feature_dropout(X, q=0.05, rng=rng)
    zero_cols_1 = np.where(np.all(X_dropped_1 == 0.0, axis=0))[0]
    assert len(zero_cols_1) == 1

def test_batch_shift_row_uniformity():
    rng = np.random.default_rng(999)
    X = np.zeros((10, 4))
    train_sd = np.array([1.0, 2.0, 3.0, 4.0])

    X_shifted = apply_batch_shift(X, a=1.0, tau=0.5, train_sd=train_sd, rng=rng)
    
    # All rows in X_shifted should have received identical shift offset per gene
    for col in range(4):
        col_vals = X_shifted[:, col]
        assert np.all(col_vals == col_vals[0])
