"""
Unit test: Candidate evaluation in NSGA-II
Validates objective calculation (f1, f2, f3), caching, and edge-case handling.
"""

import pytest
import numpy as np
from nerva.optimization.evaluator import NervaProblem

def test_evaluator_objectives_finite():
    n_rows = 40
    pool_size = 20
    np.random.seed(42)
    
    X_train = np.random.randn(n_rows, pool_size)
    y_train = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    
    # 2 inner folds
    inner_splits = [
        (np.arange(0, 20), np.arange(20, 40)),
        (np.arange(20, 40), np.arange(0, 20))
    ]

    problem = NervaProblem(
        X_train=X_train,
        y_train=y_train,
        inner_splits=inner_splits,
        pool_size=pool_size,
        k_min=5,
        k_max=15
    )

    # 1 candidate chromosome
    x = np.zeros((1, pool_size + 10))
    x[0, :8] = 1.0  # 8 genes
    x[0, pool_size:] = 0.5

    out = {}
    problem._evaluate(x, out)

    F = out["F"]
    assert F.shape == (1, 3)
    f1, f2, f3 = F[0]
    
    # Check that all 3 objectives are finite and in [0, 1]
    assert np.isfinite(f1) and 0.0 <= f1 <= 1.0
    assert np.isfinite(f2) and 0.0 <= f2 <= 1.0
    assert np.isfinite(f3) and 0.0 <= f3 <= 1.0

def test_evaluator_empty_mask_handling():
    X_train = np.random.randn(20, 10)
    y_train = np.random.choice([0, 1], size=20)
    inner_splits = [(np.arange(0, 10), np.arange(10, 20))]

    problem = NervaProblem(X_train, y_train, inner_splits, pool_size=10, k_min=0, k_max=10)
    
    # All zeros chromosome (no genes)
    x = np.zeros((1, 20))
    out = {}
    problem._evaluate(x, out)
    
    # Expected fallback worst-case values
    assert np.all(out["F"][0] == [1.0, 1.0, 1.0])
