"""
Unit test: Evolutionary operators for NSGA-II
Validates sampling, crossover, and mutation behavior for mixed binary-continuous chromosomes.
"""

import pytest
import numpy as np
from nerva.optimization.operators import MixedSampling, MixedCrossover, MixedMutation

def test_mixed_sampling():
    pool_size = 40
    n_samples = 10
    sampler = MixedSampling(pool_size=pool_size, n_params=10, k_min=5, k_max=20)
    
    samples = sampler._do(problem=None, n_samples=n_samples)
    assert samples.shape == (n_samples, pool_size + 10)

    for i in range(n_samples):
        mask = samples[i, :pool_size] > 0.5
        n_selected = np.sum(mask)
        assert 5 <= n_selected <= 20
        # Continuous hyperparameters in [0, 1]
        hp = samples[i, pool_size:]
        assert np.all(hp >= 0.0) and np.all(hp <= 1.0)

def test_mixed_crossover_shape_and_bounds():
    pool_size = 30
    crossover = MixedCrossover(pool_size=pool_size, prob=1.0, k_min=5, k_max=20)
    
    # Parents shape: (2, 1, pool_size + 10)
    p1 = np.zeros((1, 1, pool_size + 10))
    p2 = np.ones((1, 1, pool_size + 10))
    p1[0, 0, :10] = 1.0
    p2[0, 0, :10] = 1.0
    X = np.concatenate([p1, p2], axis=0)

    Y = crossover._do(problem=None, X=X)
    assert Y.shape == X.shape
    
    # Check bounds on offspring
    for parent_idx in range(2):
        mask = Y[parent_idx, 0, :pool_size] > 0.5
        assert 5 <= np.sum(mask) <= 20
        hp = Y[parent_idx, 0, pool_size:]
        assert np.all(hp >= 0.0) and np.all(hp <= 1.0)
