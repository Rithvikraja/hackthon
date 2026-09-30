"""
Unit test: Chromosome decoding and mask repair
Validates parameter decoding, range constraints, and k_min/k_max repair behavior.
"""

import pytest
import numpy as np
from nerva.optimization.chromosome import decode_chromosome, repair_mask, PARAM_RANGES

def test_decode_chromosome_ranges():
    pool_size = 50
    # Create normalized chromosome [0..1]
    raw_chromosome = np.zeros(pool_size + 10)
    # Set alternating mask
    raw_chromosome[:pool_size] = [1.0 if i % 2 == 0 else 0.0 for i in range(pool_size)]
    # Set middle values for parameters
    raw_chromosome[pool_size:] = 0.5

    mask, params = decode_chromosome(raw_chromosome, pool_size)
    
    assert len(mask) == pool_size
    assert len(params) == 10

    for key, (low, high, scale) in PARAM_RANGES.items():
        assert key in params
        val = params[key]
        if scale == "int":
            assert isinstance(val, int)
            assert low <= val <= high
        else:
            assert low <= val <= high

def test_repair_mask_enforces_bounds():
    # Test case 1: under k_min (e.g., only 2 selected, need at least 5)
    mask_sparse = np.zeros(30, dtype=bool)
    mask_sparse[0] = True
    mask_sparse[1] = True
    repaired_sparse = repair_mask(mask_sparse, k_min=5, k_max=20)
    assert np.sum(repaired_sparse) >= 5

    # Test case 2: over k_max (e.g., 25 selected, maximum is 10)
    mask_dense = np.ones(30, dtype=bool)
    repaired_dense = repair_mask(mask_dense, k_min=5, k_max=10)
    assert np.sum(repaired_dense) <= 10

    # Test case 3: empty mask
    mask_empty = np.zeros(20, dtype=bool)
    repaired_empty = repair_mask(mask_empty, k_min=5, k_max=15)
    assert np.sum(repaired_empty) >= 5
