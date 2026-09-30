"""
Unit test: NSGA-II core non-dominated sorting
Compares a simple hand-written 30-line non-domination check with pymoo's implementation on random points.
"""

import pytest
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def simple_non_dominated_front(points: np.ndarray) -> list:
    """
    Hand-written non-domination check for minimization:
    Point A dominates B if A <= B for all objectives and A < B for at least one.
    Front 1 comprises points that are not dominated by any other point.
    """
    n = len(points)
    front_1 = []
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated = True
                break
        if not is_dominated:
            front_1.append(i)
    return sorted(front_1)

def test_hand_written_vs_pymoo_nds():
    np.random.seed(42)
    # 30 random 3-objective candidates
    points = np.random.rand(30, 3)

    # 1. Hand-written
    expected_front_1 = simple_non_dominated_front(points)

    # 2. Pymoo
    nds = NonDominatedSorting()
    fronts = nds.do(points)
    pymoo_front_1 = sorted(list(fronts[0]))

    # Exact agreement
    assert expected_front_1 == pymoo_front_1
    assert len(expected_front_1) > 0
