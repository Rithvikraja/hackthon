"""
NERVA-ML Deterministic Seed Management
Controls master seed and spawns independent child RNG streams.
"""

import random
import os
import numpy as np

def seed_everything(master_seed: int = 42) -> np.random.SeedSequence:
    """Sets master seed for Python and hash seeds, returns master SeedSequence."""
    random.seed(master_seed)
    os.environ["PYTHONHASHSEED"] = str(master_seed)
    np.random.seed(master_seed)
    return np.random.SeedSequence(master_seed)

def spawn_child_rng(seed_seq: np.random.SeedSequence, n: int = 1):
    """Spawns n independent numpy Generator objects."""
    child_seeds = seed_seq.spawn(n)
    if n == 1:
        return np.random.default_rng(child_seeds[0])
    return [np.random.default_rng(s) for s in child_seeds]
