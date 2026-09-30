"""
NERVA-ML Custom Genetic Operators for Pymoo
Mixed binary-continuous Sampling, Crossover, and Mutation.
Uses explicit NumPy Generator for strict seed determinism.
"""

from typing import Optional
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from nerva.optimization.chromosome import repair_mask

class MixedSampling(Sampling):
    def __init__(self, pool_size: int, n_params: int = 10, k_min: int = 5, k_max: int = 40, seed: int = 42):
        super().__init__()
        self.pool_size = pool_size
        self.n_params = n_params
        self.k_min = k_min
        self.k_max = k_max
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, self.pool_size + self.n_params))
        for i in range(n_samples):
            # Sparse gene initialization
            k0 = self.rng.integers(self.k_min, self.k_max + 1)
            chosen = self.rng.choice(self.pool_size, size=k0, replace=False)
            X[i, chosen] = 1.0
            # Hyperparameter genes in [0, 1]
            X[i, self.pool_size:] = self.rng.random(self.n_params)
        return X

class MixedCrossover(Crossover):
    def __init__(self, pool_size: int, prob: float = 0.9, k_min: int = 5, k_max: int = 60, seed: int = 42):
        super().__init__(2, 2)
        self.pool_size = pool_size
        self.prob = prob
        self.k_min = k_min
        self.k_max = k_max
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Y = np.copy(X)

        for k in range(n_matings):
            if self.rng.random() < self.prob:
                p1, p2 = X[0, k], X[1, k]
                # Uniform crossover on mask
                mask_swap = self.rng.random(self.pool_size) < 0.5
                c1_mask = np.where(mask_swap, p1[:self.pool_size], p2[:self.pool_size])
                c2_mask = np.where(mask_swap, p2[:self.pool_size], p1[:self.pool_size])

                # Simulated binary crossover (SBX) on hyperparameters
                p1_h = p1[self.pool_size:]
                p2_h = p2[self.pool_size:]
                u = self.rng.random(len(p1_h))
                eta = 15.0
                beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
                c1_h = np.clip(0.5 * ((1 + beta) * p1_h + (1 - beta) * p2_h), 0.0, 1.0)
                c2_h = np.clip(0.5 * ((1 - beta) * p1_h + (1 + beta) * p2_h), 0.0, 1.0)

                Y[0, k, :self.pool_size] = repair_mask(c1_mask > 0.5, self.k_min, self.k_max, rng=self.rng)
                Y[0, k, self.pool_size:] = c1_h
                Y[1, k, :self.pool_size] = repair_mask(c2_mask > 0.5, self.k_min, self.k_max, rng=self.rng)
                Y[1, k, self.pool_size:] = c2_h

        return Y

class MixedMutation(Mutation):
    def __init__(self, pool_size: int, k_min: int = 5, k_max: int = 60, seed: int = 42):
        super().__init__()
        self.pool_size = pool_size
        self.k_min = k_min
        self.k_max = k_max
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)
        for i in range(len(Y)):
            mask = Y[i, :self.pool_size] > 0.5
            n_ops = 1 + self.rng.poisson(1)

            for _ in range(n_ops):
                op = self.rng.random()
                selected = np.where(mask)[0]
                unselected = np.where(~mask)[0]

                if op < 0.40 and len(unselected) > 0:
                    mask[self.rng.choice(unselected)] = True
                elif op < 0.80 and len(selected) > 0:
                    mask[self.rng.choice(selected)] = False
                elif len(selected) > 0 and len(unselected) > 0:
                    mask[self.rng.choice(selected)] = False
                    mask[self.rng.choice(unselected)] = True

            Y[i, :self.pool_size] = repair_mask(mask, self.k_min, self.k_max, rng=self.rng)

            # Polynomial mutation on hyperparameters
            params = Y[i, self.pool_size:]
            mut_mask = self.rng.random(len(params)) < 0.10
            delta = self.rng.normal(0, 0.1, size=len(params))
            params[mut_mask] = np.clip(params[mut_mask] + delta[mut_mask], 0.0, 1.0)
            Y[i, self.pool_size:] = params

        return Y
