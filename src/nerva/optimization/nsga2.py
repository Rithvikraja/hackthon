"""
NERVA-ML NSGA-II Multi-Objective Optimization Runner
Executes NSGA-II via Pymoo with custom mixed chromosome operators.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from nerva.optimization.evaluator import NervaProblem
from nerva.optimization.operators import MixedSampling, MixedCrossover, MixedMutation
from nerva.optimization.chromosome import decode_chromosome, repair_mask
from nerva.optimization.selection import select_knee_and_1se_solutions

class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        best_auc = 1.0 - np.min(F[:, 0])
        best_rob = 1.0 - np.min(F[:, 1])
        min_genes = np.min(F[:, 2]) * 60
        self.history.append({
            "generation": algorithm.n_gen,
            "best_auroc": float(best_auc),
            "best_robustness": float(best_rob),
            "min_genes": int(round(min_genes))
        })

def run_nsga2_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    pool_size: int,
    pop_size: int = 32,
    generations: int = 20,
    seed: int = 42,
    n_jobs: int = 4
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, int, int]:
    """
    Executes NSGA-II evolutionary search.
    Returns:
        pareto_solutions: List of non-dominated candidate dicts
        history_df: DataFrame of convergence metrics across generations
        knee_idx: Index of knee trade-off solution
        one_se_idx: Index of 1-SE compact solution
    """
    rng = np.random.default_rng(seed)

    problem = NervaProblem(
        X_train=X_train,
        y_train=y_train,
        inner_splits=inner_splits,
        pool_size=pool_size,
        k_min=5,
        k_max=60,
        seed=seed,
        n_jobs=n_jobs
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedSampling(pool_size=pool_size, n_params=10, k_min=5, k_max=40, seed=seed),
        crossover=MixedCrossover(pool_size=pool_size, prob=0.9, k_min=5, k_max=60, seed=seed + 1),
        mutation=MixedMutation(pool_size=pool_size, k_min=5, k_max=60, seed=seed + 2),
        eliminate_duplicates=True
    )

    hist_callback = HistoryCallback()

    res = minimize(
        problem,
        algorithm,
        ("n_gen", generations),
        seed=seed,
        callback=hist_callback,
        verbose=False
    )

    history_df = pd.DataFrame(hist_callback.history)

    # Decode Pareto solutions
    pareto_solutions = []
    opt_X = res.opt.get("X")
    opt_F = res.opt.get("F")

    for i in range(len(opt_X)):
        mask, hp = decode_chromosome(opt_X[i], pool_size)
        mask = repair_mask(mask, 5, 60, rng=rng)
        auc_val = 1.0 - opt_F[i, 0]
        rob_val = 1.0 - opt_F[i, 1]
        k_val = int(np.sum(mask))

        pareto_solutions.append({
            "index": i,
            "mask": mask,
            "hyperparameters": hp,
            "auroc": float(auc_val),
            "robustness": float(rob_val),
            "n_genes": k_val,
            "selected_indices": list(np.where(mask)[0])
        })

    aurocs = np.array([s["auroc"] for s in pareto_solutions])
    robs = np.array([s["robustness"] for s in pareto_solutions])
    genes = np.array([s["n_genes"] for s in pareto_solutions])

    knee_idx, one_se_idx = select_knee_and_1se_solutions(aurocs, robs, genes)

    return pareto_solutions, history_df, knee_idx, one_se_idx

