"""
NERVA-ML Pymoo Optimization Problem Evaluator
Evaluates candidate solutions across inner donor folds using XGBoost with:
- All 10 decoded hyperparameters + noise-augmented training (aug_sigma)
- Robustness objective computed across noise, dropout, and batch shift using Common Random Numbers (CRN)
- Chromosome hashing cache and strict seed determinism.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import xgboost as xgb
from pymoo.core.problem import Problem
from sklearn.metrics import roc_auc_score
from nerva.optimization.chromosome import decode_chromosome, repair_mask
from nerva.robustness.metrics import calculate_skill_retention

class NervaProblem(Problem):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        inner_splits: List[Tuple[np.ndarray, np.ndarray]],
        pool_size: int,
        k_min: int = 5,
        k_max: int = 60,
        seed: int = 42,
        n_jobs: int = 1
    ):
        n_var = pool_size + 10
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, xl=0.0, xu=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.inner_splits = inner_splits
        self.pool_size = pool_size
        self.k_min = k_min
        self.k_max = k_max
        self.seed = seed
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)
        self.cache: Dict[str, Tuple[float, float, float]] = {}

        # Precompute per-fold training feature standard deviations and Common Random Numbers (CRN)
        self.fold_crns = []
        for tr_idx, val_idx in self.inner_splits:
            X_tr = self.X_train[tr_idx]
            X_va = self.X_train[val_idx]
            train_sd = np.std(X_tr, axis=0)
            train_sd[train_sd == 0] = 1.0

            # CRN for this fold
            crn_noise = self.rng.normal(0, 1.0, size=X_va.shape)
            crn_perm = self.rng.permutation(pool_size)
            crn_shift = self.rng.normal(0, 1.0, size=(1, pool_size))

            self.fold_crns.append({
                "train_sd": train_sd,
                "crn_noise": crn_noise,
                "crn_perm": crn_perm,
                "crn_shift": crn_shift
            })

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = len(x)
        F = np.zeros((n_samples, 3))

        for i in range(n_samples):
            mask, hp = decode_chromosome(x[i], self.pool_size)
            mask = repair_mask(mask, self.k_min, self.k_max, rng=self.rng)
            selected_idx = np.where(mask)[0]
            k = len(selected_idx)

            if k == 0:
                F[i] = [1.0, 1.0, 1.0]
                continue

            # Cache key including all hyperparameters
            cache_key = (
                f"{tuple(selected_idx)}_"
                f"{hp['learning_rate']:.3f}_{hp['n_estimators']}_{hp['max_depth']}_"
                f"{hp['subsample']:.2f}_{hp['colsample_bytree']:.2f}_{hp['min_child_weight']}_"
                f"{hp['reg_alpha']:.3f}_{hp['reg_lambda']:.3f}_{hp['gamma']:.2f}_{hp['aug_sigma']:.3f}"
            )
            if cache_key in self.cache:
                F[i] = self.cache[cache_key]
                continue

            clean_aucs = []
            retentions = []

            for fold_i, (tr_idx, val_idx) in enumerate(self.inner_splits):
                X_tr = self.X_train[tr_idx][:, selected_idx]
                y_tr = self.y_train[tr_idx]
                X_va = self.X_train[val_idx][:, selected_idx]
                y_va = self.y_train[val_idx]

                if len(np.unique(y_va)) < 2:
                    continue

                crn = self.fold_crns[fold_i]
                s_g = crn["train_sd"][selected_idx]

                # Vicinal risk minimization / Noise-augmented training (Section 9.5)
                aug_sigma = hp.get("aug_sigma", 0.0)
                if aug_sigma > 0:
                    noise_aug = self.rng.normal(0, aug_sigma * s_g, size=X_tr.shape)
                    X_fit = np.vstack([X_tr, X_tr + noise_aug])
                    y_fit = np.concatenate([y_tr, y_tr])
                else:
                    X_fit = X_tr
                    y_fit = y_tr

                n_neg = int(np.sum(y_fit == 0))
                n_pos = int(np.sum(y_fit == 1))
                spw = float(n_neg / max(1, n_pos))

                try:
                    clf = xgb.XGBClassifier(
                        learning_rate=hp["learning_rate"],
                        n_estimators=min(hp["n_estimators"], 100),  # bounded for fast inner evaluation
                        max_depth=hp["max_depth"],
                        subsample=hp["subsample"],
                        colsample_bytree=hp["colsample_bytree"],
                        min_child_weight=hp["min_child_weight"],
                        reg_alpha=hp["reg_alpha"],
                        reg_lambda=hp["reg_lambda"],
                        gamma=hp["gamma"],
                        scale_pos_weight=spw,
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=self.seed,
                        n_jobs=1
                    )
                    clf.fit(X_fit, y_fit)
                    probs_clean = clf.predict_proba(X_va)[:, 1]
                    auc_clean = float(roc_auc_score(y_va, probs_clean))
                except Exception:
                    auc_clean = 0.5

                clean_aucs.append(auc_clean)

                # Robustness stress tests with Common Random Numbers
                # 1. Gaussian noise (alpha = 0.10)
                X_noise = X_va + 0.10 * s_g * crn["crn_noise"][:, selected_idx]
                try:
                    probs_noise = clf.predict_proba(X_noise)[:, 1]
                    auc_noise = float(roc_auc_score(y_va, probs_noise))
                except Exception:
                    auc_noise = 0.5
                r_noise = calculate_skill_retention(auc_clean, auc_noise)

                # 2. Feature dropout (q = 0.10)
                X_drop = X_va.copy()
                n_drop = max(1, int(round(0.10 * k)))
                drop_cols = [j for j in range(k) if selected_idx[j] in crn["crn_perm"][:n_drop]]
                if not drop_cols:
                    drop_cols = [0]
                X_drop[:, drop_cols] = 0.0
                try:
                    probs_drop = clf.predict_proba(X_drop)[:, 1]
                    auc_drop = float(roc_auc_score(y_va, probs_drop))
                except Exception:
                    auc_drop = 0.5
                r_drop = calculate_skill_retention(auc_clean, auc_drop)

                # 3. Affine batch shift (Level 2: a = 1.15, tau = 0.25)
                X_shift = 1.15 * X_va + 0.25 * s_g * crn["crn_shift"][:, selected_idx]
                try:
                    probs_shift = clf.predict_proba(X_shift)[:, 1]
                    auc_shift = float(roc_auc_score(y_va, probs_shift))
                except Exception:
                    auc_shift = 0.5
                r_shift = calculate_skill_retention(auc_clean, auc_shift)

                mean_r_fold = float(np.mean([r_noise, r_drop, r_shift]))
                retentions.append(mean_r_fold)

            mean_auc = float(np.mean(clean_aucs)) if clean_aucs else 0.5
            mean_ret = float(np.mean(retentions)) if retentions else 0.0

            # Objectives (minimization):
            f1 = 1.0 - mean_auc
            f2 = 1.0 - mean_ret
            f3 = k / self.k_max

            F[i] = [f1, f2, f3]
            self.cache[cache_key] = (f1, f2, f3)

        out["F"] = F
