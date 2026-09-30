"""
NERVA-ML Classical Baselines & Region-Identity Null
Evaluates label-permutation control, region-identity rule, Logistic Regression,
Random Forest, and Default XGBoost across outer donor-grouped folds.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import xgboost as xgb
from nerva.ml.features import build_candidate_feature_pool

def evaluate_all_baselines(
    X: Optional[np.ndarray],
    y: np.ndarray,
    structures: np.ndarray,
    outer_splits: List[Tuple[np.ndarray, np.ndarray]],
    z_matrix: Optional[pd.DataFrame] = None,
    disease_genes: Optional[List[str]] = None,
    pool_size: int = 200,
    custom_models: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluates baselines across outer donor-grouped folds.
    If z_matrix and disease_genes are provided, builds candidate feature pool strictly
    from each outer fold's training donors to ensure zero feature-selection leakage.
    Returns:
        DataFrame with AUROC(mean±sd), AUPRC, Balanced Accuracy.
    """
    rng = np.random.default_rng(seed)

    base_models = {
        "label-permutation": LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed),
        "region-identity": None,
        "logistic regression": LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed),
        "random forest": RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced_subsample", random_state=seed),
        "xgboost (default)": xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
    }

    if custom_models:
        base_models.update(custom_models)

    records = []

    for name, clf in base_models.items():
        aurocs = []
        auprcs = []
        bal_accs = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            # Per-fold feature pool construction if z_matrix provided
            if z_matrix is not None and disease_genes is not None:
                train_mask = np.zeros(len(y), dtype=bool)
                train_mask[train_idx] = True
                X_fold, _ = build_candidate_feature_pool(z_matrix, disease_genes, train_mask, pool_size=pool_size)
                X_tr, X_te = X_fold[train_idx], X_fold[test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]

            y_tr, y_te = y[train_idx], y[test_idx]
            struct_tr, struct_te = structures[train_idx], structures[test_idx]

            # Check if test fold has both classes
            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue

            if name == "label-permutation":
                # Shuffle labels on training donors to establish empirical chance level (~0.50)
                y_tr_shuffled = rng.permutation(y_tr)
                clf.fit(X_tr, y_tr_shuffled)
                probs = clf.predict_proba(X_te)[:, 1]
                preds = clf.predict(X_te)
            elif name == "region-identity":
                # Majority vote / mean prevalence per structure from training donors
                struct_to_vote = {}
                for s in np.unique(struct_tr):
                    s_mask = struct_tr == s
                    struct_to_vote[s] = float(np.mean(y_tr[s_mask]))
                global_mean = float(np.mean(y_tr))
                probs = np.array([struct_to_vote.get(s, global_mean) for s in struct_te])
                preds = (probs >= 0.5).astype(int)
            else:
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_te)[:, 1]
                preds = clf.predict(X_te)

            try:
                auc = float(roc_auc_score(y_te, probs))
            except ValueError:
                auc = 0.5

            try:
                ap = float(average_precision_score(y_te, probs))
            except ValueError:
                ap = float(np.mean(y_te))

            bal_acc = float(balanced_accuracy_score(y_te, preds))

            aurocs.append(auc)
            auprcs.append(ap)
            bal_accs.append(bal_acc)

        mean_auc = float(np.mean(aurocs)) if aurocs else 0.5
        sd_auc = float(np.std(aurocs)) if aurocs else 0.0

        records.append({
            "model": name,
            "auroc_mean": mean_auc,
            "auroc_sd": sd_auc,
            "auprc_mean": float(np.mean(auprcs)) if auprcs else 0.0,
            "balanced_accuracy_mean": float(np.mean(bal_accs)) if bal_accs else 0.5,
            "display_str": f"{mean_auc:.2f}±{sd_auc:.2f}"
        })

    return pd.DataFrame(records)

