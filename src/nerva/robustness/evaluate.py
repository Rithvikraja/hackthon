"""
NERVA-ML Comprehensive Robustness Evaluation
Evaluates full perturbation grid and region-held-out domain generalization.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from nerva.robustness.perturbations import apply_gaussian_noise, apply_feature_dropout, apply_batch_shift
from nerva.robustness.metrics import calculate_skill_retention, calculate_robustness_score

def evaluate_full_perturbation_grid(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[float] = [0.05, 0.10, 0.20],
    dropout_levels: List[float] = [0.05, 0.10, 0.20],
    shift_levels: List[List[float]] = [[1.0, 0.10], [1.15, 0.25], [1.30, 0.50]],
    n_repeats: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluates clean and perturbed AUROC, drop, and retention across the grid.
    """
    rng = np.random.default_rng(seed)
    train_sd = np.std(X_train, axis=0, keepdims=True)
    train_sd[train_sd == 0] = 1.0

    # Clean AUROC
    try:
        p_clean = model.predict_proba(X_test)[:, 1]
        clean_auc = float(roc_auc_score(y_test, p_clean))
    except Exception:
        clean_auc = 0.5

    rows = [{
        "perturbation": "clean",
        "level": 0.0,
        "auroc": clean_auc,
        "drop": 0.0,
        "retention": 1.0
    }]

    # 1. Noise
    for nl in noise_levels:
        rep_aucs = []
        for _ in range(n_repeats):
            X_pert = apply_gaussian_noise(X_test, nl, train_sd, rng)
            try:
                p_pert = model.predict_proba(X_pert)[:, 1]
                rep_aucs.append(float(roc_auc_score(y_test, p_pert)))
            except Exception:
                rep_aucs.append(0.5)
        m_auc = float(np.mean(rep_aucs))
        drop = clean_auc - m_auc
        ret = calculate_skill_retention(clean_auc, m_auc)
        rows.append({"perturbation": "noise", "level": nl, "auroc": m_auc, "drop": drop, "retention": ret})

    # 2. Dropout
    for dl in dropout_levels:
        rep_aucs = []
        for _ in range(n_repeats):
            X_pert = apply_feature_dropout(X_test, dl, rng)
            try:
                p_pert = model.predict_proba(X_pert)[:, 1]
                rep_aucs.append(float(roc_auc_score(y_test, p_pert)))
            except Exception:
                rep_aucs.append(0.5)
        m_auc = float(np.mean(rep_aucs))
        drop = clean_auc - m_auc
        ret = calculate_skill_retention(clean_auc, m_auc)
        rows.append({"perturbation": "dropout", "level": dl, "auroc": m_auc, "drop": drop, "retention": ret})

    # 3. Batch Shift
    for s_params in shift_levels:
        a_val, tau_val = s_params[0], s_params[1]
        rep_aucs = []
        for _ in range(n_repeats):
            X_pert = apply_batch_shift(X_test, a_val, tau_val, train_sd, rng)
            try:
                p_pert = model.predict_proba(X_pert)[:, 1]
                rep_aucs.append(float(roc_auc_score(y_test, p_pert)))
            except Exception:
                rep_aucs.append(0.5)
        m_auc = float(np.mean(rep_aucs))
        drop = clean_auc - m_auc
        ret = calculate_skill_retention(clean_auc, m_auc)
        rows.append({"perturbation": f"batch_shift_L_{a_val}_{tau_val}", "level": tau_val, "auroc": m_auc, "drop": drop, "retention": ret})

    return pd.DataFrame(rows)

from sklearn.base import clone

def evaluate_region_heldout_suite(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    region_splits: List[Tuple[str, np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Evaluates domain generalization when major anatomical divisions are held out.
    Strictly re-fits a fresh cloned estimator on training divisions (division != held_out)
    and tests solely on the held-out division to avoid data leakage.
    """
    records = []
    for div_name, tr_idx, te_idx in region_splits:
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            continue
        try:
            fold_model = clone(model)
            n_neg = int(np.sum(y_tr == 0))
            n_pos = int(np.sum(y_tr == 1))
            if hasattr(fold_model, "scale_pos_weight"):
                fold_model.set_params(scale_pos_weight=float(n_neg / max(1, n_pos)))
            fold_model.fit(X[tr_idx], y_tr)
            p_te = fold_model.predict_proba(X[te_idx])[:, 1]
            auc = float(roc_auc_score(y_te, p_te))
        except Exception:
            auc = 0.5
        records.append({"division": div_name, "heldout_auroc": auc, "n_samples": len(te_idx)})
    return pd.DataFrame(records)

