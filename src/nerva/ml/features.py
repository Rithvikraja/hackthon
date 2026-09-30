"""
NERVA-ML Feature Matrix Construction & Pool Pruning
Strictly excludes disease genes to prevent target circularity,
selects candidate pool using training donors, and prunes collinear duplicates.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

def build_candidate_feature_pool(
    z_matrix: pd.DataFrame,
    disease_gene_set: List[str],
    train_mask: np.ndarray,
    pool_size: int = 300,
    corr_threshold: float = 0.95
) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs feature matrix X and candidate gene list from training donors:
    1. Removes all disease genes
    2. Selects top M genes by variance / differential stability on training donors
    3. Prunes redundant genes with |r| > corr_threshold
    """
    d_set = set(disease_gene_set)
    all_genes = [g for g in z_matrix.columns if g not in d_set]

    # Subset training rows
    train_df = z_matrix.iloc[train_mask][all_genes]

    # Rank by training variance (unsupervised)
    variances = train_df.var(axis=0)
    top_candidates = list(variances.nlargest(min(pool_size * 2, len(all_genes))).index)

    # Correlation pruning
    corr_mat = train_df[top_candidates].corr().abs().values
    keep_indices = []
    dropped = set()

    for i in range(len(top_candidates)):
        if i in dropped:
            continue
        keep_indices.append(i)
        if len(keep_indices) >= pool_size:
            break
        # Mark collinear genes
        collinear = np.where(corr_mat[i, :] > corr_threshold)[0]
        for c_idx in collinear:
            if c_idx > i:
                dropped.add(c_idx)

    selected_genes = [top_candidates[i] for i in keep_indices][:pool_size]
    X = z_matrix[selected_genes].values

    return X, selected_genes
