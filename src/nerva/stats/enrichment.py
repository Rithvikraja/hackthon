"""
NERVA-ML Statistical Enrichment Engine
Computes raw scores S, matched null distributions, empirical p-values,
Benjamini-Hochberg FDR q-values, donor consistency metrics, and
literature reference-region validation (Option C).
High-performance vectorized implementation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def compute_enrichment_pipeline(
    z_matrix: pd.DataFrame,
    disease_gene_set: List[str],
    control_sets: List[List[str]],
    division_map: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes disease gene expression enrichment across structures.
    Returns:
        regional_df: Summary per structure with S, E, p, q, and donor consistency
        row_level_E: Series of row-level E z-scores for machine learning labels
    """
    Z = z_matrix.values
    gene_to_col = {g: i for i, g in enumerate(z_matrix.columns)}
    
    valid_d_indices = [gene_to_col[g] for g in disease_gene_set if g in gene_to_col]
    if not valid_d_indices:
        valid_d_indices = [0]

    # 1. Row-level observed score
    row_S = Z[:, valid_d_indices].mean(axis=1)

    # 2. Vectorized null distribution for each row
    n_controls = len(control_sets)
    null_scores = np.zeros((len(z_matrix), n_controls))
    
    for c_idx in range(n_controls):
        c_indices = [gene_to_col[g] for g in control_sets[c_idx] if g in gene_to_col]
        if c_indices:
            null_scores[:, c_idx] = Z[:, c_indices].mean(axis=1)

    null_mean = null_scores.mean(axis=1)
    null_std = null_scores.std(axis=1)
    null_std[null_std == 0] = 1e-6

    # Row-level enrichment Z-score
    row_level_E = (row_S - null_mean) / null_std
    row_E_series = pd.Series(row_level_E, index=z_matrix.index)

    # 3. Structure-level aggregation using fast integer indices
    struct_indices_map = {}
    for row_i, idx in enumerate(z_matrix.index):
        s_name = idx[1] if isinstance(idx, tuple) else idx
        struct_indices_map.setdefault(s_name, []).append(row_i)

    records = []
    for struct_name, row_indices in struct_indices_map.items():
        s_obs = float(np.mean(row_S[row_indices]))
        sub_null_means = null_scores[row_indices].mean(axis=0)
        n_mean = float(np.mean(sub_null_means))
        n_std = float(np.std(sub_null_means))
        if n_std == 0: 
            n_std = 1e-6

        E_val = (s_obs - n_mean) / n_std
        p_val = float((np.sum(sub_null_means >= s_obs) + 1) / (n_controls + 1))

        # Donor consistency: fraction of donors sampling this structure with E > 0
        donor_E = row_level_E[row_indices]
        n_donors_sampling = len(donor_E)
        n_positive = int(np.sum(donor_E > 0))
        consistency_str = f"{n_positive}/{n_donors_sampling}"

        records.append({
            "structure": struct_name,
            "division": division_map.get(struct_name, "Other"),
            "S": float(s_obs),
            "E": float(E_val),
            "p": float(p_val),
            "donor_consistency": consistency_str
        })

    df = pd.DataFrame(records)

    # Benjamini-Hochberg FDR correction
    reject, q_values, _, _ = multipletests(df["p"], method="fdr_bh")
    df["q"] = q_values

    df = df.sort_values(by="E", ascending=False).reset_index(drop=True)
    return df, row_E_series


def validate_reference_regions(
    regional_df: pd.DataFrame,
    reference_structures: List[str],
    n_permutations: int = 1000,
    seed: int = 42
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Validates regional DGEE rankings against pre-registered literature reference regions (Option C).
    Returns:
        obs_mean_percentile: Mean percentile rank (0..1) of literature reference regions in DGEE
        permutation_p: Empirical p-value comparing against randomly sampled region sets
        matched_details: List of per-reference matching details
    """
    rng = np.random.default_rng(seed)
    all_structs = list(regional_df["structure"])
    n_structs = len(all_structs)

    # Percentile ranking (0.0 = lowest E, 1.0 = highest E)
    sorted_df = regional_df.sort_values(by="E", ascending=True).reset_index(drop=True)
    struct_to_pct = {row["structure"]: (idx / max(1, n_structs - 1)) for idx, row in sorted_df.iterrows()}

    matched_refs = []
    matched_pcts = []

    for ref in reference_structures:
        matched_name = None
        for s in all_structs:
            s_clean = s.replace("_", " ").lower()
            ref_clean = ref.replace("_", " ").lower()
            if ref_clean in s_clean or s_clean in ref_clean:
                matched_name = s
                break
        if matched_name:
            pct = struct_to_pct[matched_name]
            matched_pcts.append(pct)
            matched_refs.append({
                "reference": ref,
                "matched_structure": matched_name,
                "percentile": float(pct)
            })

    if not matched_pcts:
        return 0.50, 1.0, []

    obs_mean_pct = float(np.mean(matched_pcts))
    k_refs = len(matched_pcts)

    # Permutation null: draw k_refs random structures 1000 times
    all_pcts = np.array(list(struct_to_pct.values()))
    null_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        sample_pcts = rng.choice(all_pcts, size=min(k_refs, len(all_pcts)), replace=False)
        null_means[i] = np.mean(sample_pcts)

    p_val = float((np.sum(null_means >= obs_mean_pct) + 1) / (n_permutations + 1))
    return obs_mean_pct, p_val, matched_refs
