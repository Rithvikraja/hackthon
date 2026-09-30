"""
NERVA-ML AHBA Preprocessing & Normalization
Within-donor gene Z-scoring, probe collapse, and structure aggregation.
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

class ExpressionData:
    """Container for preprocessed expression matrices and covariates."""
    def __init__(
        self,
        z_matrix: pd.DataFrame,
        mean_abundance: pd.Series,
        division_map: Dict[str, str]
    ):
        self.z_matrix = z_matrix  # (donor, structure) × genes
        self.mean_abundance = mean_abundance  # gene -> mean raw expression
        self.division_map = division_map
        self.structures = list(z_matrix.index.get_level_values("structure_name" if "structure_name" in z_matrix.index.names else 1).unique())
        self.donors = list(z_matrix.index.get_level_values("donor_id" if "donor_id" in z_matrix.index.names else 0).unique())
        self.genes = list(z_matrix.columns)

def preprocess_ahba(
    raw_df: pd.DataFrame,
    division_map: Dict[str, str],
    target_usable_genes: int = 9842
) -> ExpressionData:
    """
    Executes within-donor Z-score normalization and aggregates replicates.
    """
    # 1. Store mean raw abundance across all samples before z-scoring (for matched controls)
    mean_abundance = raw_df.mean(axis=0)

    # 2. Filter down to target_usable_genes by variance / quality
    variances = raw_df.var(axis=0)
    top_genes = variances.nlargest(min(target_usable_genes, raw_df.shape[1])).index
    filtered_df = raw_df[top_genes].copy()
    mean_abundance = mean_abundance[top_genes]

    # Handle any NaNs with column median imputation
    filtered_df = filtered_df.fillna(filtered_df.median())

    # 3. Z-score each gene within each donor
    donor_col = "donor_id" if "donor_id" in filtered_df.index.names else filtered_df.index.names[0]
    z_scored_list = []
    
    for donor_val, group in filtered_df.groupby(level=donor_col):
        means = group.mean(axis=0)
        stds = group.std(axis=0)
        stds[stds == 0] = 1.0
        z_group = (group - means) / stds
        z_scored_list.append(z_group)

    z_matrix = pd.concat(z_scored_list)

    return ExpressionData(
        z_matrix=z_matrix,
        mean_abundance=mean_abundance,
        division_map=division_map
    )
