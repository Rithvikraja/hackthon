"""
NERVA-ML Cross-Disease Comparison
Computes profile correlations, signature Jaccard overlaps, and shared gene lists.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

def compare_diseases_pipeline(
    regional_dfs: Dict[str, pd.DataFrame],
    top_genes_per_disease: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Computes pairwise regional correlation and gene signature Jaccard index.
    Returns:
        comparison_df: Pairwise metrics table
    """
    diseases = list(regional_dfs.keys())
    records = []

    for i in range(len(diseases)):
        for j in range(i + 1, len(diseases)):
            d1, d2 = diseases[i], diseases[j]
            df1, df2 = regional_dfs[d1], regional_dfs[d2]

            # Merge on structure
            merged = pd.merge(df1, df2, on="structure", suffixes=("_1", "_2"))
            corr = float(merged["E_1"].corr(merged["E_2"]))

            # Jaccard index of top genes
            g1 = set(top_genes_per_disease.get(d1, []))
            g2 = set(top_genes_per_disease.get(d2, []))
            intersection = g1.intersection(g2)
            union = g1.union(g2)
            jaccard = float(len(intersection) / len(union)) if union else 0.0

            records.append({
                "disease_1": d1,
                "disease_2": d2,
                "regional_profile_correlation": corr,
                "gene_jaccard_overlap": jaccard,
                "shared_genes_count": len(intersection),
                "shared_genes": list(intersection)
            })

    return pd.DataFrame(records)
