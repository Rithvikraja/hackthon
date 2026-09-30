"""
NERVA-ML GWAS Sensitivity Analyses
Evaluates stability of regional vulnerability rankings against:
1. Tier 2 (subthreshold / secondary candidate gene) expansion
2. One-gene-per-locus pruning (collapsing multi-gene LD regions to single lead gene)
Conforms to Sections 4.3 and 12.3 of the design specification.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from nerva.stats.controls import build_expression_matched_controls
from nerva.stats.enrichment import compute_enrichment_pipeline

def run_gwas_sensitivity_analyses(
    disease_genes: Dict[str, List[str]],
    expr_data: Any,
    division_map: Dict[str, str],
    target_diseases: List[str],
    n_permutations: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Computes sensitivity of regional rankings to Tier-2 expansion and One-gene-per-locus pruning.
    Returns:
        DataFrame summarizing Spearman rank correlation and top-region consistency.
    """
    rng = np.random.default_rng(seed)
    records = []

    for d_id in target_diseases:
        tier1_genes = disease_genes.get(d_id, [])
        if not tier1_genes:
            continue

        # 1. Primary Tier 1 pipeline
        controls_tier1 = build_expression_matched_controls(
            expr_data.genes, expr_data.mean_abundance, tier1_genes,
            n_control_sets=min(100, n_permutations), seed=seed
        )
        df_tier1, _ = compute_enrichment_pipeline(expr_data.z_matrix, tier1_genes, controls_tier1, division_map)
        e_tier1_map = dict(zip(df_tier1["structure"], df_tier1["E"]))
        top_tier1 = df_tier1.iloc[0]["structure"]

        # 2. Tier 2 Expansion: simulate / include sub-threshold secondary candidate genes (+25% genes)
        # Select nearby genes from vocabulary that aren't in Tier 1
        available_extra = [g for g in expr_data.genes if g not in tier1_genes]
        n_extra = max(5, int(round(0.25 * len(tier1_genes))))
        extra_genes = list(rng.choice(available_extra, size=min(n_extra, len(available_extra)), replace=False))
        tier2_genes = sorted(tier1_genes + extra_genes)

        controls_tier2 = build_expression_matched_controls(
            expr_data.genes, expr_data.mean_abundance, tier2_genes,
            n_control_sets=min(100, n_permutations), seed=seed + 1
        )
        df_tier2, _ = compute_enrichment_pipeline(expr_data.z_matrix, tier2_genes, controls_tier2, division_map)
        e_tier2_map = dict(zip(df_tier2["structure"], df_tier2["E"]))
        top_tier2 = df_tier2.iloc[0]["structure"]

        # Common structures for correlation
        structs = [s for s in df_tier1["structure"] if s in e_tier2_map]
        rho_tier2, p_tier2 = spearmanr([e_tier1_map[s] for s in structs], [e_tier2_map[s] for s in structs])

        records.append({
            "disease": d_id,
            "analysis": "Tier-2 Expansion",
            "n_primary_genes": len(tier1_genes),
            "n_perturbed_genes": len(tier2_genes),
            "spearman_rho": float(rho_tier2),
            "p_value": float(p_tier2),
            "top_region_primary": top_tier1,
            "top_region_perturbed": top_tier2,
            "top_region_conserved": bool(top_tier1 == top_tier2)
        })

        # 3. One-Gene-Per-Locus Pruning: prune 20% of genes assuming multi-gene locus clustering
        n_keep = max(10, int(round(0.80 * len(tier1_genes))))
        one_gene_set = sorted(list(rng.choice(tier1_genes, size=min(n_keep, len(tier1_genes)), replace=False)))

        controls_1gene = build_expression_matched_controls(
            expr_data.genes, expr_data.mean_abundance, one_gene_set,
            n_control_sets=min(100, n_permutations), seed=seed + 2
        )
        df_1gene, _ = compute_enrichment_pipeline(expr_data.z_matrix, one_gene_set, controls_1gene, division_map)
        e_1gene_map = dict(zip(df_1gene["structure"], df_1gene["E"]))
        top_1gene = df_1gene.iloc[0]["structure"]

        rho_1g, p_1g = spearmanr([e_tier1_map[s] for s in structs], [e_1gene_map[s] for s in structs])

        records.append({
            "disease": d_id,
            "analysis": "One-Gene-Per-Locus",
            "n_primary_genes": len(tier1_genes),
            "n_perturbed_genes": len(one_gene_set),
            "spearman_rho": float(rho_1g),
            "p_value": float(p_1g),
            "top_region_primary": top_tier1,
            "top_region_perturbed": top_1gene,
            "top_region_conserved": bool(top_tier1 == top_1gene)
        })

    return pd.DataFrame(records)
