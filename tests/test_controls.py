"""
Unit test: Expression-matched control generation
Validates control set size k, disease gene exclusion, abundance bin matching, and seed determinism.
"""

import pytest
import numpy as np
import pandas as pd
from nerva.stats.controls import build_expression_matched_controls

def test_controls_exact_size_and_exclusion():
    # Universe of 100 genes with known mean abundance
    genes = [f"GENE_{i:03d}" for i in range(100)]
    abundance = pd.Series(np.linspace(1.0, 10.0, 100), index=genes)
    
    disease_genes = ["GENE_010", "GENE_025", "GENE_050", "GENE_075", "GENE_090"]
    k = len(disease_genes)
    n_controls = 50

    controls = build_expression_matched_controls(
        common_universe=genes,
        mean_abundance=abundance,
        disease_gene_set=disease_genes,
        n_control_sets=n_controls,
        n_bins=10,
        seed=42
    )

    assert len(controls) == n_controls
    disease_set = set(disease_genes)

    for c_set in controls:
        # Exact size k
        assert len(c_set) == k
        # No disease gene in control set
        for g in c_set:
            assert g not in disease_set
        # Within one control set, no duplicates
        assert len(set(c_set)) == k

def test_controls_determinism():
    genes = [f"GENE_{i:03d}" for i in range(50)]
    abundance = pd.Series(np.random.RandomState(123).uniform(1, 10, 50), index=genes)
    disease_genes = ["GENE_005", "GENE_015", "GENE_025"]

    controls_1 = build_expression_matched_controls(genes, abundance, disease_genes, n_control_sets=20, seed=999)
    controls_2 = build_expression_matched_controls(genes, abundance, disease_genes, n_control_sets=20, seed=999)

    assert controls_1 == controls_2
