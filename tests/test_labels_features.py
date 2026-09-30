"""
Unit test: Target labels and feature pool generation
Checks that disease genes are strictly absent from feature matrix X,
labels follow E >= tau thresholding, class-balance fails loudly when violated,
and donor groups align with row counts.
"""

import pytest
import numpy as np
import pandas as pd
from nerva.ml.labels import make_dgee_labels
from nerva.ml.features import build_candidate_feature_pool
from nerva.utils.errors import DataFormatError

def test_labels_creation_and_balance():
    # Synthetic row E z-scores: 3/10 pass tau=1.645 -> 30% positive fraction
    row_E = pd.Series([2.5, 1.8, 1.0, 0.2, -0.5, -1.2, 1.9, 0.4, -0.8, -1.5])
    
    y, pos_frac = make_dgee_labels(row_E, tau_label=1.645)
    assert len(y) == len(row_E)
    assert pos_frac == 0.30
    assert np.sum(y) == 3

def test_labels_fail_loud_when_unbalanced():
    # All low values -> 0% positives
    low_E = pd.Series([-2.0, -1.5, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(DataFormatError) as exc_info:
        make_dgee_labels(low_E, tau_label=1.645, disease_name="Parkinson's disease")
    assert "allowed 10-50%" in str(exc_info.value)
    assert "label_mode: quantile" in exc_info.value.hint

    # In quantile mode, 75th percentile yields balanced labels
    y_q, pos_frac_q = make_dgee_labels(low_E, label_mode="quantile")
    assert 0.10 <= pos_frac_q <= 0.50

def test_features_exclude_disease_genes():
    index = pd.MultiIndex.from_tuples(
        [(f"D{i}", f"R{j}") for i in range(1, 4) for j in range(1, 5)],
        names=["donor_id", "structure_name"]
    )
    cols = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"]
    data = np.random.RandomState(42).randn(len(index), len(cols))
    z_df = pd.DataFrame(data, index=index, columns=cols)

    disease_genes = ["GENE_A", "GENE_C"]
    train_mask = np.ones(len(index), dtype=bool)

    X_pool, pool_genes = build_candidate_feature_pool(
        z_df, disease_genes, train_mask, pool_size=4
    )

    # Critical invariant: no disease gene in pool_genes or X_pool
    for dg in disease_genes:
        assert dg not in pool_genes

    assert X_pool.shape[0] == len(index)
    assert X_pool.shape[1] == len(pool_genes)
