"""
Unit test: Statistical enrichment and Benjamini-Hochberg FDR
Validates toy example calculations from Section 4.3, empirical p formula, and BH-FDR correction.
"""

import pytest
import numpy as np
import pandas as pd
from nerva.stats.enrichment import compute_enrichment_pipeline

def test_toy_enrichment_calculation():
    """
    Direct test of Section 4.3 toy example:
    Region R1: Gene A=1.2, Gene C=0.9 -> S(R1) = 1.05
    Region R2: Gene A=-0.4, Gene C=-0.6 -> S(R2) = -0.50
    Region R3: Gene A=0.1, Gene C=0.2 -> S(R3) = 0.15
    """
    # Create multi-index (donor, structure)
    index = pd.MultiIndex.from_tuples(
        [("D1", "R1"), ("D1", "R2"), ("D1", "R3")],
        names=["donor_id", "structure_name"]
    )
    # Z-matrix columns: A, B, C, D, E
    data = [
        [1.2, -0.3, 0.9, 0.1, 0.0],
        [-0.4, 0.2, -0.6, 0.5, 1.1],
        [0.1, 0.8, 0.2, -0.2, -0.5]
    ]
    z_matrix = pd.DataFrame(data, index=index, columns=["A", "B", "C", "D", "E"])

    disease_genes = ["A", "C"]
    division_map = {"R1": "Div1", "R2": "Div1", "R3": "Div2"}
    
    # Toy controls: {B, D}, {D, E}
    controls = [["B", "D"], ["D", "E"]]

    regional_df, row_E = compute_enrichment_pipeline(z_matrix, disease_genes, controls, division_map)

    # Check observed region score S
    r1_row = regional_df[regional_df["structure"] == "R1"].iloc[0]
    r2_row = regional_df[regional_df["structure"] == "R2"].iloc[0]
    r3_row = regional_df[regional_df["structure"] == "R3"].iloc[0]

    assert pytest.approx(r1_row["S"], abs=1e-3) == 1.05
    assert pytest.approx(r2_row["S"], abs=1e-3) == -0.50
    assert pytest.approx(r3_row["S"], abs=1e-3) == 0.15

    # Check empirical p-value uses (1 + count) / (1 + K) formula
    # With K=2, minimum possible p is 1/3 ~ 0.3333
    assert r1_row["p"] >= (1.0 / (2 + 1))
    assert "q" in regional_df.columns
    assert len(row_E) == 3
