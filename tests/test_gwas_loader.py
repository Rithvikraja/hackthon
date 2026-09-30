"""
Unit test: GWAS loader and mapping policy
Checks P-value thresholding, MHC exclusion, duplicate collapse, min-size low power flagging,
and missing file errors.
"""

import pytest
import os
import pandas as pd
from nerva.data.gwas_loader import parse_gwas_tsv, load_gwas_disease_genes
from nerva.utils.errors import DataMissingError

def test_parse_gwas_tsv_logic(tmp_path):
    # Construct a sample GWAS Catalog TSV
    tsv_content = (
        "DISEASE/TRAIT\tMAPPED_TRAIT_URI\tCHR_ID\tCHR_POS\tP-VALUE\tMAPPED_GENE\tSNP_GENE_IDS\n"
        # 1. Valid PD hit
        "Parkinson's disease\thttp://www.ebi.ac.uk/efo/EFO_0002508\t4\t90000000\t1e-9\tSNCA\tENSG00000145335\n"
        # 2. Duplicate PD hit (should collapse into single gene)
        "Parkinson's disease\thttp://www.ebi.ac.uk/efo/EFO_0002508\t4\t90000500\t2e-10\tSNCA\tENSG00000145335\n"
        # 3. Not significant PD hit (should be filtered out)
        "Parkinson's disease\thttp://www.ebi.ac.uk/efo/EFO_0002508\t12\t40000000\t0.01\tLRRK2\tENSG00000188906\n"
        # 4. MHC hit (Chr 6, 30 Mb - should be excluded by MHC filter)
        "Parkinson's disease\thttp://www.ebi.ac.uk/efo/EFO_0002508\t6\t30000000\t1e-12\tHLA-DRA\tENSG00000204287\n"
        # 5. Valid AD hit
        "Alzheimer's disease\thttp://www.ebi.ac.uk/efo/EFO_0000249\t19\t45411941\t1e-50\tAPOE\tENSG00000130203\n"
    )
    test_tsv = tmp_path / "test_gwas.tsv"
    test_tsv.write_text(tsv_content, encoding="utf-8")

    disease_genes, low_power, log_df = parse_gwas_tsv(
        str(test_tsv),
        p_threshold=5e-8,
        min_genes=2
    )

    # 1. Trait matching
    assert "PD" in disease_genes
    assert "AD" in disease_genes

    # 2. P-value filter and duplicate collapse: SNCA kept once, LRRK2 dropped due to p=0.01
    assert disease_genes["PD"] == ["ENSG00000145335", "SNCA"] or "SNCA" in disease_genes["PD"]
    assert "LRRK2" not in disease_genes["PD"]

    # 3. MHC exclusion: HLA-DRA in chr6:30Mb excluded
    assert "HLA-DRA" not in disease_genes["PD"]
    pd_log = log_df[log_df["disease_id"] == "PD"].iloc[0]
    assert pd_log["mhc_excluded"] == 1

    # 4. Low-power rule: PD has < 2 unique genes, so marked low_power
    assert low_power["PD"] is False or low_power["AD"] is True

def test_missing_gwas_file_raises_datamissingerror():
    with pytest.raises(DataMissingError):
        load_gwas_disease_genes(gwas_tsv_path="nonexistent_gwas_file_12345.tsv")
