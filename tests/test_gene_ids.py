"""
Unit test: Gene ID hub and symbol normalization
Validates HGNC resolution, alias handling, Ensembl to symbol mapping, and unmapped tracking.
"""

import pytest
from nerva.data.gene_ids import GeneIdHub

def test_gene_id_direct_and_alias_resolution():
    hub = GeneIdHub()
    hub.approved_symbols.update(["SNCA", "LRRK2", "HTT", "APOE", "MAPT"])
    hub.alias_to_approved["PARK1"] = "SNCA"
    hub.alias_to_approved["HD"] = "HTT"
    hub.ensembl_to_approved["ENSG00000145335"] = "SNCA"

    # Direct approved match
    sym, mapped = hub.normalize("SNCA")
    assert mapped is True
    assert sym == "SNCA"

    # Alias resolution to approved symbol
    sym, mapped = hub.normalize("PARK1")
    assert mapped is True
    assert sym == "SNCA"

    sym, mapped = hub.normalize("HD")
    assert mapped is True
    assert sym == "HTT"

    # Ensembl ID resolution
    sym, mapped = hub.normalize("ENSG00000145335")
    assert mapped is True
    assert sym == "SNCA"

    # Synthetic demo symbols pass through
    sym, mapped = hub.normalize("SYN_G0042")
    assert mapped is True
    assert sym == "SYN_G0042"

    # Empty query handling
    sym, mapped = hub.normalize("")
    assert mapped is False
    assert sym == ""
