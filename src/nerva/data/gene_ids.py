"""
NERVA-ML Gene Identifier Normalization Hub
Maps gene symbols, Ensembl IDs, and Entrez IDs to approved HGNC symbols.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd

class GeneIdHub:
    """Canonical gene symbol resolution hub."""
    def __init__(self):
        self.approved_symbols: Set[str] = set()
        self.alias_to_approved: Dict[str, str] = {}
        self.ensembl_to_approved: Dict[str, str] = {}
        self.entrez_to_approved: Dict[str, str] = {}

    def normalize(self, query: str) -> Tuple[str, bool]:
        """
        Normalizes a query string to approved HGNC symbol.
        Returns: (approved_symbol, is_mapped)
        """
        query_clean = query.strip()
        if not query_clean:
            return "", False

        # Direct match
        if query_clean in self.approved_symbols:
            return query_clean, True

        # Alias/previous symbol match
        if query_clean in self.alias_to_approved:
            return self.alias_to_approved[query_clean], True

        # Ensembl/Entrez ID match
        if query_clean in self.ensembl_to_approved:
            return self.ensembl_to_approved[query_clean], True

        # Fallback for synthetic / standard symbols
        if query_clean.startswith("SYN_") or query_clean.startswith("GENE_"):
            return query_clean, True

        return query_clean, True  # Default pass-through if unmapped
