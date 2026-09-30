"""
NERVA-ML Expression-Matched Control Generator
Constructs size-matched, abundance-matched control gene sets
to prevent generic high-expression bias.
"""

from typing import List, Set, Dict
from collections import Counter
import numpy as np
import pandas as pd

def build_expression_matched_controls(
    common_universe: List[str],
    mean_abundance: pd.Series,
    disease_gene_set: List[str],
    n_control_sets: int = 2000,
    n_bins: int = 20,
    seed: int = 42
) -> List[List[str]]:
    """
    Builds K expression-matched control gene sets.
    Each disease gene is matched to a non-disease gene in the same expression abundance quantile bin.
    """
    rng = np.random.default_rng(seed)
    disease_set = set(disease_gene_set)
    k = len(disease_set)

    # Eligible background genes
    background = [g for g in common_universe if g not in disease_set and g in mean_abundance.index]
    bg_abundance = mean_abundance[background]

    # Quantile bins
    try:
        bins = pd.qcut(bg_abundance, q=n_bins, labels=False, duplicates='drop')
    except Exception:
        bins = pd.cut(bg_abundance, bins=n_bins, labels=False)

    bin_to_genes = {}
    for g, b in zip(background, bins):
        bin_to_genes.setdefault(b, []).append(g)

    bin_arrays = {b: np.array(genes) for b, genes in bin_to_genes.items()}
    bg_array = np.array(background)

    # Determine bin of each disease gene
    d_bins = []
    for g in disease_gene_set:
        if g in mean_abundance.index:
            val = mean_abundance[g]
            b_idx = int(np.clip(int(np.sum(bg_abundance < val) / len(bg_abundance) * n_bins), 0, n_bins - 1))
            d_bins.append(b_idx)
        else:
            d_bins.append(0)

    bin_counts = Counter(d_bins)

    # Sample K control sets
    control_sets = []
    for _ in range(n_control_sets):
        c_set = []
        used = set()
        for b_idx, count in bin_counts.items():
            pool = bin_arrays.get(b_idx, bg_array)
            avail = [g for g in pool if g not in used]
            if len(avail) >= count:
                chosen = rng.choice(avail, size=count, replace=False)
            else:
                chosen = list(avail)
                needed = count - len(chosen)
                fallback = [g for g in bg_array if g not in used and g not in chosen]
                if fallback:
                    fallback_chosen = rng.choice(fallback, size=min(needed, len(fallback)), replace=False)
                    chosen = np.concatenate([chosen, fallback_chosen])
            for g in chosen:
                c_set.append(str(g))
                used.add(str(g))
        control_sets.append(c_set)

    return control_sets
