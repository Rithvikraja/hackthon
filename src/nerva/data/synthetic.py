"""
NERVA-ML Synthetic Demo Generator (DEMO / TEST DATA ONLY)
Generates structured latent-factor multi-donor brain expression data
and synthetic disease gene sets for end-to-end offline testing.
Strictly labeled DEMO_A, DEMO_B, DEMO_C; never confused with real clinical datasets.
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

SYNTHETIC_DIVISIONS = {
    "Cerebral Cortex": [
        "Cortical_Area_Frontal_Pole", "Cortical_Area_Superior_Frontal", "Cortical_Area_Middle_Frontal",
        "Cortical_Area_Inferior_Frontal", "Cortical_Area_Precentral", "Cortical_Area_Postcentral",
        "Cortical_Area_Superior_Temporal", "Cortical_Area_Middle_Temporal", "Cortical_Area_Inferior_Temporal",
        "Cortical_Area_Superior_Parietal", "Cortical_Area_Inferior_Parietal", "Cortical_Area_Supramarginal",
        "Cortical_Area_Angular", "Cortical_Area_Lateral_Occipital", "Cortical_Area_Cuneus", "Cortical_Area_Calcarine"
    ],
    "Basal Ganglia": [
        "Caudate_Dorsal", "Caudate_Ventral", "Putamen_Rostral", "Putamen_Caudal",
        "Globus_Pallidus_External", "Globus_Pallidus_Internal", "Substantia_Nigra_Dense", "Subthalamic_Nucleus"
    ],
    "Limbic System": [
        "Hippocampus_Field_CA1", "Hippocampus_Field_CA2_3", "Hippocampus_Field_CA4",
        "Dentate_Gyrus", "Subiculum", "Entorhinal_Cortex_Area",
        "Amygdalar_Core_Basolateral", "Amygdalar_Core_Corticomedial"
    ],
    "Diencephalon": [
        "Thalamus_Anterior", "Thalamus_Mediodorsal", "Thalamus_Pulvinar",
        "Hypothalamus_Medial", "Hypothalamus_Lateral", "Epithalamus"
    ],
    "Brainstem": [
        "Midbrain_Tegmentum", "Midbrain_Raphe", "Pons_Locus_Coeruleus",
        "Pons_Pontine_Nuclei", "Medulla_Reticular"
    ],
    "Cerebellum": [
        "Cerebellar_Hemisphere_Crus_I", "Cerebellar_Hemisphere_Crus_II", "Cerebellar_Hemisphere_Vermis",
        "Cerebellar_Deep_Nuclei_Dentate", "Cerebellar_Deep_Nuclei_Fastigial"
    ]
}

def make_synthetic_dataset(
    n_donors: int = 6,
    n_genes: int = 2000,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """
    Generates realistic multi-donor expression matrix using a low-rank latent factor model.
    Returns:
        raw_df: MultiIndex (donor, structure) × genes
        disease_genes: Dict of synthetic disease gene sets (DEMO_A, DEMO_B, DEMO_C)
        division_map: Dict of structure -> major division
    """
    rng = np.random.default_rng(seed)

    # Flatten structures
    structures = []
    division_map = {}
    for div, structs in SYNTHETIC_DIVISIONS.items():
        for s in structs:
            structures.append(s)
            division_map[s] = div

    n_structures = len(structures)
    gene_names = [f"SYN_G{i:04d}" for i in range(1, n_genes + 1)]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Latent factors (K=8 latent transcriptional biological modules)
    n_factors = 8
    region_factors = rng.normal(0, 1.0, size=(n_structures, n_factors))
    gene_loadings = rng.normal(0, 0.8, size=(n_factors, n_genes))

    base_expr = 6.5 + (region_factors @ gene_loadings)

    # Three synthetic diseases named honestly per Section 14.4
    disease_genes = {
        "DEMO_A": gene_names[0:35],
        "DEMO_B": gene_names[35:75],
        "DEMO_C": gene_names[75:110]
    }

    # Planted moderate latent elevations with realistic biological variation
    # Disease A: Factor 1 elevated in selected Basal Ganglia structures
    target_structs_a = ["Substantia_Nigra_Dense", "Caudate_Dorsal", "Putamen_Rostral", "Pons_Locus_Coeruleus"]
    for s in target_structs_a:
        if s in structures:
            s_idx = structures.index(s)
            for g in disease_genes["DEMO_A"]:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.7, 1.2)

    # Disease B: Factor 2 elevated in selected Limbic structures
    target_structs_b = ["Hippocampus_Field_CA1", "Entorhinal_Cortex_Area", "Subiculum", "Amygdalar_Core_Basolateral"]
    for s in target_structs_b:
        if s in structures:
            s_idx = structures.index(s)
            for g in disease_genes["DEMO_B"]:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.7, 1.2)

    # Disease C: Factor 3 elevated in Striatum and Deep Nuclei
    target_structs_c = ["Caudate_Ventral", "Putamen_Caudal", "Globus_Pallidus_Internal", "Cerebellar_Deep_Nuclei_Dentate"]
    for s in target_structs_c:
        if s in structures:
            s_idx = structures.index(s)
            for g in disease_genes["DEMO_C"]:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.7, 1.2)

    # Correlated co-expression module genes (non-disease features that predict the pattern)
    module_a = gene_names[110:180]
    for s in target_structs_a:
        if s in structures:
            s_idx = structures.index(s)
            for g in module_a:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.5, 1.0)

    module_b = gene_names[180:250]
    for s in target_structs_b:
        if s in structures:
            s_idx = structures.index(s)
            for g in module_b:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.5, 1.0)

    module_c = gene_names[250:320]
    for s in target_structs_c:
        if s in structures:
            s_idx = structures.index(s)
            for g in module_c:
                base_expr[s_idx, gene_to_idx[g]] += rng.uniform(0.5, 1.0)

    # Expand across donors with natural inter-donor variance and measurement noise
    donors = [9861, 10021, 12876, 14380, 15496, 15697][:n_donors]
    index_tuples = []
    rows = []

    for d in donors:
        donor_shift = rng.normal(0, 0.35, size=(1, n_genes))
        donor_noise = rng.normal(0, 0.45, size=base_expr.shape)
        donor_expr = base_expr + donor_shift + donor_noise
        
        # Inject ~1% missing values (NaNs) to test robustness against missing probes
        nan_mask = rng.random(donor_expr.shape) < 0.01
        donor_expr[nan_mask] = np.nan

        for s_idx, s in enumerate(structures):
            index_tuples.append((str(d), s))
            rows.append(donor_expr[s_idx])

    m_index = pd.MultiIndex.from_tuples(index_tuples, names=["donor_id", "structure_name"])
    raw_df = pd.DataFrame(rows, index=m_index, columns=gene_names)

    return raw_df, disease_genes, division_map
