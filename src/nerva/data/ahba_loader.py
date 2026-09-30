"""
NERVA-ML AHBA Data Loader
Parses raw Allen Human Brain Atlas (AHBA) microarray files across all 6 donors.
Strictly requires real data files in data/raw/ahba/; raises DataMissingError if absent.
"""

from typing import Dict, List, Tuple, Optional
import os
import zipfile
import pandas as pd
import numpy as np
from nerva.utils.errors import DataMissingError, DataFormatError

DEFAULT_DONORS = [9861, 10021, 12876, 14380, 15496, 15697]

def find_donor_source(donor_dir: str, donor_id: int) -> Tuple[Optional[str], bool]:
    """Finds unzipped donor directory or donor zip file."""
    # Check for unzipped folder
    candidates = [
        os.path.join(donor_dir, str(donor_id)),
        os.path.join(donor_dir, f"donor_{donor_id}"),
        os.path.join(donor_dir, f"normalized_microarray_donor_{donor_id}")
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "MicroarrayExpression.csv")):
            return c, False

    # Check for zip file
    zip_candidates = [
        os.path.join(donor_dir, f"{donor_id}.zip"),
        os.path.join(donor_dir, f"donor_{donor_id}.zip"),
        os.path.join(donor_dir, f"normalized_microarray_donor_{donor_id}.zip")
    ]
    for zc in zip_candidates:
        if os.path.isfile(zc):
            return zc, True

    return None, False


def load_ontology(ontology_path: str) -> Dict[str, str]:
    """Maps structure names and acronyms to major anatomical divisions."""
    division_map = {}
    if not os.path.exists(ontology_path):
        return division_map

    ont_df = pd.read_csv(ontology_path)
    # Map structure names/acronyms to major divisions based on top-level structures
    for _, row in ont_df.iterrows():
        acronym = str(row.get("acronym", ""))
        name = str(row.get("name", ""))
        path_str = str(row.get("structure_id_path", ""))
        
        # High level divisions based on Allen ontology IDs
        div = "Cortex"
        if "4008" in path_str:
            div = "Cerebral Cortex"
        elif "4249" in path_str:
            div = "Basal Ganglia"
        elif "4275" in path_str:
            div = "Limbic System"
        elif "4391" in path_str:
            div = "Thalamus / Diencephalon"
        elif "9001" in path_str or "9131" in path_str:
            div = "Midbrain / Brainstem"
        elif "8913" in path_str:
            div = "Cerebellum"

        if acronym:
            division_map[acronym] = div
        if name:
            division_map[name] = div

    return division_map


def load_ahba_dataset(
    raw_dir: str = "data/raw/ahba",
    donors: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int]]:
    """
    Loads real multi-donor AHBA microarray dataset from data/raw/ahba/.
    Raises DataMissingError if raw data is missing (never substitutes fake data).
    """
    if donors is None:
        donors = DEFAULT_DONORS

    if not os.path.exists(raw_dir):
        raise DataMissingError(
            f"AHBA dataset directory not found at {raw_dir}.",
            "Run: python scripts/download_data.py to acquire official AHBA normalized microarray data."
        )

    # Verify all donors are present
    donor_sources = {}
    missing_donors = []
    for d_id in donors:
        src_path, is_zip = find_donor_source(raw_dir, d_id)
        if src_path is None:
            missing_donors.append(d_id)
        else:
            donor_sources[d_id] = (src_path, is_zip)

    if missing_donors:
        raise DataMissingError(
            f"AHBA donor files missing for donors: {missing_donors} in {raw_dir}.",
            "Run: python scripts/download_data.py to download all 6 normalized microarray donor archives."
        )

    # Find ontology file
    ont_path = os.path.join(raw_dir, "Ontology.csv")
    division_map = load_ontology(ont_path) if os.path.exists(ont_path) else {}

    all_donor_dfs = []
    total_samples = 0
    probes_count = 0

    for d_id in donors:
        src_path, is_zip = donor_sources[d_id]
        if is_zip:
            with zipfile.ZipFile(src_path, 'r') as z:
                # Load SampleAnnot
                with z.open("SampleAnnot.csv") as f_annot:
                    sample_annot = pd.read_csv(f_annot)
                # Load Probes
                with z.open("Probes.csv") as f_probes:
                    probes_df = pd.read_csv(f_probes)
                # Load MicroarrayExpression
                with z.open("MicroarrayExpression.csv") as f_expr:
                    expr_df = pd.read_csv(f_expr, header=None)
        else:
            sample_annot = pd.read_csv(os.path.join(src_path, "SampleAnnot.csv"))
            probes_df = pd.read_csv(os.path.join(src_path, "Probes.csv"))
            expr_df = pd.read_csv(os.path.join(src_path, "MicroarrayExpression.csv"), header=None)

        probe_ids = expr_df.iloc[:, 0].values
        expr_mat = expr_df.iloc[:, 1:].values  # probes x samples
        probes_count = len(probe_ids)

        # Transpose to samples x probes
        samples_df = pd.DataFrame(expr_mat.T, columns=probe_ids)
        total_samples += len(samples_df)

        # Structure annotation per sample
        struct_col = "structure_acronym" if "structure_acronym" in sample_annot.columns else "structure_name"
        structures = sample_annot[struct_col].values

        # Build MultiIndex (donor, structure)
        m_index = pd.MultiIndex.from_tuples(
            [(str(d_id), s) for s in structures],
            names=["donor_id", "structure_name"]
        )
        samples_df.index = m_index
        all_donor_dfs.append(samples_df)

    raw_df = pd.concat(all_donor_dfs, axis=0)

    stats = {
        "n_donors": len(donors),
        "n_samples": total_samples,
        "n_probes": probes_count,
        "n_structures": len(set(raw_df.index.get_level_values(1)))
    }

    return raw_df, division_map, stats
