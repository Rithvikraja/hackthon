"""
NERVA-ML Constants
Fixed paths, column names, and conventions. No magic numbers.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

# Provenance labels
PROVENANCE_REAL = "REAL"
PROVENANCE_SYNTHETIC = "SYNTHETIC_DEMO"
WATERMARK_TEXT = "DEMO / SYNTHETIC DATA ONLY"

# Standard columns
COL_DONOR = "donor_id"
COL_STRUCTURE = "structure_name"
COL_DIVISION = "major_division"
COL_SCORE = "enrichment_score"
COL_Z = "enrichment_z"
COL_P = "p_value"
COL_Q = "q_value"
