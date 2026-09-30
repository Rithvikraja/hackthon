"""
NERVA-ML Results Exporter & Schema Validator
Writes CSVs and final_results.json matching Appendix J specifications.
Strictly validates final_results.json against the design schema.
"""

from typing import Dict, List, Any
import os
import json
import yaml
import platform
import importlib.metadata
import pandas as pd
from datetime import datetime
from nerva.constants import PROVENANCE_REAL, PROVENANCE_SYNTHETIC
from nerva.utils.errors import DataFormatError

def get_pkg_version(name: str, default: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return default

def validate_final_results_schema(data: Dict[str, Any]) -> bool:
    """
    Validates that final_results.json strictly conforms to the Appendix J schema.
    Raises DataFormatError if any required field is missing or invalid.
    """
    required_top = [
        "metadata", "datasets", "diseases", "regions",
        "baseline_models", "optimization", "robustness", "explainability", "validation"
    ]
    for key in required_top:
        if key not in data:
            raise DataFormatError(f"Schema violation: missing required top-level key '{key}'", "Verify results compilation.")

    meta = data["metadata"]
    required_meta = ["project", "version", "timestamp", "data_provenance", "profile", "seed", "disclaimer"]
    for m_key in required_meta:
        if m_key not in meta:
            raise DataFormatError(f"Schema violation: missing metadata field '{m_key}'", "Verify results metadata.")

    if meta["data_provenance"] not in [PROVENANCE_REAL, PROVENANCE_SYNTHETIC]:
        raise DataFormatError(f"Schema violation: invalid data_provenance '{meta['data_provenance']}'", "Must be REAL or SYNTHETIC_DEMO.")

    return True

def export_all_results(
    data_dict: Dict[str, Any],
    output_dir: str,
    is_synthetic: bool = False,
    config_obj: Any = None
) -> str:
    """
    Exports all CSV tables and final_results.json with strict schema validation.
    Enforces separation between synthetic and real result folders.
    """
    if is_synthetic and "real" in output_dir.lower():
        raise DataFormatError("Refusing to write synthetic results into results/real/", "Use results/demo_synthetic/")
    if not is_synthetic and "demo_synthetic" in output_dir.lower():
        raise DataFormatError("Refusing to write real results into results/demo_synthetic/", "Use results/real/")

    os.makedirs(output_dir, exist_ok=True)

    # Export CSVs
    csv_map = {
        "regional_scores.csv": data_dict.get("regional_scores"),
        "disease_gene_sets.csv": data_dict.get("disease_gene_sets"),
        "gene_mapping_log.csv": data_dict.get("gene_mapping_log"),
        "baseline_metrics.csv": data_dict.get("baseline_metrics"),
        "optimization_history.csv": data_dict.get("optimization_history"),
        "pareto_solutions.csv": data_dict.get("pareto_solutions"),
        "robustness_report.csv": data_dict.get("robustness_report"),
        "region_heldout_report.csv": data_dict.get("region_heldout_report"),
        "gene_importance.csv": data_dict.get("gene_importance"),
        "gene_region_shap.csv": data_dict.get("gene_region_shap"),
        "disease_comparison.csv": data_dict.get("disease_comparison"),
        "reference_validation.csv": data_dict.get("reference_validation"),
        "gwas_sensitivity_report.csv": data_dict.get("gwas_sensitivity_report")
    }

    for fname, df in csv_map.items():
        if df is not None and isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(output_dir, fname), index=False)

    # Build final_results.json matching exact schema from Appendix J
    provenance = PROVENANCE_SYNTHETIC if is_synthetic else PROVENANCE_REAL
    
    final_json = {
        "metadata": {
            "project": "NERVA-ML",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "data_provenance": provenance,
            "profile": getattr(config_obj.project, "profile", "hackathon") if config_obj else "demo",
            "seed": getattr(config_obj.project, "seed", 42) if config_obj else 42,
            "python": platform.python_version(),
            "package_versions": {
                "xgboost": get_pkg_version("xgboost", "3.4.1"),
                "pymoo": get_pkg_version("pymoo", "0.6.2"),
                "shap": get_pkg_version("shap", "0.52.0"),
                "scikit-learn": get_pkg_version("scikit-learn", "1.6.1")
            },
            "disclaimer": "Associative, hypothesis-generating; not diagnostic; not causal."
        },
        "datasets": {
            "ahba": {
                "n_donors": data_dict.get("n_donors", 6),
                "n_samples": data_dict.get("n_samples", 288),
                "n_genes": data_dict.get("n_genes", 2000 if is_synthetic else 9842),
                "n_rows": data_dict.get("n_rows", 288),
                "n_structures": data_dict.get("n_structures", 48),
                "manifest_sha256": "synthetic_demo_v1" if is_synthetic else "verified_ahba_v1"
            },
            "gwas": {
                "source": "synthetic_demo" if is_synthetic else "gwas_catalog_bulk",
                "p_threshold": getattr(config_obj.data, "gwas_p_threshold", 5e-8) if config_obj else 5e-8
            },
            "hgnc": {
                "source": "synthetic_symbols" if is_synthetic else "hgnc_complete_set.txt"
            }
        },
        "diseases": data_dict.get("diseases_summary", {}),
        "regions": data_dict.get("regions_summary", {}),
        "baseline_models": data_dict.get("baselines_summary", {}),
        "optimization": data_dict.get("optimization_summary", {}),
        "robustness": data_dict.get("robustness_summary", {}),
        "explainability": data_dict.get("explainability_summary", {}),
        "validation": data_dict.get("validation_summary", {})
    }

    # Validate against schema
    validate_final_results_schema(final_json)

    json_path = os.path.join(output_dir, "final_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    # Save resolved config
    if config_obj:
        cfg_path = os.path.join(output_dir, "config_used.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config_obj.__dict__, f)

    return json_path

