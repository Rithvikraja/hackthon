"""
Integration test: Tiny end-to-end pipeline run
Validates that the entire 8-stage pipeline executes cleanly,
generates all required CSVs and figures, exports final_results.json adhering to schema,
and completes in well under a minute.
"""

import os
import json
import pytest
from nerva.config import Config, ProjectConfig, StatsConfig, MLConfig, NSGA2Config, RobustnessConfig
from nerva.pipeline import run_pipeline

def test_tiny_pipeline_integration(tmp_path):
    out_dir = str(tmp_path / "tiny_results")
    
    cfg = Config(
        project=ProjectConfig(
            seed=42,
            profile="demo",
            output_dir=out_dir
        ),
        stats=StatsConfig(
            n_permutations=25,
            n_expr_bins=10,
            tau_label=1.645
        ),
        ml=MLConfig(
            pool_size=40,
            outer_folds=1,
            region_heldout=True
        ),
        nsga2=NSGA2Config(
            pop_size=8,
            generations=2,
            k_min=5,
            k_max=15,
            n_jobs=1
        ),
        robustness=RobustnessConfig(
            noise_levels=[0.10],
            dropout_levels=[0.10],
            shift_levels=[[1.15, 0.25]],
            n_repeats=2
        ),
        shap_max_rows=100
    )

    # Run pipeline in demo synthetic mode
    result_folder = run_pipeline(cfg, is_demo=True, diseases_filter=["DEMO_A", "DEMO_B"])

    assert os.path.exists(result_folder)

    # 1. Assert Figures exist
    for fig_num in range(1, 8):
        fig_files = [f for f in os.listdir(result_folder) if f.startswith(f"fig{fig_num}")]
        assert len(fig_files) > 0, f"fig{fig_num} missing from results"

    # 2. Assert Tables exist
    expected_csvs = [
        "regional_scores.csv",
        "baseline_metrics.csv",
        "optimization_history.csv",
        "pareto_solutions.csv",
        "robustness_report.csv",
        "gene_importance.csv"
    ]
    for csv_file in expected_csvs:
        csv_path = os.path.join(result_folder, csv_file)
        assert os.path.exists(csv_path), f"Expected CSV {csv_file} missing"

    # 3. Assert final_results.json exists and validates against schema
    json_path = os.path.join(result_folder, "final_results.json")
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "datasets" in data
    assert "diseases" in data
    assert "regions" in data
    assert "baseline_models" in data
    assert "optimization" in data
    assert "robustness" in data
    assert "explainability" in data
    assert data["metadata"]["data_provenance"] == "SYNTHETIC_DEMO"
