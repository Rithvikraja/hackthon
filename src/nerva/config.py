"""
NERVA-ML Configuration Parser and Validator
Loads YAML configs, resolves profile overrides, and validates schemas.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import os
import yaml
from pathlib import Path
from nerva.utils.errors import ConfigError

@dataclass
class ProjectConfig:
    seed: int = 42
    profile: str = "hackathon"
    output_dir: str = "results"

@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    external_dir: str = "data/external"
    processed_dir: str = "data/processed"
    donors: List[int] = field(default_factory=lambda: [9861, 10021, 12876, 14380, 15496, 15697])
    pa_min_fraction: float = 0.5
    gwas_p_threshold: float = 5.0e-8
    min_disease_genes: int = 15
    diseases: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class StatsConfig:
    n_permutations: int = 2000
    n_expr_bins: int = 20
    fdr_method: str = "fdr_bh"
    tau_label: float = 1.645
    label_mode: str = "threshold"

@dataclass
class MLConfig:
    pool_size: int = 300
    outer_folds: int = 3
    region_heldout: bool = True

@dataclass
class NSGA2Config:
    pop_size: int = 32
    generations: int = 20
    crossover_prob: float = 0.9
    k_min: int = 5
    k_max: int = 60
    n_jobs: int = 4

@dataclass
class RobustnessConfig:
    noise_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    dropout_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    shift_levels: List[List[float]] = field(default_factory=lambda: [[1.0, 0.10], [1.15, 0.25], [1.30, 0.50]])
    n_repeats: int = 10

@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    nsga2: NSGA2Config = field(default_factory=NSGA2Config)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    shap_max_rows: int = 500

def load_config(config_path: str, profile_override: Optional[str] = None) -> Config:
    """Loads configuration with profile overrides and schema validation."""
    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found at: {config_path}", "Check file path.")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profile = profile_override or raw.get("project", {}).get("profile", "hackathon")

    # Apply profile overrides
    if "profiles" in raw and profile in raw["profiles"]:
        prof_data = raw["profiles"][profile]
        for section, settings in prof_data.items():
            if section in raw:
                raw[section].update(settings)
            else:
                raw[section] = settings

    raw_proj = raw.get("project", {})
    raw_proj["profile"] = profile

    # Build dataclasses
    try:
        cfg = Config(
            project=ProjectConfig(
                seed=raw_proj.get("seed", 42),
                profile=profile,
                output_dir=raw_proj.get("output_dir", "results")
            ),
            data=DataConfig(**raw.get("data", {})),
            stats=StatsConfig(**raw.get("stats", {})),
            ml=MLConfig(**raw.get("ml", {})),
            nsga2=NSGA2Config(**raw.get("nsga2", {})),
            robustness=RobustnessConfig(
                noise_levels=raw.get("robustness", {}).get("noise_levels", [0.05, 0.10, 0.20]),
                dropout_levels=raw.get("robustness", {}).get("dropout_levels", [0.05, 0.10, 0.20]),
                shift_levels=raw.get("robustness", {}).get("shift_levels", [[1.0, 0.10], [1.15, 0.25], [1.30, 0.50]]),
                n_repeats=raw.get("robustness", {}).get("n_repeats", 10)
            ),
            shap_max_rows=raw.get("shap", {}).get("max_rows", 500)
        )
    except Exception as e:
        raise ConfigError(f"Failed to validate config: {e}", "Verify config.yaml fields.")

    if cfg.nsga2.pop_size < 8:
        raise ConfigError(f"nsga2.pop_size must be >= 8 (got {cfg.nsga2.pop_size})", "Increase pop_size.")

    return cfg
