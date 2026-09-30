# NERVA-ML: Neuro-Evolutionary Regional Vulnerability & Robustness Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![pymoo](https://img.shields.io/badge/pymoo-0.6+-red.svg)](https://pymoo.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-yellow.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

NERVA-ML is a hypothesis-generating research and educational framework. It investigates a single bounded question:
**Which molecular patterns in healthy adult human brain regions co-occur with high expression of a disease's genetically implicated genes, and how stable are compact gene signatures of that pattern when the data are noisy, incomplete, or shifted?**

> **IMPORTANT SCIENTIFIC FLAGS & BOUNDARIES**
> 1. **Research & Educational Only**: NERVA-ML is not a clinical diagnostic tool, does not predict individual patient risk or disease progression, and does not infer causal mechanisms.
> 2. **Healthy Tissue Baseline**: The Allen Human Brain Atlas (AHBA) samples postmortem tissue from 6 healthy adult donors. It shows where genes are normally expressed in healthy brains, not pathology.
> 3. **Expression Proxy (DGEE)**: Regional vulnerability is not measured directly; it is operationalized as Disease-Gene Expression Enrichment (DGEE) relative to expression-matched null sets.
> 4. **Genetic Association vs Causality**: GWAS-implicated loci represent population statistical associations, often identifying the nearest gene rather than the proven causal gene.
> 5. **Huntington's Disease Modifier Context**: HD is monogenic (HTT CAG expansion); GWAS Catalog entries represent age-of-onset/progression modifiers (e.g., DNA mismatch repair genes).

---

## 1. The Four Layers

| Layer | Question It Answers | Method | Primary Output |
| :--- | :--- | :--- | :--- |
| **1. Scientific Discovery** | Which brain regions express GWAS-implicated genes significantly above expression-matched random genes? | Abundance-matched permutation null ($K=500\text{--}10,000$) + Benjamini-Hochberg FDR | Regional DGEE scores ($E$), $p$-values, $q$-values |
| **2. Machine Learning** | Can we predict high DGEE from the *other* genes in the transcriptome, in donors never seen in training? | Regularized gradient boosting (XGBoost) under donor-grouped nested cross-validation | Out-of-donor AUROC, AUPRC, balanced accuracy |
| **3. Evolutionary Optimization** | Which compact gene subset and hyperparameters optimize the trade-off between skill, robustness, and sparsity? | NSGA-II over mixed chromosome (gene mask + 10 hyperparameters) | Pareto front of candidate signatures; knee & 1-SE selections |
| **4. Robustness Evaluation** | Does a candidate signature retain skill under noise, missing probes, batch shifts, and unseen anatomical regions? | Perturbation stress tests (Gaussian noise, gene dropout, affine batch shift, leave-one-division-out) | Skill retention score ($R$), degradation profiles |

---

## 2. Key Questions Answered Honestly

- **What is NERVA-ML learning?** It learns whether a brain region's background expression profile predicts that region's enrichment for a disease's GWAS-associated genes, isolating compact signatures carrying that signal.
- **What is NSGA-II optimizing?** It simultaneously minimizes $f_1 = 1 - \text{AUROC}_{\text{inner}}$, minimizes $f_2 = 1 - R$ (skill retention under perturbation), and minimizes $f_3 = k / k_{\max}$ (signature size).
- **What does robustness mean?** The fraction of above-chance skill ($R_p = \text{clip}((\text{AUROC}_p - 0.5) / (\text{AUROC}_{\text{clean}} - 0.5), 0, 1)$) retained under noise, probe dropout, batch-like shifts, unseen donors, and unseen brain divisions.
- **What is the final output?** Per-disease regional enrichment profiles with multiple-testing control, Pareto frontier of compact signatures, SHAP attribution rankings, and perturbation degradation tables.
- **What can we claim?** Hypothesis-generating associations between healthy regional transcriptomic baselines, GWAS-linked genes, and compact molecular predictors with stability quantified under stated stress tests.
- **What can we NOT claim?** Clinical vulnerability, individual diagnosis, disease progression, therapeutic targets, or causal mechanisms.

---

## 3. Data Architecture & Preprocessing

- **Gene Universe**: HGNC complete gene set acts as canonical mapping hub. Probes with ambiguous mappings or non-coding RNA lacking HGNC symbols are logged and dropped.
- **AHBA Microarray Preprocessing**:
  1. Probe filter: keep probes with Present/Absent call (`PACall = 1`) in $\ge 50\%$ of samples.
  2. Probe collapse: for multi-probe genes, select probe with highest differential stability (cross-donor Spearman correlation).
  3. Normalization: within-donor $Z$-scoring across samples to eliminate donor offsets while preserving pre-standardization abundance.
  4. Aggregation: compute mean expression per `(donor, structure)` pair.
- **Target Label Formulation (Fail-Loud)**:
  - $E(d, r) = (S(d, r) - \mu_{\text{null}}) / \sigma_{\text{null}}$.
  - Binary target $y = 1$ if $E(d, r) \ge \tau_{\text{label}}$ (default $1.645$).
  - Strict balance check: positive fraction must fall within $[10\%, 50\%]$; if outside, fails loudly with `DataFormatError` and advises `label_mode: quantile`.

---

## 4. Leakage Prevention Protocol

- **Target Separation**: Training labels are computed strictly from expression vs matched controls, never from literature lists of known vulnerable regions.
- **Feature Exclusion**: All disease-associated genes $G_d$ are strictly excluded from feature matrix $X$.
- **Donor-Grouped Splitting**: All splits are grouped by donor ID. Training and evaluation rows never share samples from the same donor.
- **Candidate Pool Isolation**: Feature pool construction ($M$ genes) is performed inside outer training folds only. Outer test donors are evaluated strictly once at the end.

---

## 5. Quick Start & CLI Usage

### Installation
```bash
# Clone repository and install dependencies
pip install -r requirements.txt

# Run full test suite (21 unit & integration tests)
python -m pytest -v
```

### Execution Modes

#### 1. Synthetic Demo Mode (Offline, Watermarked, ~11s)
Runs the entire 8-stage pipeline using an independent latent-factor benchmark. Outputs are saved to `results/demo_synthetic/` and explicitly watermarked:
```bash
python main.py --demo
```

#### 2. Real Data Pipeline (Requires Downloaded Datasets)
To run on real data, first download official AHBA and GWAS Catalog data files:
```bash
python scripts/download_data.py
python main.py
```
*(If raw files are not downloaded, the pipeline fails safely with a clear `DataMissingError`.)*

#### 3. Configuration & Profile Overrides
```bash
python main.py --profile full              # Rigorous full settings
python main.py --disease PD AD            # Analyze specific diseases
python main.py --seed 7                   # Override master random seed
python main.py --config configs/config.yaml
```

#### 4. Streamlit Web Dashboard & Control Center
To launch the interactive web dashboard:
```bash
python -m streamlit run app.py
```
*(Provides interactive 3D Pareto visualizer, live pipeline execution, regional DGEE heatmaps, perturbation stress degradation curves, and gene/region lookup tools.)*

---

## 6. Output Artifacts

All results are written to `results/<real|demo_synthetic>/`:
- `fig1_dgee_heatmap.png`: Regional Disease-Gene Expression Enrichment heatmap ($E$-scores, $q < 0.05$).
- `fig2_gene_importance.png`: Top predictor genes ranked by mean $| \text{SHAP} |$ with cross-fold stability.
- `fig3_convergence.png`: Multi-objective optimization generational trajectories.
- `fig4_pareto_frontier.png`: Trade-off frontier across Accuracy, Robustness, and Signature Size.
- `fig5_robustness.png`: Performance degradation curves under noise, dropout, and batch shifts.
- `fig6_disease_comparison.png`: Cross-disease profile correlation and signature overlap.
- `fig7_gene_region_shap.png`: Long-form gene-by-region SHAP attribution matrix.
- `regional_scores.csv`: Full statistical discovery table ($S, E, p, q$, donor consistency).
- `baseline_metrics.csv`: Comparative performance of label-permutation, region-identity, LR, RF, default XGBoost, and NERVA-optimized models across outer donor folds.
- `pareto_solutions.csv`: Complete set of non-dominated signatures with all 10 decoded hyperparameters and test evaluations.
- `robustness_report.csv`: Performance degradation table under Gaussian noise, probe dropout, and batch shifts.
- `region_heldout_report.csv`: Zero-leakage out-of-distribution evaluation across held-out anatomical divisions.
- `reference_validation.csv`: Empirical percentile ranks and permutation $p$-values against pre-registered literature reference structures.
- `gwas_sensitivity_report.csv`: Stability of regional rankings against Tier-2 candidate expansion and One-gene-per-locus pruning.
- `final_results.json`: Machine-readable summary validated against Appendix J JSON schema.

