"""
NERVA-ML Publication Plotting Engine
Renders Figures 1 through 7 with strict watermark enforcement for synthetic data.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nerva.constants import WATERMARK_TEXT

def apply_synthetic_watermark(fig, ax, is_synthetic: bool):
    """Applies diagonal watermark across plot if running on synthetic demo data."""
    if is_synthetic:
        fig.text(
            0.5, 0.5, WATERMARK_TEXT,
            fontsize=28, color="gray", alpha=0.25,
            ha="center", va="center", rotation=30,
            fontweight="bold", transform=fig.transFigure
        )

def plot_all_figures(
    regional_dfs: Dict[str, pd.DataFrame],
    gene_importance_dfs: Dict[str, pd.DataFrame],
    opt_history_dfs: Dict[str, pd.DataFrame],
    pareto_solutions_dfs: Dict[str, pd.DataFrame],
    robustness_dfs: Dict[str, pd.DataFrame],
    disease_comp_df: pd.DataFrame,
    gene_region_dfs: Dict[str, pd.DataFrame],
    output_dir: str,
    is_synthetic: bool = False
) -> Dict[str, str]:
    """Generates all 7 figures defined in NERVA-ML specification."""
    os.makedirs(output_dir, exist_ok=True)
    out_paths = {}

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Fig 1: DGEE Heatmap
    p1 = os.path.join(output_dir, "fig1_dgee_heatmap.png")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Assemble regional matrix
    comb_dict = {}
    for d, df in regional_dfs.items():
        comb_dict[d] = df.set_index("structure")["E"]
    c_df = pd.DataFrame(comb_dict).fillna(0.0).head(25)
    
    sns.heatmap(c_df, cmap="magma", ax=ax, cbar_kws={'label': 'Enrichment Z-Score (E)'})
    ax.set_title("Fig 1: Disease-Gene Expression Enrichment (DGEE) Landscape", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    out_paths["fig1"] = p1

    # Fig 2: Gene Importance
    p2 = os.path.join(output_dir, "fig2_gene_importance.png")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    first_d = list(gene_importance_dfs.keys())[0]
    top_genes = gene_importance_dfs[first_d].head(15)
    ax.barh(top_genes["gene"][::-1], top_genes["mean_abs_shap"][::-1], color="#3b82f6")
    ax.set_xlabel("Mean Absolute SHAP Value (Log-Odds Impact)", fontweight="bold")
    ax.set_title(f"Fig 2: Top Driver Genes - {first_d}", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    out_paths["fig2"] = p2

    # Fig 3: Convergence
    p3 = os.path.join(output_dir, "fig3_convergence.png")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    h_df = opt_history_dfs[first_d]
    ax.plot(h_df["generation"], h_df["best_auroc"], label="Best AUROC", color="#10b981", lw=2, marker='o')
    ax.plot(h_df["generation"], h_df["best_robustness"], label="Best Robustness", color="#6366f1", lw=2, marker='s')
    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Fig 3: NSGA-II Multi-Objective Evolutionary Convergence", fontsize=12, fontweight="bold")
    ax.legend()
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    out_paths["fig3"] = p3

    # Fig 4: Pareto Frontier
    p4 = os.path.join(output_dir, "fig4_pareto_frontier.png")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sol_df = pareto_solutions_dfs[first_d]
    scatter = ax.scatter(sol_df["n_genes"], sol_df["auroc"], c=sol_df["robustness"], cmap="viridis", s=100, edgecolors="k")
    fig.colorbar(scatter, ax=ax, label="Robustness Score")
    ax.set_xlabel("Signature Size (Number of Genes)", fontweight="bold")
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_title("Fig 4: Pareto Optimal Frontier (Accuracy vs Sparsity vs Robustness)", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p4)
    plt.close(fig)
    out_paths["fig4"] = p4

    # Fig 5: Robustness
    p5 = os.path.join(output_dir, "fig5_robustness.png")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    r_df = robustness_dfs[first_d]
    ax.plot(range(len(r_df)), r_df["auroc"], marker='o', lw=2, color="#ef4444")
    ax.set_xticks(range(len(r_df)))
    ax.set_xticklabels(r_df["perturbation"], rotation=30, ha="right")
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_title("Fig 5: Model Degradation Across Perturbations & Shifts", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p5)
    plt.close(fig)
    out_paths["fig5"] = p5

    # Fig 6: Disease Comparison
    p6 = os.path.join(output_dir, "fig6_disease_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    c_mat = c_df.corr()
    sns.heatmap(c_mat, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Fig 6: Cross-Disease Regional Profile Correlations", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p6)
    plt.close(fig)
    out_paths["fig6"] = p6

    # Fig 7: Gene x Region SHAP
    p7 = os.path.join(output_dir, "fig7_gene_region_shap.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    gr_df = gene_region_dfs[first_d]
    gr_pivot = gr_df.pivot(index="gene", columns="division", values="mean_shap").fillna(0.0)
    sns.heatmap(gr_pivot, cmap="icefire", ax=ax, cbar_kws={'label': 'Mean SHAP Value'})
    ax.set_title("Fig 7: Gene × Major Anatomical Division SHAP Attribution", fontsize=12, fontweight="bold")
    apply_synthetic_watermark(fig, ax, is_synthetic)
    plt.tight_layout()
    fig.savefig(p7)
    plt.close(fig)
    out_paths["fig7"] = p7

    return out_paths
