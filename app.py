"""
NERVA-ML: Neuro-Evolutionary Regional Vulnerability & Robustness Framework
Interactive Streamlit Web Dashboard & Control Center
"""

import os
import sys
import json
import yaml
import time
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure src directory is in sys.path
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nerva.config import load_config

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NERVA-ML Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR HIGH-END AESTHETICS ---
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    /* Custom Headers & Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(168, 85, 247, 0.15) 50%, rgba(16, 185, 129, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        max-width: 900px;
        line-height: 1.6;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Disclaimers & Alert Banners */
    .warning-box {
        background: rgba(234, 179, 8, 0.08);
        border-left: 4px solid #eab308;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #fef08a;
    }
    
    .info-box {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #bae6fd;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Code / Log Containers */
    .log-box {
        font-family: 'JetBrains Mono', monospace;
        background-color: #05080e;
        color: #38bdf8;
        padding: 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_shortcut=True)

# --- HELPER FUNCTIONS ---
def get_available_results():
    results_dir = BASE_DIR / "results"
    if not results_dir.exists():
        return []
    subdirs = [d.name for d in results_dir.iterdir() if d.is_dir()]
    return sorted(subdirs)

def load_result_data(folder_name):
    res_path = BASE_DIR / "results" / folder_name
    data = {}
    
    json_file = res_path / "final_results.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            data["json"] = json.load(f)
            
    csv_files = [
        "regional_scores.csv",
        "baseline_metrics.csv",
        "pareto_solutions.csv",
        "robustness_report.csv",
        "region_heldout_report.csv",
        "gene_importance.csv",
        "gene_region_shap.csv",
        "disease_comparison.csv",
        "optimization_history.csv",
        "reference_validation.csv",
        "gwas_sensitivity_report.csv"
    ]
    
    data["csvs"] = {}
    for csv in csv_files:
        p = res_path / csv
        if p.exists():
            try:
                data["csvs"][csv] = pd.read_csv(p)
            except Exception:
                pass
                
    data["imgs"] = {}
    for img_name in res_path.glob("*.png"):
        data["imgs"][img_name.name] = str(img_name)
        
    return data, res_path

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://img.icons8.com/isometric-line/100/brain.png", width=70)
st.sidebar.title("NERVA-ML Navigation")
st.sidebar.caption("Neuro-Evolutionary Regional Vulnerability & Robustness Framework")

available_folders = get_available_results()
if not available_folders:
    st.sidebar.warning("No results folders found. Run the pipeline first.")
    selected_folder = "demo_synthetic"
else:
    default_idx = 0
    if "demo_synthetic" in available_folders:
        default_idx = available_folders.index("demo_synthetic")
    selected_folder = st.sidebar.selectbox(
        "📁 Select Active Dataset Run",
        available_folders,
        index=default_idx
    )

st.sidebar.divider()
st.sidebar.subheader("⚙️ Quick Pipeline Actions")

run_mode = st.sidebar.radio("Execution Dataset Mode", ["Synthetic Demo", "Real AHBA + GWAS"], help="Synthetic mode requires no external downloads and runs in ~11s.")
profile_sel = st.sidebar.selectbox("Profile", ["demo", "hackathon", "full"], index=0 if run_mode == "Synthetic Demo" else 1)
seed_input = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)

if st.sidebar.button("🚀 Run Pipeline Now", use_container_width=True, type="primary"):
    st.session_state["trigger_pipeline_run"] = {
        "is_demo": (run_mode == "Synthetic Demo"),
        "profile": profile_sel,
        "seed": seed_input
    }

# --- HEADER HERO BANNER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">NERVA-ML Dashboard</div>
    <div class="hero-subtitle">
        Investigating transcriptomic spatial signatures of GWAS-implicated disease enrichment across healthy human brain postmortem profiles using donor-grouped multi-objective evolutionary optimization (NSGA-II) & robust XGBoost modeling.
    </div>
</div>
""", unsafe_allow_shortcut=True)

# Handle Pipeline Execution Request
if st.session_state.get("trigger_pipeline_run"):
    run_opts = st.session_state["trigger_pipeline_run"]
    del st.session_state["trigger_pipeline_run"]
    
    st.info(f"⏳ Launching pipeline run (Mode: {'Demo' if run_opts['is_demo'] else 'Real'}, Profile: {run_opts['profile']}, Seed: {run_opts['seed']})...")
    
    cmd = [sys.executable, "main.py", "--profile", run_opts['profile'], "--seed", str(run_opts['seed'])]
    if run_opts['is_demo']:
        cmd.append("--demo")
        
    log_area = st.empty()
    log_text = ""
    
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    with st.spinner("Pipeline running..."):
        for line in proc.stdout:
            log_text += line
            log_area.markdown(f"```text\n{log_text[-3000:]}\n```")
    
    proc.wait()
    if proc.returncode == 0:
        st.success("✅ Pipeline completed successfully! Refreshing data...")
        time.sleep(1)
        st.rerun()
    else:
        st.error(f"❌ Pipeline failed with exit code {proc.returncode}. Check logs above.")

# Load active data
results_data, current_res_path = load_result_data(selected_folder)
json_summary = results_data.get("json", {})
csv_dict = results_data.get("csvs", {})

# --- MAIN TABBED INTERFACE ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview",
    "⚡ Pipeline Runner",
    "🔬 Layer 1: Scientific Discovery",
    "🤖 Layer 2: Baseline ML",
    "🧬 Layer 3: Pareto Frontier",
    "🛡️ Layer 4: Robustness",
    "🔍 Explainability & Lookup",
    "📁 Data Export"
])

# ==============================================================================
# TAB 1: OVERVIEW & GOVERNANCE
# ==============================================================================
with tab1:
    st.markdown("### 📊 Pipeline Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    meta = json_summary.get("metadata", {})
    ahba_meta = json_summary.get("datasets", {}).get("ahba", {})
    opt_meta = json_summary.get("optimization", {}).get("knee", {})
    rob_meta = json_summary.get("robustness", {})
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{ahba_meta.get('n_donors', '6')}</div>
            <div class="metric-label">Healthy AHBA Donors</div>
        </div>
        """, unsafe_allow_shortcut=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{ahba_meta.get('n_structures', '48')}</div>
            <div class="metric-label">Brain Structures</div>
        </div>
        """, unsafe_allow_shortcut=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(json_summary.get('diseases', {}))}</div>
            <div class="metric-label">Diseases Analyzed</div>
        </div>
        """, unsafe_allow_shortcut=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{opt_meta.get('n_genes', 'N/A')}</div>
            <div class="metric-label">Knee Signature Size</div>
        </div>
        """, unsafe_allow_shortcut=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{rob_meta.get('robustness_score', 0):.2f}</div>
            <div class="metric-label">Skill Retention (R)</div>
        </div>
        """, unsafe_allow_shortcut=True)
        
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
        <h4>🚨 SCIENTIFIC GOVERNANCE & BOUNDARIES</h4>
        <ul>
            <li><b>Research & Educational Tool Only</b>: NERVA-ML does NOT diagnose clinical conditions, predict individual patient outcomes, or infer direct neuro-causal mechanisms.</li>
            <li><b>Healthy Baseline Reference</b>: Microarray baseline dataset is derived from healthy adult human postmortem donors (Allen Human Brain Atlas). It measures normotypical regional abundance.</li>
            <li><b>DGEE Operationalization</b>: Disease-Gene Expression Enrichment (DGEE) is calculated relative to abundance-matched background gene permutation nulls.</li>
            <li><b>Target Label Fail-Loud Protocol</b>: Regional target labels are generated strictly without literature leakage. Prevalence must satisfy 10% – 50% positive rate.</li>
        </ul>
    </div>
    """, unsafe_allow_shortcut=True)

    st.markdown("### 🧩 The Four-Layer Architecture")
    
    layer_col1, layer_col2 = st.columns(2)
    with layer_col1:
        st.markdown("""
        #### 1️⃣ Scientific Discovery (DGEE)
        - Computes abundance-matched permutation nulls ($K=500-10,000$).
        - Controls false discovery rates using Benjamini-Hochberg (BH-FDR).
        - Generates donor-consistent regional enrichment scores ($E$).

        #### 2️⃣ Baseline Machine Learning
        - Donor-grouped cross-validation prevents donor data leakage.
        - Evaluates benchmark classifiers (Permutation, Region Identity, LR, Random Forest, XGBoost).
        """)
        
    with layer_col2:
        st.markdown("""
        #### 3️⃣ Evolutionary Optimization (NSGA-II)
        - Multi-objective search over gene selection masks & XGBoost hyperparameters.
        - Minimizes prediction error ($f_1$), perturbation vulnerability ($f_2$), and gene signature size ($f_3$).
        - Yields Pareto-optimal compact gene signatures.

        #### 4️⃣ Robustness Stress-Testing
        - Perturbations: Gaussian noise, probe dropout, affine batch shift, leave-one-division-out.
        - Quantifies retention of predictive skill under out-of-distribution shifts.
        """)

# ==============================================================================
# TAB 2: PIPELINE RUNNER & CONFIGURATOR
# ==============================================================================
with tab2:
    st.markdown("### ⚡ Interactive Pipeline Launcher")
    st.caption("Configure pipeline parameters and execute experiments dynamically.")

    p_col1, p_col2 = st.columns([1, 1])
    
    with p_col1:
        st.markdown("#### Configuration Settings")
        p_mode = st.selectbox("Data Mode", ["Synthetic Demo Mode", "Real Dataset Mode"])
        p_profile = st.selectbox("Execution Profile", ["demo", "hackathon", "full"], help="'demo' = fast (~11s), 'hackathon' = balanced (~2m), 'full' = exhaustive (~15m)")
        p_seed = st.number_input("Project Master Seed", min_value=1, max_value=9999, value=42)
        p_diseases = st.text_input("Diseases Filter (space-separated, e.g. PD AD HD)", value="")

        st.markdown("#### Pipeline Command Preview")
        cmd_preview = f"python main.py --profile {p_profile} --seed {p_seed}"
        if p_mode == "Synthetic Demo Mode":
            cmd_preview += " --demo"
        if p_diseases.strip():
            cmd_preview += f" --disease {p_diseases.strip()}"
            
        st.code(cmd_preview, language="bash")

        if st.button("▶️ Execute Command", type="primary", key="btn_exec_tab2"):
            st.session_state["trigger_pipeline_run"] = {
                "is_demo": (p_mode == "Synthetic Demo Mode"),
                "profile": p_profile,
                "seed": p_seed
            }
            st.rerun()

    with p_col2:
        st.markdown("#### Active Configuration Profile Inspection")
        try:
            cfg = load_config(os.path.join(BASE_DIR, "configs", "config.yaml"), profile_override=p_profile)
            cfg_dict = {
                "project": cfg.project.__dict__,
                "data": {k: v for k, v in cfg.data.__dict__.items() if k != "diseases"},
                "stats": cfg.stats.__dict__,
                "ml": cfg.ml.__dict__,
                "nsga2": cfg.nsga2.__dict__,
                "robustness": cfg.robustness.__dict__
            }
            st.json(cfg_dict)
        except Exception as e:
            st.warning(f"Could not load config: {e}")

# ==============================================================================
# TAB 3: LAYER 1 - SCIENTIFIC DISCOVERY
# ==============================================================================
with tab3:
    st.markdown("### 🔬 Layer 1: Regional Disease-Gene Expression Enrichment (DGEE)")
    
    if "regional_scores.csv" in csv_dict:
        df_reg = csv_dict["regional_scores.csv"]
        
        # Controls
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            selected_dis = st.selectbox("Disease Filter", ["All"] + list(df_reg["disease"].unique()))
        with col_f2:
            min_e = st.slider("Minimum E-Score Threshold", float(df_reg["E"].min()), float(df_reg["E"].max()), 2.0)
        with col_f3:
            max_q = st.number_input("Maximum FDR (q-value)", value=0.05, step=0.01)

        filtered_reg = df_reg.copy()
        if selected_dis != "All":
            filtered_reg = filtered_reg[filtered_reg["disease"] == selected_dis]
        filtered_reg = filtered_reg[(filtered_reg["E"] >= min_e) & (filtered_reg["q"] <= max_q)]
        
        st.markdown(f"**Showing {len(filtered_reg)} enriched structure-disease associations**")
        
        # Heatmap plot
        fig_heat = px.density_heatmap(
            df_reg,
            x="disease",
            y="structure",
            z="E",
            color_continuous_scale="Viridis",
            title="DGEE Regional Enrichment Heatmap (E-scores)",
            height=600
        )
        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Data Table
        st.dataframe(
            filtered_reg.style.background_gradient(subset=["E"], cmap="YlOrRd"),
            use_container_width=True
        )
    else:
        st.warning("`regional_scores.csv` not found in selected results.")
        
    if "fig1_dgee_heatmap.png" in results_data.get("imgs", {}):
        with st.expander("📷 View High-Res Published Heatmap (PNG)"):
            st.image(results_data["imgs"]["fig1_dgee_heatmap.png"], use_container_width=True)

# ==============================================================================
# TAB 4: LAYER 2 - BASELINE MACHINE LEARNING
# ==============================================================================
with tab4:
    st.markdown("### 🤖 Layer 2: Baseline Model Benchmark")
    st.caption("Evaluating outer donor-grouped cross-validation across predictive classifiers.")

    if "baseline_metrics.csv" in csv_dict:
        df_base = csv_dict["baseline_metrics.csv"]
        
        col_m1, col_m2 = st.columns([2, 1])
        
        with col_m1:
            metrics_cols = [c for c in df_base.columns if c not in ["model", "disease"]]
            sel_metric = st.selectbox("Select Metric for Visual Comparison", metrics_cols, index=0 if "auroc_mean" in metrics_cols else 0)
            
            fig_bar = px.bar(
                df_base,
                x="model",
                y=sel_metric,
                color="disease" if "disease" in df_base.columns else None,
                barmode="group",
                title=f"Classifier Benchmark Comparison ({sel_metric})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_layout(template="plotly_dark", yaxis_range=[0, 1.05])
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_m2:
            st.markdown("#### Baseline Summary Table")
            st.dataframe(df_base, use_container_width=True)
            
            st.info("""
            **Key Methodological Insight:**
            - **Region-identity** model achieves high AUROC on seen brain structures by memorizing regional prevalence.
            - However, it provides zero gene drivers and fails completely on unseen anatomical divisions.
            - **XGBoost** captures transferable gene expression patterns without leakage.
            """)
    else:
        st.warning("`baseline_metrics.csv` missing.")

# ==============================================================================
# TAB 5: LAYER 3 - EVOLUTIONARY PARETO FRONTIER
# ==============================================================================
with tab5:
    st.markdown("### 🧬 Layer 3: Multi-Objective Pareto Optimization (NSGA-II)")
    st.caption("Trade-off frontier between Accuracy (f1), Robustness (f2), and Signature Sparsity (f3).")

    if "pareto_solutions.csv" in csv_dict:
        df_par = csv_dict["pareto_solutions.csv"]
        
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            fig_3d = px.scatter_3d(
                df_par,
                x="n_genes",
                y="test_auroc",
                z="test_robustness",
                color="selection_type" if "selection_type" in df_par.columns else "n_genes",
                hover_data=["solution_id"] if "solution_id" in df_par.columns else None,
                title="3D Pareto Frontier (Signature Size vs Accuracy vs Robustness)",
                color_discrete_map={"knee": "#38bdf8", "one_se": "#a855f7", "non_dominated": "#64748b"}
            )
            fig_3d.update_layout(template="plotly_dark", height=550)
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_p2:
            st.markdown("#### Selected Knee & 1-SE Solutions")
            if "selection_type" in df_par.columns:
                knee_df = df_par[df_par["selection_type"] != "non_dominated"]
                st.dataframe(knee_df, use_container_width=True)
            else:
                st.dataframe(df_par.head(10), use_container_width=True)

        st.markdown("#### Full Decoded Pareto Solutions")
        st.dataframe(df_par, use_container_width=True)
    else:
        st.warning("`pareto_solutions.csv` missing.")
        
    if "fig4_pareto_frontier.png" in results_data.get("imgs", {}):
        with st.expander("📷 View Published Pareto Frontier Plot (PNG)"):
            st.image(results_data["imgs"]["fig4_pareto_frontier.png"], use_container_width=True)

# ==============================================================================
# TAB 6: LAYER 4 - ROBUSTNESS & STRESS TESTING
# ==============================================================================
with tab6:
    st.markdown("### 🛡️ Layer 4: Perturbation Stress Testing & Degradation")
    st.caption("Quantifying signature stability under Gaussian noise, probe dropouts, and affine batch shifts.")

    if "robustness_report.csv" in csv_dict:
        df_rob = csv_dict["robustness_report.csv"]
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r1:
            fig_rob = px.line(
                df_rob,
                x="severity",
                y="auroc",
                color="perturbation_type",
                markers=True,
                title="Performance Degradation Profiles Under Perturbation",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_rob.update_layout(template="plotly_dark", yaxis_range=[0.4, 1.05])
            st.plotly_chart(fig_rob, use_container_width=True)

        with col_r2:
            st.markdown("#### Robustness Metrics Table")
            st.dataframe(df_rob, use_container_width=True)

    if "region_heldout_report.csv" in csv_dict:
        st.markdown("#### 🧠 Held-Out Anatomical Division Evaluation")
        st.dataframe(csv_dict["region_heldout_report.csv"], use_container_width=True)

    if "fig5_robustness.png" in results_data.get("imgs", {}):
        with st.expander("📷 View Published Robustness Curves (PNG)"):
            st.image(results_data["imgs"]["fig5_robustness.png"], use_container_width=True)

# ==============================================================================
# TAB 7: EXPLAINABILITY & LOOKUP
# ==============================================================================
with tab7:
    st.markdown("### 🔍 Explainability & Gene Attribution Explorer")

    if "gene_importance.csv" in csv_dict:
        df_imp = csv_dict["gene_importance.csv"]
        
        st.markdown("#### Top Signature Genes Ranked by |SHAP| Importance")
        fig_shap = px.bar(
            df_imp.head(15),
            x="mean_abs_shap",
            y="gene",
            orientation="h",
            title="Top Predictor Genes (Mean Absolute SHAP Value)",
            color="mean_abs_shap",
            color_continuous_scale="Tealgrn"
        )
        fig_shap.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)

    if "gene_region_shap.csv" in csv_dict:
        df_gr_shap = csv_dict["gene_region_shap.csv"]
        
        st.markdown("#### Dynamic Gene / Region Lookup Tool")
        
        lookup_type = st.radio("Lookup By", ["Gene Symbol", "Brain Structure"], horizontal=True)
        
        if lookup_type == "Gene Symbol":
            all_genes = sorted(df_gr_shap["gene"].unique())
            sel_gene = st.selectbox("Select Gene", all_genes)
            sub_df = df_gr_shap[df_gr_shap["gene"] == sel_gene]
            
            fig_g = px.bar(
                sub_df,
                x="structure",
                y="shap_value",
                color="shap_value",
                title=f"SHAP Impact across Regions for Gene: {sel_gene}",
                color_continuous_scale="RdBu_r"
            )
            fig_g.update_layout(template="plotly_dark")
            st.plotly_chart(fig_g, use_container_width=True)
            st.dataframe(sub_df, use_container_width=True)
        else:
            all_structs = sorted(df_gr_shap["structure"].unique())
            sel_struct = st.selectbox("Select Structure", all_structs)
            sub_df = df_gr_shap[df_gr_shap["structure"] == sel_struct]
            
            fig_s = px.bar(
                sub_df,
                x="shap_value",
                y="gene",
                orientation="h",
                color="shap_value",
                title=f"Gene Driver SHAP Values for Region: {sel_struct}",
                color_continuous_scale="RdBu_r"
            )
            fig_s.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_s, use_container_width=True)
            st.dataframe(sub_df, use_container_width=True)

# ==============================================================================
# TAB 8: ARTIFACTS & DATA EXPORT
# ==============================================================================
with tab8:
    st.markdown("### 📁 Results Artifact Explorer & Downloads")
    st.caption(f"Active Results Folder: `{current_res_path}`")

    st.markdown("#### Machine-Readable Summary (`final_results.json`)")
    if json_summary:
        st.json(json_summary)
        json_str = json.dumps(json_summary, indent=2)
        st.download_button(
            label="📥 Download final_results.json",
            data=json_str,
            file_name="final_results.json",
            mime="application/json"
        )

    st.markdown("#### Generated CSV Artifacts")
    for csv_name, df in csv_dict.items():
        with st.expander(f"📄 {csv_name} ({len(df)} rows)"):
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {csv_name}",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
                key=f"dl_{csv_name}"
            )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 16px;">
    NERVA-ML Framework v1.0.0 &bull; Licensed under MIT &bull; Python 3.11+ / XGBoost 2.0+ / pymoo 0.6+ / SHAP 0.44+
</div>
""", unsafe_allow_shortcut=True)
