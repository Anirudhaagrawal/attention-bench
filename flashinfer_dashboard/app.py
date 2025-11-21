"""Attention Bench Dashboard - Main Overview Page."""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_all_results,
    get_available_runs,
    get_summary_stats,
    extract_approaches_from_df,
)

# Page configuration
st.set_page_config(
    page_title="Attention Bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric label {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">⚡ Attention Bench Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive analysis of attention mechanism performance</p>', unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    # Look for results directory relative to dashboard location
    results_paths = [
        "results",
        "../results",
        Path(__file__).parent.parent / "results",
    ]

    for path in results_paths:
        df = load_all_results(str(path))
        if not df.empty:
            return df, str(path)

    return load_all_results("results"), "results"

df, results_path = get_data()

if df.empty:
    st.warning("No benchmark results found. Please run some benchmarks first.")
    st.info(f"Expected results in: {results_path}")
    st.stop()

# Get summary statistics
stats = get_summary_stats(df)

# Sidebar - Data Source Info
with st.sidebar:
    st.header("📊 Data Source")
    st.caption(f"Results path: `{results_path}`")
    st.metric("Total Scenarios", stats['total_scenarios'])

    # Quick filters
    st.header("Quick Filters")

    # Run selector
    available_runs = get_available_runs(results_path)
    if available_runs:
        st.session_state.selected_run = st.selectbox(
            "Select Run",
            options=available_runs,
            help="Choose a benchmark run to analyze"
        )

# Main content - Overview Metrics
st.header("📈 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Runs", stats['total_runs'])

with col2:
    st.metric("Models Tested", len(stats['models']))

with col3:
    st.metric("Workload Types", len(stats['workload_types']))

with col4:
    st.metric("Approaches", len(stats['approaches']))

# Detailed breakdown
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Models")
    for model in stats['models']:
        model_count = len(df[df['model'] == model])
        st.write(f"- **{model}**: {model_count} scenarios")

    st.subheader("TP Degrees")
    for tp in stats['tp_degrees']:
        tp_count = len(df[df['tp_degree'] == tp])
        st.write(f"- **TP={tp}**: {tp_count} scenarios")

with col2:
    st.subheader("Workload Types")
    for wtype in stats['workload_types']:
        wtype_count = len(df[df['workload_type'] == wtype])
        st.write(f"- **{wtype}**: {wtype_count} scenarios")

    st.subheader("Available Approaches")
    for approach in stats['approaches']:
        st.write(f"- {approach}")

# Quick Navigation
st.divider()
st.header("📍 Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Heatmaps.py", label="🔥 Heatmaps", icon="🔥")
    st.caption("Interactive speedup heatmaps")

with col2:
    st.page_link("pages/2_Performance.py", label="📊 Performance", icon="📊")
    st.caption("Tables and comparison charts")

with col3:
    st.page_link("pages/3_Explorer.py", label="🔍 Explorer", icon="🔍")
    st.caption("Browse raw benchmark data")

with col4:
    st.page_link("pages/4_Workloads.py", label="📋 Workloads", icon="📋")
    st.caption("Analysis by workload category")

# Recent runs summary
st.divider()
st.header("🕐 Recent Runs")

# Show latest 5 runs
for run_id in available_runs[:5]:
    run_df = df[df['run_id'] == run_id]

    if not run_df.empty:
        with st.expander(f"📁 {run_id}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Scenarios", len(run_df))

            with col2:
                models = run_df['model'].unique().tolist()
                st.metric("Models", ", ".join(models))

            with col3:
                wtypes = run_df['workload_type'].unique().tolist()
                st.metric("Workloads", ", ".join(wtypes))

# Footer
st.divider()
st.caption("Attention Bench Dashboard | Use the navigation above or sidebar to explore results")
