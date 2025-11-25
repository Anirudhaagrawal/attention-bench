"""Attention Bench Dashboard - Main Overview Page."""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_cached_data,
    get_available_runs,
    get_summary_stats,
    extract_approaches_from_df,
    shorten_approach_name,
)

# Page configuration
st.set_page_config(
    page_title="Attention Bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for metrics (optional styling)
st.markdown("""
<style>
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
st.title("⚡ Attention Bench Dashboard")
st.caption("Interactive analysis of attention mechanism performance")

# Load data
df, results_path = get_cached_data(return_path=True)

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

    # Cache clear button
    if st.button("🔄 Refresh Data", help="Clear cache and reload all results"):
        st.cache_data.clear()
        st.rerun()

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
        st.write(f"- {shorten_approach_name(approach)}")

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
                st.metric("Models", len(models))
                st.caption(", ".join(models))

            with col3:
                wtypes = run_df['workload_type'].unique().tolist()
                st.metric("Workloads", len(wtypes))
                st.caption(", ".join(wtypes))

# Footer
st.divider()
st.caption("Attention Bench Dashboard | Use the navigation above or sidebar to explore results")
