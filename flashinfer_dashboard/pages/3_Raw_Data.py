"""Attention Bench Dashboard - Raw Data Page."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    initialize_page_data,
    get_approaches_from_filters_or_df,
    shorten_approach_name,
    create_csv_download,
)

# Page configuration
st.set_page_config(
    page_title="Raw Data - Attention Bench",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📋 Raw Data")
st.caption("Detailed benchmark results for all scenarios")

# Initialize page data
df, filtered_df, _, filters = initialize_page_data(
    require_approaches=False,
    page_key="raw_data"
)

# Get approaches from filters or extract from dataframe
approaches = get_approaches_from_filters_or_df(filters, filtered_df)

# Select columns to display
base_cols = ['scenario_name', 'model', 'tp_degree', 'workload_type',
             'batch_size', 'query_length', 'kv_length', 'workload_category']

# Add approach timing columns
timing_cols = []
for approach in approaches:
    for suffix in ['_median', '_mean', '_std']:
        col = f"{approach}{suffix}"
        if col in filtered_df.columns:
            timing_cols.append(col)

display_cols = [c for c in base_cols if c in filtered_df.columns] + timing_cols

# Create display DataFrame
display_df = filtered_df[display_cols].copy()

# Rename columns for readability
rename_map = {}
for col in timing_cols:
    parts = col.rsplit('_', 1)
    if len(parts) == 2:
        approach, metric = parts
        short_name = shorten_approach_name(approach)
        rename_map[col] = f"{short_name}_{metric}"

display_df = display_df.rename(columns=rename_map)

# Convert times to ms for display
for col in display_df.columns:
    if '_median' in col or '_mean' in col or '_std' in col:
        display_df[col] = display_df[col] * 1000

# Display table with sorting
st.dataframe(
    display_df,
    use_container_width=True,
    height=600,
    hide_index=False,
)

# Download section
st.divider()
col1, col2 = st.columns([1, 1])

with col1:
    create_csv_download(display_df, "attention_bench_results.csv")

with col2:
    st.metric("Total Scenarios", len(filtered_df))

# Sidebar summary
with st.sidebar:
    st.divider()
    st.subheader("📊 Data Summary")
    st.metric("Scenarios", len(filtered_df))
    st.metric("Models", len(filtered_df['model'].unique()))
    st.metric("TP Degrees", len(filtered_df['tp_degree'].unique()))
    st.metric("Attention Kernels", len(approaches))

    if 'workload_category' in filtered_df.columns:
        categories = set()
        for cats in filtered_df['workload_category'].dropna():
            if isinstance(cats, str):
                categories.update(cats.split(','))
        st.metric("Workload Categories", len(categories))
