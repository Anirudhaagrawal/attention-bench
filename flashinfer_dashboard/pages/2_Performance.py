"""Attention Bench Dashboard - Performance Tables & Charts Page."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    initialize_page_data,
    get_approaches_from_filters_or_df,
    create_performance_bar_chart,
    create_approach_comparison_table,
    shorten_approach_name,
    create_csv_download,
    APPROACH_COLORS,
)

# Page configuration
st.set_page_config(
    page_title="Performance - Attention Bench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 Performance Analysis")
st.caption("Tables and charts comparing attention approaches")

# Initialize page data
df, filtered_df, _, filters = initialize_page_data(
    require_approaches=False,
    page_key="performance"
)

# Get approaches from filters or extract from dataframe
approaches = get_approaches_from_filters_or_df(filters, filtered_df)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Bar Charts", "📈 Detailed Results"])

with tab1:
    st.subheader("Approach Performance Summary")

    # Group by workload type
    for workload_type in filtered_df['workload_type'].unique():
        workload_df = filtered_df[filtered_df['workload_type'] == workload_type]

        st.markdown(f"### {workload_type.title()} Workload")

        summary_table = create_approach_comparison_table(workload_df, approaches)

        if not summary_table.empty:
            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True,
            )

            # Winner highlight
            if len(summary_table) > 1:
                # Find approach with lowest median
                medians = summary_table['Median (ms)'].astype(float)
                winner_idx = medians.idxmin()
                winner = summary_table.loc[winner_idx, 'Approach']
                st.success(f"**Fastest (by median):** {winner}")
        else:
            st.info("No data available for this workload type")

        st.divider()

with tab2:
    st.subheader("Performance Comparison Charts")

    # Controls
    col1, col2 = st.columns([1, 3])

    with col1:
        num_scenarios = st.slider(
            "Number of scenarios",
            min_value=5,
            max_value=50,
            value=15,
            help="How many scenarios to show in the chart"
        )

    with col2:
        sort_by = st.selectbox(
            "Sort scenarios by",
            options=["batch_size", "kv_length", "query_length"],
            index=0,
            help="How to order scenarios in the chart"
        )

    # Create charts by workload type
    for workload_type in filtered_df['workload_type'].unique():
        workload_df = filtered_df[filtered_df['workload_type'] == workload_type]

        st.markdown(f"### {workload_type.title()} Workload")

        # Sort by selected column
        if sort_by in workload_df.columns:
            workload_df = workload_df.sort_values(sort_by)

        fig = create_performance_bar_chart(workload_df, approaches, top_n=num_scenarios)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Detailed Results Table")

    # Select columns to display
    base_cols = ['scenario_name', 'model', 'tp_degree', 'workload_type',
                 'batch_size', 'query_length', 'kv_length']

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

    # Highlight minimum values per row
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=False,
    )

    # Download button
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        create_csv_download(display_df, "attention_bench_results.csv")

# Sidebar summary
with st.sidebar:
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Total Scenarios", len(filtered_df))
    st.metric("Models", len(filtered_df['model'].unique()))
    st.metric("Approaches", len(approaches))
