"""Attention Bench Dashboard - Line Graphs Page."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    initialize_page_data,
    create_line_graph,
    format_kv_length,
    export_sidebar,
)

# Page configuration
st.set_page_config(
    page_title="Line Graphs - Attention Bench",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 Line Graphs")
st.caption("Latency vs KV Cache Length for different configurations")

# Initialize page data
df, filtered_df, approaches, filters = initialize_page_data(
    require_approaches=True,
    min_approaches=1,
    page_key="line_graphs"
)

# Line graphs visualization
st.divider()

# Group by workload type (only prefill and decode)
workload_types = [wt for wt in filtered_df['workload_type'].unique().tolist()
                  if wt in ['prefill', 'decode']]

if not workload_types:
    st.warning("No prefill or decode workload data found. Line graphs are only available for these workload types.")
    st.stop()

for workload_type in workload_types:
    workload_df = filtered_df[filtered_df['workload_type'] == workload_type]

    if workload_df.empty:
        continue

    st.subheader(f"📊 {workload_type.title()} Workload")

    # Further group by model and TP degree if multiple
    model_tp_combos = workload_df.groupby(['model', 'tp_degree']).size().reset_index()

    for _, row in model_tp_combos.iterrows():
        model_name = row['model']
        tp_degree = row['tp_degree']

        combo_df = workload_df[
            (workload_df['model'] == model_name) &
            (workload_df['tp_degree'] == tp_degree)
        ]

        if combo_df.empty:
            continue

        with st.expander(f"**{model_name} (TP={tp_degree})**", expanded=True):
            # Determine dimension column based on workload type
            if workload_type == "decode":
                dimension_col = "batch_size"
                dimension_label = "Batch Sizes"
            else:  # prefill
                dimension_col = "query_length"
                dimension_label = "Query Lengths"

            # Get available dimension values
            available_dimensions = sorted(combo_df[dimension_col].unique())

            # Create selectors
            col1, col2 = st.columns(2)

            with col1:
                selected_approaches = st.multiselect(
                    "Select Approaches",
                    options=approaches,
                    default=approaches,
                    key=f"approaches_{workload_type}_{model_name}_{tp_degree}",
                    help=f"Choose which approaches to compare"
                )

            with col2:
                selected_dimensions = st.multiselect(
                    f"Select {dimension_label}",
                    options=available_dimensions,
                    default=available_dimensions[:min(3, len(available_dimensions))],  # Default to first 3
                    format_func=format_kv_length if workload_type == "prefill" else str,
                    key=f"dimensions_{workload_type}_{model_name}_{tp_degree}",
                    help=f"Choose which {dimension_label.lower()} to plot"
                )

            # Y-axis scale selector
            y_scale = st.radio(
                "Y-axis Scale",
                options=["log", "linear"],
                index=0,  # Default to log
                horizontal=True,
                key=f"y_scale_{workload_type}_{model_name}_{tp_degree}",
                help="Log scale better shows relative performance across wide latency ranges"
            )

            # Check if selections are valid
            if not selected_approaches:
                st.warning("Please select at least one approach.")
                continue

            if not selected_dimensions:
                st.warning(f"Please select at least one {dimension_label.lower()[:-1]}.")
                continue

            # Create line graph
            fig = create_line_graph(
                df=combo_df,
                workload_type=workload_type,
                selected_approaches=selected_approaches,
                selected_dimension_values=selected_dimensions,
                metric="median",
                y_scale=y_scale
            )

            st.plotly_chart(fig, use_container_width=True, key=f"line_{workload_type}_{model_name}_{tp_degree}")

# Export options
export_sidebar(filtered_df, approaches, filters)

# Tips
st.divider()
with st.expander("💡 Tips"):
    st.markdown("""
    - **X-axis**: KV cache length (formatted as 1k, 1M, etc.)
    - **Y-axis**: Latency in milliseconds
    - **Lines**: Each line represents one (approach, batch/query) combination
    - **Colors**: Match the approach colors used in heatmaps
    - **Decode**: Shows lines for different batch sizes
    - **Prefill**: Shows lines for different query lengths
    - **Hover**: Hover over points to see exact latency values
    - **Legend**: Click legend items to show/hide lines
    - **Zoom**: Use mouse scroll or pinch to zoom
    - **Pan**: Click and drag to pan around the graph
    - **Export**: Click the camera icon to save as PNG
    """)
