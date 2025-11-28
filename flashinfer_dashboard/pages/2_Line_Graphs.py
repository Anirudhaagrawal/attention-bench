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
from utils.approach_grouping import apply_approach_grouping

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

    # Apply approach grouping (using the entire workload_df)
    # This ensures consistent approach grouping across all combos
    _, grouped_approaches = apply_approach_grouping(
        workload_df, workload_type, approaches, metric="median"
    )

    # Get filter values from global topbar
    selected_approaches = filters.get('approaches', grouped_approaches)
    x_axis = filters.get('x_axis', 'kv_length')
    selected_dimensions = filters.get('selected_dimensions', [])
    y_scale = filters.get('y_scale', 'log')

    # Validation
    if not selected_approaches:
        st.warning("Please select at least one approach in the filters above.")
        continue

    if not selected_dimensions:
        st.warning("Please select at least one dimension in the filters above.")
        continue

    st.divider()  # Visual separator before graphs

    # Get model/TP combinations for iteration
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

        # Apply approach grouping to combo_df for the graph
        # (create_line_graph expects the grouped columns to exist in the dataframe)
        combo_df_grouped, _ = apply_approach_grouping(
            combo_df, workload_type, approaches, metric="median"
        )

        with st.expander(f"**{model_name} (TP={tp_degree})**", expanded=True):
            # Create line graph using shared filters
            fig = create_line_graph(
                df=combo_df_grouped,
                workload_type=workload_type,
                selected_approaches=selected_approaches,
                selected_dimension_values=selected_dimensions,
                metric="median",
                y_scale=y_scale,
                x_axis=x_axis
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"line_{workload_type}_{model_name}_{tp_degree}"
            )

# Export options
export_sidebar(filtered_df, approaches, filters)
