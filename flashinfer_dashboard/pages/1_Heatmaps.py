"""Attention Bench Dashboard - Interactive Heatmaps Page."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    initialize_page_data,
    create_speedup_heatmap,
    create_best_performer_heatmap,
    create_mixed_heatmap,
    create_mixed_best_performer_heatmap,
    create_speedup_distribution,
    format_kv_length,
    shorten_approach_name,
    apply_approach_grouping,
    export_sidebar,
    # New components
    calculate_winner_statistics,
    render_heatmap_mode_selector,
    render_prefill_selectors,
    render_winner_metrics,
)

# Page configuration
st.set_page_config(
    page_title="Heatmaps - Attention Bench",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🔥 Performance Heatmaps")
st.caption("Interactive visualization of approach performance across scenarios")

# Initialize page data
df, filtered_df, approaches, filters = initialize_page_data(
    require_approaches=True,
    min_approaches=1,
    page_key="heatmaps"
)

# Heatmap mode selector (using extracted component)
mode = render_heatmap_mode_selector()

# Heatmap visualization
st.divider()

# Group by workload type if multiple present
workload_types = filtered_df['workload_type'].unique().tolist()

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
            if mode == "Pairwise Comparison":
                # Pairwise mode - need exactly 2 approaches
                selected_approaches = filters.get('approaches', [])

                if len(selected_approaches) < 2:
                    st.warning("Please select at least 2 approaches in the sidebar for pairwise comparison.")
                    continue

                approach1 = selected_approaches[0]
                approach2 = selected_approaches[1]

                # Handle mixed workloads separately
                if workload_type == "mixed":
                    # Use extracted prefill selectors component
                    selected_query, selected_kv = render_prefill_selectors(
                        combo_df, model_name, tp_degree, key_prefix="pairwise"
                    )

                    # Create mixed heatmap
                    fig = create_mixed_heatmap(combo_df, approach1, approach2, selected_query, selected_kv)
                    st.plotly_chart(fig, use_container_width=True, key=f"mixed_heatmap_{model_name}_{tp_degree}_{selected_query}_{selected_kv}")

                    # Winner statistics using extracted components
                    # Filter by selected prefill configuration
                    filtered_combo = combo_df[
                        (combo_df['prefill_query'] == selected_query) &
                        (combo_df['prefill_kv'] == selected_kv)
                    ].copy()

                    # Apply approach grouping
                    grouped_df, grouped_approaches = apply_approach_grouping(
                        filtered_combo, workload_type, [approach1, approach2], "median"
                    )

                    # Calculate and render winner statistics (aggregate across runs for mixed)
                    win_stats = calculate_winner_statistics(
                        grouped_df, grouped_approaches,
                        metric="median",
                        aggregate_by=['decode_batch', 'decode_kv']
                    )
                    render_winner_metrics(win_stats, grouped_approaches)

                else:
                    # Regular prefill/decode heatmap
                    fig = create_speedup_heatmap(combo_df, approach1, approach2, workload_type=workload_type)
                    st.plotly_chart(fig, use_container_width=True, key=f"speedup_{workload_type}_{model_name}_{tp_degree}")

                    # Winner statistics using extracted components
                    # Apply approach grouping
                    grouped_df, grouped_approaches = apply_approach_grouping(
                        combo_df, workload_type, [approach1, approach2], "median"
                    )

                    # Determine aggregation columns based on workload type
                    if workload_type == "prefill":
                        aggregate_cols = ['query_length', 'kv_length']
                    else:  # decode
                        aggregate_cols = ['batch_size', 'kv_length']

                    # Calculate and render winner statistics (aggregate across runs)
                    win_stats = calculate_winner_statistics(
                        grouped_df, grouped_approaches,
                        metric="median",
                        aggregate_by=aggregate_cols
                    )
                    render_winner_metrics(win_stats, grouped_approaches)

            else:
                # Best performer mode
                selected_approaches = filters.get('approaches', approaches)

                if len(selected_approaches) < 2:
                    st.warning("Please select at least 2 approaches to compare.")
                    continue

                if workload_type == "mixed":
                    # Use extracted prefill selectors component
                    selected_query, selected_kv = render_prefill_selectors(
                        combo_df, model_name, tp_degree, key_prefix="best"
                    )

                    # Create mixed best performer heatmap
                    fig = create_mixed_best_performer_heatmap(combo_df, selected_approaches, selected_query, selected_kv)
                    st.plotly_chart(fig, use_container_width=True, key=f"best_mixed_{model_name}_{tp_degree}_{selected_query}_{selected_kv}")

                    # Winner statistics - filter by selected prefill configuration
                    stats_df = combo_df[
                        (combo_df['prefill_query'] == selected_query) &
                        (combo_df['prefill_kv'] == selected_kv)
                    ].copy()

                    # Apply approach grouping
                    grouped_df, grouped_approaches = apply_approach_grouping(
                        stats_df, workload_type, selected_approaches, "median"
                    )

                    # Calculate and render winner statistics (aggregate across runs for mixed)
                    win_stats = calculate_winner_statistics(
                        grouped_df, grouped_approaches,
                        metric="median",
                        aggregate_by=['decode_batch', 'decode_kv']
                    )
                    render_winner_metrics(win_stats, grouped_approaches)

                else:
                    # Regular workloads
                    fig = create_best_performer_heatmap(combo_df, selected_approaches, workload_type=workload_type)
                    st.plotly_chart(fig, use_container_width=True, key=f"best_{workload_type}_{model_name}_{tp_degree}")

                    # Winner statistics
                    # Apply approach grouping
                    grouped_df, grouped_approaches = apply_approach_grouping(
                        combo_df, workload_type, selected_approaches, "median"
                    )

                    # Determine aggregation columns based on workload type
                    if workload_type == "prefill":
                        aggregate_cols = ['query_length', 'kv_length']
                    else:  # decode
                        aggregate_cols = ['batch_size', 'kv_length']

                    # Calculate and render winner statistics (aggregate across runs)
                    win_stats = calculate_winner_statistics(
                        grouped_df, grouped_approaches,
                        metric="median",
                        aggregate_by=aggregate_cols
                    )
                    render_winner_metrics(win_stats, grouped_approaches)

# Export options
export_sidebar(filtered_df, filters.get('approaches', approaches), filters)

# Tips
st.divider()
with st.expander("💡 Tips"):
    st.markdown("""
    - **Zoom**: Use mouse scroll or pinch to zoom into specific regions
    - **Pan**: Click and drag to pan around the heatmap
    - **Hover**: Hover over cells to see detailed timing information
    - **Export**: Click the camera icon in the chart toolbar to save as PNG
    - **Filters**: Use the sidebar to narrow down results by model, TP degree, or workload type
    """)
