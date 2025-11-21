"""Attention Bench Dashboard - Interactive Heatmaps Page."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_all_results,
    apply_filters,
    create_filter_sidebar,
    create_speedup_heatmap,
    create_best_performer_heatmap,
    create_speedup_distribution,
    extract_approaches_from_df,
    shorten_approach_name,
    apply_approach_grouping,
    export_sidebar,
)

# Page configuration
st.set_page_config(
    page_title="Heatmaps - Attention Bench",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Performance Heatmaps")
st.caption("Interactive visualization of approach performance across scenarios")

# Load data
@st.cache_data
def get_data():
    results_paths = ["results", "../results", Path(__file__).parent.parent.parent / "results"]
    for path in results_paths:
        df = load_all_results(str(path))
        if not df.empty:
            return df
    return load_all_results("results")

df = get_data()

if df.empty:
    st.warning("No benchmark results found. Please run some benchmarks first.")
    st.stop()

# Create filters
filters = create_filter_sidebar(df)

# Apply filters
filtered_df = apply_filters(df, filters)

if filtered_df.empty:
    st.warning("No data matches the current filters. Try adjusting your selection.")
    st.stop()

# Heatmap mode selector
st.subheader("Visualization Mode")

col1, col2 = st.columns([1, 3])

with col1:
    mode = st.radio(
        "Mode",
        options=["Pairwise Comparison", "Best Performer"],
        help="Pairwise: Compare two approaches. Best Performer: Show winner across all approaches."
    )

with col2:
    if mode == "Pairwise Comparison":
        st.info("Select two approaches in the sidebar to compare. Green = first approach faster, Red = second approach faster.")
    else:
        st.info("Shows the best performing approach for each scenario. Color intensity indicates margin of victory.")

# Get available approaches for this filtered data
approaches = extract_approaches_from_df(filtered_df)

if not approaches:
    st.error("No approach timing data found in the filtered results.")
    st.stop()

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

                col1, col2 = st.columns([3, 1])

                with col1:
                    fig = create_speedup_heatmap(combo_df, approach1, approach2, workload_type=workload_type)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**Legend**")
                    st.markdown(f"- 🟢 Green: `{shorten_approach_name(approach1)}` faster")
                    st.markdown(f"- 🔴 Red: `{shorten_approach_name(approach2)}` faster")
                    st.markdown("- ⬜ Gray: No data")
                    st.markdown("---")
                    st.markdown(f"**Speedup = {shorten_approach_name(approach2)} / {shorten_approach_name(approach1)}**")
                    st.markdown("- > 1.0: First approach wins")
                    st.markdown("- < 1.0: Second approach wins")

                # Distribution chart
                st.subheader("Speedup Distribution")
                dist_fig = create_speedup_distribution(combo_df, approach1, approach2)
                st.plotly_chart(dist_fig, use_container_width=True)

            else:
                # Best performer mode
                selected_approaches = filters.get('approaches', approaches)

                if len(selected_approaches) < 2:
                    st.warning("Please select at least 2 approaches to compare.")
                    continue

                fig = create_best_performer_heatmap(combo_df, selected_approaches, workload_type=workload_type)
                st.plotly_chart(fig, use_container_width=True)

                # Winner statistics - apply grouping first
                st.subheader("Winner Statistics")

                # Apply approach grouping for winner statistics
                grouped_df, grouped_approaches = apply_approach_grouping(
                    combo_df, workload_type, selected_approaches, "median"
                )

                # Count wins per grouped approach
                wins = {}
                for approach in grouped_approaches:
                    col = f"{approach}_median"
                    if col in grouped_df.columns:
                        wins[approach] = 0

                for _, scenario_row in grouped_df.iterrows():
                    times = {}
                    for approach in grouped_approaches:
                        col = f"{approach}_median"
                        if col in grouped_df.columns and pd.notna(scenario_row[col]):
                            times[approach] = scenario_row[col]

                    if times:
                        winner = min(times.items(), key=lambda x: x[1])[0]
                        wins[winner] = wins.get(winner, 0) + 1

                # Display win counts
                if wins:
                    cols = st.columns(len(wins))
                    for i, (approach, win_count) in enumerate(sorted(wins.items(), key=lambda x: -x[1])):
                        with cols[i % len(cols)]:
                            st.metric(
                                shorten_approach_name(approach),
                                f"{win_count} wins",
                                f"{100 * win_count / len(grouped_df):.1f}%" if len(grouped_df) > 0 else "0%"
                            )

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
