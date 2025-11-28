"""Reusable UI components for Streamlit dashboard."""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from .formatting import shorten_approach_name, format_kv_length
from .colors import APPROACH_COLORS


def build_approach_legend(
    approaches: List[str],
    y_position: float = -0.07,
    rect_height: float = 0.02,
    rect_width: float = 0.02
) -> Tuple[List[dict], List[dict]]:
    """Build legend shapes and annotations for approach colors.

    Creates a horizontal legend at the bottom of a heatmap showing:
    - Colored rectangles for each approach
    - Text labels with shortened approach names

    Args:
        approaches: List of approach names to include in legend
        y_position: Vertical position of legend text (paper coords, default -0.07)
        rect_height: Height of colored rectangles (default 0.02)
        rect_width: Width of colored rectangles (default 0.02)

    Returns:
        Tuple of (shapes, annotations) for Plotly layout:
        - shapes: List of dict for colored rectangles
        - annotations: List of dict for text labels

    Example:
        >>> shapes, annotations = build_approach_legend(['approach1', 'approach2'])
        >>> fig.update_layout(shapes=shapes, annotations=annotations)
    """
    legend_annotations = []
    legend_shapes = []
    n_approaches = len(approaches)
    legend_spacing = 1.0 / (n_approaches + 1)  # Evenly space across width

    for idx, approach in enumerate(approaches):
        x_pos = (idx + 1) * legend_spacing
        color = APPROACH_COLORS.get(approach, "#808080")
        short_name = shorten_approach_name(approach)

        # Add colored rectangle
        legend_shapes.append(dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_pos - rect_width,
            x1=x_pos,
            y0=y_position - rect_height / 2,
            y1=y_position + rect_height / 2,
            fillcolor=color,
            line=dict(color="gray", width=1),
        ))

        # Add approach name text
        legend_annotations.append(dict(
            xref="paper",
            yref="paper",
            x=x_pos + 0.01,
            y=y_position,
            text=short_name,
            showarrow=False,
            font=dict(size=12),
            xanchor="left",
        ))

    return legend_shapes, legend_annotations


def render_heatmap_mode_selector(key: str = "heatmap_mode") -> str:
    """Render heatmap visualization mode selector.

    Args:
        key: Unique key for the widget

    Returns:
        Selected mode: "Pairwise Comparison" or "Best Performer"
    """
    st.subheader("Visualization Mode")

    col1, col2 = st.columns([1, 3])

    with col1:
        mode = st.radio(
            "Mode",
            options=["Pairwise Comparison", "Best Performer"],
            help="Pairwise: Compare two approaches. Best Performer: Show winner across all approaches.",
            key=key
        )

    with col2:
        if mode == "Pairwise Comparison":
            st.info("📊 Compares exactly 2 approaches across all scenarios")
        else:
            st.info("🏆 Shows the best performing approach for each scenario")

    return mode


def render_prefill_selectors(
    df: pd.DataFrame,
    model_name: str,
    tp_degree: int,
    key_prefix: str = "prefill"
) -> Tuple[int, int]:
    """Render prefill dimension selectors for mixed workloads.

    Args:
        df: DataFrame with mixed workload data
        model_name: Model name for unique key generation
        tp_degree: TP degree for unique key generation
        key_prefix: Prefix for widget keys

    Returns:
        Tuple of (selected_query_length, selected_kv_length)
    """
    # Get available prefill dimensions
    prefill_queries = sorted(df['prefill_query'].unique())
    prefill_kvs = sorted(df['prefill_kv'].unique())

    # Create selectors in two columns
    col1, col2 = st.columns(2)

    with col1:
        selected_query = st.selectbox(
            "Prefill Query Length",
            options=prefill_queries,
            format_func=format_kv_length,
            key=f"{key_prefix}_query_{model_name}_{tp_degree}"
        )

    with col2:
        selected_kv = st.selectbox(
            "Prefill KV Length",
            options=prefill_kvs,
            format_func=format_kv_length,
            key=f"{key_prefix}_kv_{model_name}_{tp_degree}"
        )

    return selected_query, selected_kv


def render_winner_metrics(
    win_stats: Dict[str, Any],
    approaches: List[str],
    show_header: bool = True
) -> None:
    """Render winner statistics as metric columns.

    Args:
        win_stats: Dictionary with winner statistics from calculate_winner_statistics()
        approaches: List of approach names (in display order)
        show_header: Whether to show "Winner Statistics" header
    """
    if show_header:
        st.subheader("Winner Statistics")

    wins = win_stats["wins"]
    total = win_stats["total_scenarios"]

    # Display win counts in columns
    num_cols = len(approaches)
    cols = st.columns(num_cols)

    # Display each approach's win count
    for i, approach in enumerate(approaches):
        with cols[i]:
            win_count = wins.get(approach, 0)
            percentage = 100 * win_count / total if total > 0 else 0.0

            # Display using markdown components (no arrow issues)
            st.markdown(f"**{shorten_approach_name(approach)}**")
            st.markdown(f"### {win_count} wins")
            st.caption(f"{percentage:.1f}%")


def render_heatmap_with_stats(
    fig: Any,
    win_stats: Dict[str, Any],
    approaches: List[str],
    chart_key: str,
    show_stats: bool = True
) -> None:
    """Render heatmap figure with optional winner statistics.

    Args:
        fig: Plotly figure object
        win_stats: Dictionary with winner statistics
        approaches: List of approach names
        chart_key: Unique key for the plotly chart
        show_stats: Whether to show winner statistics below heatmap
    """
    # Display heatmap
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # Display winner statistics if requested
    if show_stats and win_stats:
        render_winner_metrics(win_stats, approaches)


def render_approach_selector_for_mode(
    approaches: List[str],
    mode: str,
    key: str = "approach_selector"
) -> List[str]:
    """Render approach selector appropriate for the selected mode.

    Args:
        approaches: Available approaches
        mode: "Pairwise Comparison" or "Best Performer"
        key: Unique key for the widget

    Returns:
        List of selected approaches
    """
    if mode == "Pairwise Comparison":
        # Pairwise mode: exactly 2 approaches
        if len(approaches) < 2:
            st.warning("Please select at least 2 approaches in the sidebar to use Pairwise mode.")
            return []

        # Default to first 2
        default = approaches[:2]

        selected = st.multiselect(
            "Select 2 approaches to compare:",
            options=approaches,
            default=default,
            max_selections=2,
            format_func=shorten_approach_name,
            key=key,
            help="Choose exactly 2 approaches for pairwise comparison"
        )

        if len(selected) != 2:
            st.warning(f"Please select exactly 2 approaches. Currently selected: {len(selected)}")
            return []

        return selected
    else:
        # Best Performer mode: all selected approaches
        if len(approaches) < 2:
            st.warning("Please select at least 2 approaches in the sidebar to compare.")
            return []

        st.info(f"Comparing {len(approaches)} approaches across all scenarios to find the best performer.")
        return approaches


def render_scenario_count_badge(count: int, label: str = "scenarios") -> None:
    """Render a badge showing the number of scenarios.

    Args:
        count: Number of scenarios
        label: Label to display after the count
    """
    st.caption(f"📊 Analyzing {count} {label}")


def render_empty_state(
    message: str = "No data available",
    icon: str = "⚠️",
    suggestions: Optional[List[str]] = None
) -> None:
    """Render an empty state with optional suggestions.

    Args:
        message: Main message to display
        icon: Emoji icon
        suggestions: Optional list of suggestion strings
    """
    st.warning(f"{icon} {message}")

    if suggestions:
        st.markdown("**Suggestions:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def render_comparison_summary(
    approach1: str,
    approach2: str,
    speedup_stats: Optional[Dict[str, Any]] = None
) -> None:
    """Render summary of pairwise comparison.

    Args:
        approach1: First approach name
        approach2: Second approach name
        speedup_stats: Optional speedup statistics dictionary
    """
    st.markdown(f"### {shorten_approach_name(approach1)} vs {shorten_approach_name(approach2)}")

    if speedup_stats and speedup_stats.get("total_scenarios", 0) > 0:
        cols = st.columns(4)

        with cols[0]:
            st.metric(
                "Median Speedup",
                f"{speedup_stats['median_speedup']:.2f}x"
            )

        with cols[1]:
            st.metric(
                "Faster",
                speedup_stats['scenarios_faster'],
                f"{100 * speedup_stats['scenarios_faster'] / speedup_stats['total_scenarios']:.1f}%"
            )

        with cols[2]:
            st.metric(
                "Slower",
                speedup_stats['scenarios_slower'],
                f"{100 * speedup_stats['scenarios_slower'] / speedup_stats['total_scenarios']:.1f}%"
            )

        with cols[3]:
            st.metric(
                "Similar",
                speedup_stats['scenarios_similar'],
                f"{100 * speedup_stats['scenarios_similar'] / speedup_stats['total_scenarios']:.1f}%"
            )
