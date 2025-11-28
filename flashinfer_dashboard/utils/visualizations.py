"""Plotly visualization utilities for Attention Bench dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from .colors import (
    APPROACH_COLORS,
    APPROACH_GROUPS,
    hex_to_rgb,
    rgb_to_hex,
    adjust_color_saturation,
)
from .formatting import shorten_approach_name, format_kv_length
from .ui_components import build_approach_legend


def create_speedup_heatmap_unified(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    metric: str = "median",
    workload_type: str = None,
    prefill_query: int = None,
    prefill_kv: int = None
) -> go.Figure:
    """Unified speedup heatmap for all workload types.

    Handles decode, prefill, and mixed workloads using WorkloadConfig.

    Args:
        df: DataFrame with benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        metric: Metric to use ('median', 'mean', etc.')
        workload_type: Type of workload ('decode', 'prefill', 'mixed')
        prefill_query: For mixed workloads, prefill query length to filter
        prefill_kv: For mixed workloads, prefill KV length to filter

    Returns:
        Plotly Figure object
    """
    from .heatmap_helpers import (
        prepare_heatmap_data,
        build_heatmap_grid,
        build_hover_text_speedup,
        generate_speedup_colorscale
    )

    # Prepare and filter data
    try:
        df_filtered, final_approaches, config = prepare_heatmap_data(
            df, workload_type, [approach1, approach2], metric,
            prefill_query, prefill_kv
        )
        approach1, approach2 = final_approaches[0], final_approaches[1]
    except ValueError as e:
        # Handle missing prefill configuration
        fig = go.Figure()
        fig.add_annotation(text=str(e), x=0.5, y=0.5, showarrow=False)
        return fig

    if df_filtered.empty:
        fig = go.Figure()
        msg = "No data for this configuration" if config.is_mixed() else "No valid data"
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False)
        return fig

    # Build coordinate grid
    grid = build_heatmap_grid(df_filtered, config)

    # Initialize speedup matrix and cell data
    speedup_matrix = np.full((len(grid.y_values), len(grid.x_values)), np.nan)
    cell_data = {}  # Store (speedup, time1, time2) for annotations

    short1 = shorten_approach_name(approach1)
    short2 = shorten_approach_name(approach2)

    # Calculate speedup for each cell
    for i, y_val in enumerate(grid.y_values):
        for j, x_val in enumerate(grid.x_values):
            subset = grid.get_subset(df_filtered, i, j, config)

            if not subset.empty:
                col1 = f"{approach1}_{metric}"
                col2 = f"{approach2}_{metric}"

                if col1 in subset.columns and col2 in subset.columns:
                    time1 = subset[col1].iloc[0]
                    time2 = subset[col2].iloc[0]

                    if pd.notna(time1) and pd.notna(time2) and time1 > 0:
                        speedup = time2 / time1
                        speedup_matrix[i, j] = speedup
                        cell_data[(i, j)] = (speedup, time1, time2)

                        # Create hover text using helper
                        grid.hover_text[i][j] = build_hover_text_speedup(
                            config, y_val, x_val, speedup, time1, time2,
                            approach1, approach2
                        )

    # Generate colorscale using helper
    colorscale, zmin, zmax, tick_vals_filtered, tick_texts_filtered = generate_speedup_colorscale(speedup_matrix)
    log_speedup_matrix = np.log(speedup_matrix)

    # Create heatmap with log-scale colors
    fig = go.Figure(data=go.Heatmap(
        z=log_speedup_matrix,  # Use log values for color mapping
        x=grid.x_labels,
        y=grid.y_labels,
        colorscale=colorscale,
        zmid=0.0,  # log(1.0) = 0
        zmin=zmin,
        zmax=zmax,
        hovertext=grid.hover_text,
        hoverinfo='text',
        showscale=True,
        colorbar=dict(
            title="Speedup",
            orientation='h',
            y=-0.15,
            yanchor='top',
            thickness=15,
            len=0.8,
            tickmode='array',
            tickvals=tick_vals_filtered,
            ticktext=tick_texts_filtered,
        ),
    ))

    # Build text matrix for cell annotations and count wins
    text_matrix = [['' for _ in grid.x_values] for _ in grid.y_values]
    approach1_wins = 0
    approach2_wins = 0
    ties = 0

    for (i, j), (speedup, time1, time2) in cell_data.items():
        time1_us = time1 * 1000  # Convert ms to µs
        time2_us = time2 * 1000  # Convert ms to µs
        if abs(time1 - time2) / min(time1, time2) < 0.01:  # Within 1% = tie
            ties += 1
            text_matrix[i][j] = f"{short1}:{time1_us:.0f}µs<br>{short2}:{time2_us:.0f}µs<br>{speedup:.2f}x"
        elif time1 < time2:
            # approach1 is winner (faster)
            approach1_wins += 1
            text_matrix[i][j] = f"{short1}:{time1_us:.0f}µs<br>{short2}:{time2_us:.0f}µs<br>{speedup:.2f}x"
        else:
            # approach2 is winner (faster)
            approach2_wins += 1
            text_matrix[i][j] = f"{short2}:{time2_us:.0f}µs<br>{short1}:{time1_us:.0f}µs<br>{speedup:.2f}x"

    # Update heatmap with text
    fig.data[0].text = text_matrix
    fig.data[0].texttemplate = "%{text}"
    fig.data[0].textfont = dict(size=10, family='Arial')

    # Build title
    if config.is_mixed():
        title = (
            f"Mixed Batch: {short1} vs {short2}<br>"
            f"<sub>Prefill Query={format_kv_length(prefill_query)}, "
            f"Prefill KV={format_kv_length(prefill_kv)} | "
            f"Green: {short1} faster, Red: {short2} faster</sub>"
        )
    else:
        title = (
            f"Speedup: {short1} vs {short2}<br>"
            f"<sub>Green: {short1} faster | Red: {short2} faster</sub>"
        )

    fig.update_layout(
        title=title,
        xaxis_title=config.x_axis_label,
        yaxis_title=config.y_axis_label,
        xaxis=dict(type='category'),
        yaxis=dict(type='category', autorange='reversed'),
        height=max(600, len(grid.y_values) * 70 + 250),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True,
        uniformtext=dict(minsize=10, mode='show'),
        font=dict(size=14),
        margin=dict(l=50, r=10, t=80, b=80),
    )

    return fig


def create_speedup_heatmap(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    metric: str = "median",
    workload_type: str = None
) -> go.Figure:
    """Create interactive heatmap showing speedup between two approaches.

    Unified implementation for decode, prefill, and mixed workloads.

    Args:
        df: DataFrame with benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        metric: Metric to use ('median', 'mean', etc.)
        workload_type: Type of workload for approach grouping ('prefill', 'decode', 'mixed')

    Returns:
        Plotly Figure object
    """
    return create_speedup_heatmap_unified(
        df=df,
        approach1=approach1,
        approach2=approach2,
        metric=metric,
        workload_type=workload_type
    )


def create_best_performer_heatmap_unified(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median",
    workload_type: str = None,
    prefill_query: int = None,
    prefill_kv: int = None
) -> go.Figure:
    """Unified best performer heatmap for all workload types.

    Shows rich cell annotations:
    - Line 1: Winner name + time (e.g., "Mix3:234µs")
    - Line 2: Runner-up name + time (e.g., "Sep3:267µs")
    - Line 3: Speedup margin (e.g., "1.14x")

    Color intensity varies based on margin of victory (speedup).

    Args:
        df: DataFrame with benchmark results
        approaches: List of approach names to compare
        metric: Metric to use ('median', 'mean', etc.')
        workload_type: Type of workload ('decode', 'prefill', 'mixed')
        prefill_query: For mixed workloads, the prefill query length to filter
        prefill_kv: For mixed workloads, the prefill KV length to filter

    Returns:
        Plotly Figure object
    """
    from .heatmap_helpers import (
        prepare_heatmap_data,
        build_heatmap_grid,
        generate_approach_colorscale
    )

    # Prepare and filter data
    try:
        df_filtered, approaches, config = prepare_heatmap_data(
            df, workload_type, approaches, metric,
            prefill_query, prefill_kv
        )
    except ValueError as e:
        # Handle missing prefill configuration
        fig = go.Figure()
        fig.add_annotation(text=str(e), x=0.5, y=0.5, showarrow=False)
        return fig

    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Build coordinate grid
    grid = build_heatmap_grid(df_filtered, config)

    # Initialize data structures
    cell_data = {}  # Store (winner_name, winner_time, runner_name, runner_time, speedup) for each cell
    z_matrix = [[None for _ in grid.x_values] for _ in grid.y_values]
    text_matrix = [['' for _ in grid.x_values] for _ in grid.y_values]

    # Create approach index mapping
    approach_index = {approach: idx for idx, approach in enumerate(approaches)}

    # Find best performer for each cell
    for i, y_val in enumerate(grid.y_values):
        for j, x_val in enumerate(grid.x_values):
            subset = grid.get_subset(df_filtered, i, j, config)

            if not subset.empty:
                times = {}
                for approach in approaches:
                    col = f"{approach}_{metric}"
                    if col in subset.columns:
                        time = subset[col].iloc[0]
                        if pd.notna(time) and time > 0:
                            times[approach] = time

                if len(times) >= 2:
                    # Sort by time (fastest first)
                    sorted_times = sorted(times.items(), key=lambda x: x[1])
                    winner_name, winner_time = sorted_times[0]
                    runner_name, runner_time = sorted_times[1]
                    speedup = runner_time / winner_time

                    cell_data[(i, j)] = (winner_name, winner_time, runner_name, runner_time, speedup)

                    # Calculate intensity based on margin of victory
                    # intensity: 0.4 at speedup=1.0, up to 1.0 at speedup>=1.5
                    intensity = min(1.0, 0.4 + (speedup - 1.0) * 1.2)

                    # Encode winner + intensity in z-value
                    # z = approach_index + (intensity - 0.4) / 0.6
                    # Maps intensity [0.4, 1.0] to range [approach_index, approach_index + 0.999]
                    winner_idx = approach_index[winner_name]
                    z_matrix[i][j] = winner_idx + (intensity - 0.4) / 0.6

                    # Build text annotation
                    winner_us = winner_time * 1000  # Convert ms to µs
                    runner_us = runner_time * 1000  # Convert ms to µs
                    short_winner = shorten_approach_name(winner_name)
                    short_runner = shorten_approach_name(runner_name)
                    text_matrix[i][j] = f"{short_winner}:{winner_us:.0f}µs<br>{short_runner}:{runner_us:.0f}µs<br>{speedup:.2f}x"

                    # Create hover text
                    hover_lines = [
                        f"{config.y_axis_label}: {config.format_y_axis_value(y_val)}",
                        f"{config.x_axis_label}: {config.format_x_axis_value(x_val)}",
                        f"<br><b>Winner: {shorten_approach_name(winner_name)} ({winner_time*1000:.2f}µs)</b><br>"
                    ]
                    for rank, (app, time) in enumerate(sorted_times, 1):
                        hover_lines.append(
                            f"{rank}. {shorten_approach_name(app)}: {time*1000:.2f}µs"
                        )
                    grid.hover_text[i][j] = "<br>".join(hover_lines)

                elif len(times) == 1:
                    winner_name, winner_time = list(times.items())[0]
                    cell_data[(i, j)] = (winner_name, winner_time, None, None, 1.0)

                    # No margin, use moderate intensity (0.6)
                    winner_idx = approach_index[winner_name]
                    z_matrix[i][j] = winner_idx + (0.6 - 0.4) / 0.6

                    winner_ms = winner_time * 1000
                    short_winner = shorten_approach_name(winner_name)
                    text_matrix[i][j] = f"{short_winner}:{winner_ms:.0f}"

                    grid.hover_text[i][j] = f"{config.y_axis_label}: {config.format_y_axis_value(y_val)}<br>{config.x_axis_label}: {config.format_x_axis_value(x_val)}<br>{shorten_approach_name(winner_name)}: {winner_time*1000:.2f}ms"

    # Generate colorscale using helper
    colorscale = generate_approach_colorscale(approaches)

    # Create figure with native Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=grid.x_labels,
        y=grid.y_labels,
        colorscale=colorscale,
        # Z-values encode: approach_index + intensity_fraction (range: 0 to n_approaches)
        # Explicit zmin/zmax ensures correct mapping to colorscale segments
        zmin=0,
        zmax=len(approaches),
        showscale=False,  # Hide colorbar (colors represent categories)
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color='white', family='Arial'),
        hovertext=grid.hover_text,
        hoverinfo='text',
    ))

    # Count wins per approach
    from collections import Counter
    winner_counts = Counter()
    for (i, j), (winner_name, _, _, _, _) in cell_data.items():
        winner_counts[winner_name] += 1

    # Build winner statistics string
    total_cells = len(cell_data)
    winner_stats = []
    for approach in approaches:
        count = winner_counts.get(approach, 0)
        pct = 100 * count / total_cells if total_cells > 0 else 0
        short_name = shorten_approach_name(approach)
        winner_stats.append(f"{short_name}: {count} ({pct:.1f}%)")

    winner_stats_str = " | ".join(winner_stats)

    # Create legend for approaches
    legend_text = " | ".join([f"{shorten_approach_name(app)}" for app in approaches])

    # Build color legend using helper function
    legend_shapes, legend_annotations = build_approach_legend(
        approaches,
        y_position=-0.12,
        rect_height=0.03,  # Larger boxes for better visibility
        rect_width=0.03
    )

    # Build title based on workload type
    if config.is_mixed():
        title = f"Best Performer ({len(approaches)} approaches)<br><sub>Prefill Query={format_kv_length(prefill_query)}, Prefill KV={format_kv_length(prefill_kv)} | {legend_text}</sub>"
    else:
        title = f"Best Performer ({len(approaches)} approaches)<br><sub>{legend_text}</sub>"

    fig.update_layout(
        title=title,
        xaxis_title=config.x_axis_label,
        yaxis_title=config.y_axis_label,
        xaxis=dict(
            type='category',
            side='bottom',
        ),
        yaxis=dict(
            type='category',
            autorange='reversed',  # Smallest values at top
        ),
        height=max(600, len(grid.y_values) * 70 + 200),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True,
        font=dict(size=14),
        shapes=legend_shapes,
        annotations=legend_annotations,
    )

    return fig


def create_best_performer_heatmap(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median",
    workload_type: str = None
) -> go.Figure:
    """Create heatmap showing best performing approach for each scenario.

    Shows rich cell annotations matching matplotlib heatmaps:
    - Line 1: Winner name + time (e.g., "Mix3:234ms")
    - Line 2: Runner-up name + time (e.g., "Sep3:267ms")
    - Line 3: Speedup margin (e.g., "1.14x")

    Color intensity varies based on margin of victory (speedup).

    Args:
        df: DataFrame with benchmark results
        approaches: List of approach names to compare
        metric: Metric to use ('median', 'mean', etc.)
        workload_type: Type of workload for approach grouping ('prefill', 'decode', 'mixed')

    Returns:
        Plotly Figure object
    """
    return create_best_performer_heatmap_unified(
        df=df,
        approaches=approaches,
        metric=metric,
        workload_type=workload_type
    )


def create_performance_bar_chart(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median",
    top_n: int = 10
) -> go.Figure:
    """Create bar chart comparing approaches across scenarios.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approach names to compare
        metric: Metric to use ('median', 'mean', etc.)
        top_n: Number of scenarios to show

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Take top_n scenarios (by batch size and KV length)
    df_sorted = df.sort_values(['batch_size', 'kv_length']).head(top_n)

    # Create scenario labels
    scenario_labels = []
    for _, row in df_sorted.iterrows():
        if row['workload_type'] == 'decode':
            label = f"b{int(row['batch_size'])}_kv{format_kv_length(int(row['kv_length']))}"
        elif row['workload_type'] == 'prefill':
            label = f"b{int(row['batch_size'])}_q{format_kv_length(int(row['query_length']))}_kv{format_kv_length(int(row['kv_length']))}"
        else:
            label = row['scenario_name'][:30]
        scenario_labels.append(label)

    # Add a bar for each approach
    for approach in approaches:
        col = f"{approach}_{metric}"
        if col in df_sorted.columns:
            times_ms = df_sorted[col] * 1000  # Convert to ms
            fig.add_trace(go.Bar(
                name=shorten_approach_name(approach),
                x=scenario_labels,
                y=times_ms,
                marker_color=APPROACH_COLORS.get(approach, "#808080"),
                hovertemplate=(
                    f"<b>{shorten_approach_name(approach)}</b><br>"
                    "Scenario: %{x}<br>"
                    "Time: %{y:.2f}ms<br>"
                    "<extra></extra>"
                )
            ))

    fig.update_layout(
        title="Performance Comparison Across Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Time (ms)",
        barmode='group',
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_speedup_distribution(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    metric: str = "median"
) -> go.Figure:
    """Create histogram showing distribution of speedups.

    Args:
        df: DataFrame with benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        metric: Metric to use ('median', 'mean', etc.)

    Returns:
        Plotly Figure object
    """
    col1 = f"{approach1}_{metric}"
    col2 = f"{approach2}_{metric}"

    # Calculate speedups
    speedups = []
    for _, row in df.iterrows():
        if col1 in row and col2 in row:
            time1 = row[col1]
            time2 = row[col2]
            if pd.notna(time1) and pd.notna(time2) and time1 > 0:
                speedup = time2 / time1
                speedups.append(speedup)

    if not speedups:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Create histogram
    fig = go.Figure(data=[go.Histogram(
        x=speedups,
        nbinsx=30,
        marker_color='#3498DB',
        hovertemplate=(
            "Speedup Range: %{x}<br>"
            "Count: %{y}<br>"
            "<extra></extra>"
        )
    )])

    short1 = shorten_approach_name(approach1)
    short2 = shorten_approach_name(approach2)

    # Add vertical line at 1.0 (equal performance)
    fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                  annotation_text="Equal Performance")

    # Calculate statistics
    mean_speedup = np.mean(speedups)
    fig.add_vline(x=mean_speedup, line_dash="dot", line_color="green",
                  annotation_text=f"Mean: {mean_speedup:.2f}x")

    fig.update_layout(
        title=f"Speedup Distribution: {short1} vs {short2}",
        xaxis_title="Speedup (x)",
        yaxis_title="Count",
        height=400,
        plot_bgcolor='white',
        showlegend=False,
    )

    return fig


def create_approach_comparison_table(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median"
) -> pd.DataFrame:
    """Create summary table comparing approaches.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approach names to compare
        metric: Metric to use ('median', 'mean', etc.')

    Returns:
        DataFrame with comparison statistics
    """
    comparison = []

    for approach in approaches:
        col = f"{approach}_{metric}"
        if col in df.columns:
            times = df[col].dropna() * 1000  # Convert to ms

            if not times.empty:
                comparison.append({
                    "Approach": shorten_approach_name(approach),
                    "Mean (ms)": f"{times.mean():.2f}",
                    "Median (ms)": f"{times.median():.2f}",
                    "Min (ms)": f"{times.min():.2f}",
                    "Max (ms)": f"{times.max():.2f}",
                    "Std Dev (ms)": f"{times.std():.2f}",
                    "Scenarios": len(times),
                })

    return pd.DataFrame(comparison)


def create_mixed_heatmap(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    prefill_query: int,
    prefill_kv: int,
    metric: str = "median"
) -> go.Figure:
    """Create heatmap for mixed workloads showing decode performance.

    Wrapper around create_speedup_heatmap_unified for mixed workloads.

    Args:
        df: DataFrame with mixed benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        prefill_query: Prefill query length to filter
        prefill_kv: Prefill KV length to filter
        metric: Metric to use ('median', 'mean', etc.')

    Returns:
        Plotly Figure object
    """
    return create_speedup_heatmap_unified(
        df=df,
        approach1=approach1,
        approach2=approach2,
        metric=metric,
        workload_type="mixed",
        prefill_query=prefill_query,
        prefill_kv=prefill_kv
    )


def create_mixed_best_performer_heatmap(
    df: pd.DataFrame,
    approaches: List[str],
    prefill_query: int,
    prefill_kv: int,
    metric: str = "median"
) -> go.Figure:
    """Create best performer heatmap for mixed workloads.

    For a given prefill configuration (query, kv), shows which approach wins:
    - X-axis: decode KV length
    - Y-axis: decode batch size

    Args:
        df: DataFrame with mixed benchmark results
        approaches: List of approach names to compare
        prefill_query: Prefill query length to filter
        prefill_kv: Prefill KV length to filter
        metric: Metric to use ('median', 'mean', etc.')

    Returns:
        Plotly Figure object
    """
    return create_best_performer_heatmap_unified(
        df=df,
        approaches=approaches,
        metric=metric,
        workload_type="mixed",
        prefill_query=prefill_query,
        prefill_kv=prefill_kv
    )


def create_workload_category_chart(
    df: pd.DataFrame,
    approaches: List[str],
    category_col: str = "workload_category",
    metric: str = "median"
) -> go.Figure:
    """Create chart showing performance by workload category.

    Args:
        df: DataFrame with benchmark results including category column
        approaches: List of approach names to compare
        category_col: Column name containing categories
        metric: Metric to use ('median', 'mean', etc.')

    Returns:
        Plotly Figure object
    """
    if category_col not in df.columns:
        return go.Figure()

    categories = df[category_col].unique()
    fig = go.Figure()

    for approach in approaches:
        col = f"{approach}_{metric}"
        if col in df.columns:
            means = []
            for cat in categories:
                cat_df = df[df[category_col] == cat]
                mean_time = cat_df[col].mean() * 1000  # Convert to ms
                means.append(mean_time)

            fig.add_trace(go.Bar(
                name=shorten_approach_name(approach),
                x=categories,
                y=means,
                marker_color=APPROACH_COLORS.get(approach, "#808080"),
            ))

    fig.update_layout(
        title="Average Performance by Workload Category",
        xaxis_title="Category",
        yaxis_title="Average Time (ms)",
        barmode='group',
        height=450,
        plot_bgcolor='white',
    )

    return fig


def create_line_graph(
    df: pd.DataFrame,
    workload_type: str,
    selected_approaches: List[str],
    selected_dimension_values: List[int],
    metric: str = "median",
    y_scale: str = "log",
    x_axis: str = "kv_length"
) -> go.Figure:
    """Create line graph showing latency vs selected x-axis dimension.

    Args:
        df: DataFrame with benchmark results
        workload_type: 'decode' or 'prefill'
        selected_approaches: List of approaches to plot
        selected_dimension_values: List of dimension values for line separation (batch/query)
        metric: Metric to use ('median', 'mean', etc.')
        y_scale: Y-axis scale type ('log' or 'linear'), default 'log'
        x_axis: Column to use as x-axis ('kv_length', 'batch_size', or 'query_length')

    Returns:
        Plotly Figure object with line graph
    """
    # Note: DataFrame should already have grouped columns from caller
    # No need to call apply_approach_grouping again
    df_filtered = df.copy()

    # Determine dimension column (the non-x-axis variable that creates separate lines)
    if workload_type == "decode":
        if x_axis == "kv_length":
            dimension_col = "batch_size"
            dimension_label = "Batch"
        else:  # x_axis == "batch_size"
            dimension_col = "kv_length"
            dimension_label = "KV"
    elif workload_type == "prefill":
        if x_axis == "kv_length":
            dimension_col = "query_length"
            dimension_label = "Query"
        else:  # x_axis == "query_length"
            dimension_col = "kv_length"
            dimension_label = "KV"
    else:
        raise ValueError(f"Unsupported workload_type: {workload_type}")

    # Filter by selected dimension values
    df_plot = df_filtered[df_filtered[dimension_col].isin(selected_dimension_values)].copy()

    # Skip KV lengths 32 and 64 for prefill when KV is on an axis (these are decode cache sizes)
    if workload_type == "prefill" and x_axis == "kv_length":
        df_plot = df_plot[~df_plot['kv_length'].isin([32, 64])].copy()
    elif workload_type == "prefill" and dimension_col == "kv_length":
        df_plot = df_plot[~df_plot['kv_length'].isin([32, 64])].copy()

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get unique x-axis values
    x_values = sorted(df_plot[x_axis].unique())

    # Create figure
    fig = go.Figure()

    # Plot lines for each (approach, dimension_value) combination
    for approach in selected_approaches:
        col = f"{approach}_{metric}"
        if col not in df_plot.columns:
            continue

        for dim_value in selected_dimension_values:
            # Filter for this combination
            subset = df_plot[df_plot[dimension_col] == dim_value]

            if subset.empty:
                continue

            # Collect (x, latency) pairs
            x_vals = []
            y_vals = []

            for x_val in x_values:
                x_subset = subset[subset[x_axis] == x_val]
                if not x_subset.empty and col in x_subset.columns:
                    latency = x_subset[col].iloc[0]
                    if pd.notna(latency):
                        x_vals.append(x_val)
                        y_vals.append(latency * 1000)  # Convert to ms

            if not x_vals:
                continue

            # Create line
            color = APPROACH_COLORS.get(approach, "#808080")
            line_name = f"{shorten_approach_name(approach)} ({dimension_label}={format_kv_length(dim_value)})"

            # Prepare customdata for formatted x-axis values in hover
            customdata = [format_kv_length(x_val) for x_val in x_vals]

            # Determine x-axis label for hover
            if x_axis == "kv_length":
                x_label = "KV Length"
            elif x_axis == "batch_size":
                x_label = "Batch Size"
            elif x_axis == "query_length":
                x_label = "Query Length"
            else:
                x_label = x_axis.replace("_", " ").title()

            fig.add_trace(go.Scatter(
                x=x_vals,  # Use numeric values for proper positioning
                y=y_vals,
                mode='lines+markers',
                name=line_name,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{line_name}</b><br>"
                    f"{x_label}: %{{customdata}}<br>"  # Use formatted value from customdata
                    "Latency: %{y:.2f}ms<br>"
                    "<extra></extra>"
                )
            ))

    # Determine x-axis label for layout
    if x_axis == "kv_length":
        x_axis_title = "KV Cache Length"
    elif x_axis == "batch_size":
        x_axis_title = "Batch Size"
    elif x_axis == "query_length":
        x_axis_title = "Query Length"
    else:
        x_axis_title = x_axis.replace("_", " ").title()

    # Update layout
    fig.update_layout(
        title=f"{workload_type.title()} Latency vs {x_axis_title}",
        xaxis_title=x_axis_title,
        yaxis_title="Latency (ms)",
        height=600,
        margin=dict(l=50, r=10, t=80, b=80),
        plot_bgcolor='white',
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02
        )
    )

    # Format axes
    fig.update_xaxes(
        type="log",  # Log scale for better visualization across orders of magnitude
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickmode='array',
        tickvals=x_values,  # Position ticks at actual x values
        ticktext=[format_kv_length(x_val) for x_val in x_values]  # Show formatted labels
    )
    # Format y-axis based on scale type
    if y_scale == "log":
        # For log scale, use nice round tick values
        # Collect all y values to determine range
        all_y_vals = []
        for trace in fig.data:
            all_y_vals.extend([y for y in trace.y if y is not None and y > 0])

        if all_y_vals:
            import math
            min_y = min(all_y_vals)
            max_y = max(all_y_vals)

            # Generate tick values at powers of 10
            min_exp = math.floor(math.log10(min_y))
            max_exp = math.ceil(math.log10(max_y))

            tick_vals = []
            for exp in range(min_exp, max_exp + 1):
                tick_vals.append(10**exp)

            fig.update_yaxes(
                type="log",
                tickmode='array',
                tickvals=tick_vals,
                ticktext=[f"{v:.3g}" for v in tick_vals],  # Format as 0.1, 1, 10, 100, etc.
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        else:
            fig.update_yaxes(type="log", showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        # Linear scale - default formatting is fine
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig
