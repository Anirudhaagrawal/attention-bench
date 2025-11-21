"""Plotly visualization utilities for Attention Bench dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


# Color palette for approaches - pastel colors
APPROACH_COLORS = {
    "official_fa3": "#3498DB",                # Blue
    "flashinfer_batch_attention": "#E74C3C",  # Red
    "flashinfer_mixed_fa2": "#2ECC71",        # Green
    "flashinfer_mixed_fa3": "#F39C12",        # Orange
    "flashinfer_separated_fa2": "#9B59B6",    # Purple
    "flashinfer_separated_fa3": "#1ABC9C",    # Teal
    # Grouped approach colors
    "fi2": "#2ECC71",                         # Green (same as mix2)
    "fi3": "#F39C12",                         # Orange (same as mix3)
    "fi_dec": "#9B59B6",                      # Purple (same as sep2)
}

# Approach groupings by workload type
# For prefill: mix2==sep2 -> fi2, mix3==sep3 -> fi3
# For decode: sep2==sep3 -> fi_dec
APPROACH_GROUPS = {
    "prefill": {
        "fi2": ["flashinfer_mixed_fa2", "flashinfer_separated_fa2"],
        "fi3": ["flashinfer_mixed_fa3", "flashinfer_separated_fa3"],
    },
    "decode": {
        "fi_dec": ["flashinfer_separated_fa2", "flashinfer_separated_fa3"],
    },
}


def shorten_approach_name(approach_name: str) -> str:
    """Convert long approach names to short display names.

    Args:
        approach_name: Full approach name

    Returns:
        Shortened name
    """
    name_map = {
        "official_fa3": "OFA3",
        "flashinfer_batch_attention": "Batch",
        "flashinfer_mixed_fa2": "Mix2",
        "flashinfer_mixed_fa3": "Mix3",
        "flashinfer_separated_fa2": "Sep2",
        "flashinfer_separated_fa3": "Sep3",
        # Grouped approaches
        "fi2": "FI2",
        "fi3": "FI3",
        "fi_dec": "FI_Dec",
    }
    return name_map.get(approach_name, approach_name.replace("flashinfer_", "").title())


def apply_approach_grouping(
    df: pd.DataFrame,
    workload_type: str,
    approaches: List[str],
    metric: str = "median"
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply approach grouping based on workload type.

    For prefill: fi2 = min(mix2, sep2), fi3 = min(mix3, sep3)
    For decode: fi_dec = min(sep2, sep3)
    For mixed: no grouping

    Args:
        df: DataFrame with benchmark results
        workload_type: Type of workload ('prefill', 'decode', 'mixed')
        approaches: List of original approach names
        metric: Metric to use

    Returns:
        Tuple of (modified DataFrame, list of grouped approach names)
    """
    if workload_type not in APPROACH_GROUPS or workload_type == "mixed":
        return df, approaches

    df = df.copy()
    groups = APPROACH_GROUPS[workload_type]
    new_approaches = []
    used_originals = set()

    # Apply groupings
    for group_name, members in groups.items():
        # Check if we have data for any members
        available_members = [m for m in members if f"{m}_{metric}" in df.columns]
        if available_members:
            # Create grouped column as min of available members
            member_cols = [f"{m}_{metric}" for m in available_members]
            df[f"{group_name}_{metric}"] = df[member_cols].min(axis=1)
            new_approaches.append(group_name)
            used_originals.update(members)

    # Add ungrouped approaches that are still relevant
    for approach in approaches:
        if approach not in used_originals:
            if f"{approach}_{metric}" in df.columns:
                new_approaches.append(approach)

    return df, new_approaches


def format_kv_length(kv: int) -> str:
    """Format KV length for display (e.g., 1024 → 1k)."""
    if kv >= 1024 and kv % 1024 == 0:
        return f"{kv // 1024}k"
    return str(kv)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def adjust_color_saturation(hex_color: str, intensity: float) -> str:
    """Adjust color brightness/saturation based on intensity.

    Higher intensity = brighter, more vibrant color
    Lower intensity = darker, more muted color

    Args:
        hex_color: Base color in hex format
        intensity: Value between 0 and 1 (1 = full brightness)

    Returns:
        Adjusted color in hex format
    """
    r, g, b = hex_to_rgb(hex_color)

    # Scale the color by intensity (lower intensity = darker)
    # This creates a more visible gradient on dark backgrounds
    new_r = int(r * intensity)
    new_g = int(g * intensity)
    new_b = int(b * intensity)

    return rgb_to_hex((new_r, new_g, new_b))


def create_speedup_heatmap(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    metric: str = "median",
    workload_type: str = None
) -> go.Figure:
    """Create interactive heatmap showing speedup between two approaches.

    Shows rich cell annotations matching matplotlib heatmaps:
    - Line 1: Speedup ratio (e.g., "1.25x")
    - Line 2: Approach2 name + time
    - Line 3: Approach1 name + time

    Args:
        df: DataFrame with benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        metric: Metric to use ('median', 'mean', etc.)
        workload_type: Type of workload for approach grouping ('prefill', 'decode', 'mixed')

    Returns:
        Plotly Figure object
    """
    # Filter out invalid data (batch_size=0 or kv_length=0)
    df = df[(df['batch_size'] > 0) & (df['kv_length'] > 0)].copy()

    # Detect workload type if not provided
    if workload_type is None and 'workload_type' in df.columns:
        workload_types = df['workload_type'].unique()
        if len(workload_types) == 1:
            workload_type = workload_types[0]

    # Skip KV lengths 32 and 64 for prefill
    if workload_type == "prefill":
        df = df[~df['kv_length'].isin([32, 64])].copy()

    # Apply approach grouping based on workload type
    if workload_type and workload_type != "mixed":
        df, grouped_approaches = apply_approach_grouping(
            df, workload_type, [approach1, approach2], metric
        )
        # Map original approach names to grouped names if they were grouped
        approach1_mapped = approach1
        approach2_mapped = approach2
        for group_name, members in APPROACH_GROUPS.get(workload_type, {}).items():
            if approach1 in members and group_name in grouped_approaches:
                approach1_mapped = group_name
            if approach2 in members and group_name in grouped_approaches:
                approach2_mapped = group_name
        approach1 = approach1_mapped
        approach2 = approach2_mapped

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid data", x=0.5, y=0.5, showarrow=False)
        return fig

    # For prefill, use query_length for y-axis; for decode, use batch_size
    if workload_type == "prefill" and 'query_length' in df.columns:
        y_col = 'query_length'
        y_label = "Query Tokens"
    else:
        y_col = 'batch_size'
        y_label = "Batch Size"

    # Get unique y values and KV lengths
    y_values = sorted(df[y_col].unique())
    kv_lengths = sorted(df['kv_length'].unique())

    # Initialize matrices
    speedup_matrix = np.full((len(y_values), len(kv_lengths)), np.nan)
    hover_text = [['' for _ in kv_lengths] for _ in y_values]
    cell_data = {}  # Store (speedup, time1, time2) for annotations

    short1 = shorten_approach_name(approach1)
    short2 = shorten_approach_name(approach2)

    # Calculate speedup for each cell
    for i, y_val in enumerate(y_values):
        for j, kv in enumerate(kv_lengths):
            subset = df[(df[y_col] == y_val) & (df['kv_length'] == kv)]

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

                        # Create hover text
                        hover_text[i][j] = (
                            f"{y_label}: {format_kv_length(y_val)}<br>"
                            f"KV Length: {format_kv_length(kv)}<br>"
                            f"Speedup: {speedup:.3f}x<br>"
                            f"{short1}: {time1*1000:.2f}ms<br>"
                            f"{short2}: {time2*1000:.2f}ms"
                        )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=speedup_matrix,
        x=[format_kv_length(kv) for kv in kv_lengths],
        y=[format_kv_length(y) for y in y_values],
        colorscale='RdYlGn',
        zmid=1.0,
        hovertext=hover_text,
        hoverinfo='text',
        colorbar_title="Speedup",
        showscale=True,
    ))

    # Build text matrix for cell annotations instead of using annotations
    # Winner (faster) shown first and bold
    text_matrix = [['' for _ in kv_lengths] for _ in y_values]
    for (i, j), (speedup, time1, time2) in cell_data.items():
        time1_ms = time1 * 1000
        time2_ms = time2 * 1000
        if time1 <= time2:
            # approach1 is winner (faster)
            text_matrix[i][j] = f"{speedup:.2f}x<br><b>{short1}:{time1_ms:.0f}</b><br>{short2}:{time2_ms:.0f}"
        else:
            # approach2 is winner (faster)
            text_matrix[i][j] = f"{speedup:.2f}x<br><b>{short2}:{time2_ms:.0f}</b><br>{short1}:{time1_ms:.0f}"

    # Update heatmap with text
    fig.data[0].text = text_matrix
    fig.data[0].texttemplate = "%{text}"
    fig.data[0].textfont = dict(size=7)

    fig.update_layout(
        title=f"Speedup: {short1} vs {short2}<br><sub>Green: {short1} faster | Red: {short2} faster</sub>",
        xaxis_title="KV Length",
        yaxis_title=y_label,
        xaxis=dict(type='category'),
        yaxis=dict(type='category', autorange='reversed'),
        height=max(600, len(y_values) * 45 + 150),
        hovermode='closest',
        plot_bgcolor='white',
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
    # Filter out invalid data (batch_size=0 or kv_length=0)
    df = df[(df['batch_size'] > 0) & (df['kv_length'] > 0)].copy()

    # Detect workload type if not provided
    if workload_type is None and 'workload_type' in df.columns:
        workload_types = df['workload_type'].unique()
        if len(workload_types) == 1:
            workload_type = workload_types[0]

    # Skip KV lengths 32 and 64 for prefill
    if workload_type == "prefill":
        df = df[~df['kv_length'].isin([32, 64])].copy()

    # Apply approach grouping based on workload type
    if workload_type and workload_type != "mixed":
        df, approaches = apply_approach_grouping(df, workload_type, approaches, metric)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid data", x=0.5, y=0.5, showarrow=False)
        return fig

    # For prefill, use query_length for y-axis; for decode, use batch_size
    if workload_type == "prefill" and 'query_length' in df.columns:
        y_col = 'query_length'
        y_label = "Query Tokens"
    else:
        y_col = 'batch_size'
        y_label = "Batch Size"

    # Get unique y values and KV lengths
    y_values = sorted(df[y_col].unique())
    kv_lengths = sorted(df['kv_length'].unique())

    # Initialize data structures
    cell_data = {}  # Store (winner_name, winner_time, runner_name, runner_time, speedup) for each cell
    hover_text = [['' for _ in kv_lengths] for _ in y_values]
    cell_colors = [[None for _ in kv_lengths] for _ in y_values]

    # Find best performer for each cell
    for i, y_val in enumerate(y_values):
        for j, kv in enumerate(kv_lengths):
            subset = df[(df[y_col] == y_val) & (df['kv_length'] == kv)]

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

                    # Calculate color with intensity based on margin of victory
                    # intensity: 0.3 at speedup=1.0, up to 1.0 at speedup>=2.0
                    intensity = min(1.0, 0.3 + (speedup - 1.0) * 0.7)
                    base_color = APPROACH_COLORS.get(winner_name, "#808080")
                    cell_colors[i][j] = adjust_color_saturation(base_color, intensity)

                    # Create hover text
                    hover_lines = [
                        f"{y_label}: {format_kv_length(y_val)}",
                        f"KV Length: {format_kv_length(kv)}",
                        f"<br><b>Winner: {shorten_approach_name(winner_name)} ({winner_time*1000:.2f}ms)</b><br>"
                    ]
                    for rank, (app, time) in enumerate(sorted_times, 1):
                        hover_lines.append(
                            f"{rank}. {shorten_approach_name(app)}: {time*1000:.2f}ms"
                        )
                    hover_text[i][j] = "<br>".join(hover_lines)

                elif len(times) == 1:
                    winner_name, winner_time = list(times.items())[0]
                    cell_data[(i, j)] = (winner_name, winner_time, None, None, 1.0)
                    # No margin, use base intensity
                    base_color = APPROACH_COLORS.get(winner_name, "#808080")
                    cell_colors[i][j] = adjust_color_saturation(base_color, 0.3)
                    hover_text[i][j] = f"{y_label}: {format_kv_length(y_val)}<br>KV: {format_kv_length(kv)}<br>{shorten_approach_name(winner_name)}: {winner_time*1000:.2f}ms"

    # Create figure with shapes for colored cells
    fig = go.Figure()

    # Add invisible scatter for hover info at cell centers
    x_labels = [format_kv_length(kv) for kv in kv_lengths]
    y_tick_labels = [format_kv_length(y) for y in y_values]

    # Add shapes (rectangles) for each cell with the computed color
    shapes = []
    for i in range(len(y_values)):
        for j in range(len(kv_lengths)):
            if cell_colors[i][j] is not None:
                shapes.append(dict(
                    type="rect",
                    x0=j - 0.5,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor=cell_colors[i][j],
                    line=dict(color="white", width=1),
                    layer="below",
                ))

    # Build text and hover data for scatter
    scatter_x = []
    scatter_y = []
    scatter_text = []
    scatter_hover = []

    for i in range(len(y_values)):
        for j in range(len(kv_lengths)):
            scatter_x.append(j)
            scatter_y.append(i)
            scatter_hover.append(hover_text[i][j])

            if (i, j) in cell_data:
                winner_name, winner_time, runner_name, runner_time, speedup = cell_data[(i, j)]
                winner_ms = winner_time * 1000
                short_winner = shorten_approach_name(winner_name)

                if runner_name:
                    runner_ms = runner_time * 1000
                    short_runner = shorten_approach_name(runner_name)
                    scatter_text.append(f"{short_winner}:{winner_ms:.0f}<br>{short_runner}:{runner_ms:.0f}<br>{speedup:.2f}x")
                else:
                    scatter_text.append(f"{short_winner}:{winner_ms:.0f}")
            else:
                scatter_text.append("")

    # Add scatter trace for text labels and hover
    fig.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='text',
        text=scatter_text,
        textfont=dict(size=7, color='white'),
        hovertext=scatter_hover,
        hoverinfo='text',
        showlegend=False,
    ))

    # Create legend for approaches
    legend_text = " | ".join([f"{shorten_approach_name(app)}" for app in approaches])

    fig.update_layout(
        title=f"Best Performer ({len(approaches)} approaches)<br><sub>{legend_text}</sub>",
        xaxis_title="KV Length",
        yaxis_title=y_label,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(kv_lengths))),
            ticktext=x_labels,
            range=[-0.5, len(kv_lengths) - 0.5],
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(y_values))),
            ticktext=y_tick_labels,
            range=[len(y_values) - 0.5, -0.5],  # Reversed: smallest values at top
        ),
        shapes=shapes,
        height=max(600, len(y_values) * 45 + 150),
        hovermode='closest',
        plot_bgcolor='#2d2d2d',
    )

    return fig


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
