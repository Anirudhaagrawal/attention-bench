"""Helper functions for building heatmap visualizations.

This module extracts common matrix-building logic from heatmap generation
functions to eliminate duplication and improve maintainability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .workload_config import WorkloadConfig
from .colors import APPROACH_COLORS, APPROACH_GROUPS
from .formatting import shorten_approach_name
from .approach_grouping import apply_approach_grouping


@dataclass
class HeatmapGrid:
    """Data structure for heatmap matrix grid."""
    x_values: List[int]
    y_values: List[int]
    x_labels: List[str]
    y_labels: List[str]
    hover_text: List[List[str]]

    def cell_count(self) -> int:
        """Return total number of cells in grid."""
        return len(self.y_values) * len(self.x_values)

    def get_subset(self, df: pd.DataFrame, i: int, j: int,
                   config: WorkloadConfig) -> pd.DataFrame:
        """Get DataFrame subset for cell (i, j)."""
        return df[
            (df[config.y_axis_column] == self.y_values[i]) &
            (df[config.x_axis_column] == self.x_values[j])
        ]


def prepare_heatmap_data(
    df: pd.DataFrame,
    workload_type: str,
    approaches: List[str],
    metric: str = "median",
    prefill_query: int = None,
    prefill_kv: int = None
) -> Tuple[pd.DataFrame, List[str], WorkloadConfig]:
    """Prepare and filter data for heatmap generation.

    Handles workload config detection, mixed workload filtering,
    prefill KV length exclusions, and approach grouping.

    Args:
        df: Raw benchmark DataFrame
        workload_type: Type of workload
        approaches: List of approach names
        metric: Metric column suffix
        prefill_query: Optional prefill query for mixed
        prefill_kv: Optional prefill KV for mixed

    Returns:
        Tuple of (filtered_df, final_approaches, config)
    """
    # Detect workload type if not provided
    if workload_type is None and 'workload_type' in df.columns:
        workload_types = df['workload_type'].unique()
        if len(workload_types) == 1:
            workload_type = workload_types[0]

    if workload_type is None:
        workload_type = "decode"  # Default fallback

    # Get workload configuration
    config = WorkloadConfig.for_workload(workload_type)

    # Filter data based on workload type
    if config.is_mixed():
        # Mixed workloads: filter by prefill configuration
        if prefill_query is None or prefill_kv is None:
            raise ValueError("Missing prefill configuration for mixed workload")

        df_filtered = df[
            (df[config.prefill_query_column] == prefill_query) &
            (df[config.prefill_kv_column] == prefill_kv) &
            (df[config.y_axis_column] > 0) &
            (df[config.x_axis_column] > 0)
        ].copy()

        final_approaches = approaches
    else:
        # Regular workloads: filter out invalid data
        df_filtered = df[
            (df[config.y_axis_column] > 0) &
            (df[config.x_axis_column] > 0)
        ].copy()

        # Skip KV lengths 32 and 64 for prefill
        if workload_type == "prefill" and config.x_axis_column == "kv_length":
            df_filtered = df_filtered[~df_filtered['kv_length'].isin([32, 64])].copy()

        # Apply approach grouping based on workload type
        if workload_type and workload_type != "mixed":
            df_filtered, grouped_approaches = apply_approach_grouping(
                df_filtered, workload_type, approaches, metric
            )

            # Map original approach names to grouped names if they were grouped
            final_approaches = []
            for approach in approaches:
                mapped_approach = approach
                for group_name, members in APPROACH_GROUPS.get(workload_type, {}).items():
                    if approach in members and group_name in grouped_approaches:
                        mapped_approach = group_name
                        break
                final_approaches.append(mapped_approach)
        else:
            final_approaches = approaches

    return df_filtered, final_approaches, config


def build_heatmap_grid(
    df: pd.DataFrame,
    config: WorkloadConfig
) -> HeatmapGrid:
    """Build the coordinate grid for heatmap matrices.

    Args:
        df: Filtered DataFrame
        config: Workload configuration

    Returns:
        HeatmapGrid with axes and initialized hover_text
    """
    # Get unique x and y values
    y_values = sorted(df[config.y_axis_column].unique())
    x_values = sorted(df[config.x_axis_column].unique())

    # Format labels
    y_labels = [config.format_y_axis_value(v) for v in y_values]
    x_labels = [config.format_x_axis_value(v) for v in x_values]

    # Initialize hover text matrix
    hover_text = [['' for _ in x_values] for _ in y_values]

    return HeatmapGrid(
        x_values=x_values,
        y_values=y_values,
        x_labels=x_labels,
        y_labels=y_labels,
        hover_text=hover_text
    )


def build_hover_text_speedup(
    config: WorkloadConfig,
    y_val: int,
    x_val: int,
    speedup: float,
    time1: float,
    time2: float,
    approach1: str,
    approach2: str
) -> str:
    """Build hover text for a speedup heatmap cell.

    Args:
        config: Workload configuration for formatting
        y_val: Raw y-axis value
        x_val: Raw x-axis value
        speedup: Speedup value (time2/time1)
        time1: Time for approach1 in seconds
        time2: Time for approach2 in seconds
        approach1: First approach name
        approach2: Second approach name

    Returns:
        HTML-formatted hover text string
    """
    short1 = shorten_approach_name(approach1)
    short2 = shorten_approach_name(approach2)

    return (
        f"{config.y_axis_label}: {config.format_y_axis_value(y_val)}<br>"
        f"{config.x_axis_label}: {config.format_x_axis_value(x_val)}<br>"
        f"Speedup: {speedup:.3f}x<br>"
        f"{short1}: {time1*1000:.2f}ms<br>"
        f"{short2}: {time2*1000:.2f}ms"
    )


def build_hover_text_best_performer(
    config: WorkloadConfig,
    y_val: int,
    x_val: int,
    approach_times: List[Tuple[str, float]]
) -> str:
    """Build hover text for a best performer heatmap cell.

    Args:
        config: Workload configuration for formatting
        y_val: Raw y-axis value
        x_val: Raw x-axis value
        approach_times: List of (approach_name, time_in_seconds) tuples, sorted by time

    Returns:
        HTML-formatted hover text string
    """
    lines = [
        f"{config.y_axis_label}: {config.format_y_axis_value(y_val)}",
        f"{config.x_axis_label}: {config.format_x_axis_value(x_val)}",
    ]

    for approach, time in approach_times:
        short_name = shorten_approach_name(approach)
        lines.append(f"{short_name}: {time*1000:.2f}ms")

    return "<br>".join(lines)


def generate_speedup_colorscale(
    speedup_matrix: np.ndarray
) -> Tuple[str, float, float, List[float], List[str]]:
    """Generate log-scale colorscale for speedup heatmaps.

    Args:
        speedup_matrix: Matrix of speedup values (may contain NaN)

    Returns:
        Tuple of (colorscale_name, zmin, zmax, tick_vals, tick_texts)
    """
    # Convert to log scale for better color distribution
    # Log scale makes 0.5x and 2.0x equidistant from 1.0x
    log_speedup_matrix = np.log(speedup_matrix)  # NaN values preserved

    # Calculate symmetric range in log space (centered at 0 = log(1.0))
    valid_log_speedups = log_speedup_matrix[~np.isnan(log_speedup_matrix)]
    if len(valid_log_speedups) > 0:
        min_log = np.min(valid_log_speedups)
        max_log = np.max(valid_log_speedups)
        max_log_deviation = max(abs(min_log), abs(max_log))
        # Ensure minimum range for visibility
        max_log_deviation = max(max_log_deviation, 0.05)
        zmin = -max_log_deviation
        zmax = max_log_deviation
    else:
        # Fallback if no valid data: ±log(2) = [0.5x, 2.0x]
        zmin = -0.693
        zmax = 0.693

    # Generate colorbar tick labels for intuitive speedup display
    # Choose tick values that span the range nicely
    tick_speedups = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    tick_vals = [np.log(x) for x in tick_speedups]
    tick_texts = [f"{x:.1f}x" if x != 1.0 else "1x" for x in tick_speedups]

    # Filter ticks to only show those within our range
    valid_ticks = [(v, t) for v, t in zip(tick_vals, tick_texts) if zmin <= v <= zmax]
    if valid_ticks:
        tick_vals_filtered = [v for v, t in valid_ticks]
        tick_texts_filtered = [t for v, t in valid_ticks]
    else:
        # If no standard ticks fit, use min/mid/max
        tick_vals_filtered = [zmin, 0, zmax]
        tick_texts_filtered = [f"{np.exp(zmin):.2f}x", "1x", f"{np.exp(zmax):.2f}x"]

    return "RdBu_r", zmin, zmax, tick_vals_filtered, tick_texts_filtered


def generate_approach_colorscale(
    approaches: List[str]
) -> List[List]:
    """Generate gradient colorscale for multi-approach comparison.

    Each approach gets a color range with varying intensity.

    Args:
        approaches: List of approach names

    Returns:
        Plotly colorscale list [[position, color], ...]
    """
    from .colors import adjust_color_saturation

    n_approaches = len(approaches)
    colorscale = []

    for idx, approach in enumerate(approaches):
        base_color = APPROACH_COLORS.get(approach, "#808080")

        # Define approach range in [0, 1]
        start = idx / n_approaches
        end = (idx + 1) / n_approaches

        # Create gradient within approach range
        # Dark (low intensity) to bright (high intensity)
        colorscale.append([start, adjust_color_saturation(base_color, 0.3)])
        colorscale.append([end, adjust_color_saturation(base_color, 1.0)])

    return colorscale


def build_cell_annotations_speedup(
    cell_data: Dict[Tuple[int, int], Tuple[float, float, float]],
    x_values: List[int],
    y_values: List[int],
    approach1: str,
    approach2: str
) -> Tuple[List[List[str]], int, int, int]:
    """Build text annotations for speedup heatmap cells.

    Args:
        cell_data: Dict mapping (i,j) -> (speedup, time1, time2)
        x_values: X-axis values
        y_values: Y-values
        approach1: First approach name
        approach2: Second approach name

    Returns:
        Tuple of (text_matrix, approach1_wins, approach2_wins, ties)
    """
    text_matrix = [['' for _ in x_values] for _ in y_values]
    short1 = shorten_approach_name(approach1)
    short2 = shorten_approach_name(approach2)

    approach1_wins = 0
    approach2_wins = 0
    ties = 0

    for (i, j), (speedup, time1, time2) in cell_data.items():
        # Determine winner (with 1% tolerance for ties)
        if speedup > 1.01:
            # approach1 is faster (baseline wins)
            text_matrix[i][j] = f"{short1}<br>{speedup:.2f}x"
            approach1_wins += 1
        elif speedup < 0.99:
            # approach2 is faster
            text_matrix[i][j] = f"{short2}<br>{speedup:.2f}x"
            approach2_wins += 1
        else:
            # Tie
            text_matrix[i][j] = "~1x"
            ties += 1

    return text_matrix, approach1_wins, approach2_wins, ties


def build_cell_annotations_best_performer(
    cell_data: Dict[Tuple[int, int], Tuple[str, float, str, float, float]],
    x_values: List[int],
    y_values: List[int]
) -> Tuple[List[List[str]], Dict[str, int]]:
    """Build text annotations for best performer heatmap cells.

    Args:
        cell_data: Dict mapping (i,j) -> (winner, winner_time, runner_up, runner_time, speedup)
        x_values: X-axis values
        y_values: Y-axis values

    Returns:
        Tuple of (text_matrix, winner_counts)
    """
    text_matrix = [['' for _ in x_values] for _ in y_values]
    winner_counts = {}

    for (i, j), (winner, winner_time, runner_up, runner_time, speedup) in cell_data.items():
        short_winner = shorten_approach_name(winner)
        short_runner = shorten_approach_name(runner_up)

        # Build cell text
        if speedup > 1.01:
            # Clear winner
            text_matrix[i][j] = f"{short_winner}<br>vs {short_runner}<br>{speedup:.2f}x"
        else:
            # Tie
            text_matrix[i][j] = f"{short_winner}<br>~{short_runner}"

        # Count wins
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    return text_matrix, winner_counts


def build_heatmap_layout(
    config: WorkloadConfig,
    y_values: List[int],
    title: str,
    additional_shapes: List = None,
    additional_annotations: List = None
) -> Dict[str, Any]:
    """Build standard layout configuration for heatmaps.

    Args:
        config: Workload configuration
        y_values: Y-axis values for height calculation
        title: Plot title
        additional_shapes: Optional shapes (for legends)
        additional_annotations: Optional annotations (for legends)

    Returns:
        Dict of layout parameters for fig.update_layout()
    """
    layout = {
        'title': title,
        'xaxis_title': config.x_axis_label,
        'yaxis_title': config.y_axis_label,
        'xaxis': dict(
            tickmode='array',
            side='top',
        ),
        'yaxis': dict(
            tickmode='array',
            autorange='reversed',
        ),
        'height': len(y_values) * 70 + 200,
        'plot_bgcolor': 'white',
        'font': dict(size=12),
        'margin': dict(l=50, r=10, t=80, b=80),
    }

    if additional_shapes:
        layout['shapes'] = additional_shapes
    if additional_annotations:
        layout['annotations'] = additional_annotations

    return layout
