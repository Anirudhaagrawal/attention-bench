#!/usr/bin/env python3
"""
Generate heatmap visualization for decode-only benchmarks.
Shows FA3 vs BatchAttention performance across batch size and KV length.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
try:
    from .utils import (
        parse_decode_scenario,
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )
except ImportError:
    from utils import (
        parse_decode_scenario,
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )


def create_decode_heatmap(
    results_file: str,
    output_file: str = "plots/decode_heatmap.png",
    dark_mode: bool = False,
):
    """
    Create a 2D heatmap for decode benchmark results.

    Args:
        results_file: Path to benchmark results JSON
        output_file: Path to save the heatmap
        dark_mode: Use dark theme

    The heatmap shows:
        - Rows: Batch sizes
        - Cols: KV lengths
        - Color: Green = FA3 wins, Red = BA wins
        - Values: Speedup ratio (BA time / FA3 time)
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Parse and organize data
    # Store (speedup, ba_time, fa3_time) tuples
    speedup_data = {}
    batch_sizes = set()
    kv_lengths = set()

    for scenario in scenarios:
        name = scenario['scenario_name']
        time_stats = scenario.get('approach_time_stats', {})

        # Get FA3 and BA median times
        fa3_stats = time_stats.get('official_fa3', {})
        ba_stats = time_stats.get('flashinfer_batch_attention', {})

        fa3_time = fa3_stats.get('median') if fa3_stats else None
        ba_time = ba_stats.get('median') if ba_stats else None

        if fa3_time is None or ba_time is None:
            continue
        if fa3_time == float('inf') or ba_time == float('inf'):
            continue

        # Parse scenario
        batch, kv = parse_decode_scenario(name)
        if batch == 0 or kv == 0:
            continue

        # Calculate speedup
        speedup = calculate_speedup(fa3_time, ba_time)
        speedup_data[(batch, kv)] = (speedup, ba_time, fa3_time)

        batch_sizes.add(batch)
        kv_lengths.add(kv)

    if not speedup_data:
        print(f"No valid decode scenarios found in {results_file}")
        return

    # Create DataFrame
    batch_sizes = sorted(batch_sizes)
    kv_lengths = sorted(kv_lengths)

    # Initialize matrices with NaN for missing data
    matrix = np.full((len(batch_sizes), len(kv_lengths)), np.nan)
    ba_times_matrix = np.full((len(batch_sizes), len(kv_lengths)), np.nan)
    fa3_times_matrix = np.full((len(batch_sizes), len(kv_lengths)), np.nan)

    for i, batch in enumerate(batch_sizes):
        for j, kv in enumerate(kv_lengths):
            if (batch, kv) in speedup_data:
                speedup, ba_time, fa3_time = speedup_data[(batch, kv)]
                matrix[i, j] = speedup
                ba_times_matrix[i, j] = ba_time
                fa3_times_matrix[i, j] = fa3_time

    # Create DataFrame with formatted labels
    df = pd.DataFrame(
        matrix,
        index=[f"b{b}" for b in batch_sizes],
        columns=[format_number(kv) for kv in kv_lengths],
    )

    # Create figure manually for better control
    fig_mpl, ax = plt.subplots(figsize=(10, 8))

    # Use diverging norm centered at 1.0 (ignore NaN values)
    vmin = max(0.3, np.nanmin(matrix))
    vmax = min(2.0, np.nanmax(matrix))
    
    # Ensure vmin < vcenter < vmax for TwoSlopeNorm
    if vmin >= 1.0:
        vmin = 0.99
    if vmax <= 1.0:
        vmax = 1.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # Create heatmap with NaN handling
    cmap = plt.cm.RdYlGn.copy()  # Red (BA wins) → Yellow → Green (FA3 wins)
    cmap.set_bad(color='lightgray')  # Color for NaN/missing values

    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='nearest',
    )

    # Add value annotations
    for i in range(len(batch_sizes)):
        for j in range(len(kv_lengths)):
            value = matrix[i, j]

            # Skip missing data - show N/A
            if np.isnan(value):
                ax.text(j, i, 'N/A',
                       ha='center', va='center',
                       color='gray', fontsize=10, style='italic')
                continue

            # Choose text color based on background
            if value < 0.7 or value > 1.4:
                text_color = 'white'
            else:
                text_color = 'black'

            # Speedup ratio
            ax.text(j, i - 0.25, f'{value:.2f}x',
                   ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')

            # BA time (in ms) - teal color
            ba_ms = ba_times_matrix[i, j] * 1000
            ax.text(j, i + 0.08, f'BA:{ba_ms:.0f}',
                   ha='center', va='center',
                   color='#16A085', fontsize=7, fontweight='bold')

            # FA3 time (in ms) - coral color
            fa3_ms = fa3_times_matrix[i, j] * 1000
            ax.text(j, i + 0.35, f'FA:{fa3_ms:.0f}',
                   ha='center', va='center',
                   color='#E74C3C', fontsize=7, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(range(len(kv_lengths)))
    ax.set_xticklabels([format_number(kv) for kv in kv_lengths])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f"b{b}" for b in batch_sizes])

    # Add grid
    ax.set_xticks(np.arange(len(kv_lengths)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(batch_sizes)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Customize axes
    ax.set_xlabel('KV Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_title('Decode: FA3 vs BatchAttention Performance',
                fontsize=14, fontweight='bold', pad=15)

    # Add colorbar
    cbar = fig_mpl.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Speedup (BA/FA3)', rotation=270, labelpad=20,
                   fontsize=11, fontweight='bold')

    # Add subtitle explaining colors
    fig_mpl.text(
        0.5,
        0.02,
        'Green: FA3 faster  |  Red: BatchAttention faster  |  Value: Speedup ratio (BA/FA3)',
        ha='center',
        fontsize=10,
        style='italic',
    )

    # Save
    import os
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    fig_mpl.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    print(f"✓ Decode heatmap saved to: {output_file}")

    return fig_mpl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate decode heatmap")
    parser.add_argument(
        "--input",
        type=str,
        default="results/config_decode_all_combinations.json",
        help="Path to decode benchmark results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/decode_heatmap.png",
        help="Output file path",
    )
    parser.add_argument("--dark", action="store_true", help="Use dark mode")
    args = parser.parse_args()

    create_decode_heatmap(args.input, args.output, args.dark)
