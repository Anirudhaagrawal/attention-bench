#!/usr/bin/env python3
"""
Generate hierarchical heatmap visualization for prefill benchmarks.
Shows FA3 vs BatchAttention performance across batch, query, and KV dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict
try:
    from .utils import (
        parse_prefill_scenario,
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )
except ImportError:
    from utils import (
        parse_prefill_scenario,
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )


def create_prefill_heatmap(
    results_file: str,
    output_file: str = "plots/prefill_heatmap.png",
    dark_mode: bool = False,
):
    """
    Create hierarchical heatmap for prefill benchmark results.

    Layout: Multiple subplots (one per batch size)
        - Each subplot: query (rows) × kv (cols)
        - Color: Green = FA3 wins, Red = BA wins
        - Values: Speedup ratio (BA time / FA3 time)

    Args:
        results_file: Path to benchmark results JSON
        output_file: Path to save the heatmap
        dark_mode: Use dark theme
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Parse and organize data by batch size
    # Store (speedup, ba_time, fa3_time) tuples
    by_batch = defaultdict(dict)
    all_queries = set()
    all_kvs = set()

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
        batch, query, kv = parse_prefill_scenario(name)
        if batch == 0 or query == 0 or kv == 0:
            continue

        # Skip configurations with very small KV (32 and 64)
        if kv in [32, 64]:
            continue

        # Calculate speedup
        speedup = calculate_speedup(fa3_time, ba_time)
        by_batch[batch][(query, kv)] = (speedup, ba_time, fa3_time)

        all_queries.add(query)
        all_kvs.add(kv)

    if not by_batch:
        print(f"No valid prefill scenarios found in {results_file}")
        return

    # Sort dimensions
    batch_sizes = sorted(by_batch.keys())
    queries = sorted(all_queries)
    kvs = sorted(all_kvs)

    n_batches = len(batch_sizes)
    n_queries = len(queries)
    n_kvs = len(kvs)

    # Scale figure size based on grid dimensions for better readability
    # Larger grids need more space - increased for better visibility
    width_per_batch = max(12, n_kvs * 0.9)  # At least 12, scale with KV count
    height = max(10, n_queries * 1.0)  # At least 10, scale with query count

    # Create subplot grid
    fig, axes = plt.subplots(
        1,
        n_batches,
        figsize=(width_per_batch * n_batches, height),
        sharey=True,
    )

    # Handle single subplot case
    if n_batches == 1:
        axes = [axes]

    # Set theme
    if dark_mode:
        plt.style.use('dark_background')
        text_color = 'white'
    else:
        text_color = 'black'

    # Create heatmap for each batch size
    vmin, vmax = 0.5, 2.0  # Speedup range
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlGn.copy()  # Red-Yellow-Green
    cmap.set_bad(color='lightgray')  # Color for NaN/missing values

    for i, batch in enumerate(batch_sizes):
        ax = axes[i]

        # Create matrices for this batch (use NaN for missing data)
        matrix = np.full((len(queries), len(kvs)), np.nan)
        ba_times_matrix = np.full((len(queries), len(kvs)), np.nan)
        fa3_times_matrix = np.full((len(queries), len(kvs)), np.nan)

        for j, query in enumerate(queries):
            for k, kv in enumerate(kvs):
                if (query, kv) in by_batch[batch]:
                    speedup, ba_time, fa3_time = by_batch[batch][(query, kv)]
                    matrix[j, k] = speedup
                    ba_times_matrix[j, k] = ba_time
                    fa3_times_matrix[j, k] = fa3_time

        # Plot heatmap
        im = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='nearest',
        )

        # Add value annotations - scale font size based on grid size
        # Smaller font for larger grids
        speedup_fontsize = max(6, min(9, 140 // max(n_queries, n_kvs)))
        time_fontsize = max(5, min(7, 100 // max(n_queries, n_kvs)))
        na_fontsize = max(5, min(8, 120 // max(n_queries, n_kvs)))

        for j in range(len(queries)):
            for k in range(len(kvs)):
                value = matrix[j, k]

                # Skip missing data
                if np.isnan(value):
                    ax.text(
                        k,
                        j,
                        'N/A',
                        ha='center',
                        va='center',
                        color='gray',
                        fontsize=na_fontsize,
                        style='italic',
                    )
                    continue

                # Choose text color based on background
                if value < 0.8 or value > 1.6:
                    text_color_cell = 'white'
                else:
                    text_color_cell = 'black'

                # Speedup ratio
                ax.text(
                    k,
                    j - 0.25,
                    f'{value:.2f}x',
                    ha='center',
                    va='center',
                    color=text_color_cell,
                    fontsize=speedup_fontsize,
                    fontweight='bold',
                )

                # BA time (in ms) - teal color
                ba_ms = ba_times_matrix[j, k] * 1000
                ax.text(
                    k,
                    j + 0.08,
                    f'BA:{ba_ms:.0f}',
                    ha='center',
                    va='center',
                    color='#16A085',
                    fontsize=time_fontsize,
                    fontweight='bold',
                )

                # FA3 time (in ms) - coral color
                fa3_ms = fa3_times_matrix[j, k] * 1000
                ax.text(
                    k,
                    j + 0.35,
                    f'FA:{fa3_ms:.0f}',
                    ha='center',
                    va='center',
                    color='#E74C3C',
                    fontsize=time_fontsize,
                    fontweight='bold',
                )

        # Set ticks and labels
        ax.set_xticks(range(len(kvs)))
        ax.set_xticklabels([format_number(kv) for kv in kvs], rotation=45, ha='right')
        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels([format_number(q) for q in queries])

        # Subplot title
        ax.set_title(f'Batch = {batch}', fontsize=12, fontweight='bold', pad=10)

        # Grid
        ax.set_xticks(np.arange(len(kvs)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(queries)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # Labels
        if i == 0:
            ax.set_ylabel('Query Tokens', fontsize=11, fontweight='bold')
        ax.set_xlabel('KV Length', fontsize=11, fontweight='bold')

    # Add colorbar - position depends on number of subplots
    if n_batches == 1:
        # For single subplot, use fraction to keep colorbar contained
        cbar = fig.colorbar(im, ax=axes[0], orientation='vertical',
                           pad=0.08, fraction=0.046)
    else:
        # For multiple subplots, use across all axes
        cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                           pad=0.05, aspect=30, shrink=0.8)
    cbar.set_label('Speedup (BA/FA3)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Main title
    fig.suptitle(
        'Prefill: FA3 vs BatchAttention Performance',
        fontsize=14,
        fontweight='bold',
        y=0.96,
    )

    # Subtitle
    fig.text(
        0.5,
        0.01,
        'Green: FA3 faster  |  Red: BatchAttention faster  |  Yellow: Similar performance',
        ha='center',
        fontsize=10,
        style='italic',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Prefill hierarchical heatmap saved to: {output_file}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate prefill hierarchical heatmap")
    parser.add_argument(
        "--input",
        type=str,
        default="results/config_prefill_all_combinations.json",
        help="Path to prefill benchmark results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/prefill_heatmap.png",
        help="Output file path",
    )
    parser.add_argument("--dark", action="store_true", help="Use dark mode")
    args = parser.parse_args()

    create_prefill_heatmap(args.input, args.output, args.dark)
