#!/usr/bin/env python3
"""
Generate hierarchical heatmap visualization for mixed batch benchmarks.
Shows FA3 vs BatchAttention performance across decode and prefill dimensions.

Hierarchy:
- Per image (7 total): prefill_query_length
- Per subplot (10 per image): prefill_kv_length
- Heatmap axes: decode_batch (rows) × decode_kv (cols)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict
from typing import Tuple

try:
    from .utils import (
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )
except ImportError:
    from utils import (
        calculate_speedup,
        load_benchmark_results,
        format_number,
    )


def parse_mixed_scenario(scenario_name: str) -> Tuple[int, int, int, int]:
    """
    Parse mixed batch scenario name to extract parameters.

    Format: mixed_dec_b{batch}_kv{kv}_pref_q{query}_kv{prefill_kv}

    Examples:
        "mixed_dec_b1_kv1k_pref_q128_kv128" → (1, 1024, 128, 128)
        "mixed_dec_b64_kv32k_pref_q4k_kv16k" → (64, 32768, 4096, 16384)

    Returns:
        (decode_batch, decode_kv, prefill_query, prefill_kv)
    """
    decode_batch = 0
    decode_kv = 0
    prefill_query = 0
    prefill_kv = 0

    # Extract decode batch
    if m := re.search(r'dec_b(\d+)', scenario_name):
        decode_batch = int(m.group(1))

    # Extract decode KV (appears after dec_b, before pref)
    if m := re.search(r'dec_b\d+_kv(\d+)k', scenario_name):
        decode_kv = int(m.group(1)) * 1024
    elif m := re.search(r'dec_b\d+_kv(\d+)', scenario_name):
        decode_kv = int(m.group(1))

    # Extract prefill query
    if m := re.search(r'pref_q(\d+)k', scenario_name):
        prefill_query = int(m.group(1)) * 1024
    elif m := re.search(r'pref_q(\d+)', scenario_name):
        prefill_query = int(m.group(1))

    # Extract prefill KV (last kv in the string)
    if m := re.search(r'pref_q\d+k?_kv(\d+)k', scenario_name):
        prefill_kv = int(m.group(1)) * 1024
    elif m := re.search(r'pref_q\d+k?_kv(\d+)', scenario_name):
        prefill_kv = int(m.group(1))

    return (decode_batch, decode_kv, prefill_query, prefill_kv)


def create_mixed_heatmap(
    results_file: str,
    output_prefix: str = "plots/mixed_heatmap",
    dark_mode: bool = False,
):
    """
    Create hierarchical heatmaps for mixed batch benchmark results.

    Layout: Multiple images (one per prefill query length)
        - Each image: Multiple subplots (one per prefill KV length)
        - Each subplot: decode_batch (rows) × decode_kv (cols)
        - Color: Green = FA3 wins, Red = BA wins
        - Values: Speedup ratio (BA time / FA3 time)

    Args:
        results_file: Path to benchmark results JSON
        output_prefix: Prefix for output files (will append _query{N}.png)
        dark_mode: Use dark theme
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Parse and organize data
    # Structure: by_prefill_query[pref_q][pref_kv][(dec_batch, dec_kv)] = (speedup, ba_time, fa3_time)
    by_prefill_query = defaultdict(lambda: defaultdict(dict))
    all_decode_batches = set()
    all_decode_kvs = set()
    all_prefill_queries = set()
    all_prefill_kvs = set()

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
        dec_batch, dec_kv, pref_query, pref_kv = parse_mixed_scenario(name)
        if dec_batch == 0 or dec_kv == 0 or pref_query == 0 or pref_kv == 0:
            continue

        # Calculate speedup
        speedup = calculate_speedup(fa3_time, ba_time)
        by_prefill_query[pref_query][pref_kv][(dec_batch, dec_kv)] = (speedup, ba_time, fa3_time)

        all_decode_batches.add(dec_batch)
        all_decode_kvs.add(dec_kv)
        all_prefill_queries.add(pref_query)
        all_prefill_kvs.add(pref_kv)

    if not by_prefill_query:
        print(f"No valid mixed batch scenarios found in {results_file}")
        return

    # Sort dimensions
    decode_batches = sorted(all_decode_batches)
    decode_kvs = sorted(all_decode_kvs)
    prefill_queries = sorted(all_prefill_queries)
    prefill_kvs = sorted(all_prefill_kvs)

    n_decode_batches = len(decode_batches)
    n_decode_kvs = len(decode_kvs)
    n_prefill_kvs = len(prefill_kvs)

    # Set theme
    if dark_mode:
        plt.style.use('dark_background')

    # Create one image per prefill query length
    vmin, vmax = 0.5, 2.0  # Speedup range
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')

    for pref_query in prefill_queries:
        # Determine subplot grid layout (2 rows × 5 cols for 10 subplots)
        n_rows = 2
        n_cols = (n_prefill_kvs + n_rows - 1) // n_rows  # Ceiling division

        # Scale figure size - make it bigger for readability
        width_per_subplot = max(8, n_decode_kvs * 1.2)
        height_per_subplot = max(6, n_decode_batches * 0.8)
        fig_width = width_per_subplot * n_cols
        fig_height = height_per_subplot * n_rows + 3  # Extra space for title

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
        )

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Create heatmap for each prefill KV length
        for i, pref_kv in enumerate(prefill_kvs):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]

            # Create matrices for this prefill config
            matrix = np.full((n_decode_batches, n_decode_kvs), np.nan)
            ba_times = np.full((n_decode_batches, n_decode_kvs), np.nan)
            fa3_times = np.full((n_decode_batches, n_decode_kvs), np.nan)

            for j, dec_batch in enumerate(decode_batches):
                for k, dec_kv in enumerate(decode_kvs):
                    key = (dec_batch, dec_kv)
                    if key in by_prefill_query[pref_query][pref_kv]:
                        speedup, ba_time, fa3_time = by_prefill_query[pref_query][pref_kv][key]
                        matrix[j, k] = speedup
                        ba_times[j, k] = ba_time
                        fa3_times[j, k] = fa3_time

            # Plot heatmap
            im = ax.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                aspect='auto',
                interpolation='nearest',
            )

            # Add value annotations - larger font sizes
            speedup_fontsize = max(8, min(11, 110 // max(n_decode_batches, n_decode_kvs)))
            time_fontsize = max(6, min(8, 80 // max(n_decode_batches, n_decode_kvs)))
            na_fontsize = max(7, min(9, 90 // max(n_decode_batches, n_decode_kvs)))

            for j in range(n_decode_batches):
                for k in range(n_decode_kvs):
                    value = matrix[j, k]

                    if np.isnan(value):
                        ax.text(
                            k, j, 'N/A',
                            ha='center', va='center',
                            color='gray', fontsize=na_fontsize,
                            style='italic',
                        )
                        continue

                    # Choose text color based on background
                    if value < 0.8 or value > 1.6:
                        text_color = 'white'
                    else:
                        text_color = 'black'

                    # Speedup ratio
                    ax.text(
                        k, j - 0.25, f'{value:.2f}x',
                        ha='center', va='center',
                        color=text_color, fontsize=speedup_fontsize,
                        fontweight='bold',
                    )

                    # BA time (in ms) - teal color
                    ba_ms = ba_times[j, k] * 1000
                    ax.text(
                        k, j + 0.08, f'BA:{ba_ms:.0f}',
                        ha='center', va='center',
                        color='#16A085', fontsize=time_fontsize,
                        fontweight='bold',
                    )

                    # FA3 time (in ms) - coral color
                    fa3_ms = fa3_times[j, k] * 1000
                    ax.text(
                        k, j + 0.35, f'FA:{fa3_ms:.0f}',
                        ha='center', va='center',
                        color='#E74C3C', fontsize=time_fontsize,
                        fontweight='bold',
                    )

            # Set ticks and labels - larger font sizes
            ax.set_xticks(range(n_decode_kvs))
            ax.set_xticklabels([format_number(kv) for kv in decode_kvs],
                              rotation=45, ha='right', fontsize=11)
            ax.set_yticks(range(n_decode_batches))
            ax.set_yticklabels([str(b) for b in decode_batches], fontsize=11)

            # Subplot title
            ax.set_title(f'Prefill KV = {format_number(pref_kv)}',
                        fontsize=13, fontweight='bold', pad=12)

            # Grid
            ax.set_xticks(np.arange(n_decode_kvs) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_decode_batches) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

            # Axis labels (only on edges) - larger font sizes
            if i % n_cols == 0:
                ax.set_ylabel('Decode Batch', fontsize=12, fontweight='bold')
            if i >= (n_rows - 1) * n_cols:
                ax.set_xlabel('Decode KV', fontsize=12, fontweight='bold')

        # Hide unused subplots
        for i in range(n_prefill_kvs, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Add colorbar
        cbar = fig.colorbar(
            im, ax=axes, orientation='vertical',
            pad=0.02, aspect=30, shrink=0.8
        )
        cbar.set_label('Speedup (BA/FA3)', rotation=270, labelpad=25,
                      fontsize=13, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Main title - position higher to avoid overlap
        fig.suptitle(
            f'Mixed Batch: FA3 vs BatchAttention\nPrefill Query = {format_number(pref_query)}',
            fontsize=18,
            fontweight='bold',
            y=1.05,
        )

        # Subtitle
        fig.text(
            0.5, -0.02,
            'Green: FA3 faster  |  Red: BatchAttention faster  |  Yellow: Similar',
            ha='center', fontsize=11, style='italic',
        )

        # Layout is handled by constrained_layout=True

        # Save
        output_file = f"{output_prefix}_query{format_number(pref_query)}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")

    print(f"\nGenerated {len(prefill_queries)} heatmap images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate mixed batch hierarchical heatmaps"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/config_mixed_all_combinations.json",
        help="Path to mixed batch benchmark results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/mixed_heatmap",
        help="Output file prefix (will append _query{N}.png)",
    )
    parser.add_argument("--dark", action="store_true", help="Use dark mode")
    args = parser.parse_args()

    create_mixed_heatmap(args.input, args.output, args.dark)
