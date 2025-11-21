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
        shorten_approach_name,
        get_available_approaches,
        adjust_color_saturation,
        APPROACH_COLORS,
        find_result_files,
        process_approach_data,
        render_cell_text,
        generate_titles,
        parse_mixed_scenario,
        create_common_argparser,
        resolve_input_file,
        resolve_output_file,
        process_batch_mode,
        initialize_matrices,
        populate_matrices,
        setup_colormap_pairwise,
        render_best_performer_cells,
        add_colorbar,
    )
except ImportError:
    from utils import (
        calculate_speedup,
        load_benchmark_results,
        format_number,
        shorten_approach_name,
        get_available_approaches,
        adjust_color_saturation,
        APPROACH_COLORS,
        find_result_files,
        process_approach_data,
        render_cell_text,
        generate_titles,
        parse_mixed_scenario,
        create_common_argparser,
        resolve_input_file,
        resolve_output_file,
        process_batch_mode,
        initialize_matrices,
        populate_matrices,
        setup_colormap_pairwise,
        render_best_performer_cells,
        add_colorbar,
    )


def create_mixed_heatmap(
    results_file: str,
    output_prefix: str = "plots/mixed_heatmap",
    dark_mode: bool = False,
    approaches: list = None,
    model_name: str = None,
    tp_degree: int = None,
):
    """
    Create hierarchical heatmaps for mixed batch benchmark results.

    Auto-switches between two visualization modes:
    - 2 approaches: Pairwise speedup heatmap (diverging colormap)
    - 3+ approaches: Best performer heatmap (categorical colors showing winner)

    Layout: Multiple images (one per prefill query length)
        - Each image: Multiple subplots (one per prefill KV length)
        - Each subplot: decode_batch (rows) × decode_kv (cols)

    Args:
        results_file: Path to benchmark results JSON
        output_prefix: Prefix for output files (will append _query{N}.png)
        dark_mode: Use dark theme
        approaches: List of approaches to compare (default: auto-detect all available)
        model_name: Optional model name to display in title
        tp_degree: Optional TP degree to display in title
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Auto-detect approaches if not specified
    if approaches is None:
        approaches = get_available_approaches(scenarios)
        print(f"Auto-detected {len(approaches)} approaches: {', '.join(approaches)}")

    if len(approaches) < 2:
        print(f"Error: Need at least 2 approaches to compare, found {len(approaches)}")
        return

    # Determine visualization mode
    mode = "pairwise" if len(approaches) == 2 else "best_performer"
    print(f"Using '{mode}' mode for {len(approaches)} approaches")

    # Parse and organize data
    # Structure depends on mode:
    # - Pairwise: by_prefill_query[pref_q][pref_kv][(dec_batch, dec_kv)] = (speedup, time1, time2)
    # - Best performer: by_prefill_query[pref_q][pref_kv][(dec_batch, dec_kv)] = (winner, time1, time2, speedup, color)
    by_prefill_query = defaultdict(lambda: defaultdict(dict))
    all_decode_batches = set()
    all_decode_kvs = set()
    all_prefill_queries = set()
    all_prefill_kvs = set()

    for scenario in scenarios:
        name = scenario['scenario_name']
        time_stats = scenario.get('approach_time_stats', {})

        # Parse scenario
        dec_batch, dec_kv, pref_query, pref_kv = parse_mixed_scenario(name)
        if dec_batch == 0 or dec_kv == 0 or pref_query == 0 or pref_kv == 0:
            continue

        # Extract median times from approach_time_stats
        approach_times_raw = {}
        for app in approaches:
            app_stats = time_stats.get(app, {})
            median_time = app_stats.get('median') if app_stats else None
            if median_time is not None:
                approach_times_raw[app] = median_time

        # Process approach data using shared function
        result = process_approach_data(approach_times_raw, approaches, mode)

        if result is None:
            continue

        # Store result in appropriate format for mixed heatmap
        if mode == "pairwise":
            # Pairwise: (speedup, time2, time1)
            by_prefill_query[pref_query][pref_kv][(dec_batch, dec_kv)] = result
        else:
            # Best performer: (winner_name, runner_up_name, winner_time, second_time, speedup, color)
            # Mixed heatmap expects: (winner_name, winner_time, second_time, speedup, color)
            winner_name, runner_up_name, winner_time, second_time, speedup, color = result
            by_prefill_query[pref_query][pref_kv][(dec_batch, dec_kv)] = (
                winner_name, winner_time, second_time, speedup, color
            )

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

            # Get subplot data
            subplot_data = by_prefill_query[pref_query][pref_kv]

            # Initialize matrices using consolidated function
            matrix, color_matrix, data_matrix = initialize_matrices(mode, n_decode_batches, n_decode_kvs)
            populate_matrices(mode, matrix, color_matrix, data_matrix, decode_batches, decode_kvs, subplot_data)

            # Colormap setup and cell rendering
            if mode == "pairwise":
                cmap, norm, vmin, vmax = setup_colormap_pairwise(matrix)
                im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            else:
                render_best_performer_cells(ax, color_matrix, dark_mode)
                im = None

            # Add value annotations using shared function
            short_names = (
                shorten_approach_name(approaches[0]) if len(approaches) >= 1 else "",
                shorten_approach_name(approaches[1]) if len(approaches) >= 2 else ""
            )
            font_sizes = {'speedup': 6, 'time': 5, 'na': 6}

            for j in range(n_decode_batches):
                for k in range(n_decode_kvs):
                    data = data_matrix.get((j, k))
                    render_cell_text(ax, k, j, data, mode, short_names, font_sizes)

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

        # Add colorbar using consolidated function
        short_names = (
            shorten_approach_name(approaches[0]) if len(approaches) >= 1 else "",
            shorten_approach_name(approaches[1]) if len(approaches) >= 2 else ""
        )
        add_colorbar(fig, im, axes, short_names, mode, single_subplot=False)

        # Main title - position higher to avoid overlap
        title = f'Mixed Batch: FA3 vs BatchAttention\nPrefill Query = {format_number(pref_query)}'
        if model_name and tp_degree is not None:
            title = f'{title} - {model_name} TP={tp_degree}'
        elif model_name:
            title = f'{title} - {model_name}'
        elif tp_degree is not None:
            title = f'{title} - TP={tp_degree}'

        fig.suptitle(
            title,
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
    import os

    # Create argument parser using consolidated function
    # Note: Mixed workload doesn't have workload_category filtering
    parser = create_common_argparser('mixed', include_workload_category=False)
    args = parser.parse_args()

    # Parse approaches list if provided
    approaches = None
    if args.approaches:
        approaches = [app.strip() for app in args.approaches.split(',')]

    # Determine mode: batch or single
    batch_mode = args.run_dir and args.model is None

    if batch_mode:
        # Batch mode: process all files in run directory
        result_files = find_result_files(args.run_dir, 'mixed')

        if not result_files:
            print(f"No mixed result files found in {args.run_dir}")
            exit(1)

        print(f"Batch mode: Found {len(result_files)} result file(s)")
        os.makedirs(args.output_dir, exist_ok=True)

        for model_name, tp_degree, input_file in result_files:
            output_prefix = os.path.join(args.output_dir, f"mixed_heatmap_{model_name}_tp{tp_degree}")
            print(f"\nProcessing: {model_name} TP={tp_degree}")
            print(f"  Input: {input_file}")
            print(f"  Output prefix: {output_prefix}")

            if approaches:
                print(f"  Using specified approaches: {', '.join(approaches)}")

            create_mixed_heatmap(input_file, output_prefix, args.dark, approaches, model_name, tp_degree)

        print(f"\n✓ Generated heatmap(s) in {args.output_dir}/")

    else:
        # Single mode: process one file
        input_file = resolve_input_file(args, 'mixed')
        output_prefix = args.output if args.output else os.path.join(args.output_dir, "mixed_heatmap")

        if approaches:
            print(f"Using specified approaches: {', '.join(approaches)}")

        print(f"Reading benchmark results from: {input_file}")
        create_mixed_heatmap(input_file, output_prefix, args.dark, approaches, args.model, args.tp_degree)
