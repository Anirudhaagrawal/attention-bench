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
        load_benchmark_results,
        format_number,
        shorten_approach_name,
        get_available_approaches,
        apply_approach_grouping,
        APPROACH_COLORS,
        process_approach_data,
        render_cell_text,
        generate_titles,
        find_result_files,
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
    from .workload_categorizer import (
        categorize_workload,
    )
except ImportError:
    from utils import (
        parse_prefill_scenario,
        load_benchmark_results,
        format_number,
        shorten_approach_name,
        get_available_approaches,
        apply_approach_grouping,
        APPROACH_COLORS,
        process_approach_data,
        render_cell_text,
        generate_titles,
        find_result_files,
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
    from workload_categorizer import (
        categorize_workload,
    )


def create_prefill_heatmap(
    results_file: str,
    output_file: str = "plots/prefill_heatmap.png",
    dark_mode: bool = False,
    workload_category: str = None,
    approaches: list = None,
    model_name: str = None,
    tp_degree: int = None,
):
    """
    Create hierarchical heatmap for prefill benchmark results.
    Auto-switches between two visualization modes:
    - 2 approaches: Pairwise speedup heatmap (diverging colormap)
    - 3+ approaches: Best performer heatmap (categorical colors showing winner)

    Layout: Multiple subplots (one per batch size)
        - Each subplot: query (rows) × kv (cols)
        - Pairwise mode: Green = approach1 faster, Red = approach2 faster
        - Best performer mode: Categorical colors = winner for each cell

    Args:
        results_file: Path to benchmark results JSON
        output_file: Path to save the heatmap
        dark_mode: Use dark theme
        workload_category: Optional filter by workload ('code', 'chat', 'summarization')
        approaches: List of approaches to compare (default: auto-detect all)
        model_name: Optional model name to display in title
        tp_degree: Optional TP degree to display in title
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Auto-detect approaches if not specified
    if approaches is None:
        approaches = get_available_approaches(scenarios)
        print(f"Auto-detected {len(approaches)} approaches: {', '.join(approaches)}")

    # Apply approach grouping for prefill workload
    # fi2 = min(mix2, sep2), fi3 = min(mix3, sep3)
    scenarios, approaches = apply_approach_grouping(scenarios, 'prefill', approaches)
    print(f"After grouping: {len(approaches)} approaches: {', '.join(approaches)}")

    # Determine visualization mode
    mode = "pairwise" if len(approaches) == 2 else "best_performer"
    print(f"Using '{mode}' mode for {len(approaches)} approaches")

    # Parse and organize data by batch size
    by_batch = defaultdict(dict)
    all_queries = set()
    all_kvs = set()

    for scenario in scenarios:
        name = scenario['scenario_name']

        # Parse scenario
        batch, query, kv = parse_prefill_scenario(name)
        if batch == 0 or query == 0 or kv == 0:
            continue

        # Filter by workload category if specified
        if workload_category:
            categories = categorize_workload(batch, query, kv, 'prefill')
            if workload_category not in categories:
                continue

        # Skip configurations with very small KV (32 and 64)
        if kv in [32, 64]:
            continue

        # Extract median times from approach_time_stats
        time_stats = scenario.get('approach_time_stats', {})
        approach_times_raw = {}
        for app in approaches:
            app_stats = time_stats.get(app, {})
            median_time = app_stats.get('median') if app_stats else None
            if median_time is not None:
                approach_times_raw[app] = median_time

        # Process approach data using shared function
        result = process_approach_data(approach_times_raw, approaches, mode)

        if result is not None:
            by_batch[batch][(query, kv)] = result
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

    for i, batch in enumerate(batch_sizes):
        ax = axes[i]
        batch_data = by_batch[batch]

        # Initialize and populate matrices using consolidated functions
        matrix, color_matrix, data_matrix = initialize_matrices(mode, len(queries), len(kvs))
        populate_matrices(mode, matrix, color_matrix, data_matrix, queries, kvs, batch_data)

        # Setup visualization based on mode
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

        for j in range(len(queries)):
            for k in range(len(kvs)):
                data = data_matrix.get((j, k))
                render_cell_text(ax, k, j, data, mode, short_names, font_sizes)

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

    # Add colorbar using consolidated function
    add_colorbar(fig, im, axes, short_names, mode, single_subplot=(n_batches == 1))

    # Generate title and subtitle using shared function
    title, subtitle = generate_titles('Prefill', mode, approaches, workload_category, model_name, tp_degree)

    fig.suptitle(
        title,
        fontsize=14,
        fontweight='bold',
        y=0.96,
    )

    fig.text(
        0.5,
        0.01,
        subtitle,
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
    # Create argument parser using consolidated function
    parser = create_common_argparser('prefill', include_workload_category=True)
    args = parser.parse_args()

    # Parse approaches list if provided
    approaches = None
    if args.approaches:
        approaches = [app.strip() for app in args.approaches.split(',')]

    # Determine mode: batch or single
    batch_mode = args.run_dir and args.model is None

    if batch_mode:
        # Batch mode: process all files using consolidated function
        process_batch_mode(args, 'prefill', create_prefill_heatmap, approaches)
    else:
        # Single mode: process one file
        input_file = resolve_input_file(args, 'prefill')
        output_file = resolve_output_file(args, 'prefill')

        if approaches:
            print(f"Using specified approaches: {', '.join(approaches)}")

        print(f"Reading benchmark results from: {input_file}")
        create_prefill_heatmap(input_file, output_file, args.dark, args.workload_category, approaches,
                            args.model, args.tp_degree)
