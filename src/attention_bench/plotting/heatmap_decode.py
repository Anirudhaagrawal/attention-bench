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
        load_benchmark_results,
        format_number,
        get_available_approaches,
        apply_approach_grouping,
        APPROACH_COLORS,
        process_approach_data,
        render_cell_text,
        generate_titles,
        shorten_approach_name,
        find_result_files,
        create_common_argparser,
        resolve_input_file,
        resolve_output_file,
        process_batch_mode,
        initialize_matrices,
        populate_matrices,
        setup_colormap_pairwise,
        render_best_performer_cells,
        configure_heatmap_axes,
        add_colorbar,
    )
    from .workload_categorizer import (
        categorize_workload,
    )
except ImportError:
    from utils import (
        parse_decode_scenario,
        load_benchmark_results,
        format_number,
        get_available_approaches,
        apply_approach_grouping,
        APPROACH_COLORS,
        process_approach_data,
        render_cell_text,
        generate_titles,
        shorten_approach_name,
        find_result_files,
        create_common_argparser,
        resolve_input_file,
        resolve_output_file,
        process_batch_mode,
        initialize_matrices,
        populate_matrices,
        setup_colormap_pairwise,
        render_best_performer_cells,
        configure_heatmap_axes,
        add_colorbar,
    )
    from workload_categorizer import (
        categorize_workload,
    )


def create_decode_heatmap(
    results_file: str,
    output_file: str = "plots/decode_heatmap.png",
    dark_mode: bool = False,
    workload_category: str = None,
    approaches: list = None,
    model_name: str = None,
    tp_degree: int = None,
):
    """
    Create a 2D heatmap for decode benchmark results.
    Auto-switches between two visualization modes:
    - 2 approaches: Pairwise speedup heatmap (diverging colormap)
    - 3+ approaches: Best performer heatmap (categorical colors showing winner)

    Args:
        results_file: Path to benchmark results JSON
        output_file: Path to save the heatmap
        dark_mode: Use dark theme
        workload_category: Optional filter by workload ('code', 'chat', 'summarization')
        approaches: List of approaches to compare (default: auto-detect all)
        model_name: Optional model name to display in title
        tp_degree: Optional TP degree to display in title

    The heatmap shows:
        - Rows: Batch sizes
        - Cols: KV lengths
        - Pairwise mode: Green = approach1 faster, Red = approach2 faster
        - Best performer mode: Categorical colors = winner for each cell
    """
    # Load data
    scenarios = load_benchmark_results(results_file)

    # Auto-detect approaches if not specified
    if approaches is None:
        approaches = get_available_approaches(scenarios)
        print(f"Auto-detected {len(approaches)} approaches: {', '.join(approaches)}")

    # Apply approach grouping for decode workload
    # fi_dec = min(sep2, sep3)
    scenarios, approaches = apply_approach_grouping(scenarios, 'decode', approaches)
    print(f"After grouping: {len(approaches)} approaches: {', '.join(approaches)}")

    # Determine visualization mode
    mode = "pairwise" if len(approaches) == 2 else "best_performer"
    print(f"Using '{mode}' mode for {len(approaches)} approaches")

    # Parse and organize data
    speedup_data = {}
    batch_sizes = set()
    kv_lengths = set()

    for scenario in scenarios:
        name = scenario['scenario_name']

        # Parse scenario
        batch, kv = parse_decode_scenario(name)
        if batch == 0 or kv == 0:
            continue

        # Filter by workload category if specified
        if workload_category:
            categories = categorize_workload(batch, 1, kv, 'decode')  # query=1 for decode
            if workload_category not in categories:
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
            speedup_data[(batch, kv)] = result
            batch_sizes.add(batch)
            kv_lengths.add(kv)

    if not speedup_data:
        print(f"No valid decode scenarios found in {results_file}")
        return

    # Sort dimensions
    batch_sizes = sorted(batch_sizes)
    kv_lengths = sorted(kv_lengths)

    # Initialize and populate matrices using consolidated functions
    matrix, color_matrix, data_matrix = initialize_matrices(mode, len(batch_sizes), len(kv_lengths))
    populate_matrices(mode, matrix, color_matrix, data_matrix, batch_sizes, kv_lengths, speedup_data)

    # Create figure manually for better control
    fig_mpl, ax = plt.subplots(figsize=(10, 8))

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

    for i in range(len(batch_sizes)):
        for j in range(len(kv_lengths)):
            data = data_matrix.get((i, j))
            render_cell_text(ax, j, i, data, mode, short_names, font_sizes)

    # Set ticks and labels (keeping decode-specific "b" prefix formatting)
    ax.set_xticks(range(len(kv_lengths)))
    ax.set_xticklabels([format_number(kv) for kv in kv_lengths])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f"b{b}" for b in batch_sizes])

    # Add grid
    ax.set_xticks(np.arange(len(kv_lengths)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(batch_sizes)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Axis labels
    ax.set_xlabel('KV Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Batch Size', fontsize=12, fontweight='bold')

    # Generate title and subtitle using shared function
    title, subtitle = generate_titles('Decode', mode, approaches, workload_category, model_name, tp_degree)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add colorbar using consolidated function
    add_colorbar(fig_mpl, im, ax, short_names, mode, single_subplot=True)

    # Add subtitle
    fig_mpl.text(
        0.5,
        0.02,
        subtitle,
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
    # Create argument parser using consolidated function
    parser = create_common_argparser('decode', include_workload_category=True)
    args = parser.parse_args()

    # Parse approaches list if provided
    approaches = None
    if args.approaches:
        approaches = [app.strip() for app in args.approaches.split(',')]

    # Determine mode: batch or single
    batch_mode = args.run_dir and args.model is None

    if batch_mode:
        # Batch mode: process all files using consolidated function
        process_batch_mode(args, 'decode', create_decode_heatmap, approaches)
    else:
        # Single mode: process one file
        input_file = resolve_input_file(args, 'decode')
        output_file = resolve_output_file(args, 'decode')

        if approaches:
            print(f"Using specified approaches: {', '.join(approaches)}")

        print(f"Reading benchmark results from: {input_file}")
        create_decode_heatmap(input_file, output_file, args.dark, args.workload_category, approaches,
                            args.model, args.tp_degree)
