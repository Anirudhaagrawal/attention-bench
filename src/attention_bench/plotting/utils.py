"""Shared utilities for heatmap generation."""

import re
import json
from typing import Dict, Tuple


def parse_decode_scenario(scenario_name: str) -> Tuple[int, int]:
    """
    Parse decode scenario name to extract (batch, kv).

    Examples:
        "decode_b1_kv1k" → (1, 1024)
        "decode_batch_128" → (128, 0) - will skip if kv not found
        "decode_128_kv512" → (128, 512)
    """
    batch = 0
    kv = 0

    # Extract batch - multiple formats
    if m := re.search(r'decode_batch_(\d+)', scenario_name):
        batch = int(m.group(1))
    elif m := re.search(r'decode_b(\d+)', scenario_name):
        batch = int(m.group(1))
    elif m := re.search(r'decode_(\d+)_', scenario_name):
        batch = int(m.group(1))

    # Extract kv - multiple formats
    if m := re.search(r'_kv(\d+)M', scenario_name):
        kv = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'_kv(\d+)k', scenario_name):
        kv = int(m.group(1)) * 1024
    elif m := re.search(r'_kv(\d+)(?:_|$)', scenario_name):
        kv = int(m.group(1))
    elif m := re.search(r'seq(\d+)M', scenario_name):
        kv = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'seq(\d+)k', scenario_name):
        kv = int(m.group(1)) * 1024
    elif m := re.search(r'seq(\d+)', scenario_name):
        kv = int(m.group(1))

    return (batch, kv)


def parse_prefill_scenario(scenario_name: str) -> Tuple[int, int, int]:
    """
    Parse prefill scenario name to extract (batch, query, kv).

    Examples:
        "prefill_b1_q128_kv1k" → (1, 128, 1024)
        "prefill_b8_q2k_kv2k" → (8, 2048, 2048)
        "prefill_batch_32" → (32, 0, 0) - will skip if query/kv not found
        "prefill_32_seq512" → (32, 512, 512) - assume query=kv for seq
    """
    batch = 0
    query = 0
    kv = 0

    # Extract batch - multiple formats
    if m := re.search(r'prefill_batch_(\d+)', scenario_name):
        batch = int(m.group(1))
    elif m := re.search(r'prefill_b(\d+)', scenario_name):
        batch = int(m.group(1))
    elif m := re.search(r'prefill_(\d+)_', scenario_name):
        batch = int(m.group(1))

    # Extract query
    if m := re.search(r'_q(\d+)M', scenario_name):
        query = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'_q(\d+)k', scenario_name):
        query = int(m.group(1)) * 1024
    elif m := re.search(r'_q(\d+)', scenario_name):
        query = int(m.group(1))
    elif m := re.search(r'seq(\d+)M', scenario_name):
        # seq format: assume query = kv
        query = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'seq(\d+)k', scenario_name):
        # seq format: assume query = kv
        query = int(m.group(1)) * 1024
    elif m := re.search(r'seq(\d+)', scenario_name):
        query = int(m.group(1))

    # Extract kv
    if m := re.search(r'_kv(\d+)M', scenario_name):
        kv = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'_kv(\d+)k', scenario_name):
        kv = int(m.group(1)) * 1024
    elif m := re.search(r'_kv(\d+)(?:_|$)', scenario_name):
        kv = int(m.group(1))
    elif query > 0 and kv == 0:
        # If we found query via seq but no explicit kv, assume kv = query
        kv = query

    return (batch, query, kv)


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
    # Check for M (millions) first, then k (thousands), then raw number
    if m := re.search(r'dec_b\d+_kv(\d+)M', scenario_name):
        decode_kv = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'dec_b\d+_kv(\d+)k', scenario_name):
        decode_kv = int(m.group(1)) * 1024
    elif m := re.search(r'dec_b\d+_kv(\d+)', scenario_name):
        decode_kv = int(m.group(1))

    # Extract prefill query
    # Check for M (millions) first, then k (thousands), then raw number
    if m := re.search(r'pref_q(\d+)M', scenario_name):
        prefill_query = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'pref_q(\d+)k', scenario_name):
        prefill_query = int(m.group(1)) * 1024
    elif m := re.search(r'pref_q(\d+)', scenario_name):
        prefill_query = int(m.group(1))

    # Extract prefill KV (last kv in the string)
    # Check for M (millions) first, then k (thousands), then raw number
    if m := re.search(r'pref_q\d+[kM]?_kv(\d+)M', scenario_name):
        prefill_kv = int(m.group(1)) * 1024 * 1024
    elif m := re.search(r'pref_q\d+[kM]?_kv(\d+)k', scenario_name):
        prefill_kv = int(m.group(1)) * 1024
    elif m := re.search(r'pref_q\d+[kM]?_kv(\d+)', scenario_name):
        prefill_kv = int(m.group(1))

    return (decode_batch, decode_kv, prefill_query, prefill_kv)


def calculate_speedup(fa3_time: float, ba_time: float) -> float:
    """
    Calculate speedup ratio.

    Returns ba_time / fa3_time:
        - >1.0: FA3 wins (BA is slower)
        - <1.0: BA wins (FA3 is slower)
        - =1.0: Tie
    """
    if fa3_time == 0 or fa3_time == float('inf'):
        return 0.0
    if ba_time == 0 or ba_time == float('inf'):
        return 0.0

    return ba_time / fa3_time


def load_benchmark_results(filepath: str) -> list:
    """Load benchmark results from JSON file or multiple files.

    Supports both formats:
    - New format (with metadata): {"metadata": {...}, "results": [...]}
    - Old format (raw array): [...]
    """
    from typing import Union, List
    import os

    # Handle single file or list of files
    if isinstance(filepath, str):
        if ',' in filepath:
            # Comma-separated list
            filepaths = [f.strip() for f in filepath.split(',')]
        else:
            filepaths = [filepath]
    else:
        filepaths = filepath

    # Load and merge all scenarios
    all_scenarios = []
    scenarios_by_name = {}

    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"Warning: File not found: {fp}")
            continue

        with open(fp, 'r') as f:
            if fp.endswith('.jsonl'):
                # JSONL format: one scenario per line
                scenarios = [json.loads(line) for line in f if line.strip()]
                print(f"Loaded {fp} (JSONL format, {len(scenarios)} scenarios)")
            else:
                # JSON format
                data = json.load(f)

                # Check if this is the new format with metadata
                if isinstance(data, dict) and "metadata" in data and "results" in data:
                    # New format: extract metadata and results
                    metadata = data["metadata"]
                    scenarios = data["results"]

                    # Print metadata for user info
                    print(f"Loaded {fp}:")
                    print(f"  Model: {metadata.get('model_name', 'unknown')}")
                    print(f"  TP Degree: {metadata.get('tp_degree', 'unknown')}")
                    print(f"  Heads: {metadata.get('num_qo_heads', '?')} QO, {metadata.get('num_kv_heads', '?')} KV")
                else:
                    # Old format: raw results array
                    scenarios = data
                    print(f"Loaded {fp} (legacy format)")

        # Deduplicate by scenario name (keep first occurrence)
        for scenario in scenarios:
            name = scenario.get('scenario_name', '')
            if name and name not in scenarios_by_name:
                scenarios_by_name[name] = scenario

    return list(scenarios_by_name.values())


# Color palette for approaches (categorical colors for best_performer mode)
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


def format_number(num: int) -> str:
    """Format number for display (e.g., 1024 → 1k, 2048 → 2k)."""
    if num >= 1024 and num % 1024 == 0:
        return f"{num // 1024}k"
    return str(num)


def shorten_approach_name(approach_name: str) -> str:
    """Convert long approach names to very short display names for compact cells.

    Examples:
        flashinfer_mixed_fa3 -> Mix3
        flashinfer_batch_attention -> Batch
        official_fa3 -> OFA3
        fi2 -> FI2 (grouped)
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
    if approach_name in name_map:
        return name_map[approach_name]
    else:
        # Generic shortening: remove flashinfer_ prefix and capitalize
        return approach_name.replace("flashinfer_", "").replace("_", "").title()


def apply_approach_grouping(scenarios: list, workload_type: str, approaches: list) -> tuple:
    """Apply approach grouping to scenario data based on workload type.

    For prefill: fi2 = min(mix2, sep2), fi3 = min(mix3, sep3)
    For decode: fi_dec = min(sep2, sep3)
    For mixed: no grouping

    Args:
        scenarios: List of benchmark scenario dictionaries
        workload_type: Type of workload ('prefill', 'decode', 'mixed')
        approaches: List of original approach names

    Returns:
        Tuple of (modified scenarios, list of grouped approach names)
    """
    if workload_type not in APPROACH_GROUPS or workload_type == "mixed":
        return scenarios, approaches

    groups = APPROACH_GROUPS[workload_type]
    new_approaches = []
    used_originals = set()

    # Determine which groups we can form
    valid_groups = {}
    for group_name, members in groups.items():
        available_members = [m for m in members if m in approaches]
        if available_members:
            valid_groups[group_name] = available_members
            used_originals.update(members)
            new_approaches.append(group_name)

    # Add ungrouped approaches that are still relevant
    for approach in approaches:
        if approach not in used_originals:
            new_approaches.append(approach)

    # Modify scenarios to add grouped approach times and remove original grouped members
    modified_scenarios = []
    for scenario in scenarios:
        new_scenario = scenario.copy()
        approach_times = new_scenario.get('approach_times', {}).copy()
        approach_time_stats = new_scenario.get('approach_time_stats', {}).copy()

        # Create grouped approach times (take min of members)
        for group_name, members in valid_groups.items():
            member_times = []
            member_medians = []
            for member in members:
                # From approach_times
                time = approach_times.get(member)
                if time is not None and time != float('inf'):
                    member_times.append(time)
                # From approach_time_stats
                stats = approach_time_stats.get(member, {})
                if stats and stats.get('median') is not None:
                    member_medians.append(stats.get('median'))

            if member_times:
                approach_times[group_name] = min(member_times)

            # Also add to approach_time_stats with synthetic median
            if member_medians:
                approach_time_stats[group_name] = {'median': min(member_medians)}

            # Remove original members that were grouped
            for member in members:
                approach_times.pop(member, None)
                approach_time_stats.pop(member, None)

        new_scenario['approach_times'] = approach_times
        new_scenario['approach_time_stats'] = approach_time_stats
        modified_scenarios.append(new_scenario)

    return modified_scenarios, new_approaches


def get_available_approaches(scenarios: list) -> list:
    """Extract all approaches that have timing data from scenarios.

    Args:
        scenarios: List of benchmark scenario dictionaries

    Returns:
        Sorted list of approach names that have data
    """
    approaches = set()
    for scenario in scenarios:
        approach_times = scenario.get('approach_times', {})
        # Only include approaches with non-None times
        for approach, time in approach_times.items():
            if time is not None:
                approaches.add(approach)
    return sorted(list(approaches))


def adjust_color_saturation(hex_color: str, saturation: float) -> str:
    """Adjust the saturation of a hex color.

    Args:
        hex_color: Hex color string (e.g., "#3498DB")
        saturation: Saturation factor 0.0-1.0 (0=gray, 1=full color)

    Returns:
        Adjusted hex color string
    """
    import matplotlib.colors as mcolors

    # Convert hex to RGB
    rgb = mcolors.hex2color(hex_color)

    # Convert to HSV
    import colorsys
    hsv = colorsys.rgb_to_hsv(*rgb)

    # Adjust saturation
    adjusted_hsv = (hsv[0], hsv[1] * saturation, hsv[2])

    # Convert back to RGB and hex
    adjusted_rgb = colorsys.hsv_to_rgb(*adjusted_hsv)
    return mcolors.rgb2hex(adjusted_rgb)


def process_approach_data(approach_times_raw: dict, approaches: list, mode: str):
    """Process approach timing data for visualization.

    Args:
        approach_times_raw: Raw approach times from scenario
        approaches: List of approach names to compare
        mode: 'pairwise' or 'best_performer'

    Returns:
        Pairwise mode: (speedup, time2, time1) or None if insufficient data
        Best performer mode: (winner_name, runner_up_name, winner_time, second_time, speedup, color) or None
    """
    # Filter to only requested approaches with valid times
    approach_times = {}
    for app in approaches:
        time = approach_times_raw.get(app)
        if time is not None and time != float('inf'):
            approach_times[app] = time

    # Skip if insufficient data
    if len(approach_times) < 2:
        return None

    if mode == "pairwise":
        # Pairwise: Compare two approaches
        time1 = approach_times.get(approaches[0])
        time2 = approach_times.get(approaches[1])

        if time1 is None or time2 is None:
            return None

        speedup = calculate_speedup(time1, time2)
        return (speedup, time2, time1)

    else:  # best_performer mode
        # Find winner (fastest) and 2nd place
        sorted_times = sorted(approach_times.items(), key=lambda x: x[1])
        winner_name, winner_time = sorted_times[0]
        second_name, second_time = sorted_times[1] if len(sorted_times) > 1 else (None, None)

        # Calculate speedup (2nd / 1st)
        speedup = second_time / winner_time if second_time else 1.0

        # Get winner color with intensity based on margin
        base_color = APPROACH_COLORS.get(winner_name, "#808080")
        intensity = min(1.0, 0.3 + (speedup - 1.0) * 0.7)
        color = adjust_color_saturation(base_color, intensity)

        return (winner_name, second_name, winner_time, second_time, speedup, color)


def generate_titles(workload_type: str, mode: str, approaches: list, workload_category: str = None,
                    model_name: str = None, tp_degree: int = None):
    """Generate title and subtitle for heatmap visualization.

    Args:
        workload_type: 'Prefill', 'Decode', or 'Mixed'
        mode: 'pairwise' or 'best_performer'
        approaches: List of approach names
        workload_category: Optional workload category filter
        model_name: Optional model name (e.g., 'llama8b', 'llama70b')
        tp_degree: Optional tensor parallelism degree (e.g., 1, 2, 4, 8)

    Returns:
        (title, subtitle) tuple
    """
    # Get short names for approaches
    short_app1 = shorten_approach_name(approaches[0]) if len(approaches) >= 1 else ""
    short_app2 = shorten_approach_name(approaches[1]) if len(approaches) >= 2 else ""

    # Generate main title
    if mode == "pairwise":
        title = f'{workload_type}: {short_app1} vs {short_app2} Performance'
    else:
        title = f'{workload_type}: Best Performer ({len(approaches)} approaches)'

    # Add workload category to title if specified
    if workload_category:
        from .workload_categorizer import get_category_display_name
        cat_display = get_category_display_name(workload_category)
        if mode == "pairwise":
            title = f'{workload_type} ({cat_display}): {short_app1} vs {short_app2} Performance'
        else:
            title = f'{workload_type} ({cat_display}): Best Performer ({len(approaches)} approaches)'

    # Add model name and TP degree if provided
    if model_name and tp_degree is not None:
        title = f'{title} - {model_name} TP={tp_degree}'
    elif model_name:
        title = f'{title} - {model_name}'
    elif tp_degree is not None:
        title = f'{title} - TP={tp_degree}'

    # Generate subtitle
    if mode == "pairwise":
        subtitle = f'Green: {short_app1} faster  |  Red: {short_app2} faster  |  Value: Speedup ratio'
    else:
        subtitle = 'Colors indicate winner  |  Intensity shows margin of victory'

    return (title, subtitle)


def render_cell_text(ax, col, row, data, mode: str, short_names: tuple, font_sizes: dict):
    """Render text annotations for a single heatmap cell.

    Args:
        ax: Matplotlib axis
        col: Column index (x position)
        row: Row index (y position)
        data: Cell data - None for missing, tuple for pairwise/best_performer
        mode: 'pairwise' or 'best_performer'
        short_names: (short_app1, short_app2) for pairwise mode
        font_sizes: Dict with 'speedup', 'time', 'na' font sizes
    """
    # Missing data - show N/A
    if data is None:
        ax.text(
            col, row, 'N/A',
            ha='center', va='center',
            color='gray',
            fontsize=font_sizes.get('na', 10),
            style='italic',
        )
        return

    if mode == "pairwise":
        # Pairwise mode: (speedup, time2, time1)
        speedup, time2, time1 = data
        short_app1, short_app2 = short_names

        # Choose text color based on background
        if speedup < 0.7 or speedup > 1.4:
            text_color = 'white'
        else:
            text_color = 'black'

        # Speedup ratio
        ax.text(
            col, row - 0.25, f'{speedup:.2f}x',
            ha='center', va='center',
            color=text_color,
            fontsize=font_sizes.get('speedup', 10),
            fontweight='bold',
        )

        # Determine winner and loser
        time1_ms = time1 * 1000
        time2_ms = time2 * 1000

        if time1 <= time2:
            # approach1 is winner (faster) - show first and bold
            winner_text = f'{short_app1}:{time1_ms:.0f}'
            loser_text = f'{short_app2}:{time2_ms:.0f}'
            winner_color = '#E74C3C'
            loser_color = '#16A085'
        else:
            # approach2 is winner (faster) - show first and bold
            winner_text = f'{short_app2}:{time2_ms:.0f}'
            loser_text = f'{short_app1}:{time1_ms:.0f}'
            winner_color = '#16A085'
            loser_color = '#E74C3C'

        # Winner (first line, bold)
        ax.text(
            col, row + 0.08, winner_text,
            ha='center', va='center',
            color=winner_color,
            fontsize=font_sizes.get('time', 7),
            fontweight='bold',
        )

        # Loser (second line, normal weight)
        ax.text(
            col, row + 0.35, loser_text,
            ha='center', va='center',
            color=loser_color,
            fontsize=font_sizes.get('time', 7),
            fontweight='normal',
        )

    else:  # best_performer mode
        # Best performer mode: (winner_name, runner_up_name, winner_time, second_time, speedup, color)
        winner_name, runner_up_name, winner_time, second_time, speedup_val, color = data

        short_winner = shorten_approach_name(winner_name)
        short_runner = shorten_approach_name(runner_up_name) if runner_up_name else "N/A"

        # Line 1: Winner name + time
        first_ms = winner_time * 1000
        ax.text(
            col, row - 0.25, f'{short_winner}:{first_ms:.0f}ms',
            ha='center', va='center',
            color='white',
            fontsize=font_sizes.get('speedup', 6),
            fontweight='bold',
        )

        # Line 2: Runner-up name + time
        second_ms = second_time * 1000 if second_time else 0
        ax.text(
            col, row + 0.08, f'{short_runner}:{second_ms:.0f}ms',
            ha='center', va='center',
            color='white',
            fontsize=font_sizes.get('time', 5),
            fontweight='bold',
        )

        # Line 3: Speedup (2nd / 1st)
        ax.text(
            col, row + 0.35, f'{speedup_val:.2f}x',
            ha='center', va='center',
            color='yellow',
            fontsize=font_sizes.get('time', 5),
            fontweight='bold',
        )


def find_result_files(run_dir: str, workload_type: str):
    """Find all result files for a given workload type in a run directory.

    Args:
        run_dir: Path to run directory (e.g., results/run_2025-11-20_16-15-39)
        workload_type: 'decode', 'prefill', or 'mixed'

    Returns:
        List of tuples: [(model_name, tp_degree, file_path), ...]
    """
    import glob
    import re
    import os

    # Pattern: config_{workload_type}_multi_model_tp_{model}_tp{degree}.json
    pattern = os.path.join(run_dir, f"config_{workload_type}_multi_model_tp_*.json")
    files = glob.glob(pattern)

    result_files = []
    regex = re.compile(rf"config_{workload_type}_multi_model_tp_([^_]+)_tp(\d+)\.json")

    for file_path in files:
        filename = os.path.basename(file_path)
        match = regex.match(filename)
        if match:
            model_name = match.group(1)
            tp_degree = int(match.group(2))
            result_files.append((model_name, tp_degree, file_path))

    # Sort by model name, then TP degree
    result_files.sort(key=lambda x: (x[0], x[1]))

    return result_files


def create_common_argparser(workload_type: str, include_workload_category: bool = True):
    """Create argument parser with common arguments for heatmap scripts.

    Args:
        workload_type: 'decode', 'prefill', or 'mixed'
        include_workload_category: Whether to include --workload-category option

    Returns:
        Configured ArgumentParser instance
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=f'Generate {workload_type} performance heatmap from benchmark results'
    )

    # Input mode: either --input (single file) OR --run-dir (batch mode)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--input', type=str, default=None,
        help=f'Path to {workload_type} results JSON file (for single heatmap)'
    )
    input_group.add_argument(
        '--run-dir', type=str, default=None,
        help=f'Path to run directory containing multiple {workload_type} results (batch mode)'
    )

    # Model and TP degree (for batch mode filtering or single mode filename)
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name (e.g., llama8b, llama70b) - required with --run-dir for single file'
    )
    parser.add_argument(
        '--tp-degree', type=int, default=None,
        help='Tensor parallelism degree (e.g., 1, 2, 4, 8)'
    )

    # Approach filtering
    parser.add_argument(
        '--approaches', type=str, default=None,
        help='Comma-separated list of approaches to compare (auto-detect if not specified)'
    )

    # Output configuration
    parser.add_argument(
        '--output', type=str, default=None,
        help=f'Output file path (default: plots/{workload_type}_heatmap.png)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='plots',
        help='Output directory for batch mode (default: plots/)'
    )

    # Display options
    parser.add_argument(
        '--dark', action='store_true',
        help='Use dark mode color scheme'
    )

    # Workload category filtering (optional)
    if include_workload_category:
        parser.add_argument(
            '--workload-category', type=str,
            choices=['code', 'chat', 'summarization'],
            default=None,
            help='Filter scenarios by workload category'
        )

    return parser


def resolve_input_file(args, workload_type: str) -> str:
    """Resolve input file path from command-line arguments.

    Args:
        args: Parsed arguments from argparse
        workload_type: 'decode', 'prefill', or 'mixed'

    Returns:
        Path to input JSON file
    """
    if args.input:
        return args.input
    elif args.run_dir and args.model:
        tp = args.tp_degree if args.tp_degree is not None else 1
        filename = f"config_{workload_type}_multi_model_tp_{args.model}_tp{tp}.json"
        return f"{args.run_dir}/{filename}"
    else:
        return f"results/config_{workload_type}_all_combinations.json"


def resolve_output_file(args, workload_type: str) -> str:
    """Resolve output file path from command-line arguments.

    Args:
        args: Parsed arguments from argparse
        workload_type: 'decode', 'prefill', or 'mixed'

    Returns:
        Path to output PNG file
    """
    import os

    if args.output:
        return args.output
    else:
        return os.path.join(args.output_dir, f"{workload_type}_heatmap.png")


def process_batch_mode(args, workload_type: str, heatmap_creator_func, approaches: list = None):
    """Process multiple files in batch mode.

    Args:
        args: Parsed arguments from argparse
        workload_type: 'decode', 'prefill', or 'mixed'
        heatmap_creator_func: Function to create individual heatmap
        approaches: List of approach names to use (optional)
    """
    import os

    result_files = find_result_files(args.run_dir, workload_type)

    if not result_files:
        print(f"No {workload_type} result files found in {args.run_dir}")
        exit(1)

    print(f"Batch mode: Found {len(result_files)} result file(s)")
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, tp_degree, input_file in result_files:
        output_file = os.path.join(args.output_dir, f"{workload_type}_heatmap_{model_name}_tp{tp_degree}.png")
        print(f"\nProcessing: {model_name} TP={tp_degree}")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")

        if approaches:
            print(f"  Using specified approaches: {', '.join(approaches)}")

        # Get workload_category from args if it exists
        workload_category = getattr(args, 'workload_category', None)

        heatmap_creator_func(
            input_file, output_file, args.dark, workload_category, approaches,
            model_name, tp_degree
        )

    print(f"\n✓ Generated {len(result_files)} heatmap(s) in {args.output_dir}/")


def initialize_matrices(mode: str, dim1_size: int, dim2_size: int):
    """Initialize matrices for heatmap visualization.

    Args:
        mode: 'pairwise' or 'best_performer'
        dim1_size: Number of rows
        dim2_size: Number of columns

    Returns:
        (matrix, color_matrix, data_matrix) - matrix or color_matrix will be None depending on mode
    """
    import numpy as np

    if mode == "pairwise":
        # Pairwise: numeric matrix for colormap
        matrix = np.full((dim1_size, dim2_size), np.nan)
        data_matrix = {}
        return matrix, None, data_matrix
    else:
        # Best performer: categorical color matrix
        color_matrix = np.empty((dim1_size, dim2_size), dtype=object)
        data_matrix = {}
        return None, color_matrix, data_matrix


def populate_matrices(mode: str, matrix, color_matrix, data_matrix, dim1_values, dim2_values, data_source):
    """Populate matrices from data source.

    Args:
        mode: 'pairwise' or 'best_performer'
        matrix: Numeric matrix (for pairwise mode) or None
        color_matrix: Color matrix (for best_performer mode) or None
        data_matrix: Dictionary to store cell data
        dim1_values: List of values for dimension 1 (rows)
        dim2_values: List of values for dimension 2 (columns)
        data_source: Dictionary mapping (dim1, dim2) to result data
    """
    for i, val1 in enumerate(dim1_values):
        for j, val2 in enumerate(dim2_values):
            if (val1, val2) in data_source:
                data = data_source[(val1, val2)]

                if mode == "pairwise":
                    # data is (speedup, time2, time1)
                    speedup, time2, time1 = data
                    matrix[i, j] = speedup
                    data_matrix[(i, j)] = data
                else:
                    # data is (winner_name, runner_up_name, winner_time, second_time, speedup, color)
                    color = data[5]  # color is last element
                    color_matrix[i, j] = color
                    data_matrix[(i, j)] = data
            else:
                if mode == "best_performer":
                    color_matrix[i, j] = 'lightgray'


def setup_colormap_pairwise(matrix):
    """Setup diverging colormap for pairwise mode.

    Args:
        matrix: Numpy array with speedup values

    Returns:
        (cmap, norm, vmin, vmax) tuple
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    vmin = max(0.3, np.nanmin(matrix))
    vmax = min(2.0, np.nanmax(matrix))

    # Ensure range includes 1.0
    if vmin >= 1.0:
        vmin = 0.99
    if vmax <= 1.0:
        vmax = 1.01

    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')

    return cmap, norm, vmin, vmax


def render_best_performer_cells(ax, color_matrix, dark_mode: bool):
    """Render colored rectangles for best performer mode.

    Args:
        ax: Matplotlib axis
        color_matrix: 2D array of color strings
        dark_mode: Whether to use dark mode colors

    Returns:
        None (im will be None for colorbar compatibility)
    """
    import matplotlib.pyplot as plt

    nrows, ncols = color_matrix.shape

    for i in range(nrows):
        for j in range(ncols):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=color_matrix[i, j],
                edgecolor='white' if dark_mode else 'gray',
                linewidth=0.5
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)


def configure_heatmap_axes(ax, x_values, y_values, xlabel: str, ylabel: str,
                          rotate_x: bool = True, grid: bool = True):
    """Configure axes, ticks, labels, and grid for heatmap.

    Args:
        ax: Matplotlib axis
        x_values: List of x-axis values
        y_values: List of y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        rotate_x: Whether to rotate x-axis labels
        grid: Whether to show grid
    """
    import numpy as np

    # Set ticks and labels
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(
        [format_number(x) for x in x_values],
        rotation=45 if rotate_x else 0,
        ha='right' if rotate_x else 'center'
    )
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([format_number(y) for y in y_values])

    if grid:
        # Grid on minor ticks
        ax.set_xticks(np.arange(len(x_values)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_values)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')


def add_colorbar(fig, im, ax_or_axes, short_names: tuple, mode: str, single_subplot: bool = True):
    """Add colorbar to heatmap figure.

    Args:
        fig: Matplotlib figure
        im: Image object from imshow (or None if best_performer mode)
        ax_or_axes: Single axis or list of axes
        short_names: (short_app1, short_app2) tuple
        mode: 'pairwise' or 'best_performer'
        single_subplot: Whether this is a single subplot or multi-subplot figure
    """
    if mode != "pairwise" or im is None:
        return

    short_app1, short_app2 = short_names

    if single_subplot:
        cbar = fig.colorbar(im, ax=ax_or_axes, orientation='vertical', pad=0.02)
    else:
        # Multi-subplot
        if isinstance(ax_or_axes, list) and len(ax_or_axes) == 1:
            cbar = fig.colorbar(im, ax=ax_or_axes[0], orientation='vertical',
                              pad=0.08, fraction=0.046)
        else:
            cbar = fig.colorbar(im, ax=ax_or_axes, orientation='vertical',
                              pad=0.05, aspect=30, shrink=0.8)

    cbar.set_label(
        f'Speedup ({short_app2}/{short_app1})',
        rotation=270, labelpad=20,
        fontsize=11, fontweight='bold'
    )
