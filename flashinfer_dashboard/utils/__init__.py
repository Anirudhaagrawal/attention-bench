"""Utility modules for Attention Bench dashboard."""

from .data_loader import (
    load_all_results,
    get_available_runs,
    get_run_metadata,
    extract_approaches_from_df,
    apply_filters,
    get_summary_stats,
)

from .filters import (
    extract_approaches,
    create_approach_selector,
    create_workload_selector,
    create_model_selector,
)

from .colors import (
    APPROACH_COLORS,
    APPROACH_GROUPS,
    hex_to_rgb,
    rgb_to_hex,
    adjust_color_saturation,
)

from .constants import (
    SPEEDUP_TIE_THRESHOLD,
    SPEEDUP_FASTER_THRESHOLD,
    SPEEDUP_SLOWER_THRESHOLD,
    SECONDS_TO_MS,
    KB_BYTES,
    MB_BYTES,
    HEATMAP_BASE_HEIGHT,
    HEATMAP_ROW_HEIGHT,
)

from .formatting import (
    shorten_approach_name,
    format_kv_length,
    format_time_ms,
)

from .approach_grouping import (
    apply_approach_grouping,
)

from .visualizations import (
    create_speedup_heatmap,
    create_best_performer_heatmap,
    create_mixed_heatmap,
    create_mixed_best_performer_heatmap,
    create_performance_bar_chart,
    create_speedup_distribution,
    create_approach_comparison_table,
    create_line_graph,
)

from .export import (
    create_csv_download,
    create_json_download,
    create_report_download,
    export_sidebar,
)

from .statistics import (
    calculate_winner_statistics,
    count_wins_per_scenario,
    calculate_speedup_statistics,
)

from .ui_components import (
    build_approach_legend,
    render_heatmap_mode_selector,
    render_prefill_selectors,
    render_winner_metrics,
    render_heatmap_with_stats,
    render_approach_selector_for_mode,
    render_scenario_count_badge,
    render_empty_state,
    render_comparison_summary,
)

from .workload_config import (
    WorkloadConfig,
    get_axis_config,
)

from .page_helpers import (
    get_cached_data,
    initialize_page_data,
    get_approaches_from_filters_or_df,
)

__all__ = [
    # Data loading
    'load_all_results',
    'get_available_runs',
    'get_run_metadata',
    'extract_approaches_from_df',
    'apply_filters',
    'get_summary_stats',
    # Filters
    'extract_approaches',
    'create_approach_selector',
    'create_workload_selector',
    'create_model_selector',
    # Color Management
    'APPROACH_COLORS',
    'APPROACH_GROUPS',
    'hex_to_rgb',
    'rgb_to_hex',
    'adjust_color_saturation',
    # Constants
    'SPEEDUP_TIE_THRESHOLD',
    'SPEEDUP_FASTER_THRESHOLD',
    'SPEEDUP_SLOWER_THRESHOLD',
    'SECONDS_TO_MS',
    'KB_BYTES',
    'MB_BYTES',
    'HEATMAP_BASE_HEIGHT',
    'HEATMAP_ROW_HEIGHT',
    # Text Formatting
    'shorten_approach_name',
    'format_kv_length',
    'format_time_ms',
    # Approach Grouping
    'apply_approach_grouping',
    # Visualizations
    'create_speedup_heatmap',
    'create_best_performer_heatmap',
    'create_mixed_heatmap',
    'create_mixed_best_performer_heatmap',
    'create_performance_bar_chart',
    'create_speedup_distribution',
    'create_approach_comparison_table',
    'create_line_graph',
    # Export
    'create_csv_download',
    'create_json_download',
    'create_report_download',
    'export_sidebar',
    # Statistics
    'calculate_winner_statistics',
    'count_wins_per_scenario',
    'calculate_speedup_statistics',
    # UI Components
    'build_approach_legend',
    'render_heatmap_mode_selector',
    'render_prefill_selectors',
    'render_winner_metrics',
    'render_heatmap_with_stats',
    'render_approach_selector_for_mode',
    'render_scenario_count_badge',
    'render_empty_state',
    'render_comparison_summary',
    # Workload Configuration
    'WorkloadConfig',
    'get_axis_config',
    # Page Helpers
    'get_cached_data',
    'initialize_page_data',
    'get_approaches_from_filters_or_df',
]
