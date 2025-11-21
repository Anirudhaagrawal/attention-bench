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
    create_filter_sidebar,
    extract_approaches,
    create_approach_selector,
    create_workload_selector,
    create_model_selector,
)

from .visualizations import (
    create_speedup_heatmap,
    create_best_performer_heatmap,
    create_performance_bar_chart,
    create_speedup_distribution,
    create_approach_comparison_table,
    shorten_approach_name,
    apply_approach_grouping,
    APPROACH_COLORS,
    APPROACH_GROUPS,
)

from .export import (
    create_csv_download,
    create_json_download,
    create_report_download,
    export_sidebar,
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
    'create_filter_sidebar',
    'extract_approaches',
    'create_approach_selector',
    'create_workload_selector',
    'create_model_selector',
    # Visualizations
    'create_speedup_heatmap',
    'create_best_performer_heatmap',
    'create_performance_bar_chart',
    'create_speedup_distribution',
    'create_approach_comparison_table',
    'shorten_approach_name',
    'apply_approach_grouping',
    'APPROACH_COLORS',
    'APPROACH_GROUPS',
    # Export
    'create_csv_download',
    'create_json_download',
    'create_report_download',
    'export_sidebar',
]
