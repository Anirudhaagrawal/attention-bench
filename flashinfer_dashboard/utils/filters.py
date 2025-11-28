"""Reusable filter components for Streamlit dashboard."""

import streamlit as st
import pandas as pd
import re
from typing import Dict, Any, List, Optional, Tuple
from .visualizations import shorten_approach_name
from .formatting import format_kv_length


def format_model_name(model: str) -> str:
    """Format model names for display (e.g., 'llama8b' -> 'Llama 8B').

    Args:
        model: Raw model name from data

    Returns:
        Formatted model name with proper capitalization
    """
    if model.lower().startswith('llama'):
        match = re.match(r'llama(\d+)b', model.lower())
        if match:
            size = match.group(1)
            return f"Llama {size}B"
    return model.capitalize()


def get_current_page_name() -> str:
    """Get the current page name for page-aware filtering.

    Returns:
        Page name like "Line_Graphs" or "Heatmaps", or "Home" for main page
    """
    try:
        import inspect
        # Walk up the stack to find the page file
        frame = inspect.currentframe()
        while frame:
            filename = frame.f_code.co_filename
            if 'pages' in filename:
                # Extract like "2_Line_Graphs.py" -> "Line_Graphs"
                page_file = filename.split('/')[-1]
                if '_' in page_file:
                    page_name = page_file.split('_', 1)[1].replace('.py', '')
                    return page_name
                return 'Home'
            frame = frame.f_back
        return "Home"
    except:
        return "Home"


def get_workload_specific_range(
    df: pd.DataFrame,
    workload_type: str,
    workload_category: Optional[str],
    dimension: str
) -> Tuple[int, int, int, int]:
    """Get recommended slider range for a specific workload type/category and dimension.

    Args:
        df: DataFrame with benchmark results
        workload_type: Type of workload ('decode', 'prefill', 'mixed')
        workload_category: Category filter (e.g., 'code', 'chat', 'summarization', or None for all)
        dimension: Dimension to get range for ('batch_size', 'query_length', 'kv_length')

    Returns:
        Tuple of (full_min, full_max, recommended_min, recommended_max)
    """
    # Get full data range
    if dimension not in df.columns or df[dimension].empty:
        return (0, 0, 0, 0)

    full_min = int(df[dimension].min())
    full_max = int(df[dimension].max())

    # Define recommended ranges based on workload categorizer patterns
    # These match the categories defined in workload_categorizer.py
    category_ranges = {
        # Code workloads: smaller batches, long contexts
        'code': {
            'decode': {'batch_size': (1, 16), 'kv_length': (16384, 1048576)},
            'prefill': {'query_length': (32, 4096), 'kv_length': (16384, 1048576)},
            'mixed': {
                'batch_size': (4, 16),  # decode_batch
                'query_length': (32, 4096),  # prefill_query
                'kv_length': (16384, 1048576),  # decode_kv and prefill_kv
            },
        },
        # Chat workloads: medium batches, medium contexts
        'chat': {
            'decode': {'batch_size': (16, 128), 'kv_length': (1024, 131072)},
            'prefill': {'query_length': (32, 4096), 'kv_length': (1024, 131072)},
            'mixed': {
                'batch_size': (16, 128),  # decode_batch
                'query_length': (32, 4096),  # prefill_query
                'kv_length': (1024, 131072),  # decode_kv and prefill_kv
            },
        },
        # Summarization: large batches, very long contexts
        'summarization': {
            'decode': {'batch_size': (32, 256), 'kv_length': (8192, 524288)},
            'prefill': {'query_length': (2048, 8192), 'kv_length': (8192, 524288)},
            'mixed': {
                'batch_size': (32, 256),  # decode_batch
                'query_length': (2048, 8192),  # prefill_query
                'kv_length': (8192, 524288),  # decode_kv and prefill_kv
            },
        },
        # Long context workloads
        'large_kv': {
            'decode': {'batch_size': (1, 256), 'kv_length': (131072, 1048576)},
            'prefill': {'query_length': (32, 16384), 'kv_length': (131072, 1048576)},
        },
    }

    # Default ranges when no category selected or category doesn't apply
    default_ranges = {
        'decode': {
            'batch_size': (1, 256),
            'query_length': (1, 1),  # Always 1 for decode
            'kv_length': (1024, 262144),  # 1k to 256k
        },
        'prefill': {
            'batch_size': (1, 1),  # Always 1 for prefill
            'query_length': (32, 16384),  # 32 to 16k
            'kv_length': (1024, 262144),  # 1k to 256k
        },
        'mixed': {
            'batch_size': (1, 256),
            'query_length': (128, 16384),
            'kv_length': (1024, 262144),
        }
    }

    # Try to get category-specific range first
    rec_min, rec_max = None, None
    if workload_category and workload_category in category_ranges:
        if workload_type in category_ranges[workload_category]:
            range_dict = category_ranges[workload_category][workload_type]
            if dimension in range_dict:
                rec_min, rec_max = range_dict[dimension]

    # Fall back to default ranges
    if rec_min is None or rec_max is None:
        rec_min, rec_max = default_ranges.get(workload_type, {}).get(
            dimension, (full_min, full_max)
        )

    # Clamp recommended range to actual data range
    rec_min = max(full_min, min(rec_min, full_max))
    rec_max = min(full_max, max(rec_max, full_min))

    return (full_min, full_max, rec_min, rec_max)


def create_filter_topbar(df: pd.DataFrame) -> Dict[str, Any]:
    """Create top-aligned filter bar for benchmark data.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Dictionary with selected filter values
    """
    filters = {}

    if df.empty:
        st.warning("No data loaded")
        return filters

    # Create a container for filters with nice styling
    with st.container(border=True):
        st.markdown("#### 🔍 Filters")

        # Row 1: Primary filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Model selection
            models = sorted(df['model'].unique().tolist())
            filters['models'] = st.multiselect(
                "Models",
                options=models,
                default=models,
                format_func=format_model_name,
                help="Select which models to include"
            )

        with col2:
            # TP degree selection
            tp_degrees = sorted(df['tp_degree'].unique().tolist())
            filters['tp_degrees'] = st.multiselect(
                "TP Degrees",
                options=tp_degrees,
                default=tp_degrees,
                help="Tensor parallelism degrees"
            )

        with col3:
            # Batch type
            workload_types = sorted(df['workload_type'].unique().tolist())
            filters['workload'] = st.radio(
                "Batch Type",
                options=workload_types,
                index=0,
                horizontal=True,
                format_func=str.capitalize,
                help="Filter by batch type (Decode, Prefill, Mixed)"
            )

        with col4:
            # Workload category
            if 'workload_category' in df.columns:
                # Get unique categories (excluding empty strings and 'uncategorized')
                all_categories = set()
                for cats in df['workload_category'].dropna():
                    if isinstance(cats, str) and cats:
                        # Filter out empty strings and 'uncategorized'
                        categories = [c.strip() for c in cats.split(',')
                                     if c.strip() and c.strip().lower() != 'uncategorized']
                        all_categories.update(categories)

                if all_categories:
                    categories = ['All Data'] + sorted(all_categories)
                else:
                    categories = ['All Data']

                selected = st.selectbox(
                    "Workload Type",
                    categories,
                    index=0,
                    format_func=str.capitalize,
                    help="Filter by workload characteristics (code, chat, summarization)"
                )
                filters['workload_category'] = None if selected == 'All Data' else selected
            else:
                filters['workload_category'] = None

        # Row 2: Additional filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Batch size range - workload-aware presets
            workload_type = filters.get('workload', 'mixed')
            workload_category = filters.get('workload_category')

            # Determine column name based on workload type
            if workload_type == 'mixed':
                batch_column = 'decode_batch'
                batch_label = "Decode Batch Size Range"
            else:
                batch_column = 'batch_size'
                batch_label = "Batch Size Range"

            if batch_column in df.columns and df[batch_column].max() > 0:
                full_min, full_max, rec_min, rec_max = get_workload_specific_range(
                    df, workload_type, workload_category, 'batch_size'
                )

                # For prefill: batch is typically 1, show as caption if fixed
                if workload_type == 'prefill' and full_min == 1 and full_max == 1:
                    st.caption("Batch Size: 1 (fixed for prefill)")
                    filters['batch_range'] = (1, 1)
                else:
                    # Build dynamic help text
                    help_text = f"Filter scenarios by {batch_label.lower().replace(' range', '')}"
                    if workload_type == 'decode':
                        if workload_category == 'code':
                            help_text += "\n\n🔬 Code: typically 1-16"
                        elif workload_category == 'chat':
                            help_text += "\n\n💬 Chat: typically 16-128"
                        elif workload_category == 'summarization':
                            help_text += "\n\n📝 Summarization: typically 32-256"

                    filters['batch_range'] = st.slider(
                        batch_label,
                        min_value=full_min,
                        max_value=full_max,
                        value=(rec_min, rec_max),
                        key=f"batch_slider_{workload_type}_{workload_category}",
                        help=help_text
                    )
            else:
                filters['batch_range'] = None

        with col2:
            # Query length range - workload-aware presets
            workload_type = filters.get('workload', 'mixed')
            workload_category = filters.get('workload_category')

            # Determine column name based on workload type
            if workload_type == 'mixed':
                query_column = 'prefill_query'
                query_label = "Prefill Query Length Range"
            else:
                query_column = 'query_length'
                query_label = "Query Length Range"

            if query_column in df.columns and df[query_column].max() > 0:
                full_min, full_max, rec_min, rec_max = get_workload_specific_range(
                    df, workload_type, workload_category, 'query_length'
                )

                # For decode: query is always 1, show as caption if fixed
                if workload_type == 'decode' and full_min == 1 and full_max == 1:
                    st.caption("Query Length: 1 (fixed for decode)")
                    filters['query_range'] = (1, 1)
                else:
                    # Build dynamic help text
                    help_text = f"Filter scenarios by {query_label.lower().replace(' range', '')}"
                    if workload_type == 'prefill':
                        if workload_category == 'code':
                            help_text += "\n\n🔬 Code: typically 32-4k"
                        elif workload_category == 'chat':
                            help_text += "\n\n💬 Chat: typically 32-4k"
                        elif workload_category == 'summarization':
                            help_text += "\n\n📝 Summarization: typically 2k-8k"

                    filters['query_range'] = st.slider(
                        query_label,
                        min_value=full_min,
                        max_value=full_max,
                        value=(rec_min, rec_max),
                        key=f"query_slider_{workload_type}_{workload_category}",
                        help=help_text
                    )
            else:
                filters['query_range'] = None

        with col3:
            # KV length range - workload-aware presets
            workload_type = filters.get('workload', 'mixed')
            workload_category = filters.get('workload_category')

            # Determine column name based on workload type
            if workload_type == 'mixed':
                kv_column = 'decode_kv'
                kv_label = "Decode KV Length Range"
            else:
                kv_column = 'kv_length'
                kv_label = "KV Length Range"

            if kv_column in df.columns and df[kv_column].max() > 0:
                full_min, full_max, rec_min, rec_max = get_workload_specific_range(
                    df, workload_type, workload_category, 'kv_length'
                )

                # Build dynamic help text
                help_text = f"Filter scenarios by {kv_label.lower().replace(' range', '')}"
                if workload_category == 'code':
                    help_text += "\n\n🔬 Code: typically 16k-1M"
                elif workload_category == 'chat':
                    help_text += "\n\n💬 Chat: typically 1k-128k"
                elif workload_category == 'summarization':
                    help_text += "\n\n📝 Summarization: typically 8k-512k"
                elif workload_category == 'large_kv':
                    help_text += "\n\n📚 Long Context: typically 128k-1M"

                filters['kv_range'] = st.slider(
                    kv_label,
                    min_value=full_min,
                    max_value=full_max,
                    value=(rec_min, rec_max),
                    key=f"kv_slider_{workload_type}_{workload_category}",
                    help=help_text
                )
            else:
                filters['kv_range'] = None

        with col4:
            # Approach selection
            workload_type = filters.get('workload', 'mixed')
            approaches = get_relevant_approaches_for_workload(df, workload_type)
            if approaches:
                filters['approaches'] = st.multiselect(
                    "Attention Kernels",
                    options=approaches,
                    default=approaches,
                    format_func=shorten_approach_name,
                    help="Choose which attention kernels to analyze"
                )
            else:
                filters['approaches'] = []

        # Row 2.5: Mixed workload specific - Prefill KV slider
        workload_type = filters.get('workload', 'mixed')
        if workload_type == 'mixed' and 'prefill_kv' in df.columns and df['prefill_kv'].max() > 0:
            st.markdown("##### Mixed Workload: Prefill Parameters")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                workload_category = filters.get('workload_category')
                full_min = int(df['prefill_kv'].min())
                full_max = int(df['prefill_kv'].max())

                # Use similar ranges as decode_kv for prefill_kv
                full_min_kv, full_max_kv, rec_min, rec_max = get_workload_specific_range(
                    df, workload_type, workload_category, 'kv_length'
                )

                # Build dynamic help text
                help_text = "Filter scenarios by prefill KV cache length"
                if workload_category == 'code':
                    help_text += "\n\n🔬 Code: typically 16k-1M"
                elif workload_category == 'chat':
                    help_text += "\n\n💬 Chat: typically 1k-128k"
                elif workload_category == 'summarization':
                    help_text += "\n\n📝 Summarization: typically 8k-512k"

                filters['prefill_kv_range'] = st.slider(
                    "Prefill KV Length Range",
                    min_value=full_min,
                    max_value=full_max,
                    value=(rec_min, rec_max),
                    key=f"prefill_kv_slider_{workload_category}",
                    help=help_text
                )
        else:
            filters['prefill_kv_range'] = None

    # ===== PAGE-AWARE FILTERS =====
    # Row 3: Line Graphs specific filters (only shown on Line Graphs page)
    current_page = get_current_page_name()
    if current_page == "Line_Graphs":
        workload_type = filters.get('workload', 'decode')

        # Only show for prefill/decode (not mixed)
        if workload_type in ['prefill', 'decode']:
            st.markdown("##### 📈 Line Graph Controls")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                # X-axis selector based on workload type
                if workload_type == "decode":
                    x_axis_options = ["kv_length", "batch_size"]
                    x_axis_labels = {"kv_length": "KV Cache Length", "batch_size": "Batch Size"}
                else:  # prefill
                    x_axis_options = ["kv_length", "query_length"]
                    x_axis_labels = {"kv_length": "KV Cache Length", "query_length": "Query Length"}

                filters['x_axis'] = st.selectbox(
                    "X-axis",
                    options=x_axis_options,
                    format_func=lambda x: x_axis_labels[x],
                    index=0,  # Default to kv_length
                    key=f"global_x_axis_{workload_type}",
                    help="Choose which dimension to plot on the x-axis"
                )

            with col2:
                # Determine dimension column based on x-axis selection
                x_axis = filters['x_axis']
                if workload_type == "decode":
                    if x_axis == "kv_length":
                        dimension_col = "batch_size"
                        dimension_label = "Batch Sizes"
                        dim_format_func = str
                    else:  # x_axis == "batch_size"
                        dimension_col = "kv_length"
                        dimension_label = "KV Lengths"
                        dim_format_func = format_kv_length
                else:  # prefill
                    if x_axis == "kv_length":
                        dimension_col = "query_length"
                        dimension_label = "Query Lengths"
                        dim_format_func = format_kv_length
                    else:  # x_axis == "query_length"
                        dimension_col = "kv_length"
                        dimension_label = "KV Lengths"
                        dim_format_func = format_kv_length

                # Get available dimensions from filtered data
                if not df.empty:
                    # Apply current filters to get available dimensions
                    temp_filtered = df.copy()
                    if filters.get("workload"):
                        temp_filtered = temp_filtered[temp_filtered["workload_type"] == filters["workload"]]
                    if filters.get("models"):
                        temp_filtered = temp_filtered[temp_filtered["model"].isin(filters["models"])]
                    if filters.get("tp_degrees"):
                        temp_filtered = temp_filtered[temp_filtered["tp_degree"].isin(filters["tp_degrees"])]

                    all_available_dimensions = sorted(temp_filtered[dimension_col].unique()) if not temp_filtered.empty and dimension_col in temp_filtered.columns else []
                else:
                    all_available_dimensions = []

                if all_available_dimensions:
                    filters['selected_dimensions'] = st.multiselect(
                        f"Select {dimension_label}",
                        options=all_available_dimensions,
                        default=all_available_dimensions[:min(1, len(all_available_dimensions))],
                        format_func=dim_format_func,
                        key=f"global_dimensions_{workload_type}_{x_axis}",
                        help=f"Choose which {dimension_label.lower()} to plot (creates separate lines)"
                    )
                else:
                    filters['selected_dimensions'] = []
                    st.caption(f"No {dimension_label.lower()} available")

            with col3:
                filters['y_scale'] = st.radio(
                    "Y-axis Scale",
                    options=["log", "linear"],
                    index=0,
                    key=f"global_y_scale_{workload_type}",
                    help="Log scale better shows relative performance across wide latency ranges"
                )
        else:
            # Mixed workload - set defaults
            filters['x_axis'] = None
            filters['selected_dimensions'] = []
            filters['y_scale'] = 'log'
    else:
        # For non-Line-Graphs pages, don't include these filters
        filters['x_axis'] = None
        filters['selected_dimensions'] = []
        filters['y_scale'] = 'log'

    # Deduplication is always enabled
    filters['deduplicate'] = True

    # Show filtered count below filters
    filtered_count = get_filtered_count(df, filters)
    st.caption(f"📊 **{filtered_count:,}** scenarios match current filters")

    return filters


def get_relevant_approaches_for_workload(df: pd.DataFrame, workload_type: str) -> List[str]:
    """Get relevant grouped approaches for a specific workload type.

    Args:
        df: DataFrame with benchmark results
        workload_type: Type of workload ('prefill', 'decode', 'mixed')

    Returns:
        List of approach names (grouped for prefill/decode, original for mixed)
    """
    from .visualizations import APPROACH_GROUPS

    if df.empty:
        return []

    # Extract all available approaches from data
    all_approaches = extract_approaches(df)

    # For mixed workloads, return all approaches (no grouping)
    if workload_type not in APPROACH_GROUPS or workload_type == "mixed":
        return all_approaches

    # Apply grouping logic
    groups = APPROACH_GROUPS[workload_type]
    grouped_approaches = []
    used_originals = set()

    # Add grouped approach names
    for group_name, members in groups.items():
        # Check if any members are available in the data
        if any(m in all_approaches for m in members):
            grouped_approaches.append(group_name)
            used_originals.update(members)

    # Add ungrouped approaches (approaches not part of any group)
    for approach in all_approaches:
        if approach not in used_originals:
            grouped_approaches.append(approach)

    return sorted(grouped_approaches)


def extract_approaches(df: pd.DataFrame) -> List[str]:
    """Extract list of approaches from DataFrame columns.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Sorted list of approach names
    """
    if df.empty:
        return []

    approaches = set()

    # Look for columns ending with _mean or _median
    for col in df.columns:
        if col.endswith("_mean") or col.endswith("_median"):
            approach = col.rsplit("_", 1)[0]
            approaches.add(approach)

    return sorted(list(approaches))


def get_filtered_count(df: pd.DataFrame, filters: Dict[str, Any]) -> int:
    """Count scenarios matching current filters.

    Args:
        df: DataFrame with benchmark results
        filters: Dictionary with filter criteria

    Returns:
        Number of matching scenarios
    """
    from .data_loader import apply_filters

    try:
        filtered = apply_filters(df, filters)
        return len(filtered)
    except:
        return 0


def create_approach_selector(df: pd.DataFrame, key: str = "approach_selector") -> List[str]:
    """Create a standalone approach selector widget.

    Args:
        df: DataFrame with benchmark results
        key: Unique key for the widget

    Returns:
        List of selected approach names
    """
    approaches = extract_approaches(df)

    if not approaches:
        st.warning("No approaches found in data")
        return []

    # Default to first 2 for comparison
    default = approaches[:2] if len(approaches) >= 2 else approaches

    selected = st.multiselect(
        "Select approaches to compare",
        options=approaches,
        default=default,
        format_func=shorten_approach_name,
        key=key,
        help="Choose which attention approaches to analyze"
    )

    return selected


def create_workload_selector(key: str = "workload_selector") -> str:
    """Create a standalone batch type selector.

    Args:
        key: Unique key for the widget

    Returns:
        Selected batch type
    """
    workload = st.selectbox(
        "Batch Type",
        options=["decode", "prefill", "mixed"],
        index=0,
        key=key,
        format_func=str.capitalize,
        help="Filter by batch type"
    )

    return workload


def create_model_selector(df: pd.DataFrame, key: str = "model_selector") -> List[str]:
    """Create a standalone model selector widget.

    Args:
        df: DataFrame with benchmark results
        key: Unique key for the widget

    Returns:
        List of selected model names
    """
    models = sorted(df['model'].unique().tolist()) if not df.empty else []

    if not models:
        st.warning("No models found in data")
        return []

    selected = st.multiselect(
        "Select models",
        options=models,
        default=models,
        key=key,
        format_func=format_model_name,
        help="Choose which models to include"
    )

    return selected


