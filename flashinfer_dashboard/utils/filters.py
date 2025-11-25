"""Reusable filter components for Streamlit dashboard."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from .visualizations import shorten_approach_name


def create_filter_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    """Create sidebar with interactive filters for benchmark data.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Dictionary with selected filter values
    """
    with st.sidebar:
        st.header("🔍 Filters")

        filters = {}

        if df.empty:
            st.warning("No data loaded")
            return filters

        # Run selector
        st.subheader("Run Selection")
        runs = df['run_id'].unique().tolist()
        if runs:
            # Default to ALL runs (with deduplication enabled)
            filters['runs'] = st.multiselect(
                "Select runs",
                options=runs,
                default=runs,
                help="Choose one or more benchmark runs to analyze (duplicates auto-resolved)"
            )
        else:
            filters['runs'] = []

        # Workload type
        st.subheader("Workload Type")
        workload_types = sorted(df['workload_type'].unique().tolist())
        filters['workload'] = st.radio(
            "Type",
            options=workload_types,
            index=0,
            help="Filter by workload type (decode, prefill, mixed)"
        )

        # Model selection
        st.subheader("Model & Config")
        models = sorted(df['model'].unique().tolist())
        filters['models'] = st.multiselect(
            "Models",
            options=models,
            default=models,
            help="Select which models to include"
        )

        tp_degrees = sorted(df['tp_degree'].unique().tolist())
        filters['tp_degrees'] = st.multiselect(
            "TP Degrees",
            options=tp_degrees,
            default=tp_degrees,
            help="Tensor parallelism degrees"
        )

        # Hardware type selection
        if 'hardware_type' in df.columns:
            hardware_types = sorted(df['hardware_type'].unique().tolist())
            filters['hardware_types'] = st.multiselect(
                "Hardware",
                options=hardware_types,
                default=hardware_types,
                help="GPU hardware type (e.g., H200, A100)"
            )
        else:
            filters['hardware_types'] = []

        # Approach selection
        st.subheader("Approaches")
        # Get approaches relevant to selected workload type (with grouping applied)
        workload_type = filters.get('workload', 'mixed')
        approaches = get_relevant_approaches_for_workload(df, workload_type)
        if approaches:
            # Default to all approaches for best performer mode
            filters['approaches'] = st.multiselect(
                "Select approaches to compare",
                options=approaches,
                default=approaches,
                format_func=shorten_approach_name,
                help="Choose which attention approaches to analyze (select 2 for pairwise, 3+ for best performer)"
            )
        else:
            filters['approaches'] = []

        # Advanced filters (collapsible)
        with st.expander("⚙️ Advanced Filters"):
            # Batch size range
            if 'batch_size' in df.columns and df['batch_size'].max() > 0:
                min_batch = int(df['batch_size'].min())
                max_batch = int(df['batch_size'].max())
                filters['batch_range'] = st.slider(
                    "Batch Size Range",
                    min_value=min_batch,
                    max_value=max_batch,
                    value=(min_batch, max_batch),
                    help="Filter scenarios by batch size"
                )
            else:
                filters['batch_range'] = None

            # KV length range
            if 'kv_length' in df.columns and df['kv_length'].max() > 0:
                min_kv = int(df['kv_length'].min())
                max_kv = int(df['kv_length'].max())
                filters['kv_range'] = st.slider(
                    "KV Length Range",
                    min_value=min_kv,
                    max_value=max_kv,
                    value=(min_kv, max_kv),
                    help="Filter scenarios by KV cache length"
                )
            else:
                filters['kv_range'] = None

            # CUDA graphs filter
            filters['cuda_graphs_only'] = st.checkbox(
                "CUDA Graphs only",
                value=False,
                help="Show only scenarios using CUDA graphs"
            )

        # Deduplication is always enabled (selects best run for each scenario)
        filters['deduplicate'] = True

        # Display filter summary
        st.divider()
        filtered_count = get_filtered_count(df, filters)
        st.metric("Matching Scenarios", filtered_count)

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
    """Create a standalone workload type selector.

    Args:
        key: Unique key for the widget

    Returns:
        Selected workload type
    """
    workload = st.selectbox(
        "Workload Type",
        options=["decode", "prefill", "mixed"],
        index=0,
        key=key,
        help="Filter by workload type"
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
        help="Choose which models to include"
    )

    return selected
