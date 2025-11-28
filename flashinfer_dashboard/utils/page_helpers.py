"""Helper functions for dashboard page initialization and common patterns."""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from .data_loader import load_all_results, extract_approaches_from_df, apply_filters
from .filters import create_filter_topbar


@st.cache_data
def get_cached_data(return_path: bool = False):
    """Load benchmark data with automatic path discovery and caching.

    Tries multiple possible locations for results directory:
    1. ./results (from dashboard root)
    2. ../results (from pages directory)
    3. parent.parent.parent/results (from pages subdirectory)

    Args:
        return_path: If True, returns (df, path) tuple. If False, returns just df.

    Returns:
        DataFrame of benchmark results, or (DataFrame, path_str) if return_path=True
    """
    results_paths = [
        "results",
        "../results",
        Path(__file__).parent.parent.parent / "results"
    ]

    for path in results_paths:
        df = load_all_results(str(path))
        if not df.empty:
            if return_path:
                return df, str(path)
            return df

    # Fallback: try default path even if empty
    fallback = load_all_results("results")
    if return_path:
        return fallback, "results"
    return fallback


def initialize_page_data(
    require_approaches: bool = True,
    min_approaches: int = 0,
    page_key: str = "page",
    include_refresh_button: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    """Initialize data, filters, and approaches for a dashboard page.

    This function handles the complete page initialization flow:
    1. Load data with caching
    2. Check for empty data
    3. Create and apply filters
    4. Extract approaches
    5. Validate data availability

    Automatically stops execution with user-friendly messages if data is unavailable.

    Args:
        require_approaches: If True, requires at least min_approaches to be present
        min_approaches: Minimum number of approaches required (if require_approaches=True)
        page_key: Unique key prefix for page-specific Streamlit widgets
        include_refresh_button: If True, adds refresh data button to sidebar

    Returns:
        Tuple of (df, filtered_df, approaches, filters)
        - df: Full unfiltered DataFrame
        - filtered_df: DataFrame after applying filters
        - approaches: List of approach names found in filtered data
        - filters: Dictionary of filter selections

    Raises:
        st.stop: Stops execution if data is unavailable or invalid
    """
    # Load data
    df = get_cached_data(return_path=False)

    if df.empty:
        st.warning("No benchmark results found. Please run some benchmarks first.")
        st.stop()

    # Create filters
    filters = create_filter_topbar(df)

    # Add refresh button to sidebar if requested
    if include_refresh_button:
        with st.sidebar:
            st.divider()
            if st.button(
                "🔄 Refresh Data",
                help="Clear cache and reload all results",
                key=f"refresh_{page_key}"
            ):
                st.cache_data.clear()
                st.rerun()

    # Apply filters
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No data matches the current filters. Try adjusting your selection.")
        st.stop()

    # Extract approaches
    approaches = extract_approaches_from_df(filtered_df)

    # Validate approaches if required
    if require_approaches:
        if not approaches:
            st.error("No approach timing data found in the filtered results.")
            st.stop()

        if min_approaches > 0 and len(approaches) < min_approaches:
            st.warning(
                f"At least {min_approaches} approaches are required for this page. "
                f"Currently have {len(approaches)}. Please adjust filters."
            )
            st.stop()

    return df, filtered_df, approaches, filters


def get_approaches_from_filters_or_df(
    filters: Dict[str, Any],
    df: pd.DataFrame
) -> List[str]:
    """Get approaches from filters if specified, otherwise extract from DataFrame.

    This is a common pattern where some pages allow users to select specific
    approaches via filters, while falling back to all available approaches.

    Args:
        filters: Dictionary of filter selections
        df: DataFrame to extract approaches from if not in filters

    Returns:
        List of approach names
    """
    return filters.get('approaches', extract_approaches_from_df(df))
