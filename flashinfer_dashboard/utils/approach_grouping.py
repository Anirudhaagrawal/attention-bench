"""Approach grouping logic for FlashInfer dashboard.

This module handles grouping of equivalent approaches based on workload type.
For example, in prefill workloads, mixed_fa2 and separated_fa2 perform identically
and are grouped as "fi2".
"""

import pandas as pd
from typing import List, Tuple

from .colors import APPROACH_GROUPS


def apply_approach_grouping(
    df: pd.DataFrame,
    workload_type: str,
    approaches: List[str],
    metric: str = "median"
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply approach grouping based on workload type.

    For prefill: fi2 = min(mix2, sep2), fi3 = min(mix3, sep3)
    For decode: fi_dec = min(sep2, sep3)
    For mixed: no grouping

    Args:
        df: DataFrame with benchmark results
        workload_type: Type of workload ('prefill', 'decode', 'mixed')
        approaches: List of original approach names
        metric: Metric to use

    Returns:
        Tuple of (modified DataFrame, list of grouped approach names)
    """
    if workload_type not in APPROACH_GROUPS or workload_type == "mixed":
        return df, approaches

    df = df.copy()
    groups = APPROACH_GROUPS[workload_type]
    new_approaches = []
    used_originals = set()

    # Apply groupings
    for group_name, members in groups.items():
        # Check if we have data for any members
        available_members = [m for m in members if f"{m}_{metric}" in df.columns]
        if available_members:
            # Create grouped column as min of available members
            member_cols = [f"{m}_{metric}" for m in available_members]
            df[f"{group_name}_{metric}"] = df[member_cols].min(axis=1)
            new_approaches.append(group_name)
            used_originals.update(members)

    # Add ungrouped approaches that are still relevant
    for approach in approaches:
        if approach not in used_originals:
            # Skip if this is already a grouped approach name (avoid double-adding)
            if approach in groups:
                continue
            if f"{approach}_{metric}" in df.columns:
                new_approaches.append(approach)

    return df, new_approaches
