"""Statistical calculations for benchmark analysis."""

import pandas as pd
from typing import List, Dict, Any, Tuple


def calculate_winner_statistics(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median",
    tie_threshold: float = 0.01,
    aggregate_by: List[str] = None
) -> Dict[str, Any]:
    """Calculate winner statistics for comparing approaches.

    Args:
        df: DataFrame with benchmark results (already filtered and grouped)
        approaches: List of approach names to compare
        metric: Metric to use for comparison ("median", "mean", etc.)
        tie_threshold: Relative difference threshold for ties (default: 1%)
        aggregate_by: Optional list of columns to group by before counting wins.
                     Used to aggregate across multiple runs for the same scenario.
                     Takes minimum time across runs for each approach.

    Returns:
        Dictionary with winner statistics:
        {
            "wins": {approach: win_count, ...},
            "ties": tie_count,
            "total_scenarios": total_count,
            "win_percentages": {approach: percentage, ...},
            "tie_percentage": percentage
        }
    """
    # If aggregate_by is specified, aggregate across runs first
    if aggregate_by is not None and len(aggregate_by) > 0:
        # Build aggregation dict - take minimum time across runs for each approach
        agg_dict = {}
        for approach in approaches:
            col = f"{approach}_{metric}"
            if col in df.columns:
                agg_dict[col] = 'min'  # Take best (minimum) time across runs

        # Only aggregate if we have columns to aggregate
        if agg_dict:
            # Keep only the columns we need for grouping and aggregation
            group_cols = [col for col in aggregate_by if col in df.columns]
            if group_cols:
                df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Initialize win counters
    wins = {approach: 0 for approach in approaches}
    ties = 0
    total_scenarios = 0

    # Count wins per scenario
    for _, scenario_row in df.iterrows():
        # Get times for each approach
        times = {}
        for approach in approaches:
            col = f"{approach}_{metric}"
            if col in df.columns and pd.notna(scenario_row[col]):
                times[approach] = scenario_row[col]

        # Need at least 2 approaches with valid times to compare
        if len(times) >= 2:
            total_scenarios += 1

            # Check for ties (all times within threshold of minimum)
            time_values = list(times.values())
            min_time = min(time_values)
            max_time = max(time_values)

            if abs(max_time - min_time) / min_time < tie_threshold:
                # All approaches within tie threshold
                ties += 1
            else:
                # Find winner (lowest time)
                winner = min(times.items(), key=lambda x: x[1])[0]
                wins[winner] += 1

    # Calculate percentages
    win_percentages = {}
    for approach in approaches:
        if total_scenarios > 0:
            win_percentages[approach] = 100 * wins[approach] / total_scenarios
        else:
            win_percentages[approach] = 0.0

    tie_percentage = 100 * ties / total_scenarios if total_scenarios > 0 else 0.0

    return {
        "wins": wins,
        "ties": ties,
        "total_scenarios": total_scenarios,
        "win_percentages": win_percentages,
        "tie_percentage": tie_percentage
    }


def count_wins_per_scenario(
    df: pd.DataFrame,
    approaches: List[str],
    metric: str = "median",
    tie_threshold: float = 0.01
) -> pd.DataFrame:
    """Calculate winner for each scenario.

    Args:
        df: DataFrame with benchmark results
        approaches: List of approach names to compare
        metric: Metric to use for comparison
        tie_threshold: Relative difference threshold for ties

    Returns:
        DataFrame with additional columns:
        - "winner": Name of winning approach or "tie"
        - "min_time": Minimum time across approaches
        - "time_spread": Relative difference between max and min times
    """
    result_df = df.copy()
    winners = []
    min_times = []
    time_spreads = []

    for _, row in result_df.iterrows():
        # Get times for each approach
        times = {}
        for approach in approaches:
            col = f"{approach}_{metric}"
            if col in result_df.columns and pd.notna(row[col]):
                times[approach] = row[col]

        if len(times) >= 2:
            time_values = list(times.values())
            min_time = min(time_values)
            max_time = max(time_values)
            spread = abs(max_time - min_time) / min_time

            if spread < tie_threshold:
                winners.append("tie")
            else:
                winner = min(times.items(), key=lambda x: x[1])[0]
                winners.append(winner)

            min_times.append(min_time)
            time_spreads.append(spread)
        else:
            winners.append(None)
            min_times.append(None)
            time_spreads.append(None)

    result_df["winner"] = winners
    result_df["min_time"] = min_times
    result_df["time_spread"] = time_spreads

    return result_df


def calculate_speedup_statistics(
    df: pd.DataFrame,
    approach1: str,
    approach2: str,
    metric: str = "median"
) -> Dict[str, Any]:
    """Calculate speedup statistics between two approaches.

    Args:
        df: DataFrame with benchmark results
        approach1: First approach name (baseline)
        approach2: Second approach name (comparison)
        metric: Metric to use for comparison

    Returns:
        Dictionary with speedup statistics:
        {
            "mean_speedup": float,
            "median_speedup": float,
            "min_speedup": float,
            "max_speedup": float,
            "geometric_mean_speedup": float,
            "scenarios_faster": int,
            "scenarios_slower": int,
            "scenarios_similar": int  # within 1%
        }
    """
    col1 = f"{approach1}_{metric}"
    col2 = f"{approach2}_{metric}"

    # Filter to rows with valid data for both approaches
    valid_df = df[
        df[col1].notna() & df[col2].notna() &
        (df[col1] > 0) & (df[col2] > 0)
    ].copy()

    if valid_df.empty:
        return {
            "mean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
            "geometric_mean_speedup": None,
            "scenarios_faster": 0,
            "scenarios_slower": 0,
            "scenarios_similar": 0
        }

    # Calculate speedups
    speedups = valid_df[col1] / valid_df[col2]

    # Count scenarios
    faster = (speedups > 1.01).sum()  # More than 1% faster
    slower = (speedups < 0.99).sum()  # More than 1% slower
    similar = ((speedups >= 0.99) & (speedups <= 1.01)).sum()

    # Geometric mean (better for ratios)
    import numpy as np
    geometric_mean = np.exp(np.log(speedups).mean())

    return {
        "mean_speedup": float(speedups.mean()),
        "median_speedup": float(speedups.median()),
        "min_speedup": float(speedups.min()),
        "max_speedup": float(speedups.max()),
        "geometric_mean_speedup": float(geometric_mean),
        "scenarios_faster": int(faster),
        "scenarios_slower": int(slower),
        "scenarios_similar": int(similar),
        "total_scenarios": len(valid_df)
    }
