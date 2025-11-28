"""Data loading and caching utilities for Attention Bench dashboard."""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple
import sys
from datetime import datetime

# Add parent directory to path to import from attention_bench
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from attention_bench.plotting.utils import (
    parse_decode_scenario,
    parse_prefill_scenario,
    parse_mixed_scenario,
)
from attention_bench.plotting.workload_categorizer import categorize_workload, categorize_mixed_workload


@st.cache_data
def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all benchmark results from results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        DataFrame with all benchmark results
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return pd.DataFrame()

    # Collect all run directories (supporting both old and new structure)
    # Old structure: results/run_*/
    # New structure: results/hardware_type/run_*/
    run_dirs = []

    # First, check for hardware-specific folders (new structure)
    for hardware_dir in sorted(results_path.iterdir()):
        if hardware_dir.is_dir() and not hardware_dir.name.startswith("run_"):
            # This is a hardware folder (e.g., h200, a100)
            for run_dir in sorted(hardware_dir.glob("run_*"), reverse=True):
                run_dirs.append((run_dir, hardware_dir.name))

    # Also check for old-style run directories directly in results/ (backward compatibility)
    for run_dir in sorted(results_path.glob("run_*"), reverse=True):
        run_dirs.append((run_dir, "unknown"))

    # Sort by run directory name (timestamp) in reverse order
    run_dirs = sorted(run_dirs, key=lambda x: x[0].name, reverse=True)

    # Iterate through all run directories
    for run_dir, hardware_type in run_dirs:
        # Load all JSON and JSONL files in this run (exclude profiler traces and test files)
        result_files = sorted(list(run_dir.glob("*.json")) + list(run_dir.glob("*.jsonl")))
        for result_file in result_files:
            # Skip profiler trace files and test files
            if "profiler_trace" in result_file.name or "config_test" in result_file.name:
                continue
            try:
                with open(result_file) as f:
                    if result_file.suffix == '.jsonl':
                        # JSONL format: one scenario per line
                        scenarios = [json.loads(line) for line in f if line.strip()]
                        metadata = {}  # JSONL doesn't have metadata wrapper
                    else:
                        # JSON format
                        data = json.load(f)
                        # Check if this is the new format with metadata
                        if isinstance(data, dict) and "metadata" in data and "results" in data:
                            metadata = data["metadata"]
                            scenarios = data["results"]
                        else:
                            # Old format: no metadata
                            metadata = {}
                            scenarios = data if isinstance(data, list) else []

                # Extract metadata
                model_name = metadata.get("model_name", "unknown")
                tp_degree = metadata.get("tp_degree", 1)
                timestamp = metadata.get("timestamp", "")
                config_name = metadata.get("config_name", "")

                # Fallback: parse model and TP from filename if not in metadata
                # e.g., "config_prefill_multi_model_tp_llama8b_tp4.json" or ".jsonl"
                filename = result_file.stem
                if model_name == "unknown":
                    import re
                    model_match = re.search(r'(llama\d+b)', filename, re.IGNORECASE)
                    if model_match:
                        model_name = model_match.group(1).lower()

                if tp_degree == 1 and "_tp" in filename:
                    import re
                    tp_match = re.search(r'_tp(\d+)', filename)
                    if tp_match:
                        tp_degree = int(tp_match.group(1))

                # Determine workload type from filename or config
                filename = result_file.stem
                if "decode" in filename.lower():
                    workload_type = "decode"
                elif "prefill" in filename.lower():
                    workload_type = "prefill"
                elif "mixed" in filename.lower():
                    workload_type = "mixed"
                else:
                    # Try to infer from scenario names if filename is ambiguous
                    if scenarios and len(scenarios) > 0:
                        first_scenario = scenarios[0].get("scenario_name", "")
                        if "decode_b" in first_scenario.lower():
                            workload_type = "decode"
                        elif "prefill_b" in first_scenario.lower():
                            workload_type = "prefill"
                        elif "mixed_" in first_scenario.lower():
                            workload_type = "mixed"
                        else:
                            workload_type = "unknown"
                    else:
                        workload_type = "unknown"

                # Process each scenario
                for scenario in scenarios:
                    scenario_name = scenario.get("scenario_name", "")
                    approach_times = scenario.get("approach_times", {})
                    approach_time_stats = scenario.get("approach_time_stats", {})

                    # Parse scenario parameters based on workload type
                    batch_size = 0
                    query_length = 0
                    kv_length = 0
                    decode_batch = 0
                    decode_kv = 0
                    prefill_query = 0
                    prefill_kv = 0

                    if workload_type == "decode":
                        batch_size, kv_length = parse_decode_scenario(scenario_name)
                        query_length = 1  # Decode is always query=1
                    elif workload_type == "prefill":
                        batch_size, query_length, kv_length = parse_prefill_scenario(scenario_name)
                    elif workload_type == "mixed":
                        decode_batch, decode_kv, prefill_query, prefill_kv = parse_mixed_scenario(scenario_name)

                    # Create a row for this scenario
                    row = {
                        "run_id": run_dir.name,
                        "run_date": run_dir.name.replace("run_", ""),
                        "hardware_type": hardware_type,
                        "model": model_name,
                        "tp_degree": tp_degree,
                        "timestamp": timestamp,
                        "config_name": config_name,
                        "workload_type": workload_type,
                        "scenario_name": scenario_name,
                        "batch_size": batch_size,
                        "query_length": query_length,
                        "kv_length": kv_length,
                        "decode_batch": decode_batch,
                        "decode_kv": decode_kv,
                        "prefill_query": prefill_query,
                        "prefill_kv": prefill_kv,
                        "num_decodes": scenario.get("num_decodes", 0),
                        "num_prefills": scenario.get("num_prefills", 0),
                        "used_cuda_graphs": scenario.get("used_cuda_graphs", False),
                    }

                    # Add workload category
                    if workload_type == "decode":
                        categories = categorize_workload(batch_size, 1, kv_length, "decode")
                    elif workload_type == "prefill":
                        categories = categorize_workload(batch_size, query_length, kv_length, "prefill")
                    elif workload_type == "mixed":
                        # For mixed workloads, check all 4 dimensions separately
                        categories = categorize_mixed_workload(decode_batch, decode_kv, prefill_query, prefill_kv)
                    else:
                        categories = []

                    row["workload_category"] = ','.join(categories) if categories else ''

                    # Add approach times (mean values)
                    for approach, time in approach_times.items():
                        row[f"{approach}_mean"] = time

                    # Add approach statistics (median, std, min, max)
                    for approach, stats in approach_time_stats.items():
                        if isinstance(stats, dict):
                            row[f"{approach}_median"] = stats.get("median")
                            row[f"{approach}_std"] = stats.get("std")
                            row[f"{approach}_min"] = stats.get("min")
                            row[f"{approach}_max"] = stats.get("max")

                    results.append(row)

            except Exception as e:
                st.warning(f"Error loading {result_file}: {e}")
                continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Filter out default model runs (case-insensitive)
    if 'model' in df.columns:
        df = df[df['model'].str.lower() != 'default']

    # Convert timestamp to datetime if possible
    if "timestamp" in df.columns and not df["timestamp"].empty:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except:
            pass

    return df


def deduplicate_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate scenarios across runs by picking the run with most successful approaches.

    For each unique scenario (identified by scenario_name, model, tp_degree, hardware_type, workload_type),
    picks the run with the most non-null approach times. Tie-breaker: latest run_date.

    Args:
        df: DataFrame with all benchmark results

    Returns:
        DataFrame with deduplicated scenarios
    """
    if df.empty:
        return df

    # Get all approach columns (those ending with _mean, _median, etc.)
    approach_mean_cols = [col for col in df.columns if col.endswith("_mean")]

    if not approach_mean_cols:
        # No approach columns found, return as-is
        return df

    # Add a column to count non-null approach times (successful approaches)
    df = df.copy()
    df["_num_successful_approaches"] = df[approach_mean_cols].notna().sum(axis=1)

    # Group by unique scenario identifier
    # Include hardware_type only if it exists in the DataFrame
    group_cols = ["scenario_name", "model", "tp_degree", "workload_type"]
    if "hardware_type" in df.columns:
        group_cols.insert(3, "hardware_type")  # Insert after tp_degree

    # For each group, pick the row with most successful approaches
    # Tie-breaker: latest run_date (lexicographically largest)
    deduped = (
        df.sort_values(["_num_successful_approaches", "run_date"], ascending=[False, False])
        .groupby(group_cols, dropna=False)
        .first()
        .reset_index()
    )

    # Drop the temporary column
    deduped = deduped.drop(columns=["_num_successful_approaches"])

    return deduped


@st.cache_data
def get_available_runs(results_dir: str = "results") -> List[str]:
    """Get list of available benchmark runs.

    Args:
        results_dir: Path to results directory

    Returns:
        Sorted list of run IDs (newest first)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    runs = [d.name for d in results_path.glob("run_*") if d.is_dir()]
    return sorted(runs, reverse=True)


@st.cache_data
def get_run_metadata(run_id: str, results_dir: str = "results") -> Dict[str, Any]:
    """Get metadata for a specific run.

    Args:
        run_id: Run identifier (e.g., "run_2025-11-20_19-00-57")
        results_dir: Path to results directory

    Returns:
        Dictionary with run metadata
    """
    run_path = Path(results_dir) / run_id
    if not run_path.exists():
        return {}

    metadata = {
        "run_id": run_id,
        "files": [],
        "models": set(),
        "tp_degrees": set(),
        "workload_types": set(),
    }

    # Scan all JSON files
    for json_file in run_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if isinstance(data, dict) and "metadata" in data:
                file_metadata = data["metadata"]
                metadata["models"].add(file_metadata.get("model_name", "unknown"))
                metadata["tp_degrees"].add(file_metadata.get("tp_degree", 1))

            # Determine workload type from filename
            filename = json_file.stem
            if "decode" in filename:
                metadata["workload_types"].add("decode")
            elif "prefill" in filename:
                metadata["workload_types"].add("prefill")
            elif "mixed" in filename:
                metadata["workload_types"].add("mixed")

            metadata["files"].append(json_file.name)
        except:
            continue

    # Convert sets to sorted lists
    metadata["models"] = sorted(list(metadata["models"]))
    metadata["tp_degrees"] = sorted(list(metadata["tp_degrees"]))
    metadata["workload_types"] = sorted(list(metadata["workload_types"]))

    return metadata


def extract_approaches_from_df(df: pd.DataFrame) -> List[str]:
    """Extract list of all approaches that have timing data.

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


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to DataFrame.

    Args:
        df: DataFrame with benchmark results
        filters: Dictionary with filter criteria

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    # Workload type filter
    if filters.get("workload"):
        filtered = filtered[filtered["workload_type"] == filters["workload"]]

    # Workload category filter
    if filters.get("workload_category"):
        category = filters["workload_category"]
        filtered = filtered[filtered["workload_category"].str.contains(category, na=False)]

    # Model filter
    if filters.get("models"):
        filtered = filtered[filtered["model"].isin(filters["models"])]

    # TP degree filter
    if filters.get("tp_degrees"):
        filtered = filtered[filtered["tp_degree"].isin(filters["tp_degrees"])]

    # Hardware type filter (only if column exists)
    if filters.get("hardware_types") and "hardware_type" in filtered.columns:
        filtered = filtered[filtered["hardware_type"].isin(filters["hardware_types"])]

    # Determine column names based on workload type
    workload_type = filters.get("workload", "mixed")
    if workload_type == "mixed":
        batch_col = "decode_batch"
        query_col = "prefill_query"
        kv_col = "decode_kv"
    else:
        batch_col = "batch_size"
        query_col = "query_length"
        kv_col = "kv_length"

    # Batch size range
    if filters.get("batch_range") and batch_col in filtered.columns:
        min_batch, max_batch = filters["batch_range"]
        filtered = filtered[
            (filtered[batch_col] >= min_batch) &
            (filtered[batch_col] <= max_batch)
        ]

    # Query length range
    if filters.get("query_range") and query_col in filtered.columns:
        min_query, max_query = filters["query_range"]
        filtered = filtered[
            (filtered[query_col] >= min_query) &
            (filtered[query_col] <= max_query)
        ]

    # KV length range
    if filters.get("kv_range") and kv_col in filtered.columns:
        min_kv, max_kv = filters["kv_range"]
        filtered = filtered[
            (filtered[kv_col] >= min_kv) &
            (filtered[kv_col] <= max_kv)
        ]

    # Prefill KV range (mixed workloads only)
    if filters.get("prefill_kv_range") and "prefill_kv" in filtered.columns:
        min_pfkv, max_pfkv = filters["prefill_kv_range"]
        filtered = filtered[
            (filtered["prefill_kv"] >= min_pfkv) &
            (filtered["prefill_kv"] <= max_pfkv)
        ]

    # Apply deduplication (always on by default)
    # Picks the run with most successful approaches for each unique scenario
    if filters.get("deduplicate", True):
        filtered = deduplicate_scenarios(filtered)

    return filtered


def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics from DataFrame.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            "total_scenarios": 0,
            "total_runs": 0,
            "models": [],
            "workload_types": [],
            "approaches": [],
            "tp_degrees": [],
        }

    return {
        "total_scenarios": len(df),
        "total_runs": df["run_id"].nunique(),
        "models": sorted(df["model"].unique().tolist()),
        "workload_types": sorted(df["workload_type"].unique().tolist()),
        "approaches": extract_approaches_from_df(df),
        "tp_degrees": sorted(df["tp_degree"].unique().tolist()),
        "date_range": (
            df["run_date"].min() if "run_date" in df.columns else "",
            df["run_date"].max() if "run_date" in df.columns else ""
        ),
    }
