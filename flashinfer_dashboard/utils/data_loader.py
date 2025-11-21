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

    # Iterate through all run directories
    for run_dir in sorted(results_path.glob("run_*"), reverse=True):
        # Load all JSON files in this run
        for json_file in sorted(run_dir.glob("*.json")):
            try:
                with open(json_file) as f:
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
                # e.g., "config_prefill_multi_model_tp_llama8b_tp4.json"
                filename = json_file.stem
                if model_name == "unknown":
                    import re
                    model_match = re.search(r'(llama\d+b)', filename, re.IGNORECASE)
                    if model_match:
                        model_name = model_match.group(1).lower()

                if tp_degree == 1 and "_tp" in filename:
                    import re
                    tp_match = re.search(r'_tp(\d+)(?:\.json)?$', filename)
                    if tp_match:
                        tp_degree = int(tp_match.group(1))

                # Determine workload type from filename or config
                filename = json_file.stem
                if "decode" in filename:
                    workload_type = "decode"
                elif "prefill" in filename:
                    workload_type = "prefill"
                elif "mixed" in filename:
                    workload_type = "mixed"
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
                st.warning(f"Error loading {json_file}: {e}")
                continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Convert timestamp to datetime if possible
    if "timestamp" in df.columns and not df["timestamp"].empty:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except:
            pass

    return df


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

    # Run filter
    if filters.get("runs"):
        filtered = filtered[filtered["run_id"].isin(filters["runs"])]

    # Workload type filter
    if filters.get("workload") and filters["workload"] != "All":
        filtered = filtered[filtered["workload_type"] == filters["workload"]]

    # Model filter
    if filters.get("models"):
        filtered = filtered[filtered["model"].isin(filters["models"])]

    # TP degree filter
    if filters.get("tp_degrees"):
        filtered = filtered[filtered["tp_degree"].isin(filters["tp_degrees"])]

    # Batch size range
    if filters.get("batch_range"):
        min_batch, max_batch = filters["batch_range"]
        filtered = filtered[
            (filtered["batch_size"] >= min_batch) &
            (filtered["batch_size"] <= max_batch)
        ]

    # KV length range
    if filters.get("kv_range"):
        min_kv, max_kv = filters["kv_range"]
        filtered = filtered[
            (filtered["kv_length"] >= min_kv) &
            (filtered["kv_length"] <= max_kv)
        ]

    # CUDA graphs only
    if filters.get("cuda_graphs_only"):
        filtered = filtered[filtered["used_cuda_graphs"] == True]

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
