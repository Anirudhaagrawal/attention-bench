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
    if m := re.search(r'_kv(\d+)k', scenario_name):
        kv = int(m.group(1)) * 1024
    elif m := re.search(r'_kv(\d+)(?:_|$)', scenario_name):
        kv = int(m.group(1))
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
    if m := re.search(r'_q(\d+)k', scenario_name):
        query = int(m.group(1)) * 1024
    elif m := re.search(r'_q(\d+)', scenario_name):
        query = int(m.group(1))
    elif m := re.search(r'seq(\d+)k', scenario_name):
        # seq format: assume query = kv
        query = int(m.group(1)) * 1024
    elif m := re.search(r'seq(\d+)', scenario_name):
        query = int(m.group(1))

    # Extract kv
    if m := re.search(r'_kv(\d+)k', scenario_name):
        kv = int(m.group(1)) * 1024
    elif m := re.search(r'_kv(\d+)(?:_|$)', scenario_name):
        kv = int(m.group(1))
    elif query > 0 and kv == 0:
        # If we found query via seq but no explicit kv, assume kv = query
        kv = query

    return (batch, query, kv)


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
    """Load benchmark results from JSON file or multiple files."""
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
            data = json.load(f)

        # Deduplicate by scenario name (keep first occurrence)
        for scenario in data:
            name = scenario.get('scenario_name', '')
            if name and name not in scenarios_by_name:
                scenarios_by_name[name] = scenario

    return list(scenarios_by_name.values())


def format_number(num: int) -> str:
    """Format number for display (e.g., 1024 → 1k, 2048 → 2k)."""
    if num >= 1024 and num % 1024 == 0:
        return f"{num // 1024}k"
    return str(num)
