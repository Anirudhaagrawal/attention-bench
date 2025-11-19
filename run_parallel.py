#!/usr/bin/env python3
"""Parallel benchmark runner for multi-GPU execution.

This script runs benchmarks across multiple GPUs in parallel by:
1. Loading the config to determine total scenarios
2. Dividing scenarios among available GPUs
3. Spawning parallel processes (one per GPU)
4. Waiting for all to complete
5. Merging results and creating plots

Usage:
    python run_parallel.py --config config.yaml --num-gpus 4
    python run_parallel.py --config config.yaml --num-gpus 2 --approaches official_fa3,flashinfer_mixed_fa3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_scenario_count(config_path):
    """Load config to determine number of scenarios.

    Args:
        config_path: Path to config YAML file

    Returns:
        Number of scenarios in the config
    """
    import yaml

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Check if using programmatic generation
    if "scenario_generation" in config_data:
        # Need to generate scenarios to count them
        import config_generator
        _, scenarios_dict = config_generator.generate_scenarios(
            config_data["scenario_generation"]
        )
        return len(scenarios_dict)
    else:
        # Legacy: scenarios directly in config
        return len(config_data.get("scenarios", {}))


def divide_scenarios(total_scenarios, num_gpus):
    """Divide scenarios among GPUs.

    Args:
        total_scenarios: Total number of scenarios
        num_gpus: Number of GPUs to use

    Returns:
        List of (start, end) tuples for each GPU
    """
    scenarios_per_gpu = total_scenarios // num_gpus
    remainder = total_scenarios % num_gpus

    ranges = []
    start = 0
    for gpu_id in range(num_gpus):
        # Distribute remainder across first few GPUs
        count = scenarios_per_gpu + (1 if gpu_id < remainder else 0)
        end = start + count - 1
        ranges.append((start, end))
        start = end + 1

    return ranges


def run_benchmark_on_gpu(config_path, gpu_id, scenario_range, approaches=None,
                        use_cuda_graphs=False, enable_profiling=False):
    """Run benchmark on a specific GPU with scenario range.

    Args:
        config_path: Path to config file
        gpu_id: GPU device ID
        scenario_range: Tuple of (start, end) scenario indices
        approaches: Optional comma-separated list of approaches
        use_cuda_graphs: Enable CUDA graphs
        enable_profiling: Enable profiling

    Returns:
        subprocess.Popen object
    """
    start, end = scenario_range

    cmd = [
        sys.executable,  # Use same Python interpreter
        "run_benchmark.py",
        "--config", config_path,
        "--gpu", str(gpu_id),
        "--scenario-range", f"{start}-{end}",
    ]

    if approaches:
        cmd.extend(["--approaches", approaches])
    if use_cuda_graphs:
        cmd.append("--use-cuda-graphs")
    if enable_profiling:
        cmd.append("--enable-profiling")

    print(f"GPU {gpu_id}: Running scenarios {start}-{end} ({end-start+1} scenarios)")

    # Set CUDA_VISIBLE_DEVICES in subprocess environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Redirect stdout/stderr to log files for debugging
    log_dir = f"logs/gpu{gpu_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/output.log"

    # Open log file (will be kept open by subprocess)
    log_f = open(log_file, 'w')
    return subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)


def merge_results(config_path, num_gpus):
    """Merge partial results from all GPUs.

    Args:
        config_path: Path to config file
        num_gpus: Number of GPUs used

    Returns:
        Path to merged results file
    """
    config_name = Path(config_path).stem

    # Load partial results from each GPU
    all_results = []
    for gpu_id in range(num_gpus):
        result_file = f"results/{config_name}_gpu{gpu_id}.json"

        if not os.path.exists(result_file):
            print(f"Warning: Missing result file for GPU {gpu_id}: {result_file}")
            continue

        with open(result_file) as f:
            partial_results = json.load(f)
            all_results.extend(partial_results)

    # Save merged results
    merged_path = f"results/{config_name}.json"
    with open(merged_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Merged {len(all_results)} results into: {merged_path}")

    # Clean up partial results
    for gpu_id in range(num_gpus):
        result_file = f"results/{config_name}_gpu{gpu_id}.json"
        if os.path.exists(result_file):
            os.remove(result_file)
            print(f"Cleaned up: {result_file}")

    return merged_path


def create_plots_from_merged(config_path, merged_results_path):
    """Create plots from merged results.

    Args:
        config_path: Path to config file
        merged_results_path: Path to merged results JSON
    """
    from run_benchmark import load_config, create_plots

    # Load config to get plot settings
    config, _, _, approaches = load_config(config_path, False, False)

    # Load merged results
    with open(merged_results_path) as f:
        results_data = json.load(f)

    # Convert to ToleranceResult objects
    from run_benchmark import ToleranceResult
    results = []
    for r in results_data:
        results.append(ToleranceResult(
            scenario_name=r['scenario_name'],
            num_decodes=r['num_decodes'],
            num_prefills=r['num_prefills'],
            prefill_ratio=r['prefill_ratio'],
            approach_times=r['approach_times'],
            approach_time_stats=r['approach_time_stats'],
            page_sizes=r['page_sizes'],
            used_cuda_graphs=r['used_cuda_graphs'],
        ))

    # Create plots
    config_name = Path(config_path).stem
    create_plots(results, config.plots_dir, config_name, approaches)


def main():
    parser = argparse.ArgumentParser(description="Parallel FlashInfer Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use (default: 4)")
    parser.add_argument("--approaches", type=str, help="Comma-separated list of approaches to run")
    parser.add_argument("--use-cuda-graphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    print("=" * 80)
    print("Parallel Benchmark Runner")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Approaches: {args.approaches or 'all (from config)'}")
    print("=" * 80)

    # Determine number of scenarios
    total_scenarios = load_scenario_count(args.config)
    print(f"\nTotal scenarios: {total_scenarios}")

    if total_scenarios < args.num_gpus:
        print(f"Warning: Only {total_scenarios} scenarios, using {total_scenarios} GPU(s)")
        num_gpus = total_scenarios
    else:
        num_gpus = args.num_gpus

    # Divide scenarios among GPUs
    ranges = divide_scenarios(total_scenarios, num_gpus)

    print(f"\nDistribution:")
    for gpu_id, (start, end) in enumerate(ranges):
        print(f"  GPU {gpu_id}: scenarios {start}-{end} ({end-start+1} scenarios)")

    # Spawn processes
    print(f"\nStarting {num_gpus} parallel processes...")
    start_time = time.time()

    processes = []
    for gpu_id, scenario_range in enumerate(ranges):
        p = run_benchmark_on_gpu(
            args.config,
            gpu_id,
            scenario_range,
            args.approaches,
            args.use_cuda_graphs,
            args.enable_profiling,
        )
        processes.append(p)

    # Wait for all processes to complete
    print(f"\nWaiting for {num_gpus} processes to complete...")
    for gpu_id, p in enumerate(processes):
        return_code = p.wait()
        if return_code != 0:
            print(f"Warning: GPU {gpu_id} process failed with return code {return_code}")
        else:
            print(f"GPU {gpu_id}: Completed successfully")

    elapsed = time.time() - start_time
    print(f"\nAll processes completed in {elapsed:.1f} seconds")

    # Merge results
    print("\n" + "=" * 80)
    print("Merging Results")
    print("=" * 80)
    merged_path = merge_results(args.config, num_gpus)

    # Create plots from merged results
    print("\n" + "=" * 80)
    print("Creating Plots")
    print("=" * 80)
    create_plots_from_merged(args.config, merged_path)

    print("\n" + "=" * 80)
    print("Parallel Benchmark Complete!")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Results: {merged_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
