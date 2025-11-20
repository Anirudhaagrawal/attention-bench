#!/usr/bin/env python3
"""Ray-based benchmark orchestrator with memory estimation.

Follows Vidur's patterns:
- Ray actors with num_gpus=1 for GPU isolation
- Pre-filter scenarios by memory estimation
- Batched execution with sync points
- Explicit worker cleanup
"""

import argparse
import json
import os
import time
import ray
import yaml
from typing import Dict, List, Any, Tuple

from attention_bench.utils.memory_estimator import estimate_scenario_memory, MemoryEstimate
from attention_bench.distributed.worker import BenchmarkWorker


def load_scenarios_and_config(config_path: str) -> Tuple[Dict, Dict, List[str], Dict]:
    """Load scenarios and model config from YAML file.

    Returns:
        (scenarios, variants, approaches, model_config)
    """
    from attention_bench.config import generator as config_generator

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Get approaches
    approaches = config_data.get("approaches", [
        "flashinfer_mixed_fa2",
        "flashinfer_mixed_fa3",
        "flashinfer_separated_fa2",
        "flashinfer_separated_fa3",
        "flashinfer_batch_attention",
        "official_fa3",
    ])

    # Get model config
    model_config = config_data["model"]

    # Generate variants and scenarios programmatically
    if "scenario_generation" in config_data:
        variants, scenarios = config_generator.generate_scenarios(
            config_data["scenario_generation"]
        )
    else:
        # Fallback to manually defined variants/scenarios if present
        variants = config_data.get("variants", {})
        scenarios = config_data.get("scenarios", {})

    return scenarios, variants, approaches, model_config


def build_workload_from_variant_set(
    variant_set: Dict[str, int],
    variants: Dict
) -> Tuple[List[int], List[int]]:
    """Convert variant set to q_lengths and kv_lengths lists.

    Args:
        variant_set: Dictionary mapping variant names to counts
        variants: Dictionary of variant configurations

    Returns:
        (q_lengths, kv_lengths)
    """
    q_lengths = []
    kv_lengths = []

    # Handle both dict and list formats
    if isinstance(variant_set, list):
        for item in variant_set:
            for variant_name, count in item.items():
                variant = variants[variant_name]
                for _ in range(count):
                    # Access as dict (from config_generator) or object (from VariantConfig)
                    q_tok = variant["q_tokens"] if isinstance(variant, dict) else variant.q_tokens
                    kv_tok = variant["kv_tokens"] if isinstance(variant, dict) else variant.kv_tokens
                    q_lengths.append(q_tok)
                    kv_lengths.append(kv_tok)
    else:
        for variant_name, count in variant_set.items():
            variant = variants[variant_name]
            for _ in range(count):
                # Access as dict (from config_generator) or object (from VariantConfig)
                q_tok = variant["q_tokens"] if isinstance(variant, dict) else variant.q_tokens
                kv_tok = variant["kv_tokens"] if isinstance(variant, dict) else variant.kv_tokens
                q_lengths.append(q_tok)
                kv_lengths.append(kv_tok)

    return q_lengths, kv_lengths


def prefilter_scenarios(
    scenarios: Dict,
    variants: Dict,
    model_config: Dict,
    utilization: float = 0.90,
    overhead: float = 1.1
) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict, MemoryEstimate]]]:
    """Pre-filter scenarios by memory estimation.

    Args:
        scenarios: Dictionary of scenario configurations
        variants: Dictionary of variant configurations
        model_config: Model configuration with head counts, etc.
        utilization: GPU memory utilization threshold
        overhead: Overhead multiplier for allocations

    Returns:
        (feasible_scenarios, skipped_scenarios)
    """
    feasible = []
    skipped = []

    for scenario_name, scenario_config in scenarios.items():
        variant_set = scenario_config["variants_to_run"]
        q_lengths, kv_lengths = build_workload_from_variant_set(variant_set, variants)

        estimate = estimate_scenario_memory(
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            num_kv_heads=model_config["num_kv_heads"],
            num_qo_heads=model_config["num_qo_heads"],
            head_dim=model_config["head_dim"],
            page_size=model_config.get("page_size", 16),
            workspace_size=model_config.get("workspace_size", 256 * 1024 * 1024),
            utilization=utilization,
            overhead=overhead
        )

        if estimate.fits:
            feasible.append((scenario_name, scenario_config))
        else:
            skipped.append((scenario_name, scenario_config, estimate))

    return feasible, skipped


def main():
    parser = argparse.ArgumentParser(description="Ray-based FlashInfer Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--memory-utilization", type=float, default=0.90,
                       help="GPU memory utilization threshold (default: 0.90)")
    parser.add_argument("--memory-overhead", type=float, default=1.1,
                       help="Memory overhead multiplier (default: 1.1)")
    parser.add_argument("--approaches", type=str, help="Comma-separated list of approaches")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    workers = []  # Track workers for cleanup

    try:
        print("=" * 80)
        print("FlashInfer Ray Benchmark Orchestrator")
        print("=" * 80)

        # Load scenarios and config
        print("\nLoading configuration...")
        scenarios, variants, approaches, model_config = load_scenarios_and_config(args.config)

        # Override approaches if specified
        if args.approaches:
            approaches = [a.strip() for a in args.approaches.split(",")]

        print(f"Config: {args.config}")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Approaches: {', '.join(approaches)}")
        print(f"GPUs: {args.num_gpus}")
        print(f"Memory utilization: {args.memory_utilization * 100:.0f}%")
        print(f"Memory overhead: {args.memory_overhead}x")

        # Pre-filter scenarios by memory estimation
        print("\nPre-filtering scenarios by memory estimation...")
        feasible, skipped = prefilter_scenarios(
            scenarios, variants, model_config,
            utilization=args.memory_utilization,
            overhead=args.memory_overhead
        )

        print(f"Feasible scenarios: {len(feasible)}")
        print(f"Skipped scenarios: {len(skipped)}")

        if skipped:
            print("\nSkipped scenarios (would OOM):")
            for scenario_name, _, estimate in skipped[:10]:  # Show first 10
                print(f"  {scenario_name}: needs {estimate.total_gb:.2f}GB, have {estimate.available_gb:.2f}GB")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")

        if not feasible:
            print("\nNo feasible scenarios to run!")
            return

        # Create workers (one per GPU)
        print(f"\nCreating {args.num_gpus} workers...")
        workers = [
            BenchmarkWorker.remote(args.config)
            for _ in range(args.num_gpus)
        ]

        # Verify workers are ready
        print("Verifying workers...")
        pings = ray.get([w.ping.remote() for w in workers], timeout=60)
        for ping in pings:
            print(f"  {ping}")

        # Run benchmarks with greedy scheduling
        # Submit new tasks as soon as workers become available
        print(f"\nRunning {len(feasible)} scenarios...")
        print("=" * 80)

        results = []
        start_time = time.time()

        # Create scenario queue
        scenario_queue = list(feasible)

        # Track pending tasks: promise -> (scenario_name, worker_id)
        pending = {}

        # Initial fill: submit one task to each worker
        initial_count = min(args.num_gpus, len(scenario_queue))
        for i in range(initial_count):
            scenario_name, scenario_config = scenario_queue.pop(0)
            variant_set = scenario_config["variants_to_run"]

            promise = workers[i].profile.remote(
                scenario_name,
                variant_set,
                approaches
            )
            pending[promise] = (scenario_name, i)

        print(f"Started {initial_count} initial tasks...")

        # Greedy loop: process completions and submit new tasks
        while pending:
            # Wait for any task to complete (timeout 10 min per task)
            try:
                done, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=600)
            except Exception as e:
                print(f"ERROR: ray.wait failed: {e}")
                break

            if not done:
                # Timeout - mark all pending as timed out
                print("ERROR: Tasks timed out!")
                for promise in list(pending.keys()):
                    scenario_name, worker_id = pending.pop(promise)
                    results.append({
                        'scenario_name': scenario_name,
                        'skipped_reason': 'Timeout'
                    })
                break

            # Process completed task
            for promise in done:
                scenario_name, worker_id = pending.pop(promise)

                try:
                    result = ray.get(promise)
                except Exception as e:
                    result = {
                        'scenario_name': scenario_name,
                        'skipped_reason': f'Error: {str(e)}'
                    }

                results.append(result)
                time_str = "SKIPPED" if result.get('skipped_reason') else "OK"
                print(f"  [{len(results)}/{len(feasible)}] {scenario_name}: {time_str}")

                # Submit next task to freed worker
                if scenario_queue:
                    next_name, next_config = scenario_queue.pop(0)
                    next_variant_set = next_config["variants_to_run"]

                    new_promise = workers[worker_id].profile.remote(
                        next_name,
                        next_variant_set,
                        approaches
                    )
                    pending[new_promise] = (next_name, worker_id)

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(results)} scenarios in {elapsed:.1f}s")

        # Create results for skipped scenarios
        for scenario_name, scenario_config, estimate in skipped:
            results.append({
                'scenario_name': scenario_name,
                'num_decodes': 0,
                'num_prefills': 0,
                'prefill_ratio': 0.0,
                'approach_times': {a: None for a in approaches},
                'approach_time_stats': {a: None for a in approaches},
                'page_sizes': {},
                'used_cuda_graphs': False,
                'skipped_reason': f"OOM: needs {estimate.total_gb:.2f}GB, have {estimate.available_gb:.2f}GB",
            })

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        results_path = os.path.join(args.output_dir, f"{config_name}_ray.json")

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        successful = [r for r in results if not r.get('skipped_reason')]
        skipped_results = [r for r in results if r.get('skipped_reason')]

        print(f"Total scenarios: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Skipped: {len(skipped_results)}")

        if successful:
            # Count wins per approach
            wins = {approach: 0 for approach in approaches}
            for result in successful:
                valid_times = {k: v for k, v in result['approach_times'].items() if v is not None}
                if valid_times:
                    winner = min(valid_times, key=valid_times.get)
                    wins[winner] = wins.get(winner, 0) + 1

            print("\nWinner Count by Approach:")
            for approach in approaches:
                count = wins.get(approach, 0)
                pct = (count / len(successful) * 100) if successful else 0
                print(f"  {approach:35s}: {count:3d}/{len(successful)} ({pct:5.1f}%)")

        print("\nDone!")

    finally:
        # Clean up workers (always executed, even on exception)
        if workers:
            print("\nCleaning up workers...")
            for worker in workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass  # Ignore errors during cleanup

        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
