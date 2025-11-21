#!/usr/bin/env python3
"""Ray-based benchmark orchestrator with memory estimation.

Follows Vidur's patterns:
- Ray actors with num_gpus=1 for GPU isolation
- Pre-filter scenarios by memory estimation
- Greedy scheduling for maximum GPU utilization
- Explicit worker cleanup

Supports:
- Multiple models (Llama 8B, Llama 70B, etc.)
- Multiple TP degrees for single-GPU profiling (1, 2, 4, 8)
- Timestamped output folders
- Separate output files per (model, TP) combination

TP Profiling Note:
  TP degrees are profiling configurations, NOT multi-GPU execution!
  - TP=4 means: profile 1 GPU with 1/4 of the heads (per-GPU slice)
  - We measure single-GPU kernel performance in TP setups
  - We do NOT run actual multi-GPU TP with communication
  - --num-gpus controls parallel scenario execution (separate concern)
"""

import argparse
import json
import os
import time
from datetime import datetime
import ray
import yaml
from typing import Dict, List, Any, Tuple, Optional

from attention_bench.utils.memory_estimator import estimate_scenario_memory, MemoryEstimate
from attention_bench.utils.workload import build_workload_from_variant_set
from attention_bench.distributed.worker import BenchmarkWorker


def load_config(config_path: str) -> Dict:
    """Load complete configuration from YAML file.

    Returns:
        config_data: Complete configuration dictionary
    """
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return config_data


def parse_config(config_data: Dict) -> Tuple[Dict, Dict, List[str], Dict, List[int], Optional[str]]:
    """Parse configuration data.

    Returns:
        (scenarios, variants, approaches, models, tp_degrees, output_config)
    """
    from attention_bench.config import generator as config_generator

    # Get approaches
    approaches = config_data.get("approaches", [
        "flashinfer_mixed_fa2",
        "flashinfer_mixed_fa3",
        "flashinfer_separated_fa2",
        "flashinfer_separated_fa3",
        "flashinfer_batch_attention",
        "official_fa3",
    ])

    # Get models (new multi-model format or legacy single model)
    if "models" in config_data:
        models = config_data["models"]
    else:
        # Legacy format: single model config
        models = {"default": config_data["model"]}
        models["default"]["name"] = "default"

    # Get TP degrees (default to [1] if not specified)
    tp_degrees = config_data.get("tp_degrees", [1])

    # Get output config
    output_config = config_data.get("output", {"results_dir": "results", "plots_dir": "plots"})

    # Generate variants and scenarios programmatically
    if "scenario_generation" in config_data:
        variants, scenarios = config_generator.generate_scenarios(
            config_data["scenario_generation"]
        )
    else:
        # Fallback to manually defined variants/scenarios if present
        variants = config_data.get("variants", {})
        scenarios = config_data.get("scenarios", {})

    return scenarios, variants, approaches, models, tp_degrees, output_config


def calculate_heads_for_tp(model_config: Dict, tp_degree: int) -> Dict:
    """Calculate heads per GPU for given TP degree (for single-GPU profiling).

    IMPORTANT: This is for profiling single-GPU performance in TP setups,
    NOT for actual multi-GPU TP execution!

    When profiling TP=4:
    - We benchmark ONE GPU with 1/4 of the heads
    - This simulates what each GPU would compute in a 4-way TP setup
    - We measure kernel performance without TP communication overhead
    - We do NOT run the same kernel across 4 GPUs

    Args:
        model_config: Base model configuration with total head counts
        tp_degree: Tensor parallelism degree (1, 2, 4, 8, etc.)

    Returns:
        Modified model config with heads_per_gpu = total_heads / tp_degree
    """
    config = model_config.copy()
    config["num_qo_heads"] = model_config["num_qo_heads"] // tp_degree
    config["num_kv_heads"] = model_config["num_kv_heads"] // tp_degree

    # Validate division
    if model_config["num_qo_heads"] % tp_degree != 0:
        raise ValueError(f"num_qo_heads ({model_config['num_qo_heads']}) not divisible by TP degree ({tp_degree})")
    if model_config["num_kv_heads"] % tp_degree != 0:
        raise ValueError(f"num_kv_heads ({model_config['num_kv_heads']}) not divisible by TP degree ({tp_degree})")

    return config


def create_timestamped_output_dir(base_dir: str) -> str:
    """Create a timestamped output directory.

    Args:
        base_dir: Base results directory

    Returns:
        Path to timestamped directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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


def run_benchmark_for_model_tp(
    config_name: str,
    model_name: str,
    model_config: Dict,
    tp_degree: int,
    scenarios: Dict,
    variants: Dict,
    approaches: List[str],
    num_gpus: int,
    memory_util: float,
    memory_overhead: float,
    output_dir: str,
    scenario_gen_config: Dict
) -> Tuple[List[Dict], int, int, List]:
    """Run benchmarks for a single (model, TP) combination.

    IMPORTANT: Creates fresh workers for this TP configuration.
    Workers are killed at the end to ensure clean state for next TP degree.

    Returns:
        (results, num_successful, num_skipped, workers)
    """
    workers = []
    print("\n" + "=" * 80)
    print(f"Running: {model_name} with TP={tp_degree} (Single-GPU Profiling)")
    print("=" * 80)

    # Calculate heads for this TP degree
    tp_model_config = calculate_heads_for_tp(model_config, tp_degree)
    print(f"Model: {model_config['num_qo_heads']} QO heads, {model_config['num_kv_heads']} KV heads (total)")
    print(f"Per-GPU (TP={tp_degree}): {tp_model_config['num_qo_heads']} QO heads, {tp_model_config['num_kv_heads']} KV heads")
    print(f"Note: Profiling 1 GPU with {tp_model_config['num_kv_heads']} KV heads (simulates per-GPU workload in {tp_degree}-way TP)")

    # Pre-filter scenarios by memory estimation (using per-GPU head counts)
    print(f"\nPre-filtering scenarios (memory calculated for {tp_model_config['num_kv_heads']} KV heads per GPU)...")
    feasible, skipped = prefilter_scenarios(
        scenarios, variants, tp_model_config,  # Uses TP-adjusted heads
        utilization=memory_util,
        overhead=memory_overhead
    )

    print(f"Feasible: {len(feasible)}, Skipped: {len(skipped)}")

    if not feasible:
        print("No feasible scenarios - skipping this configuration")
        return [], 0, len(skipped)

    # Create workers if needed
    if not workers:
        print(f"\nCreating {num_gpus} workers...")
        # Create temporary config file for workers
        temp_config = {
            "model": tp_model_config,
            "profiling": {"num_warmup_iters": 5, "num_active_iters": 50},
            "output": {"results_dir": output_dir, "plots_dir": "plots"},
            "scenario_generation": scenario_gen_config  # Use real scenario config
        }

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name

        for i in range(num_gpus):
            worker = BenchmarkWorker.remote(temp_config_path)
            workers.append(worker)

        # Verify workers
        pings = ray.get([w.ping.remote() for w in workers], timeout=60)
        for ping in pings:
            print(f"  {ping}")

        # Clean up temp config
        os.unlink(temp_config_path)

    # Run greedy scheduling (same as before)
    results = []
    start_time = time.time()

    scenario_queue = list(feasible)
    pending = {}

    # Initial fill
    initial_count = min(num_gpus, len(scenario_queue))
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

    # Track workers that need recreation due to CUDA errors
    corrupted_workers = set()

    # Greedy loop
    while pending:
        try:
            done, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=600)
        except Exception as e:
            print(f"ERROR: ray.wait failed: {e}")
            break

        if not done:
            print("ERROR: Tasks timed out!")
            for promise in list(pending.keys()):
                scenario_name, worker_id = pending.pop(promise)
                results.append({'scenario_name': scenario_name, 'skipped_reason': 'Timeout'})
            break

        for promise in done:
            scenario_name, worker_id = pending.pop(promise)

            try:
                result = ray.get(promise)
            except Exception as e:
                result = {'scenario_name': scenario_name, 'skipped_reason': f'Error: {str(e)}'}

            results.append(result)
            time_str = "SKIPPED" if result.get('skipped_reason') else "OK"
            print(f"  [{len(results)}/{len(feasible)}] {scenario_name}: {time_str}")

            # Check if worker encountered CUDA error (illegal memory, etc.)
            # These errors corrupt the CUDA context - worker must be recreated
            if result.get('skipped_reason') and 'illegal' in result.get('skipped_reason', '').lower():
                if worker_id not in corrupted_workers:
                    print(f"    WARNING: Worker {worker_id} encountered CUDA error - recreating...")
                    corrupted_workers.add(worker_id)

                    # Kill corrupted worker
                    try:
                        ray.kill(workers[worker_id])
                    except:
                        pass

                    # Wait for GPU context to fully clean up
                    time.sleep(2)

                    # Create new worker with same config
                    import tempfile
                    temp_config = {
                        "model": tp_model_config,
                        "profiling": {"num_warmup_iters": 5, "num_active_iters": 50},
                        "output": {"results_dir": output_dir, "plots_dir": "plots"},
                        "scenario_generation": scenario_gen_config
                    }
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                        yaml.dump(temp_config, f)
                        temp_config_path = f.name

                    try:
                        workers[worker_id] = BenchmarkWorker.remote(temp_config_path)
                        corrupted_workers.remove(worker_id)
                        print(f"    Worker {worker_id} recreated successfully")
                        # Note: temp_config_path left in /tmp for worker to read
                        # It will be cleaned up on system reboot
                    except Exception as e:
                        print(f"    ERROR: Failed to recreate worker {worker_id}: {e}")
                        print(f"    Worker {worker_id} will remain unavailable")
                        # Clean up temp file only if worker creation failed
                        os.unlink(temp_config_path)

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

    # Add skipped scenarios to results
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

    # Save results with metadata
    results_filename = f"{config_name}_{model_name}_tp{tp_degree}.json"
    results_path = os.path.join(output_dir, results_filename)

    # Wrap results with metadata for self-documentation
    from datetime import datetime
    results_with_metadata = {
        "metadata": {
            "model_name": model_name,
            "tp_degree": tp_degree,
            "num_qo_heads": model_config["num_qo_heads"],
            "num_kv_heads": model_config["num_kv_heads"],
            "head_dim": model_config.get("head_dim", 128),  # Default to 128 if not specified
            "timestamp": datetime.now().isoformat(),
            "config_name": config_name,
        },
        "results": results
    }

    with open(results_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"Results saved to: {results_path}")

    # Count successful/skipped
    successful = [r for r in results if not r.get('skipped_reason')]
    skipped_results = [r for r in results if r.get('skipped_reason')]

    # Kill workers to ensure clean state for next TP configuration
    # Each TP degree needs fresh workers with correct head counts
    print(f"\nCleaning up {len(workers)} workers...")
    for worker in workers:
        try:
            ray.kill(worker)
        except Exception:
            pass

    return results, len(successful), len(skipped_results), []


def main():
    parser = argparse.ArgumentParser(description="Ray-based FlashInfer Benchmark with Multi-Model/TP support")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--memory-utilization", type=float, default=0.90,
                       help="GPU memory utilization threshold (default: 0.90)")
    parser.add_argument("--memory-overhead", type=float, default=1.1,
                       help="Memory overhead multiplier (default: 1.1)")
    parser.add_argument("--approaches", type=str, help="Comma-separated list of approaches")
    parser.add_argument("--models", type=str, help="Comma-separated list of models (default: all)")
    parser.add_argument("--tp-degrees", type=str, help="Comma-separated list of TP degrees (default: all)")
    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    try:
        print("=" * 80)
        print("FlashInfer Ray Benchmark Orchestrator")
        print("Multi-Model, Multi-TP Support")
        print("=" * 80)

        # Load and parse config
        print("\nLoading configuration...")
        config_data = load_config(args.config)
        scenarios, variants, approaches, models, tp_degrees, output_config = parse_config(config_data)

        # Get scenario generation config for workers
        scenario_gen_config = config_data.get("scenario_generation", {})

        # Override approaches if specified
        if args.approaches:
            approaches = [a.strip() for a in args.approaches.split(",")]

        # Filter models if specified
        if args.models:
            selected_models = {k: v for k, v in models.items()
                             if k in args.models.split(",") or v.get("name") in args.models.split(",")}
            models = selected_models

        # Filter TP degrees if specified
        if args.tp_degrees:
            tp_degrees = [int(tp.strip()) for tp in args.tp_degrees.split(",")]

        print(f"Config: {args.config}")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Models: {', '.join([m.get('name', k) for k, m in models.items()])}")
        print(f"TP degrees: {', '.join(map(str, tp_degrees))}")
        print(f"Approaches: {', '.join(approaches)}")
        print(f"GPUs: {args.num_gpus}")
        print(f"Memory utilization: {args.memory_utilization * 100:.0f}%")
        print(f"Memory overhead: {args.memory_overhead}x")

        # Create timestamped output directory
        base_results_dir = output_config.get("results_dir", "results")
        timestamped_dir = create_timestamped_output_dir(base_results_dir)
        print(f"\nOutput directory: {timestamped_dir}")

        # Extract config name (without path and extension)
        config_name = os.path.splitext(os.path.basename(args.config))[0]

        # Loop over all (model, TP) combinations
        all_summaries = []
        total_successful = 0
        total_skipped = 0

        for model_key, model_config in models.items():
            model_name = model_config.get("name", model_key)

            for tp_degree in tp_degrees:
                try:
                    # Each TP configuration gets fresh workers (created inside function)
                    results, num_successful, num_skipped, _ = run_benchmark_for_model_tp(
                        config_name=config_name,
                        model_name=model_name,
                        model_config=model_config,
                        tp_degree=tp_degree,
                        scenarios=scenarios,
                        variants=variants,
                        approaches=approaches,
                        num_gpus=args.num_gpus,
                        memory_util=args.memory_utilization,
                        memory_overhead=args.memory_overhead,
                        output_dir=timestamped_dir,
                        scenario_gen_config=scenario_gen_config
                    )

                    all_summaries.append({
                        "model": model_name,
                        "tp_degree": tp_degree,
                        "successful": num_successful,
                        "skipped": num_skipped,
                        "total": num_successful + num_skipped
                    })

                    total_successful += num_successful
                    total_skipped += num_skipped

                except Exception as e:
                    print(f"\nERROR running {model_name} TP={tp_degree}: {e}")
                    import traceback
                    traceback.print_exc()

        # Print overall summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total configurations: {len(all_summaries)}")
        print(f"Total successful scenarios: {total_successful}")
        print(f"Total skipped scenarios: {total_skipped}")

        # Print per-configuration summaries
        if all_summaries:
            print("\nPer-configuration results:")
            print(f"{'Model':<15} {'TP':<5} {'Successful':<12} {'Skipped':<10} {'Total':<10}")
            print("-" * 60)
            for summary in all_summaries:
                print(f"{summary['model']:<15} {summary['tp_degree']:<5} "
                      f"{summary['successful']:<12} {summary['skipped']:<10} {summary['total']:<10}")

        print("\nDone!")

    finally:
        # Workers are cleaned up inside run_benchmark_for_model_tp
        # after each TP configuration completes

        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
