#!/usr/bin/env python3
"""Profile the full benchmark execution with torch.profiler.

This script runs the actual benchmark code (ToleranceBenchmarkRunner) with
comprehensive torch.profiler instrumentation to find where ALL time is spent,
including the missing ~157ms of overhead.

Outputs:
  - full_benchmark_trace.json: Chrome trace for visual analysis
  - Console: Detailed breakdown of where time is spent
"""

import sys
import torch
import yaml

sys.path.insert(0, 'src/attention_bench')

from cli.run_ray import load_config, parse_config
from cli.run_benchmark import ToleranceBenchmarkRunner, BenchmarkConfig, VariantConfig

def main():
    print("=" * 80)
    print("FULL BENCHMARK PROFILING WITH TORCH.PROFILER")
    print("=" * 80)
    print()
    print("Running actual benchmark code with comprehensive profiling:")
    print("  - CPU activity with Python stack traces")
    print("  - GPU/CUDA operations")
    print("  - Memory allocations")
    print("  - RecordFunctionTracer overhead")
    print("  - Gaps/idle time")
    print()

    # Load real config
    config_path = 'configs/mixed/config_test_small.yaml'
    print(f"Loading config: {config_path}")

    # Load and parse config (run_ray.py style)
    config_data = load_config(config_path)
    scenarios, variants_dict, approaches, models, tp_degrees, output_config = parse_config(config_data)

    print(f"Loaded {len(scenarios)} scenarios, {len(approaches)} approaches")

    # Use just the first scenario for detailed profiling
    scenario_names = list(scenarios.keys())
    first_scenario = scenario_names[0]
    scenario_data = scenarios[first_scenario]

    print(f"\nProfiling scenario: {first_scenario}")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"Variant set: {scenario_data['variants_to_run']}")
    print()

    # Get first model config
    model_name = list(models.keys())[0]
    model_config = models[model_name]

    # Create BenchmarkConfig
    full_config = BenchmarkConfig(
        num_qo_heads=model_config['num_qo_heads'],
        num_kv_heads=model_config['num_kv_heads'],
        head_dim=model_config['head_dim'],
        page_size=model_config.get('page_size', 16),
        workspace_size=model_config.get('workspace_size', 256 * 1024 * 1024),
        num_warmup_iters=config_data['profiling']['num_warmup_iters'],
        num_active_iters=config_data['profiling']['num_active_iters'],
        use_cuda_graphs=False,
        enable_profiling=False,
        disable_internal_profiling=True,  # IMPORTANT: Disable RecordFunctionTracer for external profiler
        results_dir=config_data['output']['results_dir'],
        plots_dir=config_data['output']['plots_dir'],
        approach_overrides=config_data.get('approach_overrides', {}),
        enable_memory_check=False,
        memory_utilization=0.8,
        memory_overhead=1.4,
    )

    # Convert variant dicts to VariantConfig objects
    variants = {}
    for variant_name, variant_data in variants_dict.items():
        variants[variant_name] = VariantConfig(
            name=variant_name,
            q_tokens=variant_data['q_tokens'],
            kv_tokens=variant_data['kv_tokens'],
            description=variant_data.get('description', ''),
        )

    # Create runner
    runner = ToleranceBenchmarkRunner(full_config, variants)

    print("Starting torch.profiler (this will capture EVERYTHING)...")
    print()

    # Profile with full instrumentation (now safe with disable_internal_profiling=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,        # Get Python call stacks!
        profile_memory=True,    # Track memory allocations
        record_shapes=False,    # Keep overhead reasonable
    ) as prof:

        # Run the actual benchmark (RecordFunctionTracer is disabled via flag)
        result = runner.run_tolerance_test(
            first_scenario,
            scenario_data['variants_to_run'],
            approaches
        )

    print()
    print("=" * 80)
    print("PROFILING COMPLETED")
    print("=" * 80)

    # Export chrome trace
    print()
    print("Exporting chrome trace...")
    trace_path = "full_benchmark_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"✓ Saved to: {trace_path}")
    print(f"  → View in Chrome: chrome://tracing")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # Get key averages
    key_avg = prof.key_averages(group_by_stack_n=5)  # Group by call stack

    # Print top CPU operations
    print("Top operations by CPU time:")
    print("-" * 80)
    cpu_ops = [evt for evt in key_avg if evt.cpu_time_total > 1000]  # > 1ms
    cpu_ops.sort(key=lambda x: x.cpu_time_total, reverse=True)

    print(f"{'Operation':<60} {'CPU Time':>12} {'Calls':>8}")
    print("-" * 80)
    for evt in cpu_ops[:30]:
        cpu_time_ms = evt.cpu_time_total / 1000
        op_name = evt.key[:60]
        print(f"{op_name:<60} {cpu_time_ms:>11.2f}ms {evt.count:>8}")

    print()

    # Print top GPU operations
    print("Top operations by CUDA time:")
    print("-" * 80)
    cuda_ops = [evt for evt in key_avg if evt.device_time_total > 1000]  # > 1ms
    cuda_ops.sort(key=lambda x: x.device_time_total, reverse=True)

    print(f"{'Operation':<60} {'CUDA Time':>12} {'Calls':>8}")
    print("-" * 80)
    for evt in cuda_ops[:30]:
        cuda_time_ms = evt.device_time_total / 1000
        op_name = evt.key[:60]
        print(f"{op_name:<60} {cuda_time_ms:>11.2f}ms {evt.count:>8}")

    # Totals
    total_cpu = sum(evt.cpu_time_total for evt in key_avg) / 1000
    total_cuda = sum(evt.device_time_total for evt in key_avg) / 1000

    print("-" * 80)
    print(f"{'TOTAL CPU':<60} {total_cpu:>11.2f}ms")
    print(f"{'TOTAL CUDA':<60} {total_cuda:>11.2f}ms")
    print()

    # Categorize
    print("=" * 80)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 80)

    categories = {
        'RecordFunctionTracer': ['RecordFunctionTracer', 'record_function', 'profiler'],
        'Tensor creation': ['randn', 'empty', 'zeros', 'fill_', 'randperm', 'normal_'],
        'Memory operations': ['malloc', 'cudaMalloc', 'CachingAllocator', 'copy', 'cudaMemcpy'],
        'FlashInfer plan': ['plan', 'BatchPrefillWithRaggedKVCacheWrapper', 'BatchDecodeWithPagedKVCacheWrapper'],
        'Attention kernels': ['flash', 'attention', 'batch_decode', 'batch_prefill', 'Batch'],
        'Synchronization': ['synchronize', 'cudaDeviceSynchronize', 'cudaStreamSynchronize'],
        'Cleanup': ['collect', 'empty_cache', 'gc'],
        'Data creation': ['build_workload', 'create_batch_data'],
    }

    for cat, keywords in categories.items():
        matching = [e for e in key_avg if any(kw.lower() in e.key.lower() for kw in keywords)]
        if matching:
            total_cpu_cat = sum(e.cpu_time_total for e in matching) / 1000
            total_cuda_cat = sum(e.device_time_total for e in matching) / 1000
            count = sum(e.count for e in matching)
            print(f"{cat:<30} CPU:{total_cpu_cat:>10.2f}ms  CUDA:{total_cuda_cat:>10.2f}ms  ({count} calls)")

    print()
    print("=" * 80)
    print(f"Open {trace_path} in Chrome (chrome://tracing) for timeline analysis")
    print("Look for GAPS between operations - that's where idle time is!")
    print("=" * 80)
    print()

    # Print scenario results
    print("Scenario results:")
    print(f"  Scenario: {result.scenario_name}")
    print(f"  Approach times: {result.approach_times}")
    print()

if __name__ == '__main__':
    main()
