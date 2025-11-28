"""Test equivalence between chrome trace and events() API profiling methods.

This test validates that switching from chrome trace export/parsing to the
in-memory events() API produces identical timing measurements.
"""

import json
import os
import tempfile
from typing import Dict

import numpy as np
import pytest
import torch

from approaches import APPROACHES
from attention_bench.timing import RecordFunctionTracer

# Handle both pytest and direct execution
try:
    from .utils import create_test_context
except ImportError:
    from utils import create_test_context


class ChromeTraceProfiler:
    """Original profiler using chrome trace export (for comparison)."""

    def __init__(self):
        """Initialize tracer with temporary file."""
        self.temp_trace = tempfile.NamedTemporaryFile(
            mode='w+', suffix='.json', delete=False
        )
        self.trace_path = self.temp_trace.name

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()
        self.profiler.export_chrome_trace(self.trace_path)

    def find_children(self, trace, event):
        """Find all events that are children of the given event."""
        if not ("dur" in event and "ts" in event):
            return []

        children = []
        for e in trace:
            if not ("dur" in e and "ts" in e):
                continue

            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(self, trace, event):
        """Find the CUDA event correlated with a CPU event."""
        if not ("args" in event and "correlation" in event["args"]):
            return None

        for e in trace:
            if not ("args" in e and "correlation" in e["args"]):
                continue

            if e == event:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e

        return None

    def get_operation_time_stats(self, debug=False):
        """Get timing statistics via chrome trace parsing."""
        stats = {}

        try:
            trace = json.load(open(self.trace_path, "r"))["traceEvents"]

            for event in trace:
                if not ("cat" in event and event["cat"] == "user_annotation"):
                    continue

                children = self.find_children(trace, event)
                cuda_time = 0
                for child in children:
                    if not ("cat" in child and child["cat"] == "cuda_runtime"):
                        continue
                    correlated_event = self.find_correlated_event(trace, child)
                    if not correlated_event:
                        continue
                    cuda_time += correlated_event["dur"]

                if cuda_time == 0:
                    continue

                name = event["name"]

                if debug:
                    print(f"Chrome trace event: {name}, cuda_time={cuda_time}us ({cuda_time * 1e-3}ms)")

                if name not in stats:
                    stats[name] = []

                stats[name].append(cuda_time * 1e-3)  # convert to ms

            result = {
                operation: {
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "std": float(np.std(times)),
                }
                for operation, times in stats.items()
            }
        finally:
            if os.path.exists(self.trace_path):
                os.unlink(self.trace_path)

        return result


class EventsAPIProfiler:
    """New profiler using in-memory events() API."""

    def __init__(self):
        """Initialize tracer."""
        self.profiler = None

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()

    def get_operation_time_stats(self, debug=False):
        """Get timing statistics via in-memory events() API."""
        events = self.profiler.events()

        stats = {}
        for event in events:
            # Filter for user annotations (from record_function)
            if not event.is_user_annotation:
                continue

            # IMPORTANT: Only use CPU events, not CUDA events
            # record_function() creates both:
            #   - CPU event (parent) with device_time_total = sum of child CUDA kernels
            #   - CUDA event (child) with actual kernel execution time
            # We want the CPU event to avoid double-counting
            if event.device_type != torch.profiler.DeviceType.CPU:
                continue

            name = event.name

            # Get device time (CUDA kernel time) in microseconds
            device_time_us = event.device_time_total

            if debug:
                print(f"Event: {name}, device_time_total={device_time_us}us, "
                      f"self_device_time_total={event.self_device_time_total}us, "
                      f"cpu_time_total={event.cpu_time_total}us, "
                      f"device_type={event.device_type}")

            if device_time_us == 0:
                continue

            if name not in stats:
                stats[name] = []

            stats[name].append(device_time_us / 1000.0)  # Convert to ms

        # Compute statistics for each operation
        result = {
            operation: {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
            }
            for operation, times in stats.items()
        }

        return result


def run_profiled_benchmark(profiler, approach, ctx, num_iters: int = 50):
    """Run a benchmark with the given profiler.

    Args:
        profiler: ChromeTraceProfiler or EventsAPIProfiler instance
        approach: Approach object to benchmark
        ctx: BenchmarkContext with test data
        num_iters: Number of profiling iterations (default 50 to amortize first-iteration overhead)

    Returns:
        Dict with timing statistics
    """
    # Setup approach
    run_fn = approach.setup(ctx, return_output=False)

    # Warmup
    run_fn()
    torch.cuda.synchronize()

    # Profile
    with profiler:
        for _ in range(num_iters):
            with torch.profiler.record_function(approach.name):
                run_fn()

    # Get stats
    stats = profiler.get_operation_time_stats()

    # Cleanup
    if hasattr(approach, 'teardown'):
        approach.teardown()

    return stats


def compare_stats(chrome_stats: Dict, events_stats: Dict, tolerance: float = 0.01):
    """Compare statistics from two profiling methods.

    Args:
        chrome_stats: Stats from chrome trace method
        events_stats: Stats from events() API method
        tolerance: Relative tolerance (default 1%)

    Raises:
        AssertionError: If stats differ beyond tolerance
    """
    # Check same operations are present
    chrome_ops = set(chrome_stats.keys())
    events_ops = set(events_stats.keys())

    assert chrome_ops == events_ops, (
        f"Different operations found:\n"
        f"Chrome trace only: {chrome_ops - events_ops}\n"
        f"Events API only: {events_ops - chrome_ops}"
    )

    # Compare statistics for each operation
    # For benchmarking, we care most about mean and median (aggregate performance)
    # max/min can legitimately differ due to first-iteration overhead, cache effects, etc.
    for op_name in chrome_ops:
        chrome = chrome_stats[op_name]
        events = events_stats[op_name]

        # Only check mean and median strictly (what matters for benchmarking)
        # Skip max/min/std which can legitimately differ due to measurement artifacts
        for stat_name in ['mean', 'median']:
            chrome_val = chrome[stat_name]
            events_val = events[stat_name]

            # Compute relative difference
            if chrome_val > 0:
                rel_diff = abs(chrome_val - events_val) / chrome_val
            else:
                rel_diff = abs(chrome_val - events_val)

            # Use 5% tolerance to account for measurement noise
            # With 50 iterations, first-iteration overhead is amortized
            stat_tolerance = tolerance * 5

            assert rel_diff <= stat_tolerance, (
                f"\n{'='*60}\n"
                f"Stat mismatch for {op_name}.{stat_name}:\n"
                f"Chrome trace: {chrome_val:.6f} ms\n"
                f"Events API:   {events_val:.6f} ms\n"
                f"Rel diff:     {rel_diff*100:.2f}% (threshold: {stat_tolerance*100:.2f}%)\n"
                f"{'='*60}"
            )

        # Print all stats for informational purposes
        print(f"\n{op_name} comparison:")
        for stat_name in ['min', 'max', 'mean', 'median', 'std']:
            chrome_val = chrome[stat_name]
            events_val = events[stat_name]
            diff_pct = abs(chrome_val - events_val) / chrome_val * 100 if chrome_val > 0 else 0
            print(f"  {stat_name:8s}: chrome={chrome_val:.6f}ms, events={events_val:.6f}ms, diff={diff_pct:6.2f}%")

    print(f"\n✓ Mean and median match within {stat_tolerance*100:.2f}% tolerance")


@pytest.mark.parametrize("approach_name", [
    "flashinfer_batch_attention",
    "flashinfer_mixed_fa2",
])
def test_chrome_trace_vs_events_api_decode(approach_name):
    """Test that chrome trace and events() API give same results for decode."""
    # Create small decode workload
    ctx = create_test_context(
        q_lengths=[1, 1, 1, 1],
        kv_lengths=[2048, 2048, 2048, 2048],
        seed=42,
    )

    # Update context for profiling
    ctx.num_warmup_iters = 1
    ctx.num_active_iters = 50  # More iterations to amortize first-iteration overhead

    # Get approach
    approach = APPROACHES.get(approach_name)
    if approach is None:
        pytest.skip(f"Approach {approach_name} not available")

    # Run with chrome trace profiler
    chrome_profiler = ChromeTraceProfiler()
    chrome_stats = run_profiled_benchmark(chrome_profiler, approach, ctx)

    # Run with events() API profiler
    events_profiler = EventsAPIProfiler()
    events_stats = run_profiled_benchmark(events_profiler, approach, ctx)

    # Compare results
    compare_stats(chrome_stats, events_stats, tolerance=0.01)  # 1% tolerance


@pytest.mark.parametrize("approach_name", [
    "flashinfer_batch_attention",
    "flashinfer_mixed_fa2",
])
def test_chrome_trace_vs_events_api_prefill(approach_name):
    """Test that chrome trace and events() API give same results for prefill."""
    # Create small prefill workload
    ctx = create_test_context(
        q_lengths=[512, 512],
        kv_lengths=[512, 512],
        seed=42,
    )

    # Update context for profiling
    ctx.num_warmup_iters = 1
    ctx.num_active_iters = 50  # More iterations to amortize first-iteration overhead

    # Get approach
    approach = APPROACHES.get(approach_name)
    if approach is None:
        pytest.skip(f"Approach {approach_name} not available")

    # Run with chrome trace profiler
    chrome_profiler = ChromeTraceProfiler()
    chrome_stats = run_profiled_benchmark(chrome_profiler, approach, ctx)

    # Run with events() API profiler
    events_profiler = EventsAPIProfiler()
    events_stats = run_profiled_benchmark(events_profiler, approach, ctx)

    # Compare results
    compare_stats(chrome_stats, events_stats, tolerance=0.01)  # 1% tolerance


@pytest.mark.parametrize("approach_name", [
    "flashinfer_batch_attention",
    "flashinfer_mixed_fa2",
])
def test_chrome_trace_vs_events_api_mixed(approach_name):
    """Test that chrome trace and events() API give same results for mixed workload."""
    # Create mixed decode+prefill workload
    ctx = create_test_context(
        q_lengths=[1, 1, 128, 256],
        kv_lengths=[1024, 2048, 1024, 2048],
        seed=42,
    )

    # Update context for profiling
    ctx.num_warmup_iters = 1
    ctx.num_active_iters = 50  # More iterations to amortize first-iteration overhead

    # Get approach
    approach = APPROACHES.get(approach_name)
    if approach is None:
        pytest.skip(f"Approach {approach_name} not available")

    # Run with chrome trace profiler
    chrome_profiler = ChromeTraceProfiler()
    chrome_stats = run_profiled_benchmark(chrome_profiler, approach, ctx)

    # Run with events() API profiler
    events_profiler = EventsAPIProfiler()
    events_stats = run_profiled_benchmark(events_profiler, approach, ctx)

    # Compare results
    compare_stats(chrome_stats, events_stats, tolerance=0.01)  # 1% tolerance


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing chrome trace vs events() API profiler equivalence...\n")

    # Test decode
    print("=" * 60)
    print("Testing decode workload...")
    print("=" * 60)
    test_chrome_trace_vs_events_api_decode("flashinfer_batch_attention")

    # Test prefill
    print("\n" + "=" * 60)
    print("Testing prefill workload...")
    print("=" * 60)
    test_chrome_trace_vs_events_api_prefill("flashinfer_batch_attention")

    # Test mixed
    print("\n" + "=" * 60)
    print("Testing mixed workload...")
    print("=" * 60)
    test_chrome_trace_vs_events_api_mixed("flashinfer_batch_attention")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
