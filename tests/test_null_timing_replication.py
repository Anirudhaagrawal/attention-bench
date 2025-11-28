"""Test to replicate null timing issues found in production benchmarks.

This test runs real scenarios from production that produced null timing results,
allowing us to debug and fix the root cause.
"""

import pytest
import torch

from approaches import APPROACHES
from attention_bench.timing import RecordFunctionTracer

# Handle both pytest and direct execution
try:
    from .utils import create_test_context
except ImportError:
    from utils import create_test_context


# Real scenarios from production that produced nulls
# Format: (name, q_lengths, kv_lengths, expected_null_approaches)
NULL_SCENARIOS = [
    # Scenario 1: Small decode + tiny prefill
    ("null_test_1", [1, 128], [8192, 128], ["flashinfer_batch_attention"]),

    # Scenario 2: Small decode + huge prefill
    ("null_test_2", [1, 512], [8192, 1048576], ["flashinfer_batch_attention", "flashinfer_mixed_fa2"]),

    # Scenario 3: Large decode + small prefill
    ("null_test_3", [1, 128], [131072, 1024], ["flashinfer_batch_attention", "flashinfer_mixed_fa2", "flashinfer_mixed_fa3", "official_fa3"]),

    # Scenario 4: Multi-decode + large prefill query
    ("null_test_4", [1, 1, 16384], [1024, 1024, 4096], ["flashinfer_batch_attention", "flashinfer_mixed_fa2"]),

    # Scenario 5: Multi-decode + large decode KV
    ("null_test_5", [1, 1, 4096], [65536, 65536, 2048], ["flashinfer_batch_attention", "flashinfer_mixed_fa3"]),

    # Scenario 6: Extreme - huge decode + large prefill
    ("null_test_6", [1, 1, 512], [1048576, 1048576, 131072], ["flashinfer_batch_attention", "flashinfer_mixed_fa2", "flashinfer_mixed_fa3", "official_fa3"]),
]


# Approaches to test
TEST_APPROACHES = [
    "flashinfer_batch_attention",
    "flashinfer_mixed_fa2",
    "flashinfer_mixed_fa3",
    "flashinfer_separated_fa2",
    "flashinfer_separated_fa3",
    "official_fa3"
]


@pytest.mark.parametrize("scenario_name,q_lengths,kv_lengths,expected_nulls", NULL_SCENARIOS)
def test_null_timing_replication(scenario_name, q_lengths, kv_lengths, expected_nulls):
    """Test that reproduces null timing issues from production.

    This test SHOULD FAIL initially - it's designed to replicate the bug.
    After fixing the profiler, these scenarios should produce valid timings.
    """
    print(f"\n{'='*80}")
    print(f"Testing scenario: {scenario_name}")
    print(f"  q_lengths: {q_lengths}")
    print(f"  kv_lengths: {kv_lengths}")
    print(f"  Expected nulls: {expected_nulls}")
    print(f"{'='*80}\n")

    # Create test context
    ctx = create_test_context(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
    )

    nulls_found = []
    successes = []

    # Test each approach
    for approach_name in TEST_APPROACHES:
        approach = APPROACHES.get(approach_name)
        if approach is None:
            print(f"  ⊘ {approach_name}: NOT AVAILABLE (skipping)")
            continue

        try:
            # Setup approach
            run_fn = approach.setup(ctx, return_output=False)

            # Warmup - use production-level iterations (50) to match real workload
            for _ in range(50):
                run_fn()
            torch.cuda.synchronize()

            # Profile with RecordFunctionTracer
            # Use 100 active iterations to match production config
            tracer = RecordFunctionTracer()

            with tracer:
                for _ in range(100):
                    with torch.profiler.record_function(approach_name):
                        run_fn()

            # Get timing stats
            stats = tracer.get_operation_time_stats()

            # Check if approach produced timing data
            if approach_name not in stats:
                nulls_found.append(approach_name)
                print(f"  ✗ NULL: {approach_name} - no timing data captured")
            else:
                mean_time = stats[approach_name]['mean']
                successes.append((approach_name, mean_time))
                print(f"  ✓ SUCCESS: {approach_name} - {mean_time:.3f} ms")

        except Exception as e:
            print(f"  ! ERROR: {approach_name} - {str(e)}")
            # Errors are different from nulls - re-raise for investigation
            raise

    print(f"\nResults for {scenario_name}:")
    print(f"  Nulls found: {nulls_found}")
    print(f"  Expected nulls: {expected_nulls}")
    print(f"  Successes: {[name for name, _ in successes]}")

    # Verify that we replicated the null issue
    # Sort for comparison
    nulls_found_sorted = sorted(nulls_found)
    expected_nulls_sorted = sorted(expected_nulls)

    assert nulls_found_sorted == expected_nulls_sorted, (
        f"Null replication mismatch!\n"
        f"  Expected nulls: {expected_nulls_sorted}\n"
        f"  Found nulls:    {nulls_found_sorted}\n"
        f"  Missing:        {set(expected_nulls_sorted) - set(nulls_found_sorted)}\n"
        f"  Extra:          {set(nulls_found_sorted) - set(expected_nulls_sorted)}\n"
        f"\n"
        f"This test is designed to replicate the bug. If nulls don't match,\n"
        f"either the bug was already fixed OR the test scenarios are incorrect."
    )

    print(f"\n✓ Successfully replicated null timing issue for {scenario_name}")


if __name__ == "__main__":
    # Run tests directly
    for scenario_name, q_lengths, kv_lengths, expected_nulls in NULL_SCENARIOS:
        test_null_timing_replication(scenario_name, q_lengths, kv_lengths, expected_nulls)
