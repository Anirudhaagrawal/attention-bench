"""Correctness tests for attention kernel implementations.

These tests verify that all attention kernel implementations produce
equivalent outputs for the same inputs, ensuring there are no bugs.
"""

import pytest
import torch

from approaches import APPROACHES
from .utils import assert_outputs_close


def run_approach(approach_name: str, ctx):
    """Run a single approach and return its output.

    Args:
        approach_name: Name of the approach to run
        ctx: BenchmarkContext with test data

    Returns:
        Output tensor from the approach

    Raises:
        Exception: If approach fails to run
    """
    approach = APPROACHES.get(approach_name)
    if approach is None:
        raise ValueError(f"Approach '{approach_name}' not found")

    # Setup and run
    run_fn = approach.setup(ctx)
    output = run_fn()
    torch.cuda.synchronize()

    # Handle case where output might be a tuple (output, stats)
    if isinstance(output, tuple):
        output = output[0]

    return output


def test_decode_correctness(small_decode_context, available_approaches):
    """Test that all kernels produce equivalent outputs for decode workload.

    Scenario: 4 decode requests (q=1) with kv=2048
    """
    print("\n" + "=" * 60)
    print("Testing decode-only workload correctness")
    print("Scenario: batch=4, q=1, kv=2048")
    print("=" * 60)

    ctx = small_decode_context

    # Run all available approaches
    outputs = {}
    failed_approaches = []
    for approach_name in available_approaches:
        try:
            output = run_approach(approach_name, ctx)
            outputs[approach_name] = output
            print(f"✓ {approach_name}: output shape {output.shape}")
        except Exception as e:
            print(f"✗ {approach_name}: SKIPPED - {e}")
            failed_approaches.append((approach_name, str(e)))

    # Need at least 2 approaches to compare
    if len(outputs) < 2:
        pytest.skip(f"Need at least 2 working approaches to compare. Working: {len(outputs)}, Failed: {len(failed_approaches)}")

    # Compare all pairs
    approach_names = list(outputs.keys())
    reference_name = approach_names[0]
    reference_output = outputs[reference_name]

    print(f"\nUsing {reference_name} as reference")
    print("-" * 60)

    for approach_name in approach_names[1:]:
        assert_outputs_close(
            reference_output,
            outputs[approach_name],
            reference_name,
            approach_name,
            rtol=1e-2,
            atol=1e-3,
        )

    print("=" * 60)
    print(f"✓ All {len(outputs)} approaches produce equivalent outputs!")
    print("=" * 60)


def test_prefill_correctness(small_prefill_context, available_approaches):
    """Test that all kernels produce equivalent outputs for prefill workload.

    Scenario: 2 prefill requests with q=kv=512
    """
    print("\n" + "=" * 60)
    print("Testing prefill-only workload correctness")
    print("Scenario: batch=2, q=kv=512")
    print("=" * 60)

    ctx = small_prefill_context

    # Run all available approaches
    outputs = {}
    failed_approaches = []
    for approach_name in available_approaches:
        try:
            output = run_approach(approach_name, ctx)
            outputs[approach_name] = output
            print(f"✓ {approach_name}: output shape {output.shape}")
        except Exception as e:
            print(f"✗ {approach_name}: SKIPPED - {e}")
            failed_approaches.append((approach_name, str(e)))

    # Need at least 2 approaches to compare
    if len(outputs) < 2:
        pytest.skip(f"Need at least 2 working approaches to compare. Working: {len(outputs)}, Failed: {len(failed_approaches)}")

    # Compare all pairs
    approach_names = list(outputs.keys())
    reference_name = approach_names[0]
    reference_output = outputs[reference_name]

    print(f"\nUsing {reference_name} as reference")
    print("-" * 60)

    for approach_name in approach_names[1:]:
        assert_outputs_close(
            reference_output,
            outputs[approach_name],
            reference_name,
            approach_name,
            rtol=1e-2,
            atol=1e-3,
        )

    print("=" * 60)
    print(f"✓ All {len(outputs)} approaches produce equivalent outputs!")
    print("=" * 60)


def test_mixed_correctness(mixed_workload_context, available_approaches):
    """Test that all kernels produce equivalent outputs for mixed workload.

    Scenario: 2 decodes + 2 prefills (q=[1,1,128,256], kv=[1k,2k,1k,2k])
    """
    print("\n" + "=" * 60)
    print("Testing mixed decode+prefill workload correctness")
    print("Scenario: 2 decodes + 2 prefills")
    print("  q=[1, 1, 128, 256], kv=[1k, 2k, 1k, 2k]")
    print("=" * 60)

    ctx = mixed_workload_context

    # Run all available approaches
    outputs = {}
    failed_approaches = []
    for approach_name in available_approaches:
        try:
            output = run_approach(approach_name, ctx)
            outputs[approach_name] = output
            print(f"✓ {approach_name}: output shape {output.shape}")
        except Exception as e:
            print(f"✗ {approach_name}: SKIPPED - {e}")
            failed_approaches.append((approach_name, str(e)))

    # Need at least 2 approaches to compare
    if len(outputs) < 2:
        pytest.skip(f"Need at least 2 working approaches to compare. Working: {len(outputs)}, Failed: {len(failed_approaches)}")

    # Compare all pairs
    approach_names = list(outputs.keys())
    reference_name = approach_names[0]
    reference_output = outputs[reference_name]

    print(f"\nUsing {reference_name} as reference")
    print("-" * 60)

    for approach_name in approach_names[1:]:
        assert_outputs_close(
            reference_output,
            outputs[approach_name],
            reference_name,
            approach_name,
            rtol=1e-2,
            atol=1e-3,
        )

    print("=" * 60)
    print(f"✓ All {len(outputs)} approaches produce equivalent outputs!")
    print("=" * 60)
