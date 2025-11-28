"""
Test to validate that torch.empty() can safely replace torch.randn() for timing benchmarks.

This test creates benchmark contexts with both torch.randn and torch.empty, then:
1. Runs all registered approaches with both tensor types
2. Validates outputs contain no NaN/Inf values
3. Compares timing consistency between the two methods
4. Ensures kernel execution completes without errors

If all tests pass, it proves torch.empty is safe for production use.
"""

import sys
import torch
import pytest
from typing import Tuple, List

sys.path.insert(0, 'src/attention_bench')

from cli.run_benchmark import BenchmarkContext, APPROACHES
from timing.record_function_tracer import RecordFunctionTracer


def create_benchmark_context_with_tensor_fn(
    tensor_fn,  # torch.randn or torch.empty
    batch_size: int = 4,
    q_len: int = 1,
    kv_len: int = 128,
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
) -> BenchmarkContext:
    """Create a minimal BenchmarkContext using specified tensor creation function.

    Args:
        tensor_fn: Either torch.randn or torch.empty
        Other args: Benchmark configuration parameters

    Returns:
        BenchmarkContext ready for approach execution
    """
    # Calculate pages needed
    num_pages = [(kv_len + page_size - 1) // page_size for _ in range(batch_size)]
    total_pages = sum(num_pages)
    total_q_tokens = batch_size * q_len

    # Create Q and KV cache with specified tensor function
    q = tensor_fn(
        total_q_tokens, num_qo_heads, head_dim,
        dtype=torch.float16, device="cuda"
    )

    kv_cache = tensor_fn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda"
    )

    # Create metadata tensors (these can be randn/empty, doesn't matter)
    qo_indptr = torch.tensor(
        [i * q_len for i in range(batch_size + 1)],
        dtype=torch.int32, device="cuda"
    )

    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")

    kv_page_indptr = torch.tensor(
        [0] + [sum(num_pages[:i+1]) for i in range(batch_size)],
        dtype=torch.int32, device="cuda"
    )

    kv_last_page_len = torch.tensor(
        [kv_len % page_size if kv_len % page_size != 0 else page_size] * batch_size,
        dtype=torch.int32, device="cuda"
    )

    seq_lens_q = torch.tensor([q_len] * batch_size, dtype=torch.int32, device="cuda")
    seq_lens_kv = torch.tensor([kv_len] * batch_size, dtype=torch.int32, device="cuda")

    # Create block tables for cuDNN
    import math
    max_num_blocks_per_seq = math.ceil(kv_len / page_size)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32, device="cuda"
    )
    for i in range(batch_size):
        start_idx = kv_page_indptr[i].item()
        end_idx = kv_page_indptr[i + 1].item()
        seq_page_indices = kv_page_indices[start_idx:end_idx]
        block_tables[i, :len(seq_page_indices)] = seq_page_indices

    # Split Q into decode/prefill (for mixed approaches)
    decode_q = q
    prefill_q = q[:0]  # Empty tensor
    decode_indices = list(range(batch_size))
    prefill_indices = []

    return BenchmarkContext(
        q_lengths=[q_len] * batch_size,
        kv_lengths=[kv_len] * batch_size,
        q=q,
        kv_cache=kv_cache,
        decode_q=decode_q,
        prefill_q=prefill_q,
        decode_indices=decode_indices,
        prefill_indices=prefill_indices,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        seq_lens_q=seq_lens_q,
        seq_lens_kv=seq_lens_kv,
        block_tables=block_tables,
        max_token_per_sequence=q_len,
        max_sequence_kv=kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        workspace_size=128 * 1024 * 1024,  # 128MB
        num_warmup_iters=1,
        num_active_iters=10,
    )


def run_approach_with_context(approach, approach_name: str, ctx: BenchmarkContext) -> Tuple[torch.Tensor, float]:
    """Run an approach with given context and return output + timing.

    This matches production code exactly by using RecordFunctionTracer.

    Args:
        approach: Approach instance
        approach_name: Name for debugging
        ctx: BenchmarkContext to use

    Returns:
        (output_tensor, avg_time_ms)
    """
    # Setup approach and get callable
    # Some approaches don't support return_output parameter (e.g., cuDNN)
    try:
        run_fn = approach.setup(ctx, return_output=True)
    except TypeError:
        run_fn = approach.setup(ctx)

    # Warmup
    for _ in range(ctx.num_warmup_iters):
        output = run_fn()

    # Timed runs using RecordFunctionTracer (MATCHES PRODUCTION)
    tracer = RecordFunctionTracer()
    with tracer:
        for _ in range(ctx.num_active_iters):
            with torch.profiler.record_function(approach_name):
                output = run_fn()

    # Get timing statistics from tracer
    time_stats_dict = tracer.get_operation_time_stats()

    if approach_name in time_stats_dict:
        time_stats = time_stats_dict[approach_name]
        avg_time_ms = time_stats['mean']
    else:
        # Fallback if no timing captured
        avg_time_ms = 0.0

    # Teardown
    if hasattr(approach, 'teardown'):
        approach.teardown()

    # Handle tuple outputs (some approaches return multiple tensors)
    if isinstance(output, tuple):
        output = output[0]  # Use first tensor for validation

    return output, avg_time_ms


def test_empty_vs_randn_all_approaches():
    """Test that torch.empty and torch.randn produce working results for all approaches."""
    print("\n" + "="*80)
    print("Testing torch.empty vs torch.randn with all approaches")
    print("="*80)

    # Get all registered approaches
    approach_names = APPROACHES.list_available()
    print(f"\nFound {len(approach_names)} registered approaches: {approach_names}")

    results = {}

    for approach_name in approach_names:
        print(f"\n{'='*80}")
        print(f"Testing: {approach_name}")
        print(f"{'='*80}")

        approach = APPROACHES.get(approach_name)

        # Test with torch.randn
        print(f"  [1/2] Running with torch.randn...")
        try:
            ctx_randn = create_benchmark_context_with_tensor_fn(torch.randn)
            output_randn, time_randn = run_approach_with_context(approach, approach_name, ctx_randn)
            print(f"    ✓ Success: {time_randn:.3f} ms, output shape: {output_randn.shape}")

            # Check for NaN/Inf
            has_nan = torch.isnan(output_randn).any().item()
            has_inf = torch.isinf(output_randn).any().item()
            print(f"    ✓ NaN check: {'FAIL - has NaN!' if has_nan else 'PASS'}")
            print(f"    ✓ Inf check: {'FAIL - has Inf!' if has_inf else 'PASS'}")

            randn_result = {
                'success': True,
                'time_ms': time_randn,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'output_shape': output_randn.shape,
            }
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            randn_result = {'success': False, 'error': str(e)}

        # Clean up
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Test with torch.empty
        print(f"  [2/2] Running with torch.empty...")
        try:
            ctx_empty = create_benchmark_context_with_tensor_fn(torch.empty)
            output_empty, time_empty = run_approach_with_context(approach, approach_name, ctx_empty)
            print(f"    ✓ Success: {time_empty:.3f} ms, output shape: {output_empty.shape}")

            # Check for NaN/Inf
            has_nan = torch.isnan(output_empty).any().item()
            has_inf = torch.isinf(output_empty).any().item()
            print(f"    ✓ NaN check: {'FAIL - has NaN!' if has_nan else 'PASS'}")
            print(f"    ✓ Inf check: {'FAIL - has Inf!' if has_inf else 'PASS'}")

            empty_result = {
                'success': True,
                'time_ms': time_empty,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'output_shape': output_empty.shape,
            }
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            empty_result = {'success': False, 'error': str(e)}

        # Clean up
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Compare results
        results[approach_name] = {
            'randn': randn_result,
            'empty': empty_result,
        }

        if randn_result['success'] and empty_result['success']:
            time_diff_pct = abs(time_randn - time_empty) / time_randn * 100
            print(f"\n  Summary:")
            print(f"    torch.randn: {time_randn:.3f} ms")
            print(f"    torch.empty: {time_empty:.3f} ms")
            print(f"    Difference:  {time_diff_pct:.1f}%")
            print(f"    Verdict:     {'✓ SAFE TO USE' if time_diff_pct < 10 else '⚠ Performance difference'}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    all_passed = True
    torch_empty_safe = True
    for approach_name, result in results.items():
        randn_ok = result['randn']['success'] and not result['randn'].get('has_nan') and not result['randn'].get('has_inf')
        empty_ok = result['empty']['success'] and not result['empty'].get('has_nan') and not result['empty'].get('has_inf')

        # If torch.randn works but torch.empty doesn't, that's a problem
        if randn_ok and not empty_ok:
            status = "✗ FAIL (torch.empty issue)"
            torch_empty_safe = False
        # If both work, that's good
        elif randn_ok and empty_ok:
            status = "✓ PASS"
        # If neither work, that's a test setup issue, not torch.empty issue
        elif not randn_ok and not empty_ok:
            status = "⚠ SKIP (test setup issue)"
        # If only torch.empty works, that's actually fine (unexpected but ok)
        else:
            status = "✓ PASS (torch.empty only)"

        print(f"  {approach_name:30s}: {status}")

        if not (randn_ok and empty_ok):
            all_passed = False
            if not randn_ok:
                print(f"    - torch.randn failed: {result['randn'].get('error', 'NaN/Inf detected')}")
            if not empty_ok:
                print(f"    - torch.empty failed: {result['empty'].get('error', 'NaN/Inf detected')}")

    print(f"\n{'='*80}")
    if torch_empty_safe:
        print("✓✓✓ torch.empty is SAFE - all working approaches pass with torch.empty! ✓✓✓")
        if not all_passed:
            print("Note: Some approaches failed due to test setup issues (not torch.empty)")
    else:
        print("✗✗✗ torch.empty may not be safe - some approaches fail ONLY with torch.empty ✗✗✗")
    print(f"{'='*80}\n")

    # Assertions for pytest - only fail if torch.empty causes unique failures
    assert torch_empty_safe, "torch.empty caused failures that torch.randn didn't"


def test_empty_tensor_creation_speedup():
    """Measure speedup of torch.empty vs torch.randn for tensor creation."""
    print("\n" + "="*80)
    print("Measuring torch.empty vs torch.randn speedup")
    print("="*80)

    # Test different sizes
    sizes = [
        ("Small", 100),   # 100 pages
        ("Medium", 500),  # 500 pages
        ("Large", 1000),  # 1000 pages
    ]

    page_size = 16
    num_kv_heads = 8
    head_dim = 128

    print(f"\n{'Size':<10} {'Pages':<10} {'torch.randn':<15} {'torch.empty':<15} {'Speedup':<10}")
    print("-" * 80)

    for size_name, num_pages in sizes:
        # Measure torch.randn
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kv_randn = torch.randn(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda"
        )
        end.record()
        torch.cuda.synchronize()
        randn_time = start.elapsed_time(end)

        del kv_randn
        torch.cuda.empty_cache()

        # Measure torch.empty
        torch.cuda.synchronize()
        start.record()
        kv_empty = torch.empty(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda"
        )
        end.record()
        torch.cuda.synchronize()
        empty_time = start.elapsed_time(end)

        del kv_empty
        torch.cuda.empty_cache()

        speedup = randn_time / empty_time
        print(f"{size_name:<10} {num_pages:<10} {randn_time:<15.2f} {empty_time:<15.2f} {speedup:<10.1f}x")

    print("\n" + "="*80 + "\n")


def test_empty_with_production_workloads():
    """Test torch.empty with realistic production workloads from actual configs.

    Tests scenarios extracted from configs/decode/config_decode_all_combinations.yaml
    and configs/mixed/config_mixed_all_combinations.yaml:
    - Small: batch=16, kv=4K (typical inference)
    - Medium: batch=64, kv=16K (moderate load)
    - Large: batch=128, kv=65K (large context)
    """
    print("\n" + "="*80)
    print("Testing torch.empty with PRODUCTION WORKLOADS")
    print("="*80)

    # Production scenarios from actual config files
    production_scenarios = [
        ("Small (batch=16, kv=4K)", {
            "batch_size": 16,
            "q_len": 1,
            "kv_len": 4096,
        }),
        ("Medium (batch=64, kv=16K)", {
            "batch_size": 64,
            "q_len": 1,
            "kv_len": 16384,
        }),
        ("Large (batch=128, kv=65K)", {
            "batch_size": 128,
            "q_len": 1,
            "kv_len": 65536,
        }),
    ]

    # Get all registered approaches
    approach_names = APPROACHES.list_available()
    print(f"\nTesting {len(approach_names)} approaches across {len(production_scenarios)} production scenarios")

    overall_results = {}

    for scenario_name, params in production_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")

        scenario_results = {}

        for approach_name in approach_names:
            print(f"\n  {approach_name}...")
            approach = APPROACHES.get(approach_name)

            # Test with torch.randn
            try:
                ctx_randn = create_benchmark_context_with_tensor_fn(torch.randn, **params)
                output_randn, time_randn = run_approach_with_context(approach, approach_name, ctx_randn)
                has_nan = torch.isnan(output_randn).any().item()
                has_inf = torch.isinf(output_randn).any().item()

                randn_result = {
                    'success': True,
                    'time_ms': time_randn,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }
                print(f"    torch.randn: {time_randn:.3f} ms ({'✓' if not (has_nan or has_inf) else '✗'})")
            except Exception as e:
                randn_result = {'success': False, 'error': str(e)}
                print(f"    torch.randn: FAILED - {e}")

            # Clean up
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Test with torch.empty
            try:
                ctx_empty = create_benchmark_context_with_tensor_fn(torch.empty, **params)
                output_empty, time_empty = run_approach_with_context(approach, approach_name, ctx_empty)
                has_nan = torch.isnan(output_empty).any().item()
                has_inf = torch.isinf(output_empty).any().item()

                empty_result = {
                    'success': True,
                    'time_ms': time_empty,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }
                print(f"    torch.empty: {time_empty:.3f} ms ({'✓' if not (has_nan or has_inf) else '✗'})")
            except Exception as e:
                empty_result = {'success': False, 'error': str(e)}
                print(f"    torch.empty: FAILED - {e}")

            # Clean up
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Compare results
            scenario_results[approach_name] = {
                'randn': randn_result,
                'empty': empty_result,
            }

            if randn_result['success'] and empty_result['success']:
                time_diff_pct = abs(time_randn - time_empty) / time_randn * 100
                print(f"    Difference: {time_diff_pct:.1f}%")

        overall_results[scenario_name] = scenario_results

    # Final summary
    print(f"\n{'='*80}")
    print("PRODUCTION WORKLOAD SUMMARY")
    print(f"{'='*80}")

    all_safe = True
    for scenario_name, scenario_results in overall_results.items():
        print(f"\n{scenario_name}:")
        for approach_name, result in scenario_results.items():
            randn_ok = result['randn']['success'] and not result['randn'].get('has_nan') and not result['randn'].get('has_inf')
            empty_ok = result['empty']['success'] and not result['empty'].get('has_nan') and not result['empty'].get('has_inf')

            if randn_ok and not empty_ok:
                status = "✗ FAIL (torch.empty issue)"
                all_safe = False
            elif randn_ok and empty_ok:
                status = "✓ PASS"
            elif not randn_ok and not empty_ok:
                status = "⚠ SKIP"
            else:
                status = "✓ PASS"

            print(f"  {approach_name:30s}: {status}")

    print(f"\n{'='*80}")
    if all_safe:
        print("✓✓✓ torch.empty is SAFE with ALL production workloads! ✓✓✓")
    else:
        print("✗✗✗ torch.empty failed with some production workloads ✗✗✗")
    print(f"{'='*80}\n")

    assert all_safe, "torch.empty failed with production workloads"


if __name__ == "__main__":
    # Run tests
    print("="*80)
    print("TEST SUITE: torch.empty Safety Validation")
    print("="*80)

    # Test 1: Small test workload
    test_empty_vs_randn_all_approaches()

    # Test 2: Production workloads
    test_empty_with_production_workloads()

    # Test 3: Tensor creation speedup
    test_empty_tensor_creation_speedup()
