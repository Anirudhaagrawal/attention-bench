#!/usr/bin/env python3
"""FlashInfer Prefill Tolerance Benchmark (v2 - Refactored with Approach Registry)

Compares multiple attention implementations:
- FlashInfer: Mixed FA2/FA3, Separated FA2/FA3, BatchAttention, cuDNN
- Official FlashAttention-3

Usage:
    python run_benchmark.py [--config config.yaml]
    python run_benchmark.py --approaches official_fa3,flashinfer_mixed_fa3
"""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import flashinfer
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, BatchDecodeWithPagedKVCacheWrapper
from flashinfer.page import get_seq_lens

# Import approaches registry
from attention_bench.approaches import APPROACHES, BenchmarkContext

# Import timing utilities
from attention_bench.timing import RecordFunctionTracer

# Import workload utilities
from attention_bench.utils.workload import build_workload_from_variant_set


def _classify_and_reorder_requests(
    q_lengths: List[int], kv_lengths: List[int]
) -> Tuple[List[int], List[int], List[int], List[int], bool]:
    """Classify requests as decode/prefill and reorder for optimal memory layout.

    Requests are classified based on query length:
    - decode: q_len == 1
    - prefill: q_len > 1

    For pure workloads (all decode or all prefill), requests are reordered to
    group decode first, then prefill, enabling memory-efficient tensor views.
    Mixed workloads preserve original order.

    Args:
        q_lengths: Query sequence lengths for all requests
        kv_lengths: KV cache lengths for all requests

    Returns:
        Tuple of (reordered_q_lengths, reordered_kv_lengths, decode_indices,
                  prefill_indices, is_mixed_workload)
    """
    decode_indices = [i for i, q_len in enumerate(q_lengths) if q_len == 1]
    prefill_indices = [i for i, q_len in enumerate(q_lengths) if q_len > 1]

    has_decode = len(decode_indices) > 0
    has_prefill = len(prefill_indices) > 0
    is_mixed_workload = has_decode and has_prefill

    if is_mixed_workload:
        # Keep original order for mixed workloads
        reordered_q_lengths = q_lengths
        reordered_kv_lengths = kv_lengths
    else:
        # Reorder: decode first, then prefill (for memory-efficient views)
        reordered_indices = decode_indices + prefill_indices
        reordered_q_lengths = [q_lengths[i] for i in reordered_indices]
        reordered_kv_lengths = [kv_lengths[i] for i in reordered_indices]

    return (reordered_q_lengths, reordered_kv_lengths, decode_indices,
            prefill_indices, is_mixed_workload)


def _create_q_tensor_views(
    q: torch.Tensor,
    reordered_q_lengths: List[int],
    decode_indices: List[int],
    prefill_indices: List[int],
    is_mixed_workload: bool
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Create pre-split Q tensor views for decode and prefill requests.

    For pure workloads, uses contiguous views (zero-copy).
    For mixed workloads, uses advanced indexing to gather tokens.

    Args:
        q: Full Q tensor of shape (total_tokens, num_heads, head_dim)
        reordered_q_lengths: Query lengths in reordered order
        decode_indices: Original indices of decode requests
        prefill_indices: Original indices of prefill requests
        is_mixed_workload: Whether workload has both decode and prefill

    Returns:
        Tuple of (decode_q, prefill_q) - Views or None for pure workloads
    """
    num_decode_tokens = sum(1 for i in decode_indices)  # decode has q_len=1

    if is_mixed_workload:
        # Mixed workload: requests are NOT reordered, use advanced indexing
        decode_q = None
        prefill_q = None

        if num_decode_tokens > 0:
            decode_token_indices = []
            token_offset = 0
            for i, q_len in enumerate(reordered_q_lengths):
                if i in decode_indices:
                    decode_token_indices.extend(range(token_offset, token_offset + q_len))
                token_offset += q_len
            decode_q = q[decode_token_indices] if decode_token_indices else None

        if num_decode_tokens < q.shape[0]:
            prefill_token_indices = []
            token_offset = 0
            for i, q_len in enumerate(reordered_q_lengths):
                if i in prefill_indices:
                    prefill_token_indices.extend(range(token_offset, token_offset + q_len))
                token_offset += q_len
            prefill_q = q[prefill_token_indices] if prefill_token_indices else None
    else:
        # Pure workload: requests ARE reordered (decode first, then prefill)
        # Simple contiguous views work here
        decode_q = q[:num_decode_tokens] if num_decode_tokens > 0 else None
        prefill_q = q[num_decode_tokens:] if num_decode_tokens < q.shape[0] else None

    return decode_q, prefill_q


def _create_block_tables(
    batch_size: int,
    max_num_blocks: int,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor
) -> torch.Tensor:
    """Create block tables for all requests.

    Block tables map each request to its KV cache pages in a dense tensor format.

    Args:
        batch_size: Number of requests in the batch
        max_num_blocks: Maximum number of blocks per sequence
        kv_page_indptr: Pointer array for page indices (CSR format)
        kv_page_indices: Flat array of page indices

    Returns:
        block_tables tensor of shape (batch_size, max_num_blocks)
    """
    block_tables = torch.zeros(
        (batch_size, max_num_blocks),
        dtype=torch.int32, device="cuda",
    )

    for i in range(batch_size):
        start_idx = kv_page_indptr[i]
        end_idx = kv_page_indptr[i + 1]
        num_blocks = end_idx - start_idx
        block_tables[i, :num_blocks] = kv_page_indices[start_idx:end_idx]

    return block_tables


# ============================================================================
# Helper functions for run_tolerance_test decomposition
# ============================================================================

def _check_memory_feasibility(
    q_lengths: List[int],
    kv_lengths: List[int],
    config: 'BenchmarkConfig',
) -> Tuple[bool, Optional[str]]:
    """Check if workload can fit in available GPU memory.

    Args:
        q_lengths: Query sequence lengths for all requests
        kv_lengths: KV cache lengths for all requests
        config: Benchmark configuration

    Returns:
        Tuple of (fits_in_memory, skip_reason_if_not)
    """
    memory_check_enabled = getattr(config, 'enable_memory_check', False)
    if not memory_check_enabled:
        return True, None

    from attention_bench.utils.memory_estimator import estimate_scenario_memory
    estimate = estimate_scenario_memory(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_kv_heads=config.num_kv_heads,
        num_qo_heads=config.num_qo_heads,
        head_dim=config.head_dim,
        page_size=config.page_size,
        workspace_size=config.workspace_size,
        utilization=getattr(config, 'memory_utilization', 0.90),
        overhead=getattr(config, 'memory_overhead', 1.1)
    )

    if not estimate.fits:
        if estimate.exceeds_page_limit:
            skip_reason = f"PAGE_LIMIT: {estimate.num_pages:,} pages exceeds FlashInfer limit of 16,777,216 (2^24)"
        else:
            skip_reason = f"OOM: needs {estimate.total_gb:.2f}GB, have {estimate.available_gb:.2f}GB"
        return False, skip_reason

    return True, None


def _run_single_approach(
    approach,
    approach_name: str,
    ctx: 'BenchmarkContext',
    enable_output_validation: bool = False,
    disable_internal_profiling: bool = False,
) -> Dict[str, Any]:
    """Run a single approach with warmup and timing.

    Args:
        approach: The approach object from APPROACHES registry
        approach_name: Name of the approach for profiling
        ctx: BenchmarkContext with all required data
        enable_output_validation: Whether to capture output for validation
        disable_internal_profiling: Whether to skip RecordFunctionTracer for external profilers

    Returns:
        Dict with keys:
            - 'time_ms': Mean time in ms (or None if failed)
            - 'time_stats': Dict with min/max/mean/median/std (or None)
            - 'output': Output tensor if validation enabled (or None)
            - 'error': Error message if failed (or None)
    """
    result = {
        'time_ms': None,
        'time_stats': None,
        'output': None,
        'error': None,
    }

    try:
        # Optimization: Always use return_output=False during profiling for speed
        # Only get actual output at the end if validation is needed
        run_fn = approach.setup(ctx, return_output=False)

        # Warmup: run configured number of iterations to ensure kernels are compiled
        for _ in range(ctx.num_warmup_iters):
            warmup_output = run_fn()
        torch.cuda.synchronize()

        # Profile with RecordFunctionTracer (unless disabled for external profiling)
        if not disable_internal_profiling:
            # Normal path: use RecordFunctionTracer
            tracer = RecordFunctionTracer()
            with tracer:
                for _ in range(ctx.num_active_iters):
                    with torch.profiler.record_function(approach_name):
                        run_fn()

            # Get timing statistics
            time_stats_dict = tracer.get_operation_time_stats()

            if approach_name in time_stats_dict:
                time_stats = time_stats_dict[approach_name]
                result['time_ms'] = time_stats['mean']
                result['time_stats'] = time_stats
        else:
            # External profiling mode: skip RecordFunctionTracer
            for _ in range(ctx.num_active_iters):
                run_fn()

            # No timing stats available in external profiling mode
            result['time_ms'] = None
            result['time_stats'] = None

        # Capture output for validation (run one more iteration with output enabled)
        if enable_output_validation:
            # Teardown the no-output setup
            if hasattr(approach, 'teardown'):
                try:
                    approach.teardown()
                except:
                    pass

            # Re-setup with output enabled for validation
            run_fn_with_output = approach.setup(ctx, return_output=True)
            last_output = run_fn_with_output()

            if last_output is not None:
                if isinstance(last_output, tuple):
                    output_tensor = last_output[0]
                else:
                    output_tensor = last_output

                if output_tensor.numel() > 0:
                    result['output'] = output_tensor.detach().clone()

    except Exception as e:
        result['error'] = str(e)

    return result


def _cleanup_approach(approach, approach_name: str):
    """Clean up approach-specific resources after execution.

    Args:
        approach: The approach object that may need cleanup
        approach_name: Name of the approach for error reporting
    """
    # Call teardown() if the approach implements it
    if hasattr(approach, 'teardown'):
        try:
            approach.teardown()
        except Exception as e:
            print(f"  Warning: teardown() failed for {approach_name}: {e}")

    # Clean up GPU memory (gc.collect only runs at scenario end for efficiency)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _validate_approach_outputs(
    outputs: Dict[str, torch.Tensor],
    approaches_to_run: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """Validate that all approach outputs match reference.

    Args:
        outputs: Dict mapping approach names to output tensors
        approaches_to_run: Ordered list of approaches (first successful is reference)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dict with keys:
            - 'reference_name': Name of reference approach
            - 'all_match': Boolean indicating if all outputs match
            - 'comparisons': Dict mapping approach name to comparison result
    """
    result = {
        'reference_name': None,
        'all_match': True,
        'comparisons': {},
    }

    reference_output = None

    for approach_name in approaches_to_run:
        if approach_name not in outputs:
            continue

        if reference_output is None:
            result['reference_name'] = approach_name
            reference_output = outputs[approach_name]
            result['comparisons'][approach_name] = {'is_reference': True}
        else:
            current_output = outputs[approach_name]
            comparison = {'is_reference': False}

            if current_output.shape != reference_output.shape:
                comparison['match'] = False
                comparison['error'] = f"Shape mismatch: {current_output.shape} vs {reference_output.shape}"
                result['all_match'] = False
            else:
                matches = torch.allclose(current_output, reference_output, rtol=rtol, atol=atol)
                max_diff = (current_output - reference_output).abs().max().item()
                comparison['match'] = matches
                comparison['max_diff'] = max_diff
                if not matches:
                    result['all_match'] = False

            result['comparisons'][approach_name] = comparison

    return result


def _print_validation_results(validation: Dict[str, Any], num_outputs: int):
    """Print output validation results.

    Args:
        validation: Validation results from _validate_approach_outputs
        num_outputs: Total number of outputs validated
    """
    print(f"\n{'=' * 80}")
    print("OUTPUT VALIDATION:")
    print(f"  Using {validation['reference_name']} as reference")

    for approach_name, comparison in validation['comparisons'].items():
        if comparison.get('is_reference'):
            continue

        if 'error' in comparison:
            print(f"  X {approach_name}: {comparison['error']}")
        elif comparison['match']:
            print(f"  OK {approach_name}: Outputs match (max_diff={comparison['max_diff']:.2e})")
        else:
            print(f"  X {approach_name}: Outputs MISMATCH! (max_diff={comparison['max_diff']:.2e})")

    if validation['all_match'] and num_outputs > 1:
        print(f"\n  OK All {num_outputs} approaches produce matching outputs!")
    print(f"{'=' * 80}")


def _print_scenario_results(
    approaches_to_run: List[str],
    approach_times: Dict[str, Optional[float]],
    approach_time_stats: Dict[str, Optional[Dict[str, float]]],
    page_sizes: Dict[str, int],
    default_page_size: int,
    use_cuda_graphs: bool,
):
    """Print formatted scenario results summary.

    Args:
        approaches_to_run: List of approach names
        approach_times: Dict of approach name to mean time (or None)
        approach_time_stats: Dict of approach name to time stats dict
        page_sizes: Dict of approach name to page size used
        default_page_size: Default page size from config
        use_cuda_graphs: Whether CUDA graphs were enabled
    """
    print(f"\n{'=' * 80}")
    print("RESULTS:")

    for approach_name in approaches_to_run:
        if approach_name in approach_times and approach_times[approach_name] is not None:
            time_ms = approach_times[approach_name]
            ps = page_sizes.get(approach_name, default_page_size)
            stats = approach_time_stats.get(approach_name)
            if stats:
                print(f"  {approach_name:35s}: {time_ms:8.3f} ms (+-{stats['std']:6.3f}, "
                      f"median={stats['median']:8.3f}, page_size={ps})")
            else:
                print(f"  {approach_name:35s}: {time_ms:8.3f} ms (page_size={ps})")
        else:
            print(f"  {approach_name:35s}:      N/A")

    # Find winner
    valid_times = {k: v for k, v in approach_times.items() if v is not None}
    if valid_times:
        winner = min(valid_times, key=valid_times.get)
        print(f"\n  Winner: {winner} ({valid_times[winner]:.3f} ms)")
    else:
        print(f"\n  Winner: N/A (no successful runs)")

    print(f"  CUDA Graphs: {'ENABLED' if use_cuda_graphs else 'DISABLED'}")
    print(f"{'=' * 80}")


# Check for BatchAttention availability
try:
    from flashinfer import BatchAttention
    HAS_BATCH_ATTENTION = True
except ImportError:
    BatchAttention = None
    HAS_BATCH_ATTENTION = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    workspace_size: int
    num_warmup_iters: int
    num_active_iters: int
    use_cuda_graphs: bool = False
    enable_profiling: bool = False
    disable_internal_profiling: bool = False  # Disable RecordFunctionTracer for external profilers
    enable_output_validation: bool = False  # Enable output correctness checking
    results_dir: str = "results"
    plots_dir: str = "plots"
    approach_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Memory management settings
    enable_memory_check: bool = False
    memory_utilization: float = 0.90  # Target GPU memory utilization
    memory_overhead: float = 1.1  # Safety overhead multiplier
    # Validation tolerances
    validation_rtol: float = 1e-3  # Relative tolerance for output validation
    validation_atol: float = 1e-3  # Absolute tolerance for output validation


@dataclass
class VariantConfig:
    """Configuration for a request variant."""
    name: str
    q_tokens: int
    kv_tokens: int
    description: str = ""


@dataclass
class ToleranceResult:
    """Results from a tolerance test."""
    scenario_name: str
    num_decodes: int
    num_prefills: int
    prefill_ratio: float
    approach_times: Dict[str, Optional[float]]
    approach_time_stats: Dict[str, Optional[Dict[str, float]]]  # min, max, mean, median, std
    page_sizes: Dict[str, int]
    used_cuda_graphs: bool
    skipped_reason: Optional[str] = None  # Reason if scenario was skipped (e.g., OOM)


class ToleranceBenchmarkRunner:
    """Runs tolerance benchmarks."""

    def __init__(self, config: BenchmarkConfig, variants: Dict[str, VariantConfig]):
        self.config = config
        self.variants = variants

    def create_batch_data(
        self, q_lengths: List[int], kv_lengths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create tensors and metadata for a batch of requests."""
        num_pages = [(kv + self.config.page_size - 1) // self.config.page_size for kv in kv_lengths]

        # Create Q and KV cache tensors
        q = torch.empty(
            sum(q_lengths), self.config.num_qo_heads, self.config.head_dim,
            dtype=torch.float16, device="cuda"
        )

        kv_cache = torch.randn(
            sum(num_pages), 2, self.config.page_size, self.config.num_kv_heads, self.config.head_dim,
            dtype=torch.float16, device="cuda"
        )

        # Create metadata tensors
        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lengths), 0)),
            dtype=torch.int32, device="cuda"
        )

        # Use shuffled page indices to simulate realistic scattered memory allocation
        # (like vLLM/SGLang where blocks are reused and non-sequential)
        kv_page_indices = torch.randperm(sum(num_pages), dtype=torch.int32, device="cuda")

        kv_page_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(num_pages), 0)),
            dtype=torch.int32, device="cuda"
        )

        kv_last_page_len = torch.tensor(
            [kv % self.config.page_size if kv % self.config.page_size != 0 else self.config.page_size
             for kv in kv_lengths],
            dtype=torch.int32, device="cuda"
        )

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def _create_benchmark_context(
        self, all_q_lengths: List[int], all_kv_lengths: List[int], page_size: int
    ) -> BenchmarkContext:
        """Create complete benchmark context with ALL parameters (superset).

        This method computes ALL possible parameters that any approach might need.
        Each approach can then select what it needs from this complete context.

        Args:
            all_q_lengths: Query sequence lengths for all requests
            all_kv_lengths: KV cache lengths for all requests
            page_size: Page size to use for this benchmark

        Returns:
            BenchmarkContext with all computed parameters
        """
        # 1. Classify and reorder requests (decode first, then prefill)
        (reordered_q_lengths, reordered_kv_lengths, decode_indices,
         prefill_indices, is_mixed_workload) = _classify_and_reorder_requests(
            all_q_lengths, all_kv_lengths
        )

        # 2. Create batch data with reordered requests
        original_page_size = self.config.page_size
        self.config.page_size = page_size
        q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = (
            self.create_batch_data(reordered_q_lengths, reordered_kv_lengths)
        )
        self.config.page_size = original_page_size

        # 3. Create tensor views for separated approaches
        decode_q, prefill_q = _create_q_tensor_views(
            q, reordered_q_lengths, decode_indices, prefill_indices, is_mixed_workload
        )

        # 4. Compute sequence length tensors
        # Optimization: Keep tensors on GPU (no CPU transfer needed)
        seq_lens_kv = get_seq_lens(
            kv_page_indptr, kv_last_page_len, page_size
        )
        seq_lens_q = qo_indptr[1:] - qo_indptr[:-1]
        max_token_per_sequence = seq_lens_q.max().item()
        max_sequence_kv = seq_lens_kv.max().item()

        # 5. Create block tables
        batch_size = len(qo_indptr) - 1
        max_num_blocks_per_seq = math.ceil(max_sequence_kv / page_size)
        block_tables = _create_block_tables(
            batch_size, max_num_blocks_per_seq, kv_page_indptr, kv_page_indices
        )

        # 6. Assemble and return context
        return BenchmarkContext(
            q_lengths=reordered_q_lengths,
            kv_lengths=reordered_kv_lengths,
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
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            num_qo_heads=self.config.num_qo_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            page_size=page_size,
            workspace_size=self.config.workspace_size,
            num_warmup_iters=self.config.num_warmup_iters,
            num_active_iters=self.config.num_active_iters,
        )

    def run_tolerance_test(
        self,
        scenario_name: str,
        variant_set: Dict[str, int],
        approaches_to_run: List[str]
    ) -> ToleranceResult:
        """Run tolerance test for a scenario using specified approaches.

        This is the main orchestrator that:
        1. Builds workload from variant set
        2. Checks memory feasibility
        3. Creates benchmark context
        4. Runs each approach with timing
        5. Validates outputs across approaches
        6. Formats and returns results

        Args:
            scenario_name: Name of the scenario
            variant_set: Dictionary mapping variant names to counts
            approaches_to_run: List of approach names to benchmark

        Returns:
            ToleranceResult with timing and validation data
        """
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'=' * 80}")

        # 1. Build workload from variant set
        all_q_lengths, all_kv_lengths = build_workload_from_variant_set(
            variant_set, self.variants
        )

        num_decodes = sum(1 for q in all_q_lengths if q == 1)
        num_prefills = sum(1 for q in all_q_lengths if q > 1)
        prefill_ratio = num_prefills / len(all_q_lengths) if len(all_q_lengths) > 0 else 0

        print(f"\nWorkload: {num_decodes} decodes + {num_prefills} prefills ({prefill_ratio*100:.1f}% prefill)")
        print(f"Approaches to run: {', '.join(approaches_to_run)}\n")

        # 2. Check memory feasibility
        fits, skip_reason = _check_memory_feasibility(
            all_q_lengths, all_kv_lengths, self.config
        )
        if not fits:
            print(f"SKIPPED: {skip_reason}")
            return ToleranceResult(
                scenario_name=scenario_name,
                num_decodes=num_decodes,
                num_prefills=num_prefills,
                prefill_ratio=prefill_ratio,
                approach_times={a: None for a in approaches_to_run},
                approach_time_stats={a: None for a in approaches_to_run},
                page_sizes={},
                used_cuda_graphs=self.config.use_cuda_graphs,
                skipped_reason=skip_reason,
            )

        # 3. Create shared benchmark context
        page_size = self.config.page_size
        ctx = self._create_benchmark_context(all_q_lengths, all_kv_lengths, page_size)

        # 4. Run each approach
        approach_times = {}
        approach_time_stats = {}
        page_sizes = {}
        approach_outputs = {}

        for i, approach_name in enumerate(approaches_to_run, 1):
            approach = APPROACHES.get(approach_name)
            if approach is None:
                print(f"[{i}/{len(approaches_to_run)}] {approach_name}: SKIPPED (not registered)")
                continue

            print(f"[{i}/{len(approaches_to_run)}] {approach_name}...")

            # Run the approach with timing
            result = _run_single_approach(
                approach, approach_name, ctx,
                enable_output_validation=self.config.enable_output_validation,
                disable_internal_profiling=self.config.disable_internal_profiling
            )

            # Process results
            if result['error']:
                print(f"  {approach_name}: FAILED - {result['error']}")
                approach_times[approach_name] = None
                approach_time_stats[approach_name] = None
            elif result['time_ms'] is not None:
                approach_times[approach_name] = result['time_ms']
                approach_time_stats[approach_name] = result['time_stats']
                page_sizes[approach_name] = page_size

                stats = result['time_stats']
                print(f"  {approach_name}: {result['time_ms']:.3f} ms (+-{stats['std']:.3f}, "
                      f"median={stats['median']:.3f}, page_size={page_size})")

                if result['output'] is not None:
                    approach_outputs[approach_name] = result['output']
            else:
                print(f"  {approach_name}: FAILED - no timing data captured")
                approach_times[approach_name] = None
                approach_time_stats[approach_name] = None

            # Cleanup after each approach
            _cleanup_approach(approach, approach_name)

        # 5. Validate outputs across approaches
        if self.config.enable_output_validation and len(approach_outputs) > 1:
            validation = _validate_approach_outputs(approach_outputs, approaches_to_run)
            _print_validation_results(validation, len(approach_outputs))

        # 6. Print results summary
        _print_scenario_results(
            approaches_to_run, approach_times, approach_time_stats,
            page_sizes, self.config.page_size, self.config.use_cuda_graphs
        )

        # Clean up shared context
        del ctx
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return ToleranceResult(
            scenario_name=scenario_name,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            prefill_ratio=prefill_ratio,
            approach_times=approach_times,
            approach_time_stats=approach_time_stats,
            page_sizes=page_sizes,
            used_cuda_graphs=self.config.use_cuda_graphs,
        )


def load_config(config_path: str, use_cuda_graphs: bool, enable_profiling: bool = False):
    """Load configuration from YAML file with programmatic generation support."""
    from attention_bench.config import generator as config_generator

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Get approaches list (default to FlashInfer approaches if not specified)
    approaches = config_data.get("approaches", [
        "flashinfer_mixed_fa2",
        "flashinfer_mixed_fa3",
        "flashinfer_separated_fa2",
        "flashinfer_separated_fa3",
        "flashinfer_batch_attention",
    ])

    # Load model config
    model_config = config_data["model"]

    # Load memory management settings
    memory_config = config_data.get("memory", {})
    enable_memory_check = memory_config.get("enable_memory_check", False)
    memory_utilization = memory_config.get("memory_utilization", 0.90)
    memory_overhead = memory_config.get("memory_overhead", 1.1)

    config = BenchmarkConfig(
        num_qo_heads=model_config["num_qo_heads"],
        num_kv_heads=model_config["num_kv_heads"],
        head_dim=model_config["head_dim"],
        page_size=model_config.get("page_size", 16),
        workspace_size=model_config.get("workspace_size", 256 * 1024 * 1024),
        num_warmup_iters=config_data["profiling"]["num_warmup_iters"],
        num_active_iters=config_data["profiling"]["num_active_iters"],
        use_cuda_graphs=use_cuda_graphs,
        enable_profiling=enable_profiling,
        results_dir=config_data["output"]["results_dir"],
        plots_dir=config_data["output"]["plots_dir"],
        approach_overrides=config_data.get("approach_overrides", {}),
        enable_memory_check=enable_memory_check,
        memory_utilization=memory_utilization,
        memory_overhead=memory_overhead,
    )

    # Generate variants and scenarios programmatically
    if "scenario_generation" in config_data:
        variants_dict, scenarios_dict = config_generator.generate_scenarios(
            config_data["scenario_generation"]
        )
        print(f"Generated {len(variants_dict)} variants and {len(scenarios_dict)} scenarios")
    else:
        # Fallback to manually defined variants/scenarios if present
        variants_dict = config_data.get("variants", {})
        scenarios_dict = config_data.get("scenarios", {})
        if not variants_dict or not scenarios_dict:
            raise ValueError("Config must contain either 'scenario_generation' or both 'variants' and 'scenarios' sections")

    # Convert variant dicts to VariantConfig objects
    variants = {}
    for variant_name, variant_data in variants_dict.items():
        variants[variant_name] = VariantConfig(
            name=variant_name,
            q_tokens=variant_data["q_tokens"],
            kv_tokens=variant_data["kv_tokens"],
            description=variant_data.get("description", ""),
        )

    return config, variants, scenarios_dict, approaches


def save_results(results: List[ToleranceResult], output_path: str):
    """Save results to JSON file."""
    output_data = []
    for result in results:
        output_data.append({
            "scenario_name": result.scenario_name,
            "num_decodes": result.num_decodes,
            "num_prefills": result.num_prefills,
            "prefill_ratio": result.prefill_ratio,
            "approach_times": result.approach_times,
            "approach_time_stats": result.approach_time_stats,
            "page_sizes": result.page_sizes,
            "used_cuda_graphs": result.used_cuda_graphs,
        })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def _get_grid_layout(num_subplots: int) -> Tuple[int, int]:
    """Calculate grid layout (rows, cols) for given number of subplots."""
    if num_subplots == 1:
        return (1, 1)
    elif num_subplots == 2:
        return (1, 2)
    elif num_subplots <= 4:
        return (2, 2)
    elif num_subplots <= 6:
        return (2, 3)
    elif num_subplots <= 9:
        return (3, 3)
    elif num_subplots <= 12:
        return (3, 4)
    else:
        return (4, 4)


def _create_bar_chart_subplot(ax, result, approaches: List[str],
                               title_fontsize: int = 11,
                               ylabel_fontsize: int = 10,
                               xticklabel_fontsize: int = 8,
                               value_label_fontsize: int = 7,
                               value_label_format: str = '.1f',
                               include_xlabel: bool = False,
                               xlabel_fontsize: int = 12,
                               compact_title: bool = True):
    """Create a bar chart for a single scenario on the given axes.

    Args:
        ax: Matplotlib axes to plot on
        result: ToleranceResult containing scenario data and timing results
        approaches: List of approach names to plot
        title_fontsize: Font size for the title
        ylabel_fontsize: Font size for the y-axis label
        xticklabel_fontsize: Font size for x-axis tick labels
        value_label_fontsize: Font size for value labels on bars
        value_label_format: Format string for value labels (e.g., '.1f' or '.2f')
        include_xlabel: Whether to include x-axis label
        xlabel_fontsize: Font size for x-axis label (if included)
        compact_title: Whether to use compact title format (e.g., '10D + 5P' vs '10 decodes + 5 prefills')
    """
    # Get times for each approach, filtering out None values
    approach_names = []
    times = []

    for approach_name in approaches:
        if approach_name in result.approach_times and result.approach_times[approach_name] is not None:
            approach_names.append(approach_name)
            times.append(result.approach_times[approach_name])

    # Create bar chart
    x = np.arange(len(approach_names))
    bars = ax.bar(x, times, color='steelblue', alpha=0.8)

    # Customize plot
    ax.set_ylabel('Time (ms)', fontsize=ylabel_fontsize)

    # Format title based on compact mode
    if compact_title:
        title = f'{result.scenario_name}\n({result.num_decodes}D + {result.num_prefills}P)'
    else:
        title = f'{result.scenario_name}\n({result.num_decodes} decodes + {result.num_prefills} prefills)'
    ax.set_title(title, fontsize=title_fontsize)

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(approach_names, rotation=45, ha='right', fontsize=xticklabel_fontsize)

    if include_xlabel:
        ax.set_xlabel('Approach', fontsize=xlabel_fontsize)

    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:{value_label_format}}',
               ha='center', va='bottom', fontsize=value_label_fontsize)


def _create_single_plot(result, approaches: List[str], plots_dir: str, config_name: str):
    """Create a single plot for one scenario.

    Args:
        result: ToleranceResult for the scenario
        approaches: List of approach names to plot
        plots_dir: Directory to save the plot
        config_name: Configuration name for the output filename

    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    _create_bar_chart_subplot(
        ax, result, approaches,
        title_fontsize=14,
        ylabel_fontsize=12,
        xticklabel_fontsize=8,
        value_label_fontsize=9,
        value_label_format='.2f',
        include_xlabel=True,
        xlabel_fontsize=12,
        compact_title=False
    )

    plt.tight_layout()
    output_path = os.path.join(plots_dir, f"comparison_{config_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _create_subplot_grid(results: List, approaches: List[str], plots_dir: str, config_name: str):
    """Create a grid of subplots for multiple scenarios.

    Args:
        results: List of ToleranceResults for the scenarios
        approaches: List of approach names to plot
        plots_dir: Directory to save the plot
        config_name: Configuration name for the output filename

    Returns:
        Path to the saved plot
    """
    rows, cols = _get_grid_layout(len(results))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

    if len(results) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Create bar chart for each scenario
    for idx, result in enumerate(results):
        _create_bar_chart_subplot(axes[idx], result, approaches)

    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = os.path.join(plots_dir, f"comparison_{config_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _create_paginated_plots(results: List, approaches: List[str], plots_dir: str,
                            config_name: str, max_scenarios_per_page: int) -> List[str]:
    """Create paginated plots for many scenarios.

    Args:
        results: List of ToleranceResults for all scenarios
        approaches: List of approach names to plot
        plots_dir: Directory to save the plots
        config_name: Configuration name for the output filenames
        max_scenarios_per_page: Maximum number of scenarios per page

    Returns:
        List of paths to the saved plots
    """
    num_pages = math.ceil(len(results) / max_scenarios_per_page)
    output_paths = []

    for page_num in range(num_pages):
        start_idx = page_num * max_scenarios_per_page
        end_idx = min(start_idx + max_scenarios_per_page, len(results))
        page_results = results[start_idx:end_idx]

        print(f"    Creating page {page_num + 1}/{num_pages} ({len(page_results)} scenarios)...")

        # Create subplot grid for this page
        rows, cols = _get_grid_layout(len(page_results))
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

        if len(page_results) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Create bar chart for each scenario
        for idx, result in enumerate(page_results):
            _create_bar_chart_subplot(axes[idx], result, approaches)

        # Hide extra subplots
        for idx in range(len(page_results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save with page number
        output_path = os.path.join(plots_dir, f"comparison_{config_name}_page{page_num + 1}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        output_paths.append(output_path)
        print(f"      Page saved to: {output_path}")

    return output_paths


def create_plots(results: List[ToleranceResult], plots_dir: str, config_name: str, approaches: List[str]):
    """Create comparison plots with pagination support.

    Automatically selects the appropriate layout based on the number of scenarios:
    - Single scenario: Creates a large standalone plot
    - Multiple scenarios (<=12): Creates a grid of subplots on a single page
    - Many scenarios (>12): Creates multiple pages with up to 12 subplots each

    Args:
        results: List of ToleranceResults containing benchmark data
        plots_dir: Directory to save the plots
        config_name: Configuration name for output filenames
        approaches: List of approach names to include in plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nCreating plots for {len(results)} scenarios with {len(approaches)} approaches...")

    MAX_SCENARIOS_PER_PAGE = 12  # 3x4 grid maximum

    if len(results) == 1:
        print("  Using single plot layout")
        output_path = _create_single_plot(results[0], approaches, plots_dir, config_name)
        print(f"Plot saved to: {output_path}")

    elif len(results) <= MAX_SCENARIOS_PER_PAGE:
        print(f"  Using subplot layout (single page)")
        output_path = _create_subplot_grid(results, approaches, plots_dir, config_name)
        print(f"Plot saved to: {output_path}")

    else:
        print(f"  Using pagination: {math.ceil(len(results) / MAX_SCENARIOS_PER_PAGE)} pages with up to {MAX_SCENARIOS_PER_PAGE} subplots each")
        _create_paginated_plots(results, approaches, plots_dir, config_name, MAX_SCENARIOS_PER_PAGE)


def main():
    parser = argparse.ArgumentParser(description="FlashInfer Tolerance Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--use-cuda-graphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling")
    parser.add_argument("--approaches", type=str, help="Comma-separated list of approaches to run")
    parser.add_argument("--gpu", type=int, help="GPU device to use (for parallel execution)")
    parser.add_argument("--scenario-range", type=str, help="Range of scenarios to run, e.g., '0-24' (for parallel execution)")
    args = parser.parse_args()

    # Show GPU info if specified (CUDA_VISIBLE_DEVICES set by parent process)
    if args.gpu is not None:
        # CUDA_VISIBLE_DEVICES is set by run_parallel.py before spawning this process
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={cuda_visible})")

    # Load config
    config, variants, scenarios, approaches = load_config(
        args.config, args.use_cuda_graphs, args.enable_profiling
    )

    # Override approaches if specified
    if args.approaches:
        approaches = [a.strip() for a in args.approaches.split(",")]

    # Filter scenarios if range specified (for parallel execution)
    if args.scenario_range:
        start, end = map(int, args.scenario_range.split('-'))
        scenario_items = list(scenarios.items())
        scenarios = dict(scenario_items[start:end+1])
        print(f"Running scenario range: {start}-{end} ({len(scenarios)} scenarios)")

    # Print header
    print("=" * 80)
    print("FlashInfer Prefill Tolerance Benchmark (v2 - Refactored)")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"CUDA Graphs: {'ENABLED' if args.use_cuda_graphs else 'DISABLED'}")
    print(f"Profiling:   {'ENABLED' if args.enable_profiling else 'DISABLED'}")
    print(f"Loaded {len(variants)} variants")
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"\nAvailable approaches: {', '.join(APPROACHES.list_available())}")

    # Print environment info
    print("\n" + "=" * 80)
    print("Environment")
    print("=" * 80)
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # Create runner
    runner = ToleranceBenchmarkRunner(config, variants)

    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Tolerance Tests")
    print("=" * 80)

    results = []
    for scenario_name, scenario_config in scenarios.items():
        variant_set = scenario_config["variants_to_run"]
        result = runner.run_tolerance_test(scenario_name, variant_set, approaches)
        results.append(result)

        # Clean up memory between scenarios to prevent OOM
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    os.makedirs(config.results_dir, exist_ok=True)
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Add GPU suffix if running in parallel mode
    if args.gpu is not None:
        results_filename = f"{config_name}_gpu{args.gpu}.json"
    else:
        results_filename = f"{config_name}.json"

    results_path = os.path.join(config.results_dir, results_filename)
    save_results(results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Create plots (skip if running in parallel mode - plots will be created from merged results)
    if args.gpu is None:
        print("\n" + "=" * 80)
        print("Creating Plots")
        print("=" * 80)
        create_plots(results, config.plots_dir, config_name, approaches)
    else:
        print("\nSkipping plot creation (parallel mode - plots will be created from merged results)")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total tests run: {len(results)}")

    # Count wins per approach
    wins = {approach: 0 for approach in approaches}
    for result in results:
        valid_times = {k: v for k, v in result.approach_times.items() if v is not None}
        if valid_times:
            winner = min(valid_times, key=valid_times.get)
            wins[winner] = wins.get(winner, 0) + 1

    print("\nWinner Count by Approach:")
    for approach in approaches:
        count = wins.get(approach, 0)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"  {approach:35s}: {count:3d}/{len(results)} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
