#!/usr/bin/env python3
"""Memory estimation utilities for benchmark scenarios.

Pre-calculates memory requirements to filter out scenarios that would OOM.
Uses conservative settings (80% utilization, 1.4x overhead) to prevent OOMs
and account for separated approach memory overhead and FlashInfer internal allocations.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple

# FlashInfer internal page limit (empirically discovered)
# Scenarios exceeding this limit fail with "CUDA illegal memory access"
# This is a 24-bit integer constraint (2^24 = 16,777,216 pages)
FLASHINFER_MAX_PAGES = 16_777_216  # 2^24 pages


@dataclass
class MemoryEstimate:
    """Memory estimation result for a scenario."""
    kv_cache_gb: float
    q_tensor_gb: float
    output_tensor_gb: float
    workspace_gb: float
    total_gb: float
    available_gb: float
    num_pages: int
    exceeds_page_limit: bool
    fits: bool

    def __str__(self):
        if self.exceeds_page_limit:
            status = "PAGE_LIMIT_EXCEEDED"
            return (f"{status}: {self.num_pages:,} pages exceeds FlashInfer limit of {FLASHINFER_MAX_PAGES:,} pages (2^24)")
        elif not self.fits:
            status = "OOM"
            return (f"{status}: needs {self.total_gb:.2f}GB "
                    f"(KV:{self.kv_cache_gb:.2f}, Q:{self.q_tensor_gb:.2f}, Out:{self.output_tensor_gb:.2f}, WS:{self.workspace_gb:.2f}), "
                    f"available {self.available_gb:.2f}GB")
        else:
            status = "FITS"
            return (f"{status}: needs {self.total_gb:.2f}GB, {self.num_pages:,} pages "
                    f"(KV:{self.kv_cache_gb:.2f}, Q:{self.q_tensor_gb:.2f}, Out:{self.output_tensor_gb:.2f}, WS:{self.workspace_gb:.2f}), "
                    f"available {self.available_gb:.2f}GB")


def estimate_scenario_memory(
    q_lengths: List[int],
    kv_lengths: List[int],
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    page_size: int,
    workspace_size: int,
    dtype_bytes: int = 2,  # FP16
    utilization: float = 0.80,  # Conservative: use up to 80% of GPU
    overhead: float = 1.4,  # Conservative: 40% buffer for approach allocations
    device: int = 0
) -> MemoryEstimate:
    """Estimate memory requirement for a benchmark scenario on a single GPU.

    IMPORTANT: For Tensor Parallelism (TP), pass the per-GPU head counts:
      - num_kv_heads = total_kv_heads / tp_degree
      - num_qo_heads = total_qo_heads / tp_degree
    Each GPU only stores KV cache and activations for its portion of heads.

    Example for Llama-8B (32 qo_heads, 8 kv_heads) with TP=2:
      - Pass num_qo_heads=16, num_kv_heads=4 (per-GPU counts)
      - Memory will be ~2x smaller than TP=1

    Args:
        q_lengths: Query sequence lengths for all requests
        kv_lengths: KV cache lengths for all requests
        num_kv_heads: Number of KV heads (per-GPU for TP > 1)
        num_qo_heads: Number of query/output heads (per-GPU for TP > 1)
        head_dim: Head dimension
        page_size: Page size for paged attention
        workspace_size: Workspace buffer size in bytes (per-GPU, doesn't scale with TP)
        dtype_bytes: Bytes per element (2 for FP16)
        utilization: Fraction of GPU memory to use (0.80 = 80%)
        overhead: Multiplier for approach-specific allocations (1.4 = 40% extra)
        device: CUDA device index

    Returns:
        MemoryEstimate with fits=True if scenario can run without OOM
    """
    # Calculate number of pages needed
    num_pages = sum((kv + page_size - 1) // page_size for kv in kv_lengths)

    # Check FlashInfer page limit (2^24 = 16,777,216 pages)
    # Empirically discovered: scenarios exceeding this fail with CUDA illegal memory access
    exceeds_page_limit = num_pages > FLASHINFER_MAX_PAGES

    # KV cache memory: num_pages * 2 (K and V) * page_size * num_kv_heads * head_dim * dtype
    kv_cache_bytes = num_pages * 2 * page_size * num_kv_heads * head_dim * dtype_bytes

    # Q tensor memory: total_q_tokens * num_qo_heads * head_dim * dtype
    total_q_tokens = sum(q_lengths)
    q_tensor_bytes = total_q_tokens * num_qo_heads * head_dim * dtype_bytes

    # Output tensor memory: same shape as Q tensor
    output_tensor_bytes = total_q_tokens * num_qo_heads * head_dim * dtype_bytes

    # Total with overhead for workspaces, temporary buffers, etc.
    total_bytes = (kv_cache_bytes + q_tensor_bytes + output_tensor_bytes + workspace_size) * overhead

    # Get available GPU memory
    props = torch.cuda.get_device_properties(device)
    available_bytes = props.total_memory * utilization

    # Scenario fits if it passes BOTH memory check AND page limit check
    memory_fits = total_bytes < available_bytes
    fits = memory_fits and not exceeds_page_limit

    return MemoryEstimate(
        kv_cache_gb=kv_cache_bytes / 1e9,
        q_tensor_gb=q_tensor_bytes / 1e9,
        output_tensor_gb=output_tensor_bytes / 1e9,
        workspace_gb=workspace_size / 1e9,
        total_gb=total_bytes / 1e9,
        available_gb=available_bytes / 1e9,
        num_pages=num_pages,
        exceeds_page_limit=exceeds_page_limit,
        fits=fits
    )


def get_gpu_memory_info(device: int = 0) -> Tuple[float, float, float]:
    """Get current GPU memory usage.

    Returns:
        (allocated_gb, reserved_gb, total_gb)
    """
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    return (allocated / 1e9, reserved / 1e9, total / 1e9)


def check_memory_after_cleanup(device: int = 0, threshold_gb: float = 0.1) -> bool:
    """Check if memory was properly cleaned up.

    Args:
        device: CUDA device index
        threshold_gb: Maximum allowed allocated memory in GB

    Returns:
        True if memory is below threshold, False if potential leak
    """
    allocated_gb, _, _ = get_gpu_memory_info(device)

    if allocated_gb > threshold_gb:
        print(f"Warning: {allocated_gb:.2f} GB still allocated after cleanup (threshold: {threshold_gb} GB)")
        return False
    return True


if __name__ == "__main__":
    # Test memory estimation
    import argparse

    parser = argparse.ArgumentParser(description="Test memory estimation")
    parser.add_argument("--q-lengths", type=str, default="1,1,1,1,512",
                       help="Comma-separated Q lengths")
    parser.add_argument("--kv-lengths", type=str, default="4096,4096,4096,4096,8192",
                       help="Comma-separated KV lengths")
    args = parser.parse_args()

    q_lengths = [int(x) for x in args.q_lengths.split(",")]
    kv_lengths = [int(x) for x in args.kv_lengths.split(",")]

    # Default config values (typical for LLaMA-like model)
    estimate = estimate_scenario_memory(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_kv_heads=8,
        num_qo_heads=32,
        head_dim=128,
        page_size=16,
        workspace_size=256 * 1024 * 1024,  # 256MB
    )

    print(f"Scenario: {len(q_lengths)} requests")
    print(f"  Q lengths: {q_lengths}")
    print(f"  KV lengths: {kv_lengths}")
    print(f"\nMemory estimate: {estimate}")
