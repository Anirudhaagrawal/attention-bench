"""Test utilities for attention kernel testing."""

import math
from typing import List

import numpy as np
import torch

from approaches import BenchmarkContext


def create_test_context(
    q_lengths: List[int],
    kv_lengths: List[int],
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    seed: int = 42,
) -> BenchmarkContext:
    """Create a test BenchmarkContext with deterministic random data.

    Args:
        q_lengths: Query sequence lengths for each request
        kv_lengths: KV cache lengths for each request
        num_qo_heads: Number of query/output heads
        num_kv_heads: Number of key/value heads
        head_dim: Head dimension
        page_size: Page size for paged KV cache
        seed: Random seed for reproducibility

    Returns:
        BenchmarkContext with all required tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = len(q_lengths)
    assert len(kv_lengths) == batch_size

    # Create Q tensor
    total_q_tokens = sum(q_lengths)
    q = torch.randn(
        total_q_tokens, num_qo_heads, head_dim,
        dtype=torch.float16, device="cuda"
    )

    # Create KV cache
    num_pages = [(kv + page_size - 1) // page_size for kv in kv_lengths]
    total_pages = sum(num_pages)

    kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda"
    )

    # Create metadata tensors
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lengths), 0)),
        dtype=torch.int32, device="cuda"
    )

    # Use sequential page indices for determinism
    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")

    kv_page_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(num_pages), 0)),
        dtype=torch.int32, device="cuda"
    )

    kv_last_page_len = torch.tensor(
        [kv % page_size if kv % page_size != 0 else page_size for kv in kv_lengths],
        dtype=torch.int32, device="cuda"
    )

    # Compute optional tensors
    seq_lens_q = qo_indptr[1:] - qo_indptr[:-1]
    seq_lens_kv = torch.tensor(kv_lengths, dtype=torch.int32, device="cuda")

    max_token_per_sequence = max(q_lengths)
    max_sequence_kv = max(kv_lengths)
    max_num_blocks_per_seq = math.ceil(max_sequence_kv / page_size)

    # Create block tables
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32, device="cuda"
    )

    for i in range(batch_size):
        start_idx = kv_page_indptr[i]
        end_idx = kv_page_indptr[i + 1]
        num_blocks = end_idx - start_idx
        block_tables[i, :num_blocks] = kv_page_indices[start_idx:end_idx]

    # Split into decode and prefill requests
    decode_indices = [i for i, q_len in enumerate(q_lengths) if q_len == 1]
    prefill_indices = [i for i, q_len in enumerate(q_lengths) if q_len > 1]

    # Create views into q tensor for decode and prefill
    # Note: For tests, we don't need to reorder, just create the views
    if decode_indices:
        decode_tokens = [qo_indptr[i].item() for i in decode_indices]
        decode_q = q[decode_tokens] if len(decode_tokens) == 1 else torch.stack([q[t] for t in decode_tokens])
    else:
        decode_q = None

    if prefill_indices:
        prefill_slices = []
        for i in prefill_indices:
            start = qo_indptr[i].item()
            end = qo_indptr[i + 1].item()
            prefill_slices.append(q[start:end])
        prefill_q = torch.cat(prefill_slices, dim=0) if prefill_slices else None
    else:
        prefill_q = None

    return BenchmarkContext(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
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
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        workspace_size=256 * 1024 * 1024,
        num_warmup_iters=1,
        num_active_iters=1,
    )


def assert_outputs_close(
    output1: torch.Tensor,
    output2: torch.Tensor,
    approach1_name: str,
    approach2_name: str,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> None:
    """Assert two outputs are numerically close with detailed error reporting.

    Args:
        output1: First output tensor
        output2: Second output tensor
        approach1_name: Name of first approach
        approach2_name: Name of second approach
        rtol: Relative tolerance (default 1%)
        atol: Absolute tolerance (default 1e-3 = 0.001)

    Raises:
        AssertionError: If outputs differ beyond tolerances
    """
    # Check shapes match
    assert output1.shape == output2.shape, (
        f"Shape mismatch: {approach1_name}={output1.shape} vs "
        f"{approach2_name}={output2.shape}"
    )

    # Compute error statistics
    abs_diff = torch.abs(output1 - output2)
    max_abs_error = abs_diff.max().item()

    # Avoid division by zero in relative error
    rel_diff = abs_diff / (torch.abs(output2) + 1e-8)
    max_rel_error = rel_diff.max().item()

    mean_abs_error = abs_diff.mean().item()

    # Check if close
    is_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)

    if not is_close:
        # Find worst mismatch location
        worst_idx = torch.argmax(abs_diff.flatten())
        worst_idx_3d = np.unravel_index(worst_idx.cpu().numpy(), output1.shape)

        raise AssertionError(
            f"\n{'='*60}\n"
            f"Output mismatch: {approach1_name} vs {approach2_name}\n"
            f"{'='*60}\n"
            f"Max absolute error: {max_abs_error:.6e} (threshold: {atol:.6e})\n"
            f"Max relative error: {max_rel_error:.6e} (threshold: {rtol:.6e})\n"
            f"Mean absolute error: {mean_abs_error:.6e}\n"
            f"\nWorst mismatch at index {worst_idx_3d}:\n"
            f"  {approach1_name}: {output1.flatten()[worst_idx].item():.6f}\n"
            f"  {approach2_name}: {output2.flatten()[worst_idx].item():.6f}\n"
            f"  Difference: {abs_diff.flatten()[worst_idx].item():.6e}\n"
            f"{'='*60}"
        )

    # Print success message with error stats
    print(
        f"✓ {approach1_name} vs {approach2_name}: "
        f"max_err={max_abs_error:.2e}, max_rel_err={max_rel_error:.2e}"
    )
