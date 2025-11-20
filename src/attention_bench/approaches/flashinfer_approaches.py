#!/usr/bin/env python3
"""FlashInfer attention approach implementations."""

from typing import Callable
import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)

from .base import APPROACHES, BenchmarkContext

# Check for BatchAttention availability
try:
    from flashinfer import BatchAttention
    HAS_BATCH_ATTENTION = True
except ImportError:
    BatchAttention = None
    HAS_BATCH_ATTENTION = False


@APPROACHES.register
class FlashInferMixedFA2:
    """FlashInfer: Prefill wrapper (FA2 backend) for ALL requests."""

    name = "flashinfer_mixed_fa2"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup FA2 wrapper and return benchmark callable."""
        # Create wrapper
        workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")

        # Plan - FA2 only needs basic parameters (ignores seq_lens, block_tables, etc.)
        wrapper.plan(
            ctx.qo_indptr,
            ctx.kv_page_indptr,
            ctx.kv_page_indices,
            ctx.kv_last_page_len,
            ctx.num_qo_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            ctx.page_size,
            causal=True,
        )

        # Return callable
        def run_iteration():
            return wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration


@APPROACHES.register
class FlashInferMixedFA3:
    """FlashInfer: Prefill wrapper (FA3 backend) for ALL requests."""

    name = "flashinfer_mixed_fa3"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup FA3 wrapper and return benchmark callable."""
        # Create wrapper
        workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3")

        # Plan - FA3 benefits from seq_lens for paging
        wrapper.plan(
            ctx.qo_indptr,
            ctx.kv_page_indptr,
            ctx.kv_page_indices,
            ctx.kv_last_page_len,
            ctx.num_qo_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            ctx.page_size,
            causal=True,
            seq_lens=ctx.seq_lens_kv,  # Use from superset
        )

        # Return callable
        def run_iteration():
            return wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration


@APPROACHES.register
class FlashInferSeparatedFA2:
    """FlashInfer: Separate decode + prefill wrappers with FA2 backend."""

    name = "flashinfer_separated_fa2"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup separated decode + prefill wrappers and return benchmark callable."""
        # Separate indices for decode (q=1) and prefill (q>1)
        decode_indices = [i for i, q_len in enumerate(ctx.q_lengths) if q_len == 1]
        prefill_indices = [i for i, q_len in enumerate(ctx.q_lengths) if q_len > 1]

        decode_wrapper = None
        prefill_wrapper = None
        decode_output = None
        prefill_output = None

        # Create decode wrapper if we have decodes
        if decode_indices:
            # Extract decode data from shared context (instead of creating new tensors)
            decode_q, decode_kv, _, decode_kv_indptr, decode_kv_indices, decode_last_page = (
                self._extract_subset_from_context(decode_indices, ctx)
            )

            # Create decode wrapper
            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
            decode_wrapper.plan(
                decode_kv_indptr,
                decode_kv_indices,
                decode_last_page,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                ctx.page_size,
            )

        # Create prefill wrapper if we have prefills
        if prefill_indices:
            # Extract prefill data from shared context (instead of creating new tensors)
            prefill_q, prefill_kv, prefill_qo_indptr, prefill_kv_indptr, prefill_kv_indices, prefill_last_page = (
                self._extract_subset_from_context(prefill_indices, ctx)
            )

            # Create prefill wrapper
            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
            prefill_wrapper.plan(
                prefill_qo_indptr,
                prefill_kv_indptr,
                prefill_kv_indices,
                prefill_last_page,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                ctx.page_size,
                causal=True,
            )

        # Return callable that runs both wrappers
        def run_iteration():
            if decode_wrapper is not None:
                decode_wrapper.run(decode_q, decode_kv)
            if prefill_wrapper is not None:
                prefill_wrapper.run(prefill_q, prefill_kv)
            # Return dummy output (timing is what matters)
            return torch.empty(0)

        return run_iteration

    def _extract_subset_from_context(self, indices, ctx):
        """Extract a subset of requests from the shared context.

        Instead of creating new random tensors, this extracts the relevant portions
        from ctx.q and ctx.kv_cache to ensure all approaches benchmark on the same data.

        Args:
            indices: List of request indices to extract (e.g., decode_indices or prefill_indices)
            ctx: Shared BenchmarkContext

        Returns:
            Tuple of (q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len)
        """
        # Extract Q tokens for selected requests
        q_slices = []
        for idx in indices:
            start = ctx.qo_indptr[idx].item()
            end = ctx.qo_indptr[idx + 1].item()
            q_slices.append(ctx.q[start:end])
        q = torch.cat(q_slices, dim=0) if q_slices else torch.empty(0, ctx.num_qo_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        # Extract KV cache pages for selected requests
        kv_page_list = []
        new_kv_page_indices = []
        new_page_offset = 0

        for idx in indices:
            # Get the range of pages for this request
            page_start = ctx.kv_page_indptr[idx].item()
            page_end = ctx.kv_page_indptr[idx + 1].item()

            # Get the physical page indices for this request
            logical_pages = ctx.kv_page_indices[page_start:page_end]

            # Extract the actual pages from kv_cache
            for page_idx in logical_pages:
                kv_page_list.append(ctx.kv_cache[page_idx.item()])
                # Map to new contiguous indices
                new_kv_page_indices.append(new_page_offset)
                new_page_offset += 1

        kv_cache = torch.stack(kv_page_list, dim=0) if kv_page_list else torch.empty(0, 2, ctx.page_size, ctx.num_kv_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        # Build new metadata for the subset
        q_lengths = [ctx.q_lengths[i] for i in indices]
        qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lengths), 0)), dtype=torch.int32, device="cuda")

        # Build kv_page_indptr for subset
        num_pages_per_request = []
        for idx in indices:
            page_start = ctx.kv_page_indptr[idx].item()
            page_end = ctx.kv_page_indptr[idx + 1].item()
            num_pages_per_request.append(page_end - page_start)
        kv_page_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_pages_per_request), 0)), dtype=torch.int32, device="cuda")

        kv_page_indices = torch.tensor(new_kv_page_indices, dtype=torch.int32, device="cuda") if new_kv_page_indices else torch.empty(0, dtype=torch.int32, device="cuda")

        # Extract last page lengths for selected requests
        kv_last_page_len = ctx.kv_last_page_len[indices] if len(indices) > 0 else torch.empty(0, dtype=torch.int32, device="cuda")

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len


@APPROACHES.register
class FlashInferSeparatedFA3:
    """FlashInfer: Separate decode + prefill wrappers with FA3 backend."""

    name = "flashinfer_separated_fa3"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup separated decode + prefill wrappers and return benchmark callable."""
        # Separate indices for decode (q=1) and prefill (q>1)
        decode_indices = [i for i, q_len in enumerate(ctx.q_lengths) if q_len == 1]
        prefill_indices = [i for i, q_len in enumerate(ctx.q_lengths) if q_len > 1]

        decode_wrapper = None
        prefill_wrapper = None

        # Create decode wrapper if we have decodes
        if decode_indices:
            # Extract decode data from shared context (instead of creating new tensors)
            decode_q, decode_kv, _, decode_kv_indptr, decode_kv_indices, decode_last_page = (
                self._extract_subset_from_context(decode_indices, ctx)
            )

            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3")
            decode_wrapper.plan(
                decode_kv_indptr,
                decode_kv_indices,
                decode_last_page,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                ctx.page_size,
            )

        # Create prefill wrapper if we have prefills
        if prefill_indices:
            # Extract prefill data from shared context (instead of creating new tensors)
            prefill_q, prefill_kv, prefill_qo_indptr, prefill_kv_indptr, prefill_kv_indices, prefill_last_page = (
                self._extract_subset_from_context(prefill_indices, ctx)
            )

            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3")
            prefill_wrapper.plan(
                prefill_qo_indptr,
                prefill_kv_indptr,
                prefill_kv_indices,
                prefill_last_page,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                ctx.page_size,
                causal=True,
            )

        # Return callable that runs both wrappers
        def run_iteration():
            if decode_wrapper is not None:
                decode_wrapper.run(decode_q, decode_kv)
            if prefill_wrapper is not None:
                prefill_wrapper.run(prefill_q, prefill_kv)
            return torch.empty(0)

        return run_iteration

    def _extract_subset_from_context(self, indices, ctx):
        """Extract a subset of requests from the shared context.

        Instead of creating new random tensors, this extracts the relevant portions
        from ctx.q and ctx.kv_cache to ensure all approaches benchmark on the same data.

        Args:
            indices: List of request indices to extract (e.g., decode_indices or prefill_indices)
            ctx: Shared BenchmarkContext

        Returns:
            Tuple of (q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len)
        """
        # Extract Q tokens for selected requests
        q_slices = []
        for idx in indices:
            start = ctx.qo_indptr[idx].item()
            end = ctx.qo_indptr[idx + 1].item()
            q_slices.append(ctx.q[start:end])
        q = torch.cat(q_slices, dim=0) if q_slices else torch.empty(0, ctx.num_qo_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        # Extract KV cache pages for selected requests
        kv_page_list = []
        new_kv_page_indices = []
        new_page_offset = 0

        for idx in indices:
            # Get the range of pages for this request
            page_start = ctx.kv_page_indptr[idx].item()
            page_end = ctx.kv_page_indptr[idx + 1].item()

            # Get the physical page indices for this request
            logical_pages = ctx.kv_page_indices[page_start:page_end]

            # Extract the actual pages from kv_cache
            for page_idx in logical_pages:
                kv_page_list.append(ctx.kv_cache[page_idx.item()])
                # Map to new contiguous indices
                new_kv_page_indices.append(new_page_offset)
                new_page_offset += 1

        kv_cache = torch.stack(kv_page_list, dim=0) if kv_page_list else torch.empty(0, 2, ctx.page_size, ctx.num_kv_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        # Build new metadata for the subset
        q_lengths = [ctx.q_lengths[i] for i in indices]
        qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lengths), 0)), dtype=torch.int32, device="cuda")

        # Build kv_page_indptr for subset
        num_pages_per_request = []
        for idx in indices:
            page_start = ctx.kv_page_indptr[idx].item()
            page_end = ctx.kv_page_indptr[idx + 1].item()
            num_pages_per_request.append(page_end - page_start)
        kv_page_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_pages_per_request), 0)), dtype=torch.int32, device="cuda")

        kv_page_indices = torch.tensor(new_kv_page_indices, dtype=torch.int32, device="cuda") if new_kv_page_indices else torch.empty(0, dtype=torch.int32, device="cuda")

        # Extract last page lengths for selected requests
        kv_last_page_len = ctx.kv_last_page_len[indices] if len(indices) > 0 else torch.empty(0, dtype=torch.int32, device="cuda")

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len


@APPROACHES.register
class FlashInferBatchAttention:
    """FlashInfer: BatchAttention unified wrapper (if available)."""

    name = "flashinfer_batch_attention"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup BatchAttention wrapper and return benchmark callable."""
        if not HAS_BATCH_ATTENTION:
            raise RuntimeError("BatchAttention not available in current FlashInfer version")

        # Calculate sequence lengths
        seq_lens = torch.tensor(ctx.kv_lengths, dtype=torch.int32, device="cuda")

        # Create BatchAttention wrapper
        batch_wrapper = BatchAttention(kv_layout="NHD")

        # Plan (persistent plan)
        batch_wrapper.plan(
            ctx.qo_indptr,
            ctx.kv_page_indptr,
            ctx.kv_page_indices,
            seq_lens,
            ctx.num_qo_heads,
            ctx.num_kv_heads,
            ctx.head_dim,  # head_dim_qk
            ctx.head_dim,  # head_dim_vo
            ctx.page_size,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

        # Return callable
        def run_iteration():
            return batch_wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration
