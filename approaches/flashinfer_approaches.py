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
            # Filter decode data from context
            decode_q_lengths = [ctx.q_lengths[i] for i in decode_indices]
            decode_kv_lengths = [ctx.kv_lengths[i] for i in decode_indices]

            # Need to create separate batch data for decode
            # This is inefficient but matches the separated approach pattern
            from run_benchmark import ToleranceBenchmarkRunner
            # Create a minimal runner just for batch data creation
            temp_config = type('Config', (), {
                'page_size': ctx.page_size,
                'num_qo_heads': ctx.num_qo_heads,
                'num_kv_heads': ctx.num_kv_heads,
                'head_dim': ctx.head_dim,
            })()
            temp_runner = type('Runner', (), {'config': temp_config})()
            temp_runner.create_batch_data = lambda q, kv: self._create_decode_batch(q, kv, ctx)

            decode_q, decode_kv, _, decode_kv_indptr, decode_kv_indices, decode_last_page = (
                self._create_decode_batch(decode_q_lengths, decode_kv_lengths, ctx)
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
            # Filter prefill data from context
            prefill_q_lengths = [ctx.q_lengths[i] for i in prefill_indices]
            prefill_kv_lengths = [ctx.kv_lengths[i] for i in prefill_indices]

            prefill_q, prefill_kv, prefill_qo_indptr, prefill_kv_indptr, prefill_kv_indices, prefill_last_page = (
                self._create_prefill_batch(prefill_q_lengths, prefill_kv_lengths, ctx)
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

    def _create_decode_batch(self, q_lengths, kv_lengths, ctx):
        """Create batch data for decode requests."""
        num_pages = [(kv + ctx.page_size - 1) // ctx.page_size for kv in kv_lengths]

        q = torch.randn(sum(q_lengths), ctx.num_qo_heads, ctx.head_dim, dtype=torch.float16, device="cuda")
        # Correct KV cache layout: (num_pages, 2, page_size, num_kv_heads, head_dim)
        kv_cache = torch.randn(sum(num_pages), 2, ctx.page_size, ctx.num_kv_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lengths), 0)), dtype=torch.int32, device="cuda")
        # Use shuffled indices for realistic scattered memory
        kv_page_indices = torch.randperm(sum(num_pages), dtype=torch.int32, device="cuda")
        kv_page_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_pages), 0)), dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [kv % ctx.page_size if kv % ctx.page_size != 0 else ctx.page_size for kv in kv_lengths],
            dtype=torch.int32, device="cuda"
        )

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def _create_prefill_batch(self, q_lengths, kv_lengths, ctx):
        """Create batch data for prefill requests."""
        return self._create_decode_batch(q_lengths, kv_lengths, ctx)


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
            decode_q_lengths = [ctx.q_lengths[i] for i in decode_indices]
            decode_kv_lengths = [ctx.kv_lengths[i] for i in decode_indices]

            decode_q, decode_kv, _, decode_kv_indptr, decode_kv_indices, decode_last_page = (
                self._create_batch(decode_q_lengths, decode_kv_lengths, ctx)
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
            prefill_q_lengths = [ctx.q_lengths[i] for i in prefill_indices]
            prefill_kv_lengths = [ctx.kv_lengths[i] for i in prefill_indices]

            prefill_q, prefill_kv, prefill_qo_indptr, prefill_kv_indptr, prefill_kv_indices, prefill_last_page = (
                self._create_batch(prefill_q_lengths, prefill_kv_lengths, ctx)
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

    def _create_batch(self, q_lengths, kv_lengths, ctx):
        """Create batch data for requests."""
        num_pages = [(kv + ctx.page_size - 1) // ctx.page_size for kv in kv_lengths]

        q = torch.randn(sum(q_lengths), ctx.num_qo_heads, ctx.head_dim, dtype=torch.float16, device="cuda")
        kv_cache = torch.randn(sum(num_pages), 2, ctx.page_size, ctx.num_kv_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        qo_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lengths), 0)), dtype=torch.int32, device="cuda")
        # Use shuffled indices for realistic scattered memory
        kv_page_indices = torch.randperm(sum(num_pages), dtype=torch.int32, device="cuda")
        kv_page_indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_pages), 0)), dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [kv % ctx.page_size if kv % ctx.page_size != 0 else ctx.page_size for kv in kv_lengths],
            dtype=torch.int32, device="cuda"
        )

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
