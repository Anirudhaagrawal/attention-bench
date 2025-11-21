#!/usr/bin/env python3
"""FlashInfer attention approach implementations."""

from typing import Callable
import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)

from .base import APPROACHES, BenchmarkContext, FlashInferSeparatedBase

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

    def setup(self, ctx: BenchmarkContext, return_output: bool = False) -> Callable[[], torch.Tensor]:
        """Setup FA2 wrapper and return benchmark callable."""
        # Note: Mixed approaches always return real outputs (return_output parameter ignored)
        # Create wrapper
        workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")

        # Plan - FA2 only needs basic parameters (ignores seq_lens, block_tables, etc.)
        self.wrapper.plan(
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
            return self.wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration

    def teardown(self) -> None:
        """Clean up FlashInfer wrapper state."""
        if hasattr(self, 'wrapper'):
            # FlashInfer wrappers don't have end_forward() in all versions
            # Just delete the reference to allow cleanup
            del self.wrapper


@APPROACHES.register
class FlashInferMixedFA3:
    """FlashInfer: Prefill wrapper (FA3 backend) for ALL requests."""

    name = "flashinfer_mixed_fa3"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext, return_output: bool = False) -> Callable[[], torch.Tensor]:
        """Setup FA3 wrapper and return benchmark callable."""
        # Note: Mixed approaches always return real outputs (return_output parameter ignored)
        # Create wrapper
        workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3")

        # Plan - FA3 benefits from seq_lens for paging
        self.wrapper.plan(
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
            return self.wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration

    def teardown(self) -> None:
        """Clean up FlashInfer wrapper state."""
        if hasattr(self, 'wrapper'):
            del self.wrapper


@APPROACHES.register
class FlashInferSeparatedFA2(FlashInferSeparatedBase):
    """FlashInfer: Separate decode + prefill wrappers with FA2 backend."""

    name = "flashinfer_separated_fa2"
    default_page_size = 16

    @property
    def backend(self) -> str:
        """Use FA2 backend."""
        return "fa2"


@APPROACHES.register
class FlashInferSeparatedFA3(FlashInferSeparatedBase):
    """FlashInfer: Separate decode + prefill wrappers with FA3 backend."""

    name = "flashinfer_separated_fa3"
    default_page_size = 16

    @property
    def backend(self) -> str:
        """Use FA3 backend."""
        return "fa3"


@APPROACHES.register
class FlashInferBatchAttention:
    """FlashInfer: BatchAttention unified wrapper (if available)."""

    name = "flashinfer_batch_attention"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext, return_output: bool = False) -> Callable[[], torch.Tensor]:
        """Setup BatchAttention wrapper and return benchmark callable."""
        # Note: BatchAttention always returns real outputs (return_output parameter ignored)
        if not HAS_BATCH_ATTENTION:
            raise RuntimeError("BatchAttention not available in current FlashInfer version")

        # Calculate sequence lengths
        seq_lens = torch.tensor(ctx.kv_lengths, dtype=torch.int32, device="cuda")

        # Create BatchAttention wrapper
        self.batch_wrapper = BatchAttention(kv_layout="NHD")

        # Plan (persistent plan)
        self.batch_wrapper.plan(
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
            return self.batch_wrapper.run(ctx.q, ctx.kv_cache)

        return run_iteration

    def teardown(self) -> None:
        """Clean up FlashInfer wrapper state."""
        if hasattr(self, 'batch_wrapper'):
            del self.batch_wrapper
