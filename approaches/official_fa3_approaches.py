#!/usr/bin/env python3
"""Official FlashAttention-3 approach implementations."""

from typing import Callable
import torch

from .base import APPROACHES, BenchmarkContext

# Try to import official FA3
try:
    from flash_attn import flash_attn_with_kvcache
    HAS_OFFICIAL_FA3 = True
except ImportError:
    HAS_OFFICIAL_FA3 = False


@APPROACHES.register
class OfficialFA3:
    """Official FlashAttention-3 with paged KV cache."""

    name = "official_fa3"
    default_page_size = 256  # Official FA3 requires page_size divisible by 256

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup Official FA3 and return callable for benchmarking."""
        if not HAS_OFFICIAL_FA3:
            raise RuntimeError(
                "Official FlashAttention-3 not available. Install with: pip install flash-attn"
            )

        if ctx.page_size % 256 != 0:
            raise ValueError(
                f"Official FA3 requires page_size divisible by 256, got {ctx.page_size}"
            )

        batch_size = len(ctx.q_lengths)

        # Calculate number of pages per sequence
        num_pages_per_seq = [
            (kv + ctx.page_size - 1) // ctx.page_size for kv in ctx.kv_lengths
        ]
        total_pages = sum(num_pages_per_seq)

        # Create paged KV cache (Official FA3 layout: different from FlashInfer)
        # Official FA3: (num_pages, page_size, num_kv_heads, head_dim)
        # FlashInfer: (num_pages, 2, num_kv_heads, page_size, head_dim)
        k_cache = torch.randn(
            total_pages, ctx.page_size, ctx.num_kv_heads, ctx.head_dim,
            dtype=torch.float16, device="cuda"
        )
        v_cache = torch.randn(
            total_pages, ctx.page_size, ctx.num_kv_heads, ctx.head_dim,
            dtype=torch.float16, device="cuda"
        )

        # Create block table (batch_size, max_num_pages)
        max_num_pages = max(num_pages_per_seq)
        block_table = torch.zeros(
            batch_size, max_num_pages, dtype=torch.int32, device="cuda"
        )

        page_offset = 0
        for i, num_pages in enumerate(num_pages_per_seq):
            block_table[i, :num_pages] = torch.arange(
                page_offset, page_offset + num_pages, dtype=torch.int32
            )
            page_offset += num_pages

        # Cache sequence lengths
        cache_seqlens = torch.tensor(
            ctx.kv_lengths, dtype=torch.int32, device="cuda"
        )

        # Reshape Q from (total_q_tokens, num_qo_heads, head_dim)
        # to (batch_size, max_seqlen_q, num_qo_heads, head_dim)
        max_q_len = max(ctx.q_lengths)
        q_batched = torch.zeros(
            batch_size, max_q_len, ctx.num_qo_heads, ctx.head_dim,
            dtype=torch.float16, device="cuda"
        )

        q_offset = 0
        for i, q_len in enumerate(ctx.q_lengths):
            q_batched[i, :q_len] = ctx.q[q_offset : q_offset + q_len]
            q_offset += q_len

        # Return callable that runs one iteration
        def run_iteration():
            return flash_attn_with_kvcache(
                q_batched,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True,
            )

        return run_iteration
