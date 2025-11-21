#!/usr/bin/env python3
"""Official FlashAttention-3 approach implementations."""

from typing import Callable
import sys
import torch

from .base import APPROACHES, BenchmarkContext

# Import FA3 from hopper directory
try:
    sys.path.insert(0, '/scratch/anirudha/flash-attention/hopper')
    from flash_attn_interface import flash_attn_with_kvcache
    HAS_OFFICIAL_FA3 = True
except ImportError:
    HAS_OFFICIAL_FA3 = False


@APPROACHES.register
class OfficialFA3:
    """Official FlashAttention-3 with paged KV cache (Hopper optimized)."""

    name = "official_fa3"
    default_page_size = 16  # Using 256 for now

    def setup(self, ctx: BenchmarkContext, return_output: bool = False) -> Callable[[], torch.Tensor]:
        """Setup Official FA3 with varlen format (NO padding!) using shared data from context.

        Note: Official FA3 always returns real outputs (return_output parameter ignored for compatibility).
        """
        if not HAS_OFFICIAL_FA3:
            raise RuntimeError(
                "Official FlashAttention-3 not available. Install with: pip install flash-attn"
            )

        batch_size = len(ctx.q_lengths)
        max_seqlen_q = max(ctx.q_lengths)

        # Split interleaved KV cache format into separate K and V tensors
        # Context format: (num_pages, 2, page_size, num_kv_heads, head_dim)
        # Official FA3 expects: (num_pages, page_size, num_kv_heads, head_dim)
        k_cache = ctx.kv_cache[:, 0]  # Extract K
        v_cache = ctx.kv_cache[:, 1]  # Extract V

        # Convert FlashInfer's (indptr + indices) format to page_table format
        # FlashInfer uses: kv_page_indptr[i]:kv_page_indptr[i+1] gives indices for batch i
        # Official FA3 expects: page_table[i, :] is a padded array of page indices
        num_pages_per_seq = [
            (ctx.kv_page_indptr[i+1] - ctx.kv_page_indptr[i]).item()
            for i in range(batch_size)
        ]
        max_num_pages = max(num_pages_per_seq)

        page_table = torch.zeros(
            batch_size, max_num_pages, dtype=torch.int32, device="cuda"
        )

        # Fill page_table using ctx.kv_page_indices (same data as FlashInfer!)
        for i in range(batch_size):
            start = ctx.kv_page_indptr[i].item()
            end = ctx.kv_page_indptr[i+1].item()
            num_pages = end - start
            page_table[i, :num_pages] = ctx.kv_page_indices[start:end]

        # Cache sequence lengths (actual KV lengths)
        cache_seqlens = torch.tensor(ctx.kv_lengths, dtype=torch.int32, device="cuda")

        # FA3: Use varlen mode (NO padding!)
        def run_iteration():
            return flash_attn_with_kvcache(
                q=ctx.q,                      # Packed format (NO padding!)
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                cu_seqlens_q=ctx.qo_indptr,   # Enables varlen mode!
                max_seqlen_q=max_seqlen_q,
                causal=True,
                num_splits=0,
            )

        return run_iteration
