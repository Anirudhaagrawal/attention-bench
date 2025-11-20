"""Pure cuDNN attention implementation (no FlashInfer)."""

import math
from typing import Callable

import cudnn
import torch

from .base import APPROACHES, BenchmarkContext


@APPROACHES.register
class DirectCuDNN:
    """Pure cuDNN attention using Python frontend API.

    No FlashInfer wrapper - direct cuDNN implementation.
    Converts paged KV cache to contiguous format for cuDNN.
    """

    name = "direct_cudnn"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup pure cuDNN attention with native paged KV cache support."""

        batch_size = len(ctx.kv_lengths)
        max_q_len = max(ctx.q_lengths)

        # Convert Q from packed [total_q, heads, dim] to batched [batch, heads, max_q, dim]
        q_batched = torch.zeros(batch_size, ctx.num_qo_heads, max_q_len, ctx.head_dim,
                                dtype=torch.float16, device="cuda")
        q_offset = 0
        for i in range(batch_size):
            q_len = ctx.q_lengths[i]
            q_seq = ctx.q[q_offset:q_offset + q_len]  # [q_len, num_qo_heads, head_dim]
            q_batched[i, :, :q_len, :] = q_seq.transpose(0, 1)  # [num_qo_heads, q_len, head_dim]
            q_offset += q_len

        # Reorganize FlashInfer's global page pool into cuDNN's INTERLEAVED container format
        # cuDNN expects interleaved layout: [batch0_block0, batch1_block0, ..., batch0_block1, batch1_block1, ...]
        # NOT batch-stacked: [batch0_block0, batch0_block1, ..., batch1_block0, batch1_block1, ...]

        # Calculate blocks per batch (must be same for all sequences in this implementation)
        max_blocks = max((kv_len + ctx.page_size - 1) // ctx.page_size for kv_len in ctx.kv_lengths)
        total_blocks = batch_size * max_blocks

        # Create interleaved containers (separate K and V, no GQA expansion)
        k_container = torch.empty(total_blocks, ctx.num_kv_heads, ctx.page_size, ctx.head_dim,
                                  dtype=torch.float16, device="cuda")
        v_container = torch.empty(total_blocks, ctx.num_kv_heads, ctx.page_size, ctx.head_dim,
                                  dtype=torch.float16, device="cuda")

        # Copy pages from FlashInfer global pool to INTERLEAVED layout
        for i in range(batch_size):
            kv_len = ctx.kv_lengths[i]
            num_blocks = (kv_len + ctx.page_size - 1) // ctx.page_size

            # Get this sequence's page indices from FlashInfer's global pool
            start_page = ctx.kv_page_indptr[i].item()
            end_page = ctx.kv_page_indptr[i + 1].item()
            page_indices = ctx.kv_page_indices[start_page:end_page]

            # Copy each page to INTERLEAVED container positions
            for block_idx, page_id in enumerate(page_indices):
                # Interleaved index: batch_idx + block_idx * batch_size
                container_idx = i + block_idx * batch_size

                # Extract K and V from interleaved format
                # ctx.kv_cache[page_id]: [2, page_size, num_kv_heads, head_dim]
                k_page = ctx.kv_cache[page_id, 0]  # [page_size, num_kv_heads, head_dim]
                v_page = ctx.kv_cache[page_id, 1]  # [page_size, num_kv_heads, head_dim]

                # Transpose to cuDNN format: [num_kv_heads, page_size, head_dim]
                k_container[container_idx] = k_page.transpose(0, 1)
                v_container[container_idx] = v_page.transpose(0, 1)

        # Create INTERLEAVED page tables in cuDNN format: [batch, 1, max_blocks, 1]
        # Using strided view for correct interleaved access pattern
        page_table_temp = torch.arange(batch_size * max_blocks, dtype=torch.int32, device="cuda")
        page_table_temp = page_table_temp.reshape(max_blocks, 1, batch_size, 1)
        page_table_temp = page_table_temp.transpose(0, 2)  # [batch, 1, max_blocks, 1]

        # Create with specific stride pattern (batch, 1, blocks, 1) with stride (max_blocks, max_blocks, 1, 1)
        alt_stride = (max_blocks, max_blocks, 1, 1)
        page_table_dims = (batch_size, 1, max_blocks, 1)
        page_table = torch.randn(max_blocks * batch_size).int().cuda()\
                     .as_strided(page_table_dims, alt_stride)
        page_table.copy_(page_table_temp)

        k_table = page_table
        v_table = page_table  # Same for K and V

        # Sequence lengths
        seq_len_q = ctx.seq_lens_q  # [batch] int32
        seq_len_kv = ctx.seq_lens_kv  # [batch] int32

        # Max sequence length for KV
        max_seq_len_kv = ctx.max_sequence_kv

        # Debug output - verify interleaved format
        print(f"[DirectCuDNN] Batch size: {batch_size}, Max blocks: {max_blocks}, Total: {total_blocks}")
        print(f"[DirectCuDNN] q_batched: {q_batched.shape} [batch, qo_heads={ctx.num_qo_heads}, max_q, dim]")
        print(f"[DirectCuDNN] k_container: {k_container.shape} [total_blocks (interleaved), kv_heads={ctx.num_kv_heads}, page_size, dim]")
        print(f"[DirectCuDNN] k_table: {k_table.shape} [batch, 1, max_blocks, 1], stride: {k_table.stride()}")
        print(f"[DirectCuDNN] Page table batch 0: {k_table[0, 0, :min(8, max_blocks), 0].tolist()}")
        if batch_size > 1:
            print(f"[DirectCuDNN] Page table batch 1: {k_table[1, 0, :min(8, max_blocks), 0].tolist()}")
        print(f"[DirectCuDNN] seq_len_kv: {seq_len_kv.tolist()}, max: {max_seq_len_kv}")

        # Attention scale
        scale = 1.0 / math.sqrt(ctx.head_dim)

        # Build cuDNN graph
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=handle,
        )

        # Define tensors with explicit UIDs
        Q = graph.tensor_like(q_batched)
        Q.set_uid(100)

        # K and V are paged containers (non-contiguous memory blocks)
        K = graph.tensor_like(k_container)
        K.set_uid(101)

        V = graph.tensor_like(v_container)
        V.set_uid(102)

        # Page tables
        K_table = graph.tensor_like(k_table)
        K_table.set_uid(103)

        V_table = graph.tensor_like(v_table)
        V_table.set_uid(104)

        # Sequence lengths (required for paged attention)
        SeqLenQ = graph.tensor_like(seq_len_q)
        SeqLenQ.set_uid(105)

        SeqLenKV = graph.tensor_like(seq_len_kv)
        SeqLenKV.set_uid(106)

        # SDPA with paged attention
        # REQUIRED: paged caches need padding mask + variable seq lengths
        O, Stats = graph.sdpa(
            name="sdpa",
            q=Q,
            k=K,
            v=V,
            is_inference=True,
            attn_scale=scale,
            use_causal_mask=True,
            use_padding_mask=True,  # Required for paged attention
            seq_len_q=SeqLenQ,  # Required for paged attention
            seq_len_kv=SeqLenKV,  # Required for paged attention
            paged_attention_k_table=K_table,
            paged_attention_v_table=V_table,
            paged_attention_max_seq_len_kv=max_seq_len_kv,
        )

        # Set output with explicit UID
        O.set_uid(200).set_output(True).set_dim(q_batched.shape).set_stride(q_batched.stride()).set_data_type(cudnn.data_type.HALF)

        # Build graph
        graph.build([cudnn.heur_mode.A])

        # Allocate workspace
        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

        # Allocate output tensor
        o_batched = torch.empty_like(q_batched)

        def run_iteration():
            """Execute cuDNN attention with paged KV cache."""
            graph.execute(
                {
                    100: q_batched,
                    101: k_container,
                    102: v_container,
                    103: k_table,
                    104: v_table,
                    105: seq_len_q,
                    106: seq_len_kv,
                    200: o_batched,
                },
                workspace,
            )
            return o_batched

        return run_iteration
