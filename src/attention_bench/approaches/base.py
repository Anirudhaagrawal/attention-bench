#!/usr/bin/env python3
"""Base protocol and registry for attention approaches."""

from typing import Protocol, Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import torch


@dataclass
class BenchmarkContext:
    """Complete superset of all parameters any approach might need.

    This context contains all possible data and metadata that any attention
    approach might require. Each approach can select what it needs.
    """

    # Original inputs
    q_lengths: List[int]
    kv_lengths: List[int]

    # Batch data tensors
    q: torch.Tensor  # (total_q_tokens, num_qo_heads, head_dim)
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    # Either: (num_pages, 2, num_kv_heads, page_size, head_dim)
    # Or: tuple of (k_cache, v_cache) each (num_pages, num_kv_heads, page_size, head_dim)

    # Pre-split Q tensor views for separated approaches (memory optimization)
    decode_q: Optional[torch.Tensor]  # View into q for decode requests (q_len==1)
    prefill_q: Optional[torch.Tensor]  # View into q for prefill requests (q_len>1)
    decode_indices: List[int]  # Original indices of decode requests
    prefill_indices: List[int]  # Original indices of prefill requests

    # Page metadata
    qo_indptr: torch.Tensor  # (batch_size + 1,) uint32
    kv_page_indptr: torch.Tensor  # (batch_size + 1,) uint32
    kv_page_indices: torch.Tensor  # (total_pages,) uint32
    kv_last_page_len: torch.Tensor  # (batch_size,) uint32

    # Computed optional tensors (superset - all computed unconditionally)
    seq_lens_q: torch.Tensor  # (batch_size,) uint32
    seq_lens_kv: torch.Tensor  # (batch_size,) uint32
    block_tables: torch.Tensor  # (batch_size, max_num_blocks) uint32
    max_token_per_sequence: int
    max_sequence_kv: int

    # Model configuration
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    workspace_size: int

    # Profiling configuration
    num_warmup_iters: int
    num_active_iters: int


class AttentionApproach(Protocol):
    """Protocol defining interface for attention implementations.

    Each approach must implement:
    - name: Unique identifier (e.g., "flashinfer_mixed_fa2", "official_fa3")
    - default_page_size: Preferred page size for this approach
    - setup(): NEW method to set up and return callable (preferred)
    - benchmark(): OLD method for backward compatibility (deprecated)
    """

    name: str
    default_page_size: int

    def setup(self, ctx: BenchmarkContext, return_output: bool = False) -> Callable[[], torch.Tensor]:
        """Setup the approach and return a callable for benchmarking.

        This is the NEW interface. The runner will:
        1. Create a complete BenchmarkContext with all possible parameters
        2. Call setup() to get a benchmark callable
        3. Run warmup iterations using the callable
        4. Time the callable over active iterations
        5. Call teardown() if available for cleanup

        Args:
            ctx: Complete benchmark context with superset of all parameters.
                 Each approach selects what it needs from this context.
            return_output: If True, the callable must return actual attention outputs.
                          If False (default), can return dummy tensor to save memory.
                          Set to True for correctness testing, False for benchmarking.

        Returns:
            A callable that runs one iteration and returns output tensor.
            The runner will call this function multiple times for warmup and timing.

        Example:
            def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
                # Create wrapper with selected parameters from ctx
                wrapper = SomeWrapper(...)
                wrapper.plan(ctx.qo_indptr, ctx.kv_page_indptr, ...)

                # Return callable that captures wrapper and data
                def run_iteration():
                    return wrapper.run(ctx.q, ctx.kv_cache)

                return run_iteration
        """
        ...

    def teardown(self) -> None:
        """Optional cleanup after benchmarking completes.

        This method is called after all iterations complete to allow approaches
        to clean up stateful resources (e.g., FlashInfer wrapper internal state).
        Implementing this method is optional but recommended for approaches with
        persistent state.
        """
        pass

    def benchmark(
        self,
        q_lengths: List[int],
        kv_lengths: List[int],
        runner: Any,  # ToleranceBenchmarkRunner
        config: Dict[str, Any],
    ) -> float:
        """Run benchmark and return time in milliseconds.

        DEPRECATED: This method is kept for backward compatibility.
        New implementations should use setup() instead.

        Args:
            q_lengths: Query sequence lengths for each request
            kv_lengths: KV cache lengths for each request
            runner: ToleranceBenchmarkRunner instance (provides helper methods)
            config: Configuration dict with benchmark parameters
                - page_size: Page size to use (may override default_page_size)
                - num_warmup_iters: Warmup iterations
                - num_active_iters: Active iterations for timing
                - num_qo_heads, num_kv_heads, head_dim: Model dimensions
                - workspace_size: Workspace size for FlashInfer wrappers
                - use_cuda_graphs: Whether to use CUDA graphs
                - enable_profiling: Whether to enable profiling
                - scenario_name: Name of current scenario

        Returns:
            Average time in milliseconds
        """
        ...


class ApproachRegistry:
    """Registry of available attention approaches."""

    def __init__(self):
        self._approaches: Dict[str, AttentionApproach] = {}

    def register(self, approach_class: type) -> type:
        """Decorator to register an approach class.

        Usage:
            @APPROACHES.register
            class MyApproach:
                name = "my_approach"
                default_page_size = 16
                def benchmark(...): ...
        """
        approach = approach_class()
        if approach.name in self._approaches:
            raise ValueError(f"Approach '{approach.name}' already registered")
        self._approaches[approach.name] = approach
        return approach_class

    def get(self, name: str) -> Optional[AttentionApproach]:
        """Get approach by name."""
        return self._approaches.get(name)

    def list_available(self) -> List[str]:
        """List all registered approach names."""
        return sorted(self._approaches.keys())

    def get_all(self) -> Dict[str, AttentionApproach]:
        """Get all registered approaches."""
        return self._approaches.copy()


# Global registry instance
APPROACHES = ApproachRegistry()


class FlashInferSeparatedBase:
    """Base class for FlashInfer separated decode + prefill approaches.

    This class extracts common logic for approaches that use separate
    decode and prefill wrappers. Subclasses only need to specify the backend.
    """

    @property
    def backend(self) -> str:
        """Backend to use for wrappers (e.g., 'fa2', 'fa3')."""
        raise NotImplementedError("Subclasses must define backend property")

    def setup(self, ctx: BenchmarkContext, return_output: bool = False):
        """Setup separated decode + prefill wrappers and return benchmark callable.

        OPTIMIZATION: Uses pre-split Q tensor views from context (ctx.decode_q, ctx.prefill_q)
        instead of creating copies via torch.cat().
        """
        import torch
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper,
        )

        # Context is pre-ordered: decode requests first, then prefill
        num_decode = len(ctx.decode_indices)
        num_prefill = len(ctx.prefill_indices)

        decode_wrapper = None
        prefill_wrapper = None

        # Create decode wrapper if we have decodes
        if num_decode > 0:
            # Use pre-split Q tensor view (no copy!)
            decode_q = ctx.decode_q

            # Extract decode metadata from context using ORIGINAL request indices
            _, _, _, decode_kv_indptr, decode_kv_indices, decode_last_page = (
                self._extract_subset_from_context(ctx.decode_indices, ctx)
            )

            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend=self.backend)
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
        if num_prefill > 0:
            # Use pre-split Q tensor view (no copy!)
            prefill_q = ctx.prefill_q

            # Extract prefill metadata from context using ORIGINAL request indices
            _, _, prefill_qo_indptr, prefill_kv_indptr, prefill_kv_indices, prefill_last_page = (
                self._extract_subset_from_context(ctx.prefill_indices, ctx)
            )

            workspace = torch.empty(ctx.workspace_size, dtype=torch.uint8, device="cuda")
            prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend=self.backend)
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

        # Store wrappers and tensors for cleanup
        self.decode_wrapper = decode_wrapper
        self.prefill_wrapper = prefill_wrapper
        self.decode_q = decode_q if num_decode > 0 else None
        self.prefill_q = prefill_q if num_prefill > 0 else None
        self.kv_cache = ctx.kv_cache  # Shared KV cache (no copy!)

        # Store context info for reordering outputs
        self.ctx = ctx

        # Return callable that runs both wrappers
        if return_output:
            # Correctness testing: return outputs in ORIGINAL request order
            def run_iteration():
                # Run wrappers
                decode_out = None
                prefill_out = None

                if self.decode_wrapper is not None:
                    decode_out = self.decode_wrapper.run(self.decode_q, self.kv_cache)
                if self.prefill_wrapper is not None:
                    prefill_out = self.prefill_wrapper.run(self.prefill_q, self.kv_cache)

                # Reorder outputs back to original request order
                total_requests = len(ctx.decode_indices) + len(ctx.prefill_indices)
                result = []
                for req_idx in range(total_requests):
                    if req_idx in ctx.decode_indices:
                        # Decode request - output is 1 token
                        decode_pos = ctx.decode_indices.index(req_idx)
                        result.append(decode_out[decode_pos:decode_pos+1])
                    else:
                        # Prefill request - output is q_length tokens
                        prefill_pos = ctx.prefill_indices.index(req_idx)
                        # Calculate token range for this prefill request
                        start_token = sum(ctx.q_lengths[ctx.prefill_indices[i]]
                                        for i in range(prefill_pos))
                        end_token = start_token + ctx.q_lengths[req_idx]
                        result.append(prefill_out[start_token:end_token])

                return torch.cat(result, dim=0) if result else torch.empty(0)
        else:
            # Benchmarking: discard outputs to save memory
            def run_iteration():
                if self.decode_wrapper is not None:
                    self.decode_wrapper.run(self.decode_q, self.kv_cache)
                if self.prefill_wrapper is not None:
                    self.prefill_wrapper.run(self.prefill_q, self.kv_cache)
                return torch.empty(0)

        return run_iteration

    def teardown(self) -> None:
        """Clean up FlashInfer wrapper state."""
        if hasattr(self, 'decode_wrapper'):
            del self.decode_wrapper
        if hasattr(self, 'prefill_wrapper'):
            del self.prefill_wrapper

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
        import torch

        # Extract Q tokens for selected requests
        q_slices = []
        for idx in indices:
            start = ctx.qo_indptr[idx].item()
            end = ctx.qo_indptr[idx + 1].item()
            q_slices.append(ctx.q[start:end])
        q = torch.cat(q_slices, dim=0) if q_slices else torch.empty(0, ctx.num_qo_heads, ctx.head_dim, dtype=torch.float16, device="cuda")

        # Extract KV cache page indices for selected requests
        # OPTIMIZATION: Reuse ctx.kv_cache directly instead of copying pages
        # FlashInfer supports sparse page indices, no need to remap to [0,1,2,...]
        subset_kv_page_indices = []

        for idx in indices:
            # Get the range of pages for this request
            page_start = ctx.kv_page_indptr[idx].item()
            page_end = ctx.kv_page_indptr[idx + 1].item()

            # Get the physical page indices for this request (keep sparse indices)
            logical_pages = ctx.kv_page_indices[page_start:page_end]
            subset_kv_page_indices.extend(logical_pages.tolist())

        # Reuse the full KV cache tensor (no copy!)
        kv_cache = ctx.kv_cache

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

        kv_page_indices = torch.tensor(subset_kv_page_indices, dtype=torch.int32, device="cuda") if subset_kv_page_indices else torch.empty(0, dtype=torch.int32, device="cuda")

        # Extract last page lengths for selected requests
        kv_last_page_len = ctx.kv_last_page_len[indices] if len(indices) > 0 else torch.empty(0, dtype=torch.int32, device="cuda")

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len
