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

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup the approach and return a callable for benchmarking.

        This is the NEW interface. The runner will:
        1. Create a complete BenchmarkContext with all possible parameters
        2. Call setup() to get a benchmark callable
        3. Run warmup iterations using the callable
        4. Time the callable over active iterations

        Args:
            ctx: Complete benchmark context with superset of all parameters.
                 Each approach selects what it needs from this context.

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
                default_page_size = 256
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
