"""Centralized constants for attention benchmarking.

This module defines all magic numbers and default values used throughout
the attention_bench package. Using these constants instead of hardcoded
values ensures consistency across modules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConstants:
    """Memory-related constants for GPU operations."""

    # Default workspace size for FlashInfer wrappers (256 MB)
    DEFAULT_WORKSPACE_SIZE: int = 256 * 1024 * 1024

    # Runtime memory settings (less conservative, for actual execution)
    DEFAULT_MEMORY_UTILIZATION: float = 0.90
    DEFAULT_MEMORY_OVERHEAD: float = 1.1

    # Estimation memory settings (more conservative, for pre-filtering)
    ESTIMATION_UTILIZATION: float = 0.80
    ESTIMATION_OVERHEAD: float = 1.4


@dataclass(frozen=True)
class ValidationConstants:
    """Constants for output validation and correctness checking."""

    # Tolerances for comparing floating point outputs
    DEFAULT_RTOL: float = 1e-3  # Relative tolerance (0.1%)
    DEFAULT_ATOL: float = 1e-3  # Absolute tolerance


@dataclass(frozen=True)
class ProfilingConstants:
    """Constants for profiling and timing."""

    # Default iteration counts
    DEFAULT_WARMUP_ITERS: int = 5
    DEFAULT_ACTIVE_ITERS: int = 50


@dataclass(frozen=True)
class PlottingConstants:
    """Constants for plot generation."""

    # Default page sizes
    PLOTS_PER_PAGE: int = 12

    # Font sizes
    TITLE_FONTSIZE: int = 14
    LABEL_FONTSIZE: int = 12
    TICK_FONTSIZE: int = 10


# Singleton instances for easy access
MEMORY = MemoryConstants()
VALIDATION = ValidationConstants()
PROFILING = ProfilingConstants()
PLOTTING = PlottingConstants()
