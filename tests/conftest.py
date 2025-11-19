"""Pytest configuration and fixtures for attention kernel tests."""

import pytest
import torch

from approaches import APPROACHES
from .utils import create_test_context


@pytest.fixture(scope="session")
def available_approaches():
    """Get list of available approaches for correctness testing.

    Excludes:
    - cuDNN (known to not work)
    - Separated approaches (use different input data, incompatible with correctness testing)
    - Approaches that are not installed

    Returns:
        List of approach names that are available and compatible with correctness testing
    """
    all_approaches = APPROACHES.list_available()

    # Exclude cuDNN (known to not work) and separated approaches (incompatible with correctness testing)
    # Separated approaches create their own input data instead of using shared ctx data
    excluded = ["direct_cudnn", "flashinfer_separated_fa2", "flashinfer_separated_fa3"]

    # Check which approaches are actually available
    available = []
    for name in all_approaches:
        if name in excluded:
            continue

        approach = APPROACHES.get(name)
        if approach is not None:
            # Special handling for official_fa3: check if flash_attn is installed
            if name == "official_fa3":
                try:
                    import flash_attn
                    available.append(name)
                except ImportError:
                    # Skip if flash_attn not installed
                    pass
            else:
                available.append(name)

    return available


@pytest.fixture
def small_decode_context():
    """Create a small decode-only test scenario.

    Returns:
        BenchmarkContext with 4 decode requests, kv=2048
    """
    return create_test_context(
        q_lengths=[1, 1, 1, 1],
        kv_lengths=[2048, 2048, 2048, 2048],
        seed=42,
    )


@pytest.fixture
def small_prefill_context():
    """Create a small prefill-only test scenario.

    Returns:
        BenchmarkContext with 2 prefill requests, q=kv=512
    """
    return create_test_context(
        q_lengths=[512, 512],
        kv_lengths=[512, 512],
        seed=42,
    )


@pytest.fixture
def mixed_workload_context():
    """Create a mixed decode+prefill test scenario.

    Returns:
        BenchmarkContext with 2 decodes + 2 prefills
    """
    return create_test_context(
        q_lengths=[1, 1, 128, 256],
        kv_lengths=[1024, 2048, 1024, 2048],
        seed=42,
    )


@pytest.fixture(autouse=True)
def reset_cuda():
    """Reset CUDA state before each test to avoid interference."""
    yield
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
