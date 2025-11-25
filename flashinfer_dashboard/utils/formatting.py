"""Text formatting utilities for FlashInfer dashboard.

This module contains simple formatting functions used across the dashboard.
"""

from .constants import SECONDS_TO_MS, KB_BYTES, MB_BYTES


def shorten_approach_name(approach_name: str) -> str:
    """Convert long approach names to short display names.

    Args:
        approach_name: Full approach name (e.g., "flashinfer_mixed_fa2")

    Returns:
        Shortened display name (e.g., "Mix2")

    Examples:
        >>> shorten_approach_name("flashinfer_mixed_fa2")
        'Mix2'
        >>> shorten_approach_name("official_fa3")
        'OFA3'
    """
    name_map = {
        "official_fa3": "OFA3",
        "flashinfer_batch_attention": "Batch",
        "flashinfer_mixed_fa2": "Mix2",
        "flashinfer_mixed_fa3": "Mix3",
        "flashinfer_separated_fa2": "Sep2",
        "flashinfer_separated_fa3": "Sep3",
        # Grouped approaches
        "fi2": "FI2",
        "fi3": "FI3",
        "fi_dec": "FI_Dec",
    }
    return name_map.get(approach_name, approach_name.replace("flashinfer_", "").title())


def format_kv_length(kv: int) -> str:
    """Format KV length for display (e.g., 1024 → 1k, 1048576 → 1M).

    Args:
        kv: KV length in tokens

    Returns:
        Formatted string with k/M suffix

    Examples:
        >>> format_kv_length(1024)
        '1k'
        >>> format_kv_length(1048576)
        '1M'
        >>> format_kv_length(500)
        '500'
    """
    if kv >= MB_BYTES and kv % MB_BYTES == 0:
        return f"{kv // MB_BYTES}M"
    elif kv >= KB_BYTES and kv % KB_BYTES == 0:
        return f"{kv // KB_BYTES}k"
    return str(kv)


def format_time_ms(seconds: float, precision: int = 2) -> str:
    """Convert seconds to milliseconds formatted string.

    Args:
        seconds: Time in seconds
        precision: Number of decimal places (default: 2)

    Returns:
        Formatted string with 'ms' suffix

    Examples:
        >>> format_time_ms(0.123)
        '123.00ms'
        >>> format_time_ms(0.00456, precision=3)
        '4.560ms'
        >>> format_time_ms(1.5, precision=0)
        '1500ms'
    """
    return f"{seconds * SECONDS_TO_MS:.{precision}f}ms"
