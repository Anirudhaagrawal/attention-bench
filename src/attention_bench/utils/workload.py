"""Workload building utilities for attention benchmarks.

This module provides shared utilities for building workloads from
variant configurations, used by both the CLI benchmark runner and
Ray-based distributed executor.
"""

from typing import Any, Dict, List, Tuple, Union


def build_workload_from_variant_set(
    variant_set: Union[Dict[str, int], List[Dict[str, int]]],
    variants: Dict[str, Any],
) -> Tuple[List[int], List[int]]:
    """Build workload from variant set configuration.

    Converts a variant set specification into lists of query and KV lengths
    for benchmark execution.

    Args:
        variant_set: Variant specification in one of two formats:
            - Dict format: {"variant_name": count, ...}
            - List format: [{"variant_name": count}, ...]
        variants: Dictionary mapping variant names to configurations.
            Each variant can be either:
            - A dict with "q_tokens" and "kv_tokens" keys
            - An object with q_tokens and kv_tokens attributes (VariantConfig)

    Returns:
        Tuple of (q_lengths, kv_lengths) where each is a list of integers
        representing the query and KV cache lengths for all requests.

    Example:
        >>> variants = {
        ...     "decode_short": {"q_tokens": 1, "kv_tokens": 1024},
        ...     "prefill_long": {"q_tokens": 512, "kv_tokens": 2048},
        ... }
        >>> variant_set = {"decode_short": 10, "prefill_long": 5}
        >>> q_lengths, kv_lengths = build_workload_from_variant_set(variant_set, variants)
        >>> len(q_lengths)  # 10 + 5 = 15
        15
    """
    q_lengths: List[int] = []
    kv_lengths: List[int] = []

    def _extract_tokens(variant: Any) -> Tuple[int, int]:
        """Extract q_tokens and kv_tokens from variant (dict or object)."""
        if isinstance(variant, dict):
            return variant["q_tokens"], variant["kv_tokens"]
        else:
            # Object with attributes (e.g., VariantConfig)
            return variant.q_tokens, variant.kv_tokens

    if isinstance(variant_set, list):
        # List format: [{"variant_name": count}, ...]
        for item in variant_set:
            for variant_name, count in item.items():
                variant = variants[variant_name]
                q_tok, kv_tok = _extract_tokens(variant)
                for _ in range(count):
                    q_lengths.append(q_tok)
                    kv_lengths.append(kv_tok)
    else:
        # Dict format: {"variant_name": count, ...}
        for variant_name, count in variant_set.items():
            variant = variants[variant_name]
            q_tok, kv_tok = _extract_tokens(variant)
            for _ in range(count):
                q_lengths.append(q_tok)
                kv_lengths.append(kv_tok)

    return q_lengths, kv_lengths
