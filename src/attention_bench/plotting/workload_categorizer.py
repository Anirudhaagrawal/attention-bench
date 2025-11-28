"""
Workload categorization for benchmark results.

Categorizes scenarios into realistic workload types:
- Code: Low decode batch, high KV (long context code completion)
- Chat: Medium batch, medium KV (conversational AI)
- Summarization: High batch, high chunk/KV (batch summarization)
"""

from typing import List


def categorize_workload(batch: int, query: int, kv: int, scenario_type: str) -> List[str]:
    """
    Categorize a scenario into workload types based on parameters.

    Args:
        batch: Batch size (decode batch for decode scenarios, always 1 for prefill)
        query: Query/chunk size (always 1 for decode, chunk size for prefill)
        kv: KV cache length
        scenario_type: 'decode' or 'prefill'

    Returns:
        List of matching category names (can match multiple categories)

    Categories:

    Decode (batch varies, query=1):
        - Code: batch(4-16), kv(16k-1M)
        - Chat: batch(16-128), kv(1k-128k)
        - Summarization: batch(32-256), kv(8k-512k)

    Prefill (batch=1, chunk varies):
        - Code: chunk(32-4k), kv(16k-1M)
        - Chat: chunk(32-4k), kv(1k-128k)
        - Summarization: chunk(2k-8k), kv(8k-512k)
    """

    # Define category ranges
    categories = {
        'code': {
            'decode_batch': (4, 16),
            'prefill_chunk': (32, 4096),
            'kv': (16 * 1024, 1024 * 1024),  # 16k to 1M
        },
        'chat': {
            'decode_batch': (16, 128),
            'prefill_chunk': (32, 4096),
            'kv': (1024, 128 * 1024),  # 1k to 128k
        },
        'summarization': {
            'decode_batch': (32, 256),
            'prefill_chunk': (2048, 8192),
            'kv': (8 * 1024, 512 * 1024),  # 8k to 512k
        },
    }

    matches = []

    for cat_name, ranges in categories.items():
        kv_match = ranges['kv'][0] <= kv <= ranges['kv'][1]

        if scenario_type in ['decode', 'mixed']:
            # For decode/mixed: check decode batch size + kv
            batch_match = ranges['decode_batch'][0] <= batch <= ranges['decode_batch'][1]
            if batch_match and kv_match:
                matches.append(cat_name)

        elif scenario_type == 'prefill':
            # For prefill: check chunk size + kv (batch is always 1)
            chunk_match = ranges['prefill_chunk'][0] <= query <= ranges['prefill_chunk'][1]
            if chunk_match and kv_match:
                matches.append(cat_name)

    return matches


def categorize_mixed_workload(decode_batch: int, decode_kv: int, prefill_query: int, prefill_kv: int) -> List[str]:
    """
    Categorize a mixed workload scenario based on all dimensions.

    For mixed workloads, we consider all 4 dimensions separately:
    - decode_batch: The decode batch size
    - decode_kv: The decode KV cache length
    - prefill_query: The prefill query/chunk size
    - prefill_kv: The prefill KV cache length

    A scenario matches a category only if ALL 4 dimensions fall within the ranges.

    Args:
        decode_batch: Decode batch size
        decode_kv: Decode KV cache length
        prefill_query: Prefill query/chunk size
        prefill_kv: Prefill KV cache length

    Returns:
        List of matching category names
    """
    # Define category ranges for mixed workloads
    categories = {
        'code': {
            'decode_batch': (4, 16),
            'decode_kv': (16 * 1024, 1024 * 1024),  # 16k to 1M
            'prefill_query': (32, 4096),
            'prefill_kv': (16 * 1024, 1024 * 1024),  # 16k to 1M
        },
        'chat': {
            'decode_batch': (16, 128),
            'decode_kv': (1024, 128 * 1024),  # 1k to 128k
            'prefill_query': (32, 4096),
            'prefill_kv': (1024, 128 * 1024),  # 1k to 128k
        },
        'summarization': {
            'decode_batch': (32, 256),
            'decode_kv': (8 * 1024, 512 * 1024),  # 8k to 512k
            'prefill_query': (2048, 8192),
            'prefill_kv': (8 * 1024, 512 * 1024),  # 8k to 512k
        },
    }

    matches = []

    for cat_name, ranges in categories.items():
        # Check if ALL 4 dimensions match
        batch_match = ranges['decode_batch'][0] <= decode_batch <= ranges['decode_batch'][1]
        decode_kv_match = ranges['decode_kv'][0] <= decode_kv <= ranges['decode_kv'][1]
        query_match = ranges['prefill_query'][0] <= prefill_query <= ranges['prefill_query'][1]
        prefill_kv_match = ranges['prefill_kv'][0] <= prefill_kv <= ranges['prefill_kv'][1]

        if batch_match and decode_kv_match and query_match and prefill_kv_match:
            matches.append(cat_name)

    return matches


def get_category_display_name(category: str) -> str:
    """Get a formatted display name for a category."""
    return category.capitalize()


def get_category_description(category: str, scenario_type: str) -> str:
    """Get a description of a category's parameter ranges."""

    descriptions = {
        'code': {
            'decode': 'Decode batch(4-16), KV(16k-1M) - Long context code completion',
            'prefill': 'Chunk(32-4k), KV(16k-1M) - Long context code completion',
        },
        'chat': {
            'decode': 'Decode batch(16-128), KV(1k-128k) - Conversational AI',
            'prefill': 'Chunk(32-4k), KV(1k-128k) - Conversational AI',
        },
        'summarization': {
            'decode': 'Decode batch(32-256), KV(8k-512k) - Batch summarization',
            'prefill': 'Chunk(2k-8k), KV(8k-512k) - Batch summarization',
        },
    }

    return descriptions.get(category, {}).get(scenario_type, '')


# Available workload categories
WORKLOAD_CATEGORIES = ['code', 'chat', 'summarization']
