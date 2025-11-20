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

        if scenario_type == 'decode':
            # For decode: check decode batch size + kv
            batch_match = ranges['decode_batch'][0] <= batch <= ranges['decode_batch'][1]
            if batch_match and kv_match:
                matches.append(cat_name)

        elif scenario_type == 'prefill':
            # For prefill: check chunk size + kv (batch is always 1)
            chunk_match = ranges['prefill_chunk'][0] <= query <= ranges['prefill_chunk'][1]
            if chunk_match and kv_match:
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
