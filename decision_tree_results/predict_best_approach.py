#!/usr/bin/env python3
"""
Auto-generated decision tree for selecting optimal FlashInfer approach.
Generated from benchmark results.
"""

def select_best_approach(prefill_q, prefill_kv, decode_kv, decode_batch, total_decode_tokens, total_prefill_tokens, num_prefills):
    """
    Select the best FlashInfer approach based on workload characteristics.
    
    Args:
        prefill_q: Query tokens per prefill request (0 if decode-only)
        prefill_kv: KV cache length for prefill requests (0 if decode-only)
        decode_kv: KV cache length for decode requests (0 if prefill-only)
        decode_batch: Number of decode requests (0 if prefill-only)
        total_decode_tokens: Total decode tokens = decode_batch * decode_kv
        total_prefill_tokens: Total prefill tokens = prefill_q * prefill_kv
        num_prefills: Number of prefill requests
    
    Returns:
        str: Name of best approach
    """
    if prefill_kv <= 1536.00:
        if total_decode_tokens <= 196608.00:
            if total_prefill_tokens <= 196608.00:
                return 'flashinfer_batch_attention'
            else:  # total_prefill_tokens > 196608.00
                return 'official_fa3'
        else:  # total_decode_tokens > 196608.00
            if total_decode_tokens <= 393216.00:
                return 'flashinfer_batch_attention'
            else:  # total_decode_tokens > 393216.00
                return 'flashinfer_batch_attention'
    else:  # prefill_kv > 1536.00
        if total_decode_tokens <= 786432.00:
            if total_prefill_tokens <= 3145728.00:
                return 'official_fa3'
            else:  # total_prefill_tokens > 3145728.00
                return 'official_fa3'
        else:  # total_decode_tokens > 786432.00
            if total_prefill_tokens <= 25165824.00:
                return 'flashinfer_batch_attention'
            else:  # total_prefill_tokens > 25165824.00
                return 'official_fa3'
