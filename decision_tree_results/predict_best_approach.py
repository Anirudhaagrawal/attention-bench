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
    if total_prefill_tokens <= 786432.00:
        if decode_batch <= 24.00:
            if decode_kv <= 12288.00:
                if prefill_kv <= 384.00:
                    if total_decode_tokens <= 24576.00:
                        return 'flashinfer_batch_attention'
                    else:  # total_decode_tokens > 24576.00
                        return 'official_fa3'
                else:  # prefill_kv > 384.00
                    if prefill_kv <= 768.00:
                        return 'official_fa3'
                    else:  # prefill_kv > 768.00
                        return 'official_fa3'
            else:  # decode_kv > 12288.00
                if decode_batch <= 3.00:
                    if prefill_kv <= 1536.00:
                        return 'flashinfer_batch_attention'
                    else:  # prefill_kv > 1536.00
                        return 'flashinfer_batch_attention'
                else:  # decode_batch > 3.00
                    if prefill_q <= 192.00:
                        return 'official_fa3'
                    else:  # prefill_q > 192.00
                        return 'flashinfer_batch_attention'
        else:  # decode_batch > 24.00
            if total_prefill_tokens <= 196608.00:
                if prefill_kv <= 768.00:
                    if total_decode_tokens <= 196608.00:
                        return 'flashinfer_batch_attention'
                    else:  # total_decode_tokens > 196608.00
                        return 'flashinfer_batch_attention'
                else:  # prefill_kv > 768.00
                    return 'flashinfer_batch_attention'
            else:  # total_prefill_tokens > 196608.00
                if decode_batch <= 48.00:
                    if decode_kv <= 12288.00:
                        return 'official_fa3'
                    else:  # decode_kv > 12288.00
                        return 'flashinfer_batch_attention'
                else:  # decode_batch > 48.00
                    if decode_batch <= 96.00:
                        return 'flashinfer_batch_attention'
                    else:  # decode_batch > 96.00
                        return 'flashinfer_batch_attention'
    else:  # total_prefill_tokens > 786432.00
        if total_decode_tokens <= 1572864.00:
            if total_prefill_tokens <= 3145728.00:
                if total_decode_tokens <= 786432.00:
                    if decode_kv <= 12288.00:
                        return 'official_fa3'
                    else:  # decode_kv > 12288.00
                        return 'official_fa3'
                else:  # total_decode_tokens > 786432.00
                    if prefill_q <= 384.00:
                        return 'flashinfer_batch_attention'
                    else:  # prefill_q > 384.00
                        return 'flashinfer_batch_attention'
            else:  # total_prefill_tokens > 3145728.00
                if decode_kv <= 24576.00:
                    if total_decode_tokens <= 786432.00:
                        return 'official_fa3'
                    else:  # total_decode_tokens > 786432.00
                        return 'official_fa3'
                else:  # decode_kv > 24576.00
                    if total_prefill_tokens <= 12582912.00:
                        return 'official_fa3'
                    else:  # total_prefill_tokens > 12582912.00
                        return 'official_fa3'
        else:  # total_decode_tokens > 1572864.00
            if total_prefill_tokens <= 25165824.00:
                if prefill_kv <= 24576.00:
                    if prefill_kv <= 12288.00:
                        return 'flashinfer_batch_attention'
                    else:  # prefill_kv > 12288.00
                        return 'flashinfer_batch_attention'
                else:  # prefill_kv > 24576.00
                    return 'flashinfer_batch_attention'
            else:  # total_prefill_tokens > 25165824.00
                if total_decode_tokens <= 3145728.00:
                    return 'official_fa3'
                else:  # total_decode_tokens > 3145728.00
                    return 'official_fa3'
