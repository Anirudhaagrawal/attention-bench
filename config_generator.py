"""
Programmatic configuration generator for FlashInfer benchmarks.

Generates variants and scenarios from parameter ranges and strategies,
eliminating the need for massive manually-written YAML configs.
"""

from typing import Dict, List, Tuple
from itertools import product


def generate_scenarios(gen_config: dict) -> Tuple[Dict, Dict]:
    """Generate variants and scenarios based on strategy.

    Args:
        gen_config: The "scenario_generation" section from YAML config

    Returns:
        (variants_dict, scenarios_dict) in YAML-compatible format
    """
    strategy = gen_config["strategy"]

    if strategy == "all_combinations":
        return _generate_all_combinations(gen_config)
    elif strategy == "mixed_combinations":
        return _generate_mixed_combinations(gen_config)
    elif strategy == "varying_kv":
        return _generate_varying_kv(gen_config)
    elif strategy == "sweep_seqlen":
        return _generate_sweep_seqlen(gen_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _generate_all_combinations(config: dict) -> Tuple[Dict, Dict]:
    """Generate all combinations for pure decode or pure prefill workloads.

    For decode: batch_sizes × kv_tokens (q_tokens = 1 always)
    For prefill: batch_sizes × q_tokens × kv_tokens
    """
    workload_type = config["workload_type"]
    params = config["parameters"]

    batch_sizes = params.get("batch_sizes", [1])
    kv_tokens_list = params["kv_tokens"]

    variants = {}
    scenarios = {}

    if workload_type == "decode":
        # Decode: q_tokens is always 1
        q_tokens_list = [1]
        prefix = "decode"

    elif workload_type == "prefill":
        # Prefill: use specified q_tokens
        q_tokens_list = params["q_tokens"]
        prefix = "prefill"

    else:
        raise ValueError(f"Unknown workload_type: {workload_type}")

    # Generate variants (one per q×kv combination)
    for q, kv in product(q_tokens_list, kv_tokens_list):
        variant_name = f"{prefix}_q{_format_size(q)}_kv{_format_size(kv)}"
        variants[variant_name] = {
            "q_tokens": q,
            "kv_tokens": kv,
            "description": f"{prefix.capitalize()} ({_format_size(q)} q, {_format_size(kv)} kv)"
        }

    # Generate scenarios (one per batch×q×kv combination)
    for batch, q, kv in product(batch_sizes, q_tokens_list, kv_tokens_list):
        variant_name = f"{prefix}_q{_format_size(q)}_kv{_format_size(kv)}"
        scenario_name = f"{prefix}_b{batch}_q{_format_size(q)}_kv{_format_size(kv)}"

        scenarios[scenario_name] = {
            "name": f"{prefix.capitalize()} B{batch} Q{_format_size(q).upper()} KV{_format_size(kv).upper()}",
            "description": f"Batch {batch}, {_format_size(q)} Q, {_format_size(kv)} KV",
            "variants_to_run": [{variant_name: batch}]
        }

    return variants, scenarios


def _generate_mixed_combinations(config: dict) -> Tuple[Dict, Dict]:
    """Generate mixed decode+prefill combinations.

    Creates scenarios with 1 decode request + 1 prefill request.
    Cross product of all decode and prefill parameter combinations.
    """
    decode_params = config["decode_params"]
    prefill_params = config["prefill_params"]

    # Extract parameters
    decode_batches = decode_params["batch_sizes"]
    decode_kvs = decode_params["kv_tokens"]

    prefill_qs = prefill_params["q_tokens"]
    prefill_kvs = prefill_params["kv_tokens"]

    variants = {}
    scenarios = {}

    # Generate decode variants (q=1 always)
    for kv in decode_kvs:
        variant_name = f"decode_q1_kv{_format_size(kv)}"
        variants[variant_name] = {
            "q_tokens": 1,
            "kv_tokens": kv,
            "description": f"Decode (1 q, {_format_size(kv)} kv)"
        }

    # Generate prefill variants
    for q, kv in product(prefill_qs, prefill_kvs):
        variant_name = f"prefill_q{_format_size(q)}_kv{_format_size(kv)}"
        variants[variant_name] = {
            "q_tokens": q,
            "kv_tokens": kv,
            "description": f"Prefill ({_format_size(q)} q, {_format_size(kv)} kv)"
        }

    # Generate mixed scenarios (decode × prefill combinations)
    for dec_batch, dec_kv, pref_q, pref_kv in product(decode_batches, decode_kvs, prefill_qs, prefill_kvs):
        decode_variant = f"decode_q1_kv{_format_size(dec_kv)}"
        prefill_variant = f"prefill_q{_format_size(pref_q)}_kv{_format_size(pref_kv)}"

        scenario_name = f"mixed_dec_b{dec_batch}_kv{_format_size(dec_kv)}_pref_q{_format_size(pref_q)}_kv{_format_size(pref_kv)}"

        scenarios[scenario_name] = {
            "name": f"Mixed: {dec_batch}D({_format_size(dec_kv)}) + 1P({_format_size(pref_q)}q,{_format_size(pref_kv)}kv)",
            "description": f"{dec_batch} decodes ({_format_size(dec_kv)} KV) + 1 prefill ({_format_size(pref_q)} Q, {_format_size(pref_kv)} KV)",
            "variants_to_run": [
                {decode_variant: dec_batch},
                {prefill_variant: 1}
            ]
        }

    return variants, scenarios


def _generate_varying_kv(config: dict) -> Tuple[Dict, Dict]:
    """Generate scenarios with fixed batch/q and varying KV.

    Used for studying KV cache scaling with fixed batch size.
    """
    params = config["parameters"]

    batch_size = params["batch_sizes"][0]  # Should be single value
    q_tokens = params["q_tokens"][0]  # Should be single value
    kv_tokens_list = params["kv_tokens"]

    workload_type = "decode" if q_tokens == 1 else "prefill"
    prefix = workload_type

    variants = {}
    scenarios = {}

    # Generate variants
    for kv in kv_tokens_list:
        variant_name = f"{prefix}_q{_format_size(q_tokens)}_kv{_format_size(kv)}"
        variants[variant_name] = {
            "q_tokens": q_tokens,
            "kv_tokens": kv,
            "description": f"{prefix.capitalize()} ({_format_size(q_tokens)} q, {_format_size(kv)} kv)"
        }

    # Generate scenarios
    for kv in kv_tokens_list:
        variant_name = f"{prefix}_q{_format_size(q_tokens)}_kv{_format_size(kv)}"
        scenario_name = f"{prefix}_{batch_size}_kv{_format_size(kv)}"

        scenarios[scenario_name] = {
            "name": f"{prefix.capitalize()} B{batch_size} KV{_format_size(kv).upper()}",
            "description": f"Batch {batch_size} with {_format_size(kv)} KV cache",
            "variants_to_run": [{variant_name: batch_size}]
        }

    return variants, scenarios


def _generate_sweep_seqlen(config: dict) -> Tuple[Dict, Dict]:
    """Generate scenarios sweeping sequence length (where q = kv).

    Used for studying scaling with sequence length in prefill.
    """
    params = config["parameters"]

    batch_size = params["batch_sizes"][0]  # Should be single value
    seq_lengths = params["seq_lengths"]  # q = kv values

    variants = {}
    scenarios = {}

    # Generate variants (q = kv for each)
    for seq_len in seq_lengths:
        variant_name = f"prefill_q{_format_size(seq_len)}_kv{_format_size(seq_len)}"
        variants[variant_name] = {
            "q_tokens": seq_len,
            "kv_tokens": seq_len,
            "description": f"Prefill ({_format_size(seq_len)} tokens)"
        }

    # Generate scenarios
    for seq_len in seq_lengths:
        variant_name = f"prefill_q{_format_size(seq_len)}_kv{_format_size(seq_len)}"
        scenario_name = f"prefill_b{batch_size}_seq{_format_size(seq_len)}"

        scenarios[scenario_name] = {
            "name": f"Prefill B{batch_size} Seq{_format_size(seq_len).upper()}",
            "description": f"Batch {batch_size} with {_format_size(seq_len)} token sequences",
            "variants_to_run": [{variant_name: batch_size}]
        }

    return variants, scenarios


def _format_size(num_tokens: int) -> str:
    """Convert token count to short string (e.g., 1024 -> '1k', 1048576 -> '1M').

    Args:
        num_tokens: Number of tokens

    Returns:
        Human-readable short form (lowercase for consistency)
    """
    if num_tokens >= 1024 * 1024:
        return f"{num_tokens // (1024 * 1024)}M"
    elif num_tokens >= 1024:
        return f"{num_tokens // 1024}k"
    else:
        return str(num_tokens)
