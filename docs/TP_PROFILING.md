# Tensor Parallelism (TP) Profiling

## Overview

This benchmark supports profiling **single-GPU performance** for different TP configurations. This is NOT multi-GPU execution - we simulate what each GPU would compute in a TP setup.

## What TP Profiling Means

### TP=4 Example (Llama 8B: 32 qo_heads, 8 kv_heads)

**What we do:**
- Profile **1 GPU** with **1/4 of the heads** (8 qo_heads, 2 kv_heads)
- Measure kernel execution time for this reduced head count
- Simulate what each GPU computes in a 4-way TP setup
- **NO multi-GPU communication** (no all-reduce, no inter-GPU data transfer)

**What we DON'T do:**
- Run the same kernel across 4 GPUs simultaneously
- Measure TP communication overhead
- Perform actual tensor parallelism with multiple GPUs

## Why This Matters

In real TP deployments:
- Each GPU stores **1/TP** of the model weights and KV cache
- Each GPU computes on **1/TP** of the heads
- GPUs communicate via all-reduce after attention computation

Our profiling measures the **per-GPU compute time**, which is the dominant factor in TP performance.

## Memory Scaling with TP

For a scenario with large KV cache:

| TP Degree | Heads per GPU | Memory per GPU | Notes |
|-----------|---------------|----------------|-------|
| TP=1 | 8 KV heads | 19.93 GB | Full model on 1 GPU |
| TP=2 | 4 KV heads | 10.11 GB | ~2x memory reduction |
| TP=4 | 2 KV heads | 5.20 GB | ~4x memory reduction |
| TP=8 | 1 KV head | 2.89 GB | ~7x memory reduction |

Components that scale:
- ✅ **KV cache**: Divided by TP (largest component)
- ✅ **Q tensors**: Divided by TP
- ✅ **Output tensors**: Divided by TP
- ❌ **Workspace buffers**: Fixed per-GPU (256MB)

## Configuration

```yaml
models:
  llama_8b:
    num_qo_heads: 32  # Total heads across all GPUs
    num_kv_heads: 8   # Total KV heads across all GPUs

# Profile these TP configurations (single-GPU each)
tp_degrees: [1, 2, 4, 8]
```

## Command Line Usage

```bash
# Run benchmarks with 4 GPUs for parallel execution
# Each TP configuration still runs on 1 GPU with divided heads
python3 src/attention_bench/cli/run_ray.py \
  --config configs/mixed/config_mixed_multi_model_tp.yaml \
  --num-gpus 4

# --num-gpus controls parallel scenario execution (NOT TP execution!)
# TP degrees in config control head division for profiling
```

## Two Separate Concepts

### 1. TP Degree (in config)
- Controls **head division** for profiling
- Each TP configuration is a separate benchmark
- All run on **single GPUs** (just with different head counts)

### 2. --num-gpus (command line)
- Controls **parallel execution** for faster benchmarking
- Runs **different scenarios** simultaneously
- Each scenario still runs on **1 GPU**

## Output

Separate result files per (model, TP) combination:
```
results/run_2025-11-20_14-26-29/
├── config_mixed_all_combinations_llama8b_tp1.json
├── config_mixed_all_combinations_llama8b_tp2.json
├── config_mixed_all_combinations_llama8b_tp4.json
├── config_mixed_all_combinations_llama8b_tp8.json
├── config_mixed_all_combinations_llama70b_tp1.json
├── config_mixed_all_combinations_llama70b_tp2.json
└── ...
```

Each file contains single-GPU kernel timings for that TP configuration.

## Use Cases

This profiling data helps answer:
1. How does per-GPU compute time change with TP degree?
2. Which attention approach is best for each TP configuration?
3. What's the memory footprint per GPU at different TP degrees?
4. Can we run larger batch sizes or longer contexts with higher TP?

For end-to-end TP performance, you'd need to add:
- Communication overhead (all-reduce, etc.)
- Pipeline parallelism if used
- Load balancing across GPUs
