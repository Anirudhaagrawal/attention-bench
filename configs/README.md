# FlashInfer Benchmark Configurations

This directory contains YAML configurations for benchmarking FlashInfer attention implementations across different workload types, models, and TP degrees.

## Directory Structure

```
configs/
├── decode/                    # Decode (q_len=1) workloads
├── prefill/                   # Prefill (q_len>1) workloads
└── mixed/                     # Mixed (decode+prefill) workloads
```

## Configuration Types

### 1. Multi-Model/TP Configurations (Recommended)

Support multiple models and TP degrees with timestamped output folders.

| Config | Workload | Models | TP Degrees | Scenarios |
|--------|----------|--------|------------|-----------|
| `decode/config_decode_multi_model_tp.yaml` | Pure decode | Llama 8B, 70B | 1, 2, 4, 8 | 9 batch × 12 KV = 108 |
| `prefill/config_prefill_multi_model_tp.yaml` | Pure prefill | Llama 8B, 70B | 1, 2, 4, 8 | 10 Q × 14 KV = 140 |
| `mixed/config_mixed_multi_model_tp.yaml` | Mixed | Llama 8B, 70B | 1, 2, 4, 8 | 9×12 decode × 10×14 prefill |

**Output Format:**
```
results/run_YYYY-MM-DD_HH-MM-SS/
├── config_decode_multi_model_tp_llama8b_tp1.json
├── config_decode_multi_model_tp_llama8b_tp2.json
├── config_decode_multi_model_tp_llama8b_tp4.json
├── config_decode_multi_model_tp_llama8b_tp8.json
├── config_decode_multi_model_tp_llama70b_tp1.json
└── ...
```

### 2. Test Configurations

Smaller scenario sets for quick validation.

| Config | Workload | Scenarios | Purpose |
|--------|----------|-----------|---------|
| `decode/config_decode_test_multi_model_tp.yaml` | Decode | 9 | Quick test (3 batch × 3 KV) |
| `prefill/config_prefill_test_multi_model_tp.yaml` | Prefill | 9 | Quick test (3 Q × 3 KV) |
| `mixed/config_test_multi_model_tp.yaml` | Mixed | 4 | Quick test (2 decode × 2 prefill) |

### 3. Legacy Configurations (Single Model, TP=1)

Original configs for single model profiling only.

| Config | Workload | Notes |
|--------|----------|-------|
| `decode/config_decode_all_combinations.yaml` | Decode | Llama 8B, TP=1 only |
| `prefill/config_prefill_all_combinations.yaml` | Prefill | Llama 8B, TP=1 only |
| `mixed/config_mixed_all_combinations.yaml` | Mixed | Llama 8B, TP=1 only |

## Model Configurations

### Llama 8B
- **QO Heads**: 32 total (32 for TP=1, 16 for TP=2, 8 for TP=4, 4 for TP=8)
- **KV Heads**: 8 total (8 for TP=1, 4 for TP=2, 2 for TP=4, 1 for TP=8)
- **GQA Ratio**: 4:1
- **Head Dim**: 128

### Llama 70B
- **QO Heads**: 64 total (64 for TP=1, 32 for TP=2, 16 for TP=4, 8 for TP=8)
- **KV Heads**: 8 total (8 for TP=1, 4 for TP=2, 2 for TP=4, 1 for TP=8)
- **GQA Ratio**: 8:1
- **Head Dim**: 128

## TP Profiling Important Notes

**TP degrees are profiling configurations, NOT multi-GPU execution!**

When you specify `tp_degrees: [1, 2, 4, 8]`:
- Each TP degree is profiled on **a single GPU**
- We measure kernel performance with **divided head counts**
- We do **NOT** run actual multi-GPU TP with communication
- This gives per-GPU compute time in TP setups

Example for TP=4 with Llama 8B:
- Profile 1 GPU with 8 qo_heads, 2 kv_heads (1/4 of total)
- Measure single-GPU kernel time
- Simulates per-GPU workload in 4-way TP

## Usage Examples

### Run Full Benchmarks (All Models, All TP Degrees)

```bash
# Decode workloads across 4 GPUs (parallel execution)
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_multi_model_tp.yaml \
  --num-gpus 4

# Prefill workloads
python3 src/attention_bench/cli/run_ray.py \
  --config configs/prefill/config_prefill_multi_model_tp.yaml \
  --num-gpus 4

# Mixed workloads
python3 src/attention_bench/cli/run_ray.py \
  --config configs/mixed/config_mixed_multi_model_tp.yaml \
  --num-gpus 4
```

### Filter Specific Models or TP Degrees

```bash
# Only Llama 8B
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_multi_model_tp.yaml \
  --models llama_8b \
  --num-gpus 4

# Only TP=1,2
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_multi_model_tp.yaml \
  --tp-degrees 1,2 \
  --num-gpus 4

# Specific model and TP degrees
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_multi_model_tp.yaml \
  --models llama_70b \
  --tp-degrees 4,8 \
  --num-gpus 4
```

### Quick Testing

```bash
# Test decode with 2 GPUs
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_test_multi_model_tp.yaml \
  --num-gpus 2

# Should complete in < 1 minute
```

### Filter Approaches

```bash
# Only test specific approaches
python3 src/attention_bench/cli/run_ray.py \
  --config configs/decode/config_decode_multi_model_tp.yaml \
  --approaches flashinfer_batch_attention,flashinfer_separated_fa3 \
  --num-gpus 4
```

## Memory Considerations

Memory requirements scale with TP degree:

| TP Degree | Llama 8B Memory/GPU | Llama 70B Memory/GPU |
|-----------|---------------------|----------------------|
| TP=1 | ~20 GB (128k ctx, 32 batch) | ~40 GB |
| TP=2 | ~10 GB | ~20 GB |
| TP=4 | ~5 GB | ~10 GB |
| TP=8 | ~3 GB | ~5 GB |

Higher TP degrees allow:
- Larger batch sizes
- Longer context lengths
- Larger models on the same GPU

## Adding New Models

Edit any `*_multi_model_tp.yaml` config:

```yaml
models:
  my_model:
    name: "mymodel"
    num_qo_heads: 40  # Must be divisible by all TP degrees
    num_kv_heads: 10  # Must be divisible by all TP degrees
    head_dim: 128
    page_size: 16
    workspace_size: 536870912
```

Ensure head counts are divisible by your TP degrees!

## Output Organization

All runs create timestamped folders:
```
results/
├── run_2025-11-20_14-26-29/    # Automatic timestamp
│   ├── config_decode_multi_model_tp_llama8b_tp1.json
│   ├── config_decode_multi_model_tp_llama8b_tp2.json
│   └── ...
└── run_2025-11-20_15-30-45/
    └── ...
```

Each JSON file contains:
- Scenario name and parameters
- Timing results per approach
- Statistical data (mean, median, std)
- Winner selection
- Memory information
