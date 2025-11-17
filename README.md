# FlashInfer Heterogeneity Experiment

## Overview

Unified benchmark comparing FlashInfer (FA2, FA3, cuDNN backends) and Official FlashAttention-3 for heterogeneous batch workloads.

## Usage

### Basic Usage

```bash
# Run with specific config
python run_benchmark.py --config config_decode_all_combinations.yaml

# Override approaches from command line
python run_benchmark.py --approaches official_fa3,flashinfer_mixed_fa3

# Enable CUDA graphs
python run_benchmark.py --use-cuda-graphs
```

### Available Approaches

**FlashInfer:**
- `flashinfer_mixed_fa2` - Prefill wrapper with FA2 backend for all requests
- `flashinfer_mixed_fa3` - Prefill wrapper with FA3 backend for all requests
- `flashinfer_mixed_cudnn` - Prefill wrapper with cuDNN backend for all requests
- `flashinfer_separated_fa2` - Separate decode/prefill wrappers (FA2 prefill)
- `flashinfer_separated_fa3` - Separate decode/prefill wrappers (FA3 prefill)
- `flashinfer_batch_attention` - BatchAttention unified wrapper

**Official:**
- `official_fa3` - Official FlashAttention-3 library

### Available Configs

- `config_decode_all_combinations.yaml` - Comprehensive decode workload sweep
- `config_decode_varying_batch.yaml` - Vary batch size for decode
- `config_decode_varying_kv.yaml` - Vary KV cache length for decode
- `config_prefill_all_combinations.yaml` - Comprehensive prefill workload sweep
- `config_prefill_varying_batch.yaml` - Vary batch size for prefill
- `config_prefill_varying_seqlen.yaml` - Vary sequence length for prefill
- `config_mixed_all_combinations.yaml` - Mixed decode + prefill workloads

### Output

- `results/tolerance_clean_eager_TIMESTAMP.json` - Raw benchmark data
- `plots/comparison_CONFIGNAME.png` - Performance comparison plot

## Configuration

### YAML Format

```yaml
model:
  num_qo_heads: 32
  num_kv_heads: 8
  head_dim: 128
  page_size: 256  # Global default
  workspace_size: 536870912

profiling:
  num_warmup_iters: 5
  num_active_iters: 50

output:
  results_dir: "results"
  plots_dir: "plots"

# Which approaches to benchmark
approaches:
  - flashinfer_mixed_fa2
  - flashinfer_mixed_fa3
  - flashinfer_mixed_cudnn
  - official_fa3

# Optional: per-approach overrides
approach_overrides:
  flashinfer_mixed_fa2:
    page_size: 16

variants:
  decode_q1_kv2k:
    q_tokens: 1
    kv_tokens: 2048
    description: "Decode (1 q, 2K kv)"

scenarios:
  scenario_name:
    variants_to_run:
      - decode_q1_kv2k: 128
```

## Repository Structure

```
.
├── run_benchmark.py                   # Main benchmark tool
├── approaches/                        # Approach implementations
│   ├── __init__.py
│   ├── base.py                       # Protocol + Registry
│   ├── flashinfer_approaches.py      # FlashInfer implementations
│   └── official_fa3_approaches.py    # Official FA3 implementation
├── config_*.yaml                      # Benchmark configurations
├── results/                           # Benchmark results (JSON)
├── plots/                             # Generated plots
└── archived/                          # Old/temporary files
```

## Requirements

- PyTorch with CUDA support
- FlashInfer (v0.5.1 or later)
- flash-attn (v2.8.3 or later) - for Official FA3
- NVIDIA H100 GPU (for FA3/Hopper features)
- Python 3.8+
- matplotlib, numpy, PyYAML

## Installation

```bash
pip install torch flashinfer-python flash-attn matplotlib numpy pyyaml
```

## Adding New Approaches

To add a new approach:

1. Create a class in `approaches/flashinfer_approaches.py` or create a new file
2. Implement the `AttentionApproach` protocol:
   - `name: str` - Unique identifier
   - `default_page_size: int` - Default page size
   - `benchmark(q_lengths, kv_lengths, runner, config) -> float` - Benchmark method
3. Register with `@APPROACHES.register` decorator
4. Add to config YAML `approaches` list

Example:
```python
@APPROACHES.register
class MyNewApproach:
    name = "my_new_approach"
    default_page_size = 256

    def benchmark(self, q_lengths, kv_lengths, runner, config):
        # Implementation
        return time_in_ms
```
