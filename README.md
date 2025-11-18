# FlashInfer Heterogeneity Experiment

## Overview

Unified benchmark comparing FlashInfer (FA2, FA3, cuDNN backends) and Official FlashAttention-3 for heterogeneous batch workloads.

## Performance Analysis

We evaluated all attention kernel approaches across 2480 different workload scenarios to determine the best strategy for production use.

### Recommendation: Use Official FA3 for All Workloads

**TL;DR:** Always use Official FA3. It provides the best overall performance with minimal complexity.

### Official FA3: Overall Strategy Performance

Using FA3 for **all scenarios** (vs perfect per-scenario selection):

- **Win Rate:** 65.4% (1621/2480 scenarios where FA3 is optimal)
- **p50 (Median):** 1.000x slowdown (0.0% slower)
- **p90:** 1.385x slowdown (38.5% slower)
- **Average:** 1.131x slowdown (13.1% slower)

**Interpretation:** In 50% of scenarios, FA3 is optimal or near-optimal (≤0.0% slower). In 90% of scenarios, performance loss is ≤38.5%.

### When FA3 Wins (65.4% of scenarios)

FA3 is optimal in these scenarios:
- Large prefill workloads (prefill_kv > 1536)
- Mixed decode/prefill workloads
- Most decode-only scenarios with small-to-medium batch sizes

**Performance margin when FA3 wins:**
- **p50:** 1.11x faster than runner-up
- **p90:** 1.46x faster than runner-up
- **Average:** 1.19x faster than runner-up

### When FA3 Loses (859 scenarios, 34.6%)

FA3 is suboptimal in 859 scenarios, but the performance penalty is typically small:

| Slowdown Range | Count | Percentage |
|----------------|-------|------------|
| 1.0-1.1x (negligible) | 199 | 8.0% |
| 1.1-1.2x (minor) | 190 | 7.7% |
| 1.2-1.5x (moderate) | 320 | 12.9% |
| 1.5-2.0x (significant) | 92 | 3.7% |
| 2.0x+ (severe) | 58 | 2.3% |

**Alternative winners when FA3 loses:**
- flashinfer_batch_attention: 555 scenarios (64.6% of FA3 losses)
- flashinfer_mixed_fa2: 157 scenarios (18.3% of FA3 losses)
- flashinfer_separated_fa3: 134 scenarios (15.6% of FA3 losses)
- flashinfer_mixed_fa3: 10 scenarios (1.2% of FA3 losses)
- flashinfer_separated_fa2: 3 scenarios (0.3% of FA3 losses)

**Worst-case scenarios** where FA3 is slowest:
1. `mixed_dec_b1_kv32k_pref_q512_kv512`: 7.42x slower (winner: flashinfer_batch_attention)
2. `mixed_dec_b1_kv32k_pref_q512_kv1k`: 5.43x slower (winner: flashinfer_separated_fa3)
3. `mixed_dec_b1_kv16k_pref_q512_kv512`: 4.97x slower (winner: flashinfer_batch_attention)
4. `mixed_dec_b2_kv32k_pref_q512_kv512`: 4.92x slower (winner: flashinfer_batch_attention)
5. `mixed_dec_b1_kv32k_pref_q1k_kv1k`: 4.56x slower (winner: flashinfer_separated_fa3)

### Why Not BatchAttention?

Some might consider using BatchAttention always, but this performs significantly worse:

**Overall Strategy (using BatchAttention for everything):**
- **Win Rate:** 22.4% (555/2478 scenarios)
- **p50 (Median):** 1.434x slowdown (43.4% slower)
- **p90:** 1.846x slowdown (84.6% slower)
- **Average:** 1.413x slowdown (41.3% slower)

**When BatchAttention wins (22.4% of scenarios):**
- **p50:** 1.13x faster than runner-up
- **p90:** 1.33x faster than runner-up

BatchAttention severely underperforms on many workloads:

- **1 scenario** where BA is >3x slower
- **110 scenarios** where BA is 2-3x slower
- **Total: 111 scenarios** with >2x slowdown (4.5%)

### Comparison Summary

| Strategy | Win Rate | p50 Slowdown | p90 Slowdown | Avg Perf Loss | Complexity |
|----------|----------|--------------|--------------|---------------|------------|
| **Official FA3 (Recommended)** | **65.4%** | **1.000x** | **1.385x** | **13.1%** | **Simple** |
| BatchAttention | 22.4% | 1.434x | 1.846x | 41.3% | Simple |
| Perfect Selection | 100.0% | 1.000x | 1.000x | 0.0% | Complex |

### Decision Tree Results

We also trained a decision tree classifier to predict the optimal approach:
- **Cross-validation accuracy:** 73.1%
- **Training accuracy:** 85.8%
- **Most important features:** prefill_kv (59.6%), total_decode_tokens (22.9%)

While the decision tree achieves reasonable accuracy, the complexity is not justified:
- Requires 3-level decision tree with multiple feature checks
- Only improves from 13.1% to ~8-9% average performance loss
- Adds implementation complexity and maintenance burden

**For production systems, we recommend simply using Official FA3 for all workloads.**

### Analysis Files

- `decision_tree_results/` - Decision tree analysis and visualizations
- `learn_decision_tree.py` - Decision tree training script
- `archived/analysis_scripts/` - Analysis scripts used to generate this report

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
