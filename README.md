# Attention Kernel Benchmark

## Overview

Comprehensive benchmark framework for comparing attention kernel implementations across heterogeneous batch workloads. Currently supports FlashInfer (FA2, FA3, cuDNN backends) and Official FlashAttention-3.

## Performance Analysis

We evaluated all attention kernel approaches across 2480 different workload scenarios to determine the best strategy for production use.

### Recommendation: Use Official FA3 for All Workloads

**TL;DR:** Always use Official FA3. It provides the best overall performance with minimal complexity.

### Official FA3: Overall Strategy Performance

**What happens when you use FA3 for all workloads?**

We tested FA3 on 2,480 different scenarios and measured how it performs vs always picking the optimal kernel for each scenario.

#### Performance Distribution

| FA3 Performance | Scenarios | % | What This Means |
|-----------------|-----------|---|-----------------|
| **1.0x (optimal)** | 1,621 | 65.4% | FA3 is the fastest choice - no penalty |
| **1.0-1.1x** | 199 | 8.0% | Within 10% of optimal - negligible difference |
| **1.1-1.2x** | 190 | 7.7% | 10-20% slower - minor but acceptable |
| **1.2-1.5x** | 320 | 12.9% | 20-50% slower - moderate slowdown |
| **1.5-2.0x** | 92 | 3.7% | 50-100% slower - significant slowdown |
| **2.0x+** | 58 | 2.3% | >2x slower - severe cases (worst: 7.42x) |

**Cumulative View:**
- ✅ **73.4%** of scenarios: FA3 is optimal or near-optimal (≤10% slower)
- ✅ **81.1%** of scenarios: Within 20% of optimal
- ✅ **94.0%** of scenarios: Within 50% of optimal
- ⚠️ **6.0%** of scenarios: >50% slower than optimal

#### Summary Statistics

- **Median (p50):** 1.000x
  - *Why 1.0x?* Since FA3 wins 65% of scenarios, the middle scenario is one where FA3 is optimal
  - *What it means:* More than half of all workloads run at optimal speed with FA3

- **90th Percentile (p90):** 1.385x
  - *What it means:* 90% of scenarios have ≤38.5% slowdown; only 10% are worse

- **Average:** 1.131x (13.1% slower)
  - *How calculated:* Mean slowdown across all 2,480 scenarios
  - *What it means:* On average, using FA3 everywhere costs 13% performance vs perfect selection

- **Worst Case:** 7.42x slower
  - *When:* Small prefill (512 tokens) + large decode (32K KV, 1 batch)
  - *Frequency:* Very rare (0.04% of scenarios)

#### What the Numbers Mean

- **"1.0x"** = FA3 is the fastest option (no slowdown)
- **"1.2x"** = FA3 takes 20% longer than the optimal kernel
- **"Median 1.0x"** = In a typical scenario, FA3 performs optimally
- **"p90 1.385x"** = 9 out of 10 scenarios have ≤38.5% slowdown

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

Some might consider using BatchAttention always, but this performs significantly worse.

**What happens when you use BatchAttention for all workloads?**

We tested the same 2,480 scenarios using BatchAttention everywhere:

#### Performance Distribution

| BA Performance | Scenarios | % | What This Means |
|----------------|-----------|---|-----------------|
| **1.0x (optimal)** | 555 | 22.4% | BA is the fastest choice - no penalty |
| **1.0-1.2x** | 371 | 15.0% | Within 20% of optimal - acceptable |
| **1.2-1.5x** | 858 | 34.6% | 20-50% slower - moderate slowdown |
| **1.5-2.0x** | 583 | 23.5% | 50-100% slower - significant slowdown |
| **2.0x+** | 111 | 4.5% | >2x slower - severe cases (worst: 3.07x) |

**Key Observations:**
- ❌ Only **22.4%** of scenarios: BA is optimal
- ❌ Only **37.4%** of scenarios: Within 20% of optimal
- ⚠️ **62.6%** of scenarios: >20% slower than optimal
- ⚠️ **28.0%** of scenarios: >50% slower than optimal

#### Summary Statistics

- **Median (p50):** 1.434x (43.4% slower)
  - *Why so high?* BA only wins 22% of scenarios, so the median falls in the "loss" region
  - *What it means:* A typical workload runs 43% slower with BA than optimal

- **90th Percentile (p90):** 1.846x (84.6% slower)
  - *What it means:* 90% of scenarios are within 85% of optimal; 10% are even worse

- **Average:** 1.413x (41.3% slower)
  - *Comparison:* **3.2x worse** than FA3's 13.1% average loss

**When BatchAttention wins (22% of scenarios):**
- Wins by 1.13-1.33x vs runner-up (smaller margin than FA3's 1.11-1.46x)
- Primarily wins on: small prefills with large decode batches (≥64)

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

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd flashinfer-heterogeneity-experiment

# Install in development mode
pip install -e .

# Verify installation
attention-bench --help
attention-bench-ray --help
```

## Usage

### Quick Start

```bash
# Run single-GPU benchmark with specific config
attention-bench --config configs/decode/config_decode_all_combinations.yaml

# Run multi-GPU distributed benchmark with Ray
attention-bench-ray --config configs/mixed/config_mixed_all_combinations.yaml --num-gpus 4

# Override approaches from command line
attention-bench --config configs/prefill/config_prefill_all_combinations.yaml \
    --approaches official_fa3,flashinfer_mixed_fa3

# Enable CUDA graphs
attention-bench --config configs/decode/config_decode_all_combinations.yaml --use-cuda-graphs
```

### Single-GPU Benchmarking

The `attention-bench` command runs benchmarks sequentially on a single GPU:

```bash
attention-bench \
    --config configs/mixed/config_mixed_all_combinations.yaml \
    --approaches official_fa3,flashinfer_separated_fa3 \
    --use-cuda-graphs
```

**Options:**
- `--config CONFIG`: Path to YAML configuration file (required)
- `--approaches APPROACHES`: Comma-separated list of approaches to benchmark
- `--use-cuda-graphs`: Enable CUDA graphs for better performance
- `--enable-profiling`: Enable profiling with PyTorch profiler
- `--gpu GPU`: Specific GPU device to use (for parallel execution)
- `--scenario-range RANGE`: Range of scenarios to run, e.g., '0-24' (for parallel execution)

### Multi-GPU Distributed Benchmarking

The `attention-bench-ray` command uses Ray for distributed execution across multiple GPUs:

```bash
attention-bench-ray \
    --config configs/mixed/config_mixed_all_combinations.yaml \
    --num-gpus 4 \
    --memory-utilization 0.85 \
    --output-dir results/distributed
```

**Benefits:**
- **GPU isolation**: Each worker gets exclusive access to one GPU via Ray's resource management
- **Memory safety**: Pre-filters scenarios by memory estimation to prevent OOM
- **Batched execution**: Processes scenarios in batches with sync points for cleanup
- **Fault tolerance**: Explicit worker cleanup and error handling

**Options:**
- `--config CONFIG`: Path to YAML configuration file (required)
- `--num-gpus NUM_GPUS`: Number of GPUs to use for distributed execution
- `--memory-utilization THRESHOLD`: GPU memory utilization threshold (default: 0.90)
- `--memory-overhead MULTIPLIER`: Memory overhead multiplier for estimation (default: 1.1)
- `--approaches APPROACHES`: Comma-separated list of approaches to benchmark
- `--output-dir DIR`: Output directory for results (default: results/)

**Memory Estimation:**
The Ray orchestrator pre-filters scenarios based on available GPU memory:
```bash
# Be more conservative with memory (80% utilization)
attention-bench-ray --config configs/mixed/config_mixed_all_combinations.yaml \
    --num-gpus 2 \
    --memory-utilization 0.80 \
    --memory-overhead 1.2
```

### Running All Configs (Batch Execution)

The `scripts/run_all_configs.sh` script runs benchmarks for all configuration files automatically and generates timestamped logs:

```bash
# Run all configs sequentially on 1 GPU (slower but safe)
./scripts/run_all_configs.sh

# Run all configs in parallel on 4 GPUs (faster)
./scripts/run_all_configs.sh "" 4

# Override approaches for all configs
./scripts/run_all_configs.sh "official_fa3,flashinfer_separated_fa3" 4

# Run only specific config type
./scripts/run_all_configs.sh "" 4 decode   # Only decode configs
./scripts/run_all_configs.sh "" 4 prefill  # Only prefill configs
./scripts/run_all_configs.sh "" 4 mixed    # Only mixed configs
```

**Script Parameters:**
1. **Approaches** (optional): Comma-separated list of approaches to override config defaults
2. **Number of GPUs** (optional, default: 1): Number of GPUs for parallel execution
3. **Config type** (optional, default: all): Filter configs by type (decode/prefill/mixed/all)

**Output:**

The script creates a timestamped directory in `logs/` with individual log files for each config:

```
logs/20241119_230000/
├── config_decode_all_combinations.log
├── config_prefill_all_combinations.log
├── config_mixed_all_combinations.log
└── summary.txt
```

**Summary file** (`summary.txt`) contains:
- Timestamp and configuration
- Success/failure counts
- List of failed configs (if any)
- List of all log files

**Example output:**
```bash
$ ./scripts/run_all_configs.sh "" 4 mixed
Attention Benchmark Suite - Running All Configs
================================================================
Approaches: Using approaches from config files
Execution mode: Parallel (4 GPUs per config via Ray)
Config type: mixed
Logs directory: logs/20241119_230000

Found 2 config file(s):
  - configs/mixed/config_mixed_all_combinations.yaml
  - configs/mixed/config_test_ray.yaml

[1/2] Running: configs/mixed/config_mixed_all_combinations.yaml
Command: attention-bench-ray --config configs/mixed/config_mixed_all_combinations.yaml --num-gpus 4
...
✓ SUCCESS: config_mixed_all_combinations (1234s)

[2/2] Running: configs/mixed/config_test_ray.yaml
...
✓ SUCCESS: config_test_ray (45s)

Summary
================================================================
Successful: 2/2
Failed: 0/2

Logs directory: logs/20241119_230000
Summary file: logs/20241119_230000/summary.txt
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

### Testing Correctness

Verify that all attention approaches produce identical outputs:

```bash
# Run output correctness test (tests all 6 approaches)
python test_correctness.py

# Test compares:
# - flashinfer_separated_fa2
# - flashinfer_separated_fa3
# - flashinfer_mixed_fa2
# - flashinfer_mixed_fa3
# - flashinfer_batch_attention
# - official_fa3
```

**Output validation is also automatically enabled during benchmarks** when `enable_output_validation=True` in the config. This compares all approach outputs using `torch.allclose(rtol=1e-3, atol=1e-3)`.

### Profiling

Enable detailed performance profiling using PyTorch's profiler:

```bash
# Profile a single-GPU benchmark run
attention-bench \
    --config configs/decode/config_decode_all_combinations.yaml \
    --enable-profiling

# Profile distributed Ray execution
attention-bench-ray \
    --config configs/mixed/config_mixed_all_combinations.yaml \
    --num-gpus 4 \
    --enable-profiling
```

**Profiling Configuration:**

Control profiling behavior in your YAML config:

```yaml
profiling:
  num_warmup_iters: 5      # Warmup iterations (not profiled)
  num_active_iters: 50     # Active iterations (profiled and timed)
  enable_profiling: false  # Enable PyTorch profiler (default: false)
```

**Output:**

Profiler traces are saved to `profiler_traces/` directory:
- Chrome trace format (`.json`) for visualization in `chrome://tracing`
- Per-approach traces showing CUDA kernel timing, memory operations, and CPU overhead

**Viewing Traces:**

1. Open Chrome/Chromium browser
2. Navigate to `chrome://tracing`
3. Load the JSON trace file from `profiler_traces/`
4. Analyze kernel execution timeline, memory transfers, and performance bottlenecks

**Notes:**
- Profiling adds overhead; use separate runs for accurate performance benchmarks
- Traces can be large (100MB+) for complex scenarios
- Only available for single-GPU benchmarks (not yet supported in Ray distributed mode)

### Visualization and Analysis

Generate heatmaps from benchmark results:

```bash
# === BASIC USAGE ===

# Auto-detect all approaches in results file
# - 2 approaches found → pairwise speedup heatmap (diverging colormap)
# - 3+ approaches found → best performer heatmap (categorical colors showing winner)
python src/attention_bench/plotting/heatmap_mixed.py \
    --input results/config_mixed_all_combinations_ray.json

# Specify which approaches to compare (2 = pairwise mode)
python src/attention_bench/plotting/heatmap_decode.py \
    --input results/config_decode_all_combinations.json \
    --approaches official_fa3,flashinfer_batch_attention

# Best performer mode with 6 approaches (auto-switches to categorical visualization)
python src/attention_bench/plotting/heatmap_prefill.py \
    --input results/config_prefill_all_combinations.json

# === MULTI-MODEL / TP SUPPORT ===

# Generate heatmaps for specific model and TP degree
python src/attention_bench/plotting/heatmap_mixed.py \
    --run-dir results/run_2025-11-20_16-15-39 \
    --model llama8b \
    --tp-degree 4

# Compare specific approaches for llama70b with TP=8
python src/attention_bench/plotting/heatmap_decode.py \
    --run-dir results/run_2025-11-20_16-15-39 \
    --model llama70b \
    --tp-degree 8 \
    --approaches flashinfer_mixed_fa3,flashinfer_separated_fa3

# === ADVANCED OPTIONS ===

# Dark mode
python src/attention_bench/plotting/heatmap_mixed.py \
    --input results/config_mixed_all_combinations_ray.json \
    --dark

# Custom output location
python src/attention_bench/plotting/heatmap_decode.py \
    --input results/config_decode_all_combinations.json \
    --output plots/custom/decode_heatmap.png

# Filter by workload category
python src/attention_bench/plotting/heatmap_prefill.py \
    --input results/config_prefill_all_combinations.json \
    --workload-category code
```

**Heatmap Modes:**
- **Pairwise Mode** (2 approaches): Shows speedup ratio with diverging colormap (green=faster, red=slower)
- **Best Performer Mode** (3+ approaches): Shows which approach is fastest in each cell with categorical colors
  - Cell displays: Winner name, 1st place time, 2nd place time, speedup margin
  - Color intensity indicates margin of victory

**Multi-Model/TP Results:**
When using `--run-dir`, the heatmap script automatically reads results from files matching the pattern:
`{run_dir}/config_{type}_multi_model_tp_{model}_tp{degree}.json`

Train decision tree for approach selection:

```bash
python -m attention_bench.analysis.decision_tree \
    --results-dir results/ \
    --output-dir decision_tree_results/
```

### Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration of benchmark results:

```bash
# Install dashboard dependencies
pip install -r flashinfer_dashboard/requirements.txt

# Run the dashboard
cd flashinfer_dashboard
streamlit run app.py
```

**Dashboard Features:**

- **Overview**: Summary metrics, quick navigation, recent runs
- **Heatmaps**: Interactive Plotly speedup heatmaps with zoom/pan/hover
- **Performance**: Sortable tables, bar charts, CSV export
- **Explorer**: Search scenarios, browse raw data, statistics
- **Workloads**: Analysis by category (code/chat/summarization) with recommendations

**Dashboard Structure:**

```
flashinfer_dashboard/
├── app.py                    # Main overview page
├── pages/
│   ├── 1_Heatmaps.py        # Interactive speedup heatmaps
│   ├── 2_Performance.py      # Tables & comparison charts
│   ├── 3_Explorer.py         # Raw data browser
│   └── 4_Workloads.py        # Category analysis
├── utils/
│   ├── data_loader.py        # Load & cache JSON results
│   ├── filters.py            # Sidebar filter components
│   ├── visualizations.py     # Plotly chart builders
│   └── export.py             # CSV/report export
└── requirements.txt
```

The dashboard automatically loads results from `results/run_*/` directories and provides:

- **Real-time filtering** without regenerating plots
- **Side-by-side comparisons** across models, TP degrees, and approaches
- **Export capabilities** (CSV, JSON, Markdown reports)
- **Workload categorization** (code, chat, summarization) with optimal approach recommendations

### Available Configs

Configs are organized by workload type in `configs/`:

**Decode Workloads** (`configs/decode/`):
- `config_decode_all_combinations.yaml` - Comprehensive decode workload sweep
- `config_decode_multi_model_tp.yaml` - Multi-model (Llama 8B, 70B) with TP profiling [1,2,4,8]

**Prefill Workloads** (`configs/prefill/`):
- `config_prefill_all_combinations.yaml` - Comprehensive prefill workload sweep
- `config_prefill_multi_model_tp.yaml` - Multi-model with TP profiling

**Mixed Workloads** (`configs/mixed/`):
- `config_mixed_all_combinations.yaml` - Mixed decode + prefill workloads
- `config_mixed_multi_model_tp.yaml` - Multi-model with TP profiling
- `config_test_ray.yaml` - Small test config for Ray validation

**Multi-Model/TP Configs:**

The `*_multi_model_tp.yaml` configs benchmark multiple models (Llama 8B, Llama 70B) at different tensor parallelism (TP) degrees [1, 2, 4, 8]:

```bash
# Run multi-model/TP benchmarks with Ray
attention-bench-ray --config configs/decode/config_decode_multi_model_tp.yaml --num-gpus 4
```

**Important:** TP profiling simulates per-GPU workload in a TP setup (e.g., TP=4 means 1/4 of the heads per GPU). This measures single-GPU kernel performance with reduced head counts - **not** actual multi-GPU distributed execution with communication overhead.

**Results:** Saved per model and TP degree:
```
results/run_2025-11-20_16-15-39/
├── config_decode_multi_model_tp_llama8b_tp1.json
├── config_decode_multi_model_tp_llama8b_tp2.json
├── config_decode_multi_model_tp_llama8b_tp4.json
├── config_decode_multi_model_tp_llama70b_tp8.json
└── ...
```

### Output

Benchmark results are saved to:
- `results/<scenario_type>/*.json` - Raw benchmark data with timing statistics
- `plots/<scenario_type>/*.png` - Performance comparison visualizations

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
attention-bench/
├── bin/                               # Executable entry points
│   ├── attention-bench                # Single-GPU benchmark CLI
│   └── attention-bench-ray            # Multi-GPU Ray orchestrator
├── configs/                           # Benchmark configurations
│   ├── decode/                        # Decode-only workloads
│   ├── prefill/                       # Prefill-only workloads
│   └── mixed/                         # Mixed decode+prefill workloads
├── data/                              # Generated data (gitignored)
│   ├── results/                       # Benchmark results (JSON)
│   ├── plots/                         # Generated visualizations
│   ├── logs/                          # Execution logs
│   └── profiler_traces/               # PyTorch profiler traces
├── docs/                              # Documentation
├── flashinfer_dashboard/              # Interactive Streamlit dashboard
│   ├── app.py                         # Main overview page
│   ├── pages/                         # Dashboard pages
│   │   ├── 1_Heatmaps.py             # Interactive heatmaps
│   │   ├── 2_Performance.py          # Tables & charts
│   │   ├── 3_Explorer.py             # Data browser
│   │   └── 4_Workloads.py            # Category analysis
│   ├── utils/                         # Dashboard utilities
│   └── requirements.txt               # Dashboard dependencies
├── scripts/                           # Utility scripts
│   ├── plot_by_workload.py           # Filtered heatmap generation
│   └── run_all_configs.sh            # Batch execution script
├── src/
│   └── attention_bench/               # Main Python package
│       ├── approaches/                # Kernel implementations
│       │   ├── base.py               # Protocol + Registry
│       │   ├── flashinfer_approaches.py
│       │   └── official_fa3_approaches.py
│       ├── timing/                    # Timing utilities
│       │   ├── benchmark.py          # Benchmark runner
│       │   └── context.py            # Shared context
│       ├── plotting/                  # Visualization tools
│       │   ├── heatmap_decode.py
│       │   ├── heatmap_prefill.py
│       │   └── heatmap_mixed.py
│       ├── config/                    # Configuration tools
│       │   └── generator.py          # Scenario generator
│       ├── distributed/               # Ray distributed execution
│       │   └── worker.py             # Ray benchmark worker
│       ├── cli/                       # Command-line interfaces
│       │   ├── run_benchmark.py      # Single-GPU CLI
│       │   └── run_ray.py            # Multi-GPU Ray CLI
│       ├── analysis/                  # Analysis tools
│       │   └── decision_tree.py      # Decision tree training
│       └── utils/                     # Utility functions
│           └── memory_estimator.py   # Memory estimation
├── tests/                             # Unit tests
│   ├── test_approaches_work.py
│   └── test_correctness.py
├── setup.py                           # Package configuration
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## Requirements

**System Requirements:**
- NVIDIA GPU with CUDA support (H100 recommended for FA3/Hopper features)
- Python 3.8+
- CUDA 11.8+ or 12.x

**Python Dependencies:**
- `torch>=2.0.0` - PyTorch with CUDA support
- `flashinfer>=0.1.0` - FlashInfer library
- `flash-attn>=2.8.3` - Official FlashAttention-3 (optional)
- `ray>=2.0.0` - Ray for distributed execution
- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `pandas>=1.3.0` - Data analysis
- `pyyaml>=5.4` - YAML configuration parsing
- `scikit-learn>=0.24.0` - Decision tree analysis
- `streamlit>=1.28.0` - Interactive dashboard (optional)
- `plotly>=5.17.0` - Interactive visualizations (optional)

**Installation:**

```bash
# Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install FlashInfer
pip install flashinfer-python

# Install flash-attn (optional, for Official FA3 benchmarks)
pip install flash-attn --no-build-isolation

# Install the benchmark package in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,flashinfer]"
```

## Adding New Approaches

To add a new attention kernel implementation:

1. Create a class in `src/attention_bench/approaches/` (e.g., `my_kernel_approaches.py`)
2. Implement the `AttentionApproach` protocol:
   - `name: str` - Unique identifier
   - `default_page_size: int` - Default page size
   - `setup(ctx: BenchmarkContext) -> Callable[[], torch.Tensor]` - Setup method
3. Register with `@APPROACHES.register` decorator
4. Add to config YAML `approaches` list

Example:
```python
from attention_bench.approaches import APPROACHES, BenchmarkContext
from typing import Callable
import torch

@APPROACHES.register
class MyNewApproach:
    """My custom attention kernel implementation."""

    name = "my_new_approach"
    default_page_size = 16

    def setup(self, ctx: BenchmarkContext) -> Callable[[], torch.Tensor]:
        """Setup method called once before benchmark iterations.

        Args:
            ctx: BenchmarkContext with shared Q, KV tensors and metadata

        Returns:
            Callable that runs one attention iteration
        """
        # Initialize your kernel with ctx.q, ctx.kv_cache, ctx.qo_indptr, etc.
        # ...

        def run_iteration():
            # Run attention and return output
            return my_kernel.forward(ctx.q, ctx.kv_cache)

        return run_iteration
```

**Key Points:**
- All approaches benchmark on **identical shared data** from `BenchmarkContext`
- The `setup()` method is called once; `run_iteration()` is called repeatedly for timing
- Use `ctx.q`, `ctx.kv_cache`, `ctx.qo_indptr`, `ctx.kv_page_indptr`, etc. from shared context
- Never create new random tensors - extract from shared context for fairness

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_approaches_work.py -v

# Run with coverage
pytest tests/ --cov=attention_bench --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Layout

- **`src/attention_bench/`**: Main package source code
- **`bin/`**: Executable entry points (installed to PATH)
- **`configs/`**: Benchmark configuration files (version controlled)
- **`data/`**: Generated data (gitignored - not version controlled)
- **`scripts/`**: Utility scripts for batch operations
- **`tests/`**: Unit and integration tests

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors during benchmarking:

1. **Use Ray with memory estimation:**
   ```bash
   attention-bench-ray --config configs/mixed/config_mixed_all_combinations.yaml \
       --num-gpus 2 \
       --memory-utilization 0.80  # Be more conservative
   ```

2. **Reduce workspace size** in config:
   ```yaml
   model:
     workspace_size: 268435456  # 256MB instead of 512MB
   ```

3. **Filter scenarios** by size:
   - Edit config to remove large batch sizes or KV lengths
   - Use `--scenario-range` to run smaller batches

### Import Errors

If you get import errors:

```bash
# Reinstall in development mode
pip install -e .

# Verify installation
python -c "from attention_bench.approaches import APPROACHES; print('OK')"
```

### Ray Timeouts

If Ray workers timeout:

1. **Increase timeout** in `run_ray.py` (line ~246)
2. **Reduce batch size** to fewer scenarios per sync point
3. **Check GPU availability**: `nvidia-smi`

## Citation

If you use this benchmark framework in your research, please cite:

```bibtex
@misc{attention-bench,
  title={Attention Kernel Benchmark: Comprehensive Performance Analysis for Heterogeneous Workloads},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/flashinfer-heterogeneity-experiment}
}
```
