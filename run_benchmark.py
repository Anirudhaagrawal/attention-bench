#!/usr/bin/env python3
"""FlashInfer Prefill Tolerance Benchmark (v2 - Refactored with Approach Registry)

Compares multiple attention implementations:
- FlashInfer: Mixed FA2/FA3, Separated FA2/FA3, BatchAttention, cuDNN
- Official FlashAttention-3

Usage:
    python run_benchmark.py [--config config.yaml]
    python run_benchmark.py --approaches official_fa3,flashinfer_mixed_fa3
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import flashinfer
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, BatchDecodeWithPagedKVCacheWrapper
from flashinfer.page import get_seq_lens

# Import approaches registry
from approaches import APPROACHES, BenchmarkContext

# Check for BatchAttention availability
try:
    from flashinfer import BatchAttention
    HAS_BATCH_ATTENTION = True
except ImportError:
    BatchAttention = None
    HAS_BATCH_ATTENTION = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    workspace_size: int
    num_warmup_iters: int
    num_active_iters: int
    use_cuda_graphs: bool = False
    enable_profiling: bool = False
    results_dir: str = "results"
    plots_dir: str = "plots"
    approach_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VariantConfig:
    """Configuration for a request variant."""
    name: str
    q_tokens: int
    kv_tokens: int
    description: str = ""


@dataclass
class ToleranceResult:
    """Results from a tolerance test."""
    scenario_name: str
    num_decodes: int
    num_prefills: int
    prefill_ratio: float
    approach_times: Dict[str, Optional[float]]
    page_sizes: Dict[str, int]
    used_cuda_graphs: bool


class ToleranceBenchmarkRunner:
    """Runs tolerance benchmarks."""

    def __init__(self, config: BenchmarkConfig, variants: Dict[str, VariantConfig]):
        self.config = config
        self.variants = variants

    def create_batch_data(
        self, q_lengths: List[int], kv_lengths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create tensors and metadata for a batch of requests."""
        num_pages = [(kv + self.config.page_size - 1) // self.config.page_size for kv in kv_lengths]

        # Create Q and KV cache tensors
        q = torch.randn(
            sum(q_lengths), self.config.num_qo_heads, self.config.head_dim,
            dtype=torch.float16, device="cuda"
        )

        kv_cache = torch.randn(
            sum(num_pages), 2, self.config.page_size, self.config.num_kv_heads, self.config.head_dim,
            dtype=torch.float16, device="cuda"
        )

        # Create metadata tensors
        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lengths), 0)),
            dtype=torch.int32, device="cuda"
        )

        # Use shuffled page indices to simulate realistic scattered memory allocation
        # (like vLLM/SGLang where blocks are reused and non-sequential)
        kv_page_indices = torch.randperm(sum(num_pages), dtype=torch.int32, device="cuda")

        kv_page_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(num_pages), 0)),
            dtype=torch.int32, device="cuda"
        )

        kv_last_page_len = torch.tensor(
            [kv % self.config.page_size if kv % self.config.page_size != 0 else self.config.page_size
             for kv in kv_lengths],
            dtype=torch.int32, device="cuda"
        )

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def _create_benchmark_context(
        self, all_q_lengths: List[int], all_kv_lengths: List[int], page_size: int
    ) -> BenchmarkContext:
        """Create complete benchmark context with ALL parameters (superset).

        This method computes ALL possible parameters that any approach might need.
        Each approach can then select what it needs from this complete context.

        Args:
            all_q_lengths: Query sequence lengths for all requests
            all_kv_lengths: KV cache lengths for all requests
            page_size: Page size to use for this benchmark

        Returns:
            BenchmarkContext with all computed parameters
        """
        # Temporarily override page_size for batch data creation
        original_page_size = self.config.page_size
        self.config.page_size = page_size

        # Create batch data
        q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = (
            self.create_batch_data(all_q_lengths, all_kv_lengths)
        )

        # Restore original page_size
        self.config.page_size = original_page_size

        # Compute ALL optional tensors unconditionally (superset)
        seq_lens_kv = get_seq_lens(
            kv_page_indptr.cpu(), kv_last_page_len.cpu(), page_size
        ).to("cuda")

        seq_lens_q = (qo_indptr[1:] - qo_indptr[:-1])

        max_token_per_sequence = seq_lens_q.max().item()
        max_sequence_kv = seq_lens_kv.max().item()

        # Create block_tables
        batch_size = len(qo_indptr) - 1
        max_num_blocks_per_seq = math.ceil(max_sequence_kv / page_size)

        block_tables = torch.zeros(
            (batch_size, max_num_blocks_per_seq),
            dtype=torch.int32, device="cuda",
        )

        # Fill block_tables
        for i in range(batch_size):
            start_idx = kv_page_indptr[i]
            end_idx = kv_page_indptr[i + 1]
            num_blocks = end_idx - start_idx
            block_tables[i, :num_blocks] = kv_page_indices[start_idx:end_idx]

        return BenchmarkContext(
            q_lengths=all_q_lengths,
            kv_lengths=all_kv_lengths,
            q=q,
            kv_cache=kv_cache,
            qo_indptr=qo_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            seq_lens_q=seq_lens_q,
            seq_lens_kv=seq_lens_kv,
            block_tables=block_tables,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            num_qo_heads=self.config.num_qo_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            page_size=page_size,
            workspace_size=self.config.workspace_size,
            num_warmup_iters=self.config.num_warmup_iters,
            num_active_iters=self.config.num_active_iters,
        )

    def run_tolerance_test(
        self,
        scenario_name: str,
        variant_set: Dict[str, int],
        approaches_to_run: List[str]
    ) -> ToleranceResult:
        """Run tolerance test for a scenario using specified approaches.

        Args:
            scenario_name: Name of the scenario
            variant_set: Dictionary mapping variant names to counts
            approaches_to_run: List of approach names to benchmark
        """
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'=' * 80}")

        # Build workload
        all_q_lengths = []
        all_kv_lengths = []

        # Handle both dict and list formats
        if isinstance(variant_set, list):
            # List format: [{"variant_name": count}, ...]
            for item in variant_set:
                for variant_name, count in item.items():
                    variant = self.variants[variant_name]
                    for _ in range(count):
                        all_q_lengths.append(variant.q_tokens)
                        all_kv_lengths.append(variant.kv_tokens)
        else:
            # Dict format: {"variant_name": count, ...}
            for variant_name, count in variant_set.items():
                variant = self.variants[variant_name]
                for _ in range(count):
                    all_q_lengths.append(variant.q_tokens)
                    all_kv_lengths.append(variant.kv_tokens)

        num_decodes = sum(1 for q in all_q_lengths if q == 1)
        num_prefills = sum(1 for q in all_q_lengths if q > 1)
        prefill_ratio = num_prefills / len(all_q_lengths) if len(all_q_lengths) > 0 else 0

        print(f"\nWorkload: {num_decodes} decodes + {num_prefills} prefills ({prefill_ratio*100:.1f}% prefill)")
        print(f"Approaches to run: {', '.join(approaches_to_run)}\n")

        # Run each approach
        approach_times = {}
        page_sizes = {}

        for i, approach_name in enumerate(approaches_to_run, 1):
            approach = APPROACHES.get(approach_name)
            if approach is None:
                print(f"[{i}/{len(approaches_to_run)}] {approach_name}: SKIPPED (not registered)")
                continue

            print(f"[{i}/{len(approaches_to_run)}] {approach_name}...")

            # Determine page_size: check overrides, then approach default, then global default
            page_size = self.config.page_size
            if approach_name in self.config.approach_overrides:
                page_size = self.config.approach_overrides[approach_name].get("page_size", page_size)
            else:
                page_size = getattr(approach, "default_page_size", page_size)

            # Run the approach
            try:
                # All approaches use setup() interface
                ctx = self._create_benchmark_context(all_q_lengths, all_kv_lengths, page_size)

                # Approach sets up and returns callable
                run_fn = approach.setup(ctx)

                # Runner handles warmup
                for _ in range(ctx.num_warmup_iters):
                    run_fn()
                torch.cuda.synchronize()

                # Runner handles timing
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(ctx.num_active_iters):
                    run_fn()
                end.record()
                torch.cuda.synchronize()

                time_ms = start.elapsed_time(end) / ctx.num_active_iters

                approach_times[approach_name] = time_ms
                page_sizes[approach_name] = page_size
                print(f"  {approach_name}: {time_ms:.3f} ms (page_size={page_size})")

            except Exception as e:
                print(f"  {approach_name}: FAILED - {e}")
                approach_times[approach_name] = None

        # Print summary
        print(f"\n{'=' * 80}")
        print("RESULTS:")
        for approach_name in approaches_to_run:
            if approach_name in approach_times and approach_times[approach_name] is not None:
                time_ms = approach_times[approach_name]
                ps = page_sizes.get(approach_name, self.config.page_size)
                print(f"  {approach_name:35s}: {time_ms:8.3f} ms (page_size={ps})")
            else:
                print(f"  {approach_name:35s}:      N/A")

        # Find winner
        valid_times = {k: v for k, v in approach_times.items() if v is not None}
        if valid_times:
            winner = min(valid_times, key=valid_times.get)
            print(f"\n  Winner: {winner} ({valid_times[winner]:.3f} ms)")
        else:
            print(f"\n  Winner: N/A (no successful runs)")

        print(f"  CUDA Graphs: {'ENABLED' if self.config.use_cuda_graphs else 'DISABLED'}")
        print(f"{'=' * 80}")

        return ToleranceResult(
            scenario_name=scenario_name,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            prefill_ratio=prefill_ratio,
            approach_times=approach_times,
            page_sizes=page_sizes,
            used_cuda_graphs=self.config.use_cuda_graphs,
        )


def load_config(config_path: str, use_cuda_graphs: bool, enable_profiling: bool = False):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Get approaches list (default to FlashInfer approaches if not specified)
    approaches = config_data.get("approaches", [
        "flashinfer_mixed_fa2",
        "flashinfer_mixed_fa3",
        "flashinfer_separated_fa2",
        "flashinfer_separated_fa3",
        "flashinfer_batch_attention",
    ])

    # Load model config
    model_config = config_data["model"]
    config = BenchmarkConfig(
        num_qo_heads=model_config["num_qo_heads"],
        num_kv_heads=model_config["num_kv_heads"],
        head_dim=model_config["head_dim"],
        page_size=model_config.get("page_size", 16),
        workspace_size=model_config.get("workspace_size", 256 * 1024 * 1024),
        num_warmup_iters=config_data["profiling"]["num_warmup_iters"],
        num_active_iters=config_data["profiling"]["num_active_iters"],
        use_cuda_graphs=use_cuda_graphs,
        enable_profiling=enable_profiling,
        results_dir=config_data["output"]["results_dir"],
        plots_dir=config_data["output"]["plots_dir"],
        approach_overrides=config_data.get("approach_overrides", {}),
    )

    # Load variants
    variants = {}
    for variant_name, variant_data in config_data["variants"].items():
        variants[variant_name] = VariantConfig(
            name=variant_name,
            q_tokens=variant_data["q_tokens"],
            kv_tokens=variant_data["kv_tokens"],
            description=variant_data.get("description", ""),
        )

    # Load scenarios
    scenarios = config_data["scenarios"]

    return config, variants, scenarios, approaches


def save_results(results: List[ToleranceResult], output_path: str):
    """Save results to JSON file."""
    output_data = []
    for result in results:
        output_data.append({
            "scenario_name": result.scenario_name,
            "num_decodes": result.num_decodes,
            "num_prefills": result.num_prefills,
            "prefill_ratio": result.prefill_ratio,
            "approach_times": result.approach_times,
            "page_sizes": result.page_sizes,
            "used_cuda_graphs": result.used_cuda_graphs,
        })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def _get_grid_layout(num_subplots: int) -> Tuple[int, int]:
    """Calculate grid layout (rows, cols) for given number of subplots."""
    if num_subplots == 1:
        return (1, 1)
    elif num_subplots == 2:
        return (1, 2)
    elif num_subplots <= 4:
        return (2, 2)
    elif num_subplots <= 6:
        return (2, 3)
    elif num_subplots <= 9:
        return (3, 3)
    elif num_subplots <= 12:
        return (3, 4)
    else:
        return (4, 4)


def create_plots(results: List[ToleranceResult], plots_dir: str, config_name: str, approaches: List[str]):
    """Create comparison plots with pagination support."""
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nCreating plots for {len(results)} scenarios with {len(approaches)} approaches...")

    MAX_SCENARIOS_PER_PAGE = 12  # 3x4 grid maximum

    if len(results) == 1:
        print("  Using single plot layout")
        # Single scenario - single bar chart
        result = results[0]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Get times for each approach
        approach_names = []
        times = []

        for approach_name in approaches:
            if approach_name in result.approach_times and result.approach_times[approach_name] is not None:
                approach_names.append(approach_name)
                times.append(result.approach_times[approach_name])

        # Create bar chart
        x = np.arange(len(approach_names))
        bars = ax.bar(x, times, color='steelblue', alpha=0.8)

        # Customize plot
        ax.set_xlabel('Approach', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title(f'{result.scenario_name}\n({result.num_decodes} decodes + {result.num_prefills} prefills)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(approach_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(plots_dir, f"comparison_{config_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to: {output_path}")

    elif len(results) <= MAX_SCENARIOS_PER_PAGE:
        print(f"  Using subplot layout (single page)")
        # Multiple scenarios - single page with subplots
        rows, cols = _get_grid_layout(len(results))

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if len(results) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, result in enumerate(results):
            ax = axes[idx]

            # Get times for each approach
            approach_names = []
            times = []

            for approach_name in approaches:
                if approach_name in result.approach_times and result.approach_times[approach_name] is not None:
                    approach_names.append(approach_name)
                    times.append(result.approach_times[approach_name])

            # Create bar chart
            x = np.arange(len(approach_names))
            bars = ax.bar(x, times, color='steelblue', alpha=0.8)

            # Customize subplot
            ax.set_ylabel('Time (ms)', fontsize=10)
            ax.set_title(f'{result.scenario_name}\n({result.num_decodes}D + {result.num_prefills}P)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(approach_names, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)

        # Hide extra subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = os.path.join(plots_dir, f"comparison_{config_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to: {output_path}")

    else:
        # Many scenarios - use pagination
        num_pages = math.ceil(len(results) / MAX_SCENARIOS_PER_PAGE)
        print(f"  Using pagination: {num_pages} pages with up to {MAX_SCENARIOS_PER_PAGE} subplots each")

        for page_num in range(num_pages):
            start_idx = page_num * MAX_SCENARIOS_PER_PAGE
            end_idx = min(start_idx + MAX_SCENARIOS_PER_PAGE, len(results))
            page_results = results[start_idx:end_idx]

            print(f"    Creating page {page_num + 1}/{num_pages} ({len(page_results)} scenarios)...")

            # Create subplot grid for this page
            rows, cols = _get_grid_layout(len(page_results))
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

            if len(page_results) == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, result in enumerate(page_results):
                ax = axes[idx]

                # Get times for each approach
                approach_names = []
                times = []

                for approach_name in approaches:
                    if approach_name in result.approach_times and result.approach_times[approach_name] is not None:
                        approach_names.append(approach_name)
                        times.append(result.approach_times[approach_name])

                # Create bar chart
                x = np.arange(len(approach_names))
                bars = ax.bar(x, times, color='steelblue', alpha=0.8)

                # Customize subplot
                ax.set_ylabel('Time (ms)', fontsize=10)
                ax.set_title(f'{result.scenario_name}\n({result.num_decodes}D + {result.num_prefills}P)', fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels(approach_names, rotation=45, ha='right', fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=7)

            # Hide extra subplots
            for idx in range(len(page_results), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()

            # Save with page number
            output_path = os.path.join(plots_dir, f"comparison_{config_name}_page{page_num + 1}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"      Page saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FlashInfer Tolerance Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--use-cuda-graphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling")
    parser.add_argument("--approaches", type=str, help="Comma-separated list of approaches to run")
    args = parser.parse_args()

    # Load config
    config, variants, scenarios, approaches = load_config(
        args.config, args.use_cuda_graphs, args.enable_profiling
    )

    # Override approaches if specified
    if args.approaches:
        approaches = [a.strip() for a in args.approaches.split(",")]

    # Print header
    print("=" * 80)
    print("FlashInfer Prefill Tolerance Benchmark (v2 - Refactored)")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"CUDA Graphs: {'ENABLED' if args.use_cuda_graphs else 'DISABLED'}")
    print(f"Profiling:   {'ENABLED' if args.enable_profiling else 'DISABLED'}")
    print(f"Loaded {len(variants)} variants")
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"\nAvailable approaches: {', '.join(APPROACHES.list_available())}")

    # Print environment info
    print("\n" + "=" * 80)
    print("Environment")
    print("=" * 80)
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # Create runner
    runner = ToleranceBenchmarkRunner(config, variants)

    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Tolerance Tests")
    print("=" * 80)

    results = []
    for scenario_name, scenario_config in scenarios.items():
        variant_set = scenario_config["variants_to_run"]
        result = runner.run_tolerance_test(scenario_name, variant_set, approaches)
        results.append(result)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    graph_suffix = "_cuda_graphs" if args.use_cuda_graphs else "_eager"
    results_path = os.path.join(
        config.results_dir,
        f"tolerance_clean{graph_suffix}_{timestamp}.json"
    )
    save_results(results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Create plots
    print("\n" + "=" * 80)
    print("Creating Plots")
    print("=" * 80)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    create_plots(results, config.plots_dir, config_name, approaches)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total tests run: {len(results)}")

    # Count wins per approach
    wins = {approach: 0 for approach in approaches}
    for result in results:
        valid_times = {k: v for k, v in result.approach_times.items() if v is not None}
        if valid_times:
            winner = min(valid_times, key=valid_times.get)
            wins[winner] = wins.get(winner, 0) + 1

    print("\nWinner Count by Approach:")
    for approach in approaches:
        count = wins.get(approach, 0)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"  {approach:35s}: {count:3d}/{len(results)} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
