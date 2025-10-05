#!/usr/bin/env python3
"""FlashInfer Heterogeneity Benchmark

Benchmarks FlashInfer's performance on heterogeneous batches by comparing:
- Mixed approach: Single wrapper processing all requests
- Bucketed approach: Separate wrappers for homogeneous request groups

Usage:
    python run_benchmark.py [--config CONFIG_PATH]
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import flashinfer


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
    results_dir: str
    plots_dir: str


@dataclass
class VariantConfig:
    """Configuration for a request variant."""

    name: str
    q_tokens: int
    kv_tokens: int
    description: str


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    scenario_name: str
    variant_set_id: int
    variant_counts: Dict[str, int]
    mixed_time_ms: float
    bucketed_time_ms: float
    total_q_tokens: int
    total_kv_tokens: int
    num_requests: int
    num_buckets: int

    def to_json(self) -> Dict[str, Any]:
        """Convert ExperimentResult to JSON-serializable dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "variant_set_id": self.variant_set_id,
            "variant_counts": self.variant_counts,
            "mixed_time_ms": self.mixed_time_ms,
            "bucketed_time_ms": self.bucketed_time_ms,
            "total_q_tokens": self.total_q_tokens,
            "total_kv_tokens": self.total_kv_tokens,
            "num_requests": self.num_requests,
            "num_buckets": self.num_buckets,
        }


class BenchmarkRunner:
    """Runs FlashInfer heterogeneity benchmarks."""

    def __init__(self, config: BenchmarkConfig, variants: Dict[str, VariantConfig]):
        self.config = config
        self.variants = variants

    def create_batch_data(
        self, q_lengths: List[int], kv_lengths: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """Create tensors and metadata for a batch of requests."""
        num_pages = [
            (kv + self.config.page_size - 1) // self.config.page_size
            for kv in kv_lengths
        ]

        q = torch.randn(
            sum(q_lengths),
            self.config.num_qo_heads,
            self.config.head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        kv_cache = torch.randn(
            sum(num_pages),
            2,
            self.config.page_size,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype=torch.float16,
            device="cuda",
        )

        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lengths), 0)),
            dtype=torch.int32,
            device="cuda",
        )
        kv_page_indices = torch.arange(sum(num_pages), dtype=torch.int32, device="cuda")
        kv_page_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(num_pages), 0)),
            dtype=torch.int32,
            device="cuda",
        )
        kv_last_page_len = torch.tensor(
            [kv % self.config.page_size if kv % self.config.page_size != 0 else self.config.page_size for kv in kv_lengths],
            dtype=torch.int32,
            device="cuda",
        )

        return q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def build_workload(
        self, variant_set: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[str]]:
        """Build workload from variant set specification."""
        all_q_lengths = []
        all_kv_lengths = []
        all_variant_names = []

        for variant_name, count in variant_set.items():
            variant = self.variants[variant_name]
            for _ in range(count):
                all_q_lengths.append(variant.q_tokens)
                all_kv_lengths.append(variant.kv_tokens)
                all_variant_names.append(variant_name)

        return all_q_lengths, all_kv_lengths, all_variant_names

    def benchmark_approach(
        self,
        all_q_lengths: List[int],
        all_kv_lengths: List[int],
        all_variant_names: List[str]
    ) -> Tuple[float, int]:
        """Benchmark inference approach by grouping requests by variant name.

        For mixed approach, pass identical variant names for all requests.
        For bucketed approach, pass actual variant names to group by.

        Args:
            all_q_lengths: Query token lengths for all requests
            all_kv_lengths: KV token lengths for all requests
            all_variant_names: Variant names for grouping requests into buckets

        Returns:
            Tuple of (average inference time in milliseconds, number of buckets used)
        """
        # Organize requests by variant name
        variant_q = defaultdict(list)
        variant_kv = defaultdict(list)

        for q_len, kv_len, variant_name in zip(all_q_lengths, all_kv_lengths, all_variant_names):
            variant_q[variant_name].append(q_len)
            variant_kv[variant_name].append(kv_len)

        # Create wrappers for each bucket
        wrappers = {}
        batch_data = {}
        variant_names = list(variant_q.keys())

        for variant_name in variant_names:
            q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = (
                self.create_batch_data(variant_q[variant_name], variant_kv[variant_name])
            )

            workspace = torch.empty(
                self.config.workspace_size, dtype=torch.uint8, device="cuda"
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
            wrapper.plan(
                qo_indptr,
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                self.config.num_qo_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.page_size,
                causal=True,
            )

            wrappers[variant_name] = wrapper
            batch_data[variant_name] = (q, kv_cache)

        # Warmup
        for _ in range(self.config.num_warmup_iters):
            for variant_name in variant_names:
                q, kv_cache = batch_data[variant_name]
                wrappers[variant_name].run(q, kv_cache)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.config.num_active_iters):
            for variant_name in variant_names:
                q, kv_cache = batch_data[variant_name]
                wrappers[variant_name].run(q, kv_cache)
        end.record()
        torch.cuda.synchronize()

        avg_time_ms = start.elapsed_time(end) / self.config.num_active_iters
        num_buckets = len(variant_names)

        return avg_time_ms, num_buckets

    def run_experiment(
        self, scenario_name: str, variant_set_id: int, variant_set: Dict[str, int]
    ) -> ExperimentResult:
        """Run complete experiment for a variant set."""
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario_name} | Variant Set {variant_set_id}")
        print(f"{'=' * 80}")

        # Build workload
        all_q_lengths, all_kv_lengths, all_variant_names = self.build_workload(variant_set)

        print(f"\nWorkload: {len(all_q_lengths)} requests")
        for variant_name, count in variant_set.items():
            variant = self.variants[variant_name]
            print(f"  {variant_name}: {count}x ({variant.description})")
        print(f"  Total Q tokens:  {sum(all_q_lengths):,}")
        print(f"  Total KV tokens: {sum(all_kv_lengths):,}")

        # Run benchmarks
        print("\nBenchmarking mixed approach...")
        mixed_time, _ = self.benchmark_approach(
            all_q_lengths, all_kv_lengths, ["mixed"] * len(all_q_lengths)
        )
        print(f"  Mixed time: {mixed_time:.3f} ms")

        print("\nBenchmarking bucketed approach...")
        bucketed_time, num_buckets = self.benchmark_approach(
            all_q_lengths, all_kv_lengths, all_variant_names
        )
        print(f"  Bucketed time: {bucketed_time:.3f} ms")

        return ExperimentResult(
            scenario_name=scenario_name,
            variant_set_id=variant_set_id,
            variant_counts=variant_set,
            mixed_time_ms=mixed_time,
            bucketed_time_ms=bucketed_time,
            total_q_tokens=sum(all_q_lengths),
            total_kv_tokens=sum(all_kv_lengths),
            num_requests=len(all_q_lengths),
            num_buckets=num_buckets,
        )


def load_config(config_path: str) -> Tuple[BenchmarkConfig, Dict[str, VariantConfig], Dict[str, Any]]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    benchmark_config = BenchmarkConfig(
        num_qo_heads=config_data["model"]["num_qo_heads"],
        num_kv_heads=config_data["model"]["num_kv_heads"],
        head_dim=config_data["model"]["head_dim"],
        page_size=config_data["model"]["page_size"],
        workspace_size=config_data["model"]["workspace_size"],
        num_warmup_iters=config_data["profiling"]["num_warmup_iters"],
        num_active_iters=config_data["profiling"]["num_active_iters"],
        results_dir=config_data["output"]["results_dir"],
        plots_dir=config_data["output"]["plots_dir"],
    )

    variants = {
        name: VariantConfig(
            name=name,
            q_tokens=data["q_tokens"],
            kv_tokens=data["kv_tokens"],
            description=data["description"],
        )
        for name, data in config_data["variants"].items()
    }

    scenarios = config_data["scenarios"]

    return benchmark_config, variants, scenarios


def save_results(results: List[ExperimentResult], output_path: str) -> None:
    """Save experiment results to JSON file."""
    results_data = [r.to_json() for r in results]

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

class PlotGenerator:
    """Generates visualization plots for benchmark results."""

    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    @staticmethod
    def _add_bar_labels(ax, bars, format_str="{:.1f}"):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, format_str.format(height),
                   ha="center", va="bottom", fontsize=7)

    def _plot_comparison_bars(self, ax, scenario_data, scenario_name=None):
        """Plot comparison bars for mixed vs bucketed performance."""
        x = np.arange(len(scenario_data))
        bars1 = ax.bar(x - 0.175, [r.mixed_time_ms for r in scenario_data], 0.35,
                      label="Mixed", color="#ff7f0e")
        bars2 = ax.bar(x + 0.175, [r.bucketed_time_ms for r in scenario_data], 0.35,
                      label="Bucketed", color="#2ca02c")

        ax.set(xlabel="Variant Set", ylabel="Latency (ms)",
              xticks=x, xticklabels=[f"Set {r.variant_set_id}" for r in scenario_data])
        if scenario_name:
            ax.set_title(scenario_name, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        self._add_bar_labels(ax, bars1)
        self._add_bar_labels(ax, bars2)

    def _plot_speedup_bars(self, ax, scenario_data, scenario_name=None):
        """Plot speedup/improvement bars."""
        speedup_factors = [(r.mixed_time_ms / r.bucketed_time_ms - 1) * 100 for r in scenario_data]
        colors = ["#2ca02c" if f > 0 else "#d62728" for f in speedup_factors]

        x = np.arange(len(scenario_data))
        bars = ax.bar(x, speedup_factors, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set(xlabel="Variant Set", ylabel="Bucketed Improvement (%)",
              xticks=x, xticklabels=[f"Set {r.variant_set_id}" for r in scenario_data])
        if scenario_name:
            ax.set_title(f"{scenario_name}\nPositive = Bucketed Faster, Negative = Mixed Faster",
                        fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, speedup_factors):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:+.1f}%",
                   ha="center", va="bottom" if val > 0 else "top", fontsize=8)

    def _save_multi_scenario_plot(self, scenario_results, plot_func, filename, title, figsize_height=6):
        """Generic method to create and save multi-scenario plots."""
        num_scenarios = len(scenario_results)
        fig, axes = plt.subplots(num_scenarios, 1, figsize=(16, figsize_height * num_scenarios))
        axes = [axes] if num_scenarios == 1 else axes

        for ax, (scenario_name, scenario_data) in zip(axes, scenario_results.items()):
            plot_func(ax, scenario_data, scenario_name)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    def generate_all_plots(self, results: List[ExperimentResult]) -> None:
        """Generate all visualization plots from experiment results."""
        # Group by scenario
        scenario_results = defaultdict(list)
        for result in results:
            scenario_results[result.scenario_name].append(result)

        # Multi-scenario plots
        self._save_multi_scenario_plot(scenario_results, self._plot_comparison_bars,
                                      "comparison_by_scenario.png",
                                      "FlashInfer: Mixed vs Bucketed Performance", 6)
        self._save_multi_scenario_plot(scenario_results, self._plot_speedup_bars,
                                      "speedup_by_scenario.png",
                                      "Bucketed vs Mixed: Performance Improvement", 5)

        # Individual scenario plots
        print("\nGenerating per-scenario plots...")
        per_scenario_dir = os.path.join(self.plots_dir, "per_scenario")
        os.makedirs(per_scenario_dir, exist_ok=True)

        for scenario_name, scenario_data in scenario_results.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            self._plot_comparison_bars(ax1, scenario_data)
            self._plot_speedup_bars(ax2, scenario_data)
            plt.suptitle(scenario_name, fontsize=13, fontweight="bold")
            plt.tight_layout()

            safe_filename = scenario_name.replace(" ", "_").replace("/", "_").lower()
            plt.savefig(os.path.join(per_scenario_dir, f"{safe_filename}.png"),
                       dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Plots saved to: {self.plots_dir}/")
        print(f"  - comparison_by_scenario.png")
        print(f"  - speedup_by_scenario.png")
        print(f"  - per_scenario/*.png ({len(scenario_results)} individual scenario plots)")


def create_plots(results: List[ExperimentResult], plots_dir: str) -> None:
    """Create visualization plots from experiment results."""
    plot_generator = PlotGenerator(plots_dir)
    plot_generator.generate_all_plots(results)


def main():
    parser = argparse.ArgumentParser(description="Run FlashInfer heterogeneity benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FlashInfer Heterogeneity Benchmark")
    print("=" * 80)
    print(f"\nLoading configuration from: {args.config}")

    # Load configuration
    benchmark_config, variants, scenarios = load_config(args.config)

    print(f"Loaded {len(variants)} variants")
    print(f"Loaded {len(scenarios)} scenarios")

    # Create output directories
    os.makedirs(benchmark_config.results_dir, exist_ok=True)
    os.makedirs(benchmark_config.plots_dir, exist_ok=True)

    # Print environment info
    print("\n" + "=" * 80)
    print("Environment")
    print("=" * 80)
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # Run experiments
    print("\n" + "=" * 80)
    print("Running Experiments")
    print("=" * 80)

    runner = BenchmarkRunner(benchmark_config, variants)
    all_results = []

    for scenario_name, scenario_data in scenarios.items():
        variant_sets = scenario_data["variants_to_run"]

        for variant_set_id, variant_set in enumerate(variant_sets):
            result = runner.run_experiment(scenario_name, variant_set_id, variant_set)
            all_results.append(result)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(benchmark_config.results_dir, f"results_{timestamp}.json")
    save_results(all_results, results_path)

    # Create plots
    print("\n" + "=" * 80)
    print("Creating Plots")
    print("=" * 80)
    create_plots(all_results, benchmark_config.plots_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total experiments run: {len(all_results)}")

if __name__ == "__main__":
    main()
