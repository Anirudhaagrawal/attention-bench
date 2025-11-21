#!/usr/bin/env python3
"""Ray actor for benchmark execution with GPU isolation.

Each worker gets exclusive access to one GPU via Ray's resource management.
Follows Vidur's pattern of creating tensors inside the worker (not pickling them).
"""

import ray
import torch
import gc
import os
from typing import Dict, List, Optional, Any


@ray.remote(num_cpus=1, num_gpus=1)
class BenchmarkWorker:
    """Ray actor that runs benchmarks on a single GPU.

    Key patterns from Vidur:
    - num_gpus=1 ensures exclusive GPU access
    - Configuration passed as primitives, tensors created inside
    - Explicit cleanup after each scenario
    """

    def __init__(self, config_path: str, use_cuda_graphs: bool = False, enable_profiling: bool = False):
        """Initialize worker with benchmark configuration.

        Args:
            config_path: Path to YAML configuration file
            use_cuda_graphs: Whether to use CUDA graphs
            enable_profiling: Whether to enable profiling
        """
        # Import here to ensure clean CUDA context in worker
        from attention_bench.cli.run_benchmark import load_config, ToleranceBenchmarkRunner

        # Validate CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for worker")

        # Load configuration
        self.config_path = config_path
        config, variants, scenarios, approaches = load_config(
            config_path, use_cuda_graphs, enable_profiling
        )

        # Create runner
        self.runner = ToleranceBenchmarkRunner(config, variants)
        self.scenarios = scenarios
        self.available_approaches = approaches

        # Track GPU info
        self.gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        print(f"BenchmarkWorker initialized on GPU {self.gpu_id}")

    def profile(
        self,
        scenario_name: str,
        variant_set: Dict[str, int],
        approaches: List[str]
    ) -> Dict[str, Any]:
        """Run benchmark for a single scenario.

        Args:
            scenario_name: Name of the scenario
            variant_set: Dictionary mapping variant names to counts
            approaches: List of approach names to benchmark

        Returns:
            Dictionary with benchmark results (serializable, no tensors)
        """
        try:
            result = self.runner.run_tolerance_test(
                scenario_name, variant_set, approaches
            )

            # Convert dataclass to dict for serialization
            return {
                'scenario_name': result.scenario_name,
                'num_decodes': result.num_decodes,
                'num_prefills': result.num_prefills,
                'prefill_ratio': result.prefill_ratio,
                'approach_times': result.approach_times,
                'approach_time_stats': result.approach_time_stats,
                'page_sizes': result.page_sizes,
                'used_cuda_graphs': result.used_cuda_graphs,
                'skipped_reason': getattr(result, 'skipped_reason', None),
            }

        except Exception as e:
            print(f"Error in scenario {scenario_name}: {e}")
            return {
                'scenario_name': scenario_name,
                'num_decodes': 0,
                'num_prefills': 0,
                'prefill_ratio': 0.0,
                'approach_times': {a: None for a in approaches},
                'approach_time_stats': {a: None for a in approaches},
                'page_sizes': {},
                'used_cuda_graphs': False,
                'skipped_reason': f"Error: {str(e)}",
            }

        finally:
            # Explicit cleanup after each scenario (Vidur pattern)
            # Multiple syncs are critical for greedy scheduling to prevent
            # CUDA illegal memory access from stale FlashInfer workspace pointers
            torch.cuda.synchronize()  # Wait for all kernels
            gc.collect()              # Force Python GC
            torch.cuda.empty_cache()  # Free cached memory
            torch.cuda.synchronize()  # Ensure everything is done
            gc.collect()              # Final GC pass

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory info in GB
        """
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    def get_scenario_info(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get scenario configuration.

        Args:
            scenario_name: Name of the scenario

        Returns:
            Scenario configuration or None if not found
        """
        return self.scenarios.get(scenario_name)

    def ping(self) -> str:
        """Health check for worker.

        Returns:
            Status message with GPU info
        """
        mem = self.get_memory_info()
        return (f"Worker on GPU {self.gpu_id}: "
                f"{mem['allocated_gb']:.2f}/{mem['total_gb']:.2f} GB allocated")
