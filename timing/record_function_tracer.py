import json
import os
import uuid

import numpy as np
import torch


class RecordFunctionTracer:
    """Accurate timing using torch.profiler with CUDA correlation.

    This tracer uses torch.profiler to capture CPU and CUDA events, then
    correlates them to get accurate CUDA kernel execution times. This is
    more accurate than simple CUDA events because it properly accounts for
    kernel launch overhead and async execution.

    Usage:
        tracer = RecordFunctionTracer(output_dir="results")
        with tracer:
            for _ in range(num_iters):
                with torch.profiler.record_function("my_operation"):
                    my_function()

        stats = tracer.get_operation_time_stats()
        # stats = {"my_operation": {"min": ..., "max": ..., "mean": ..., ...}}
    """

    def __init__(self, output_dir: str = "."):
        """Initialize tracer.

        Args:
            output_dir: Directory to save profiler traces
        """
        trace_id = str(uuid.uuid4())[:8]
        os.makedirs(f"{output_dir}/profiler_traces", exist_ok=True)
        self.trace_path = (
            f"{output_dir}/profiler_traces/profiler_trace_{trace_id}.json"
        )

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()
        self.profiler.export_chrome_trace(self.trace_path)

    def find_children(self, trace, event):
        """Find all events that are children of the given event."""
        if not ("dur" in event and "ts" in event):
            return []

        children = []
        for e in trace:
            if not ("dur" in e and "ts" in e):
                continue

            # if the ts of the child is completely within the ts of the parent
            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(self, trace, event):
        """Find the CUDA event correlated with a CPU event."""
        if not ("args" in event and "correlation" in event["args"]):
            return None

        for e in trace:
            if not ("args" in e and "correlation" in e["args"]):
                continue

            if e == event:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e

        return None

    def get_operation_time_stats(self):
        """Get timing statistics for all recorded operations.

        Returns:
            dict: Mapping from operation name to statistics dict with keys:
                  min, max, mean, median, std (all in milliseconds)
        """
        stats = {}

        trace = json.load(open(self.trace_path, "r"))["traceEvents"]

        for event in trace:
            # Look for user annotations (from record_function)
            if not ("cat" in event and event["cat"] == "user_annotation"):
                continue

            # Find all CUDA runtime calls within this annotation
            children = self.find_children(trace, event)
            cuda_time = 0
            for child in children:
                if not ("cat" in child and child["cat"] == "cuda_runtime"):
                    continue
                # Find the correlated CUDA kernel execution
                correlated_event = self.find_correlated_event(trace, child)
                if not correlated_event:
                    continue
                cuda_time += correlated_event["dur"]

            if cuda_time == 0:
                continue

            name = event["name"]

            if name not in stats:
                stats[name] = []

            stats[name].append(cuda_time * 1e-3)  # convert to ms

        # Compute statistics for each operation
        return {
            operation: {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
            }
            for operation, times in stats.items()
        }
