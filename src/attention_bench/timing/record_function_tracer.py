import numpy as np
import torch


class RecordFunctionTracer:
    """Accurate timing using torch.profiler with in-memory events API.

    This tracer uses torch.profiler to capture CPU and CUDA events, then
    accesses them via the in-memory events() API.

    Usage:
        tracer = RecordFunctionTracer()
        with tracer:
            for _ in range(num_iters):
                with torch.profiler.record_function("my_operation"):
                    my_function()

        stats = tracer.get_operation_time_stats()
        # stats = {"my_operation": {"min": ..., "max": ..., "mean": ..., ...}}
    """

    def __init__(self):
        """Initialize tracer."""
        pass

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.profiler.__exit__(exc_type, exc_val, exc_tb)


    def get_operation_time_stats(self):
        """Get timing statistics for all recorded operations.

        Returns:
            dict: Mapping from operation name to statistics dict with keys:
                  min, max, mean, median, std (all in milliseconds)
        """
        torch.cuda.synchronize()  # Ensure profiler finished aggregating
        events = self.profiler.events()
        stats = {}

        # Diagnostic: Count events by type
        user_annotation_count = 0
        cuda_event_count = 0
        cpu_event_count = 0

        for event in events:
            if not event.is_user_annotation:
                continue

            user_annotation_count += 1

            if event.device_type == torch.profiler.DeviceType.CUDA:
                cuda_event_count += 1
            elif event.device_type == torch.profiler.DeviceType.CPU:
                cpu_event_count += 1

            # Use CUDA events for direct kernel timing
            if event.device_type != torch.profiler.DeviceType.CUDA:
                continue

            name = event.name
            device_time_us = event.device_time_total

            if device_time_us == 0:
                # Diagnostic logging when we encounter zero device time
                print(f"⚠️  Zero device_time for CUDA event: {name}")
                print(f"   device_type: {event.device_type}")
                print(f"   cpu_time_total: {event.cpu_time_total} µs")
                continue

            if name not in stats:
                stats[name] = []

            stats[name].append(device_time_us / 1000.0)  # Convert to ms

        # If we got no stats, log diagnostic info
        if not stats:
            print(f"⚠️  NO TIMING DATA CAPTURED")
            print(f"   Total user annotations: {user_annotation_count}")
            print(f"   CUDA events: {cuda_event_count}")
            print(f"   CPU events: {cpu_event_count}")

            # Show all user annotation events for debugging
            print(f"   All user annotation events:")
            for event in events:
                if event.is_user_annotation:
                    print(f"     - {event.name}: device_type={event.device_type}, device_time={event.device_time_total}µs")

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
