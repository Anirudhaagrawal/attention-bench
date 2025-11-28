"""Test to compare device_time_total vs cuda_time_total in PyTorch profiler.

This test validates whether using cuda_time_total instead of device_time_total
would give the same timing results.
"""

import torch
import numpy as np


def get_timing_device_time(profiler):
    """Current method: CPU events with device_time_total."""
    events = profiler.events()
    times = []

    for event in events:
        if not event.is_user_annotation:
            continue
        if event.device_type != torch.profiler.DeviceType.CPU:
            continue

        device_time_us = event.device_time_total
        if device_time_us == 0:
            continue

        times.append(device_time_us / 1000.0)  # Convert to ms

    return times


def get_timing_cuda_time(profiler):
    """Proposed method: Using cuda_time_total."""
    events = profiler.events()
    times = []

    for event in events:
        if not event.is_user_annotation:
            continue
        if event.device_type != torch.profiler.DeviceType.CPU:
            continue

        # Try cuda_time_total instead
        cuda_time_us = event.cuda_time_total
        if cuda_time_us == 0:
            continue

        times.append(cuda_time_us / 1000.0)  # Convert to ms

    return times


def print_all_events(profiler):
    """Print detailed info about all events for debugging."""
    events = profiler.events()

    print("\n" + "=" * 80)
    print("ALL PROFILER EVENTS:")
    print("=" * 80)

    for i, event in enumerate(events):
        if event.is_user_annotation:
            print(f"\nEvent {i}: {event.name}")
            print(f"  is_user_annotation: {event.is_user_annotation}")
            print(f"  device_type: {event.device_type}")
            print(f"  device_time_total: {event.device_time_total} µs")
            # Note: cuda_time_total is deprecated, use device_time_total
            try:
                print(f"  cuda_time_total (deprecated): {event.cuda_time_total} µs")
            except AttributeError:
                print(f"  cuda_time_total: Not available")


def test_timing_comparison():
    """Compare device_time_total vs cuda_time_total for simple CUDA operation."""

    print("\n" + "=" * 80)
    print("TESTING: device_time_total vs cuda_time_total")
    print("=" * 80)

    # Setup simple CUDA operation (matrix multiplication)
    device = torch.device("cuda:0")
    dtype = torch.float16

    size = 4096
    A = torch.randn(size, size, device=device, dtype=dtype)
    B = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()

    # Profile with torch.profiler
    print("\nRunning profiler...")
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    )

    profiler.__enter__()

    num_iters = 10
    for _ in range(num_iters):
        with torch.profiler.record_function("test_operation"):
            C = torch.matmul(A, B)

    profiler.__exit__(None, None, None)
    torch.cuda.synchronize()

    # Print all events for debugging
    print_all_events(profiler)

    # Get timing using both methods
    print("\n" + "=" * 80)
    print("TIMING COMPARISON:")
    print("=" * 80)

    times_device = get_timing_device_time(profiler)
    times_cuda = get_timing_cuda_time(profiler)

    print(f"\nMethod A (device_time_total):")
    print(f"  Number of samples: {len(times_device)}")
    if times_device:
        print(f"  Mean: {np.mean(times_device):.4f} ms")
        print(f"  Std: {np.std(times_device):.4f} ms")
        print(f"  Min: {np.min(times_device):.4f} ms")
        print(f"  Max: {np.max(times_device):.4f} ms")
    else:
        print("  ⚠️  NO TIMING DATA CAPTURED!")

    print(f"\nMethod B (cuda_time_total):")
    print(f"  Number of samples: {len(times_cuda)}")
    if times_cuda:
        print(f"  Mean: {np.mean(times_cuda):.4f} ms")
        print(f"  Std: {np.std(times_cuda):.4f} ms")
        print(f"  Min: {np.min(times_cuda):.4f} ms")
        print(f"  Max: {np.max(times_cuda):.4f} ms")
    else:
        print("  ⚠️  NO TIMING DATA CAPTURED!")

    # Compare
    if times_device and times_cuda:
        mean_device = np.mean(times_device)
        mean_cuda = np.mean(times_cuda)
        diff_percent = abs(mean_device - mean_cuda) / mean_device * 100

        print(f"\n{'=' * 80}")
        print("COMPARISON RESULT:")
        print(f"{'=' * 80}")
        print(f"Difference: {diff_percent:.2f}%")

        if diff_percent < 5:
            print("✅ Timings match within 5% tolerance")
        else:
            print(f"⚠️  Timings differ by {diff_percent:.2f}%")
    else:
        print(f"\n{'=' * 80}")
        print("⚠️  Cannot compare - one or both methods returned no data")
        print(f"{'=' * 80}")

if __name__ == "__main__":
    test_timing_comparison()
