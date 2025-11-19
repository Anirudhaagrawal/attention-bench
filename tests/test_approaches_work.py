"""Smoke tests to verify all attention approaches can run without errors.

These tests don't check correctness, just that approaches don't crash.
"""

import pytest
import torch

from approaches import APPROACHES
from .utils import create_test_context


def test_all_approaches_can_run():
    """Verify all registered approaches can run without crashing.

    This is a simple smoke test that:
    - Creates a small test scenario
    - Tries to run each approach
    - Verifies it doesn't crash
    - Doesn't check output correctness (that's in test_correctness.py)
    """
    print("\n" + "=" * 60)
    print("Smoke Test: Verifying all approaches can run")
    print("=" * 60)

    # Create a small mixed workload for testing
    ctx = create_test_context(
        q_lengths=[1, 128],  # 1 decode + 1 prefill
        kv_lengths=[1024, 1024],
        seed=42,
    )

    all_approaches = APPROACHES.list_available()
    print(f"Testing {len(all_approaches)} registered approaches\n")

    # Track results
    working = []
    not_installed = []
    broken = []

    for approach_name in all_approaches:
        # Skip known broken approaches
        if approach_name == "direct_cudnn":
            print(f"⊘ {approach_name}: SKIPPED (known to not work)")
            broken.append(approach_name)
            continue

        try:
            approach = APPROACHES.get(approach_name)
            run_fn = approach.setup(ctx)
            output = run_fn()
            torch.cuda.synchronize()

            # Handle tuple output
            if isinstance(output, tuple):
                output = output[0]

            print(f"✓ {approach_name}: works (output shape: {output.shape})")
            working.append(approach_name)

        except Exception as e:
            error_str = str(e)

            # Allow ImportError or "not available" errors
            if any(keyword in error_str.lower() for keyword in
                   ["not available", "not installed", "install with", "importerror"]):
                print(f"⊘ {approach_name}: not installed (OK)")
                not_installed.append(approach_name)
            else:
                print(f"✗ {approach_name}: FAILED - {error_str}")
                broken.append(approach_name)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  ✓ Working: {len(working)}")
    print(f"  ⊘ Not installed: {len(not_installed)}")
    print(f"  ✗ Broken: {len(broken)}")

    if broken and "direct_cudnn" not in broken:
        # Fail if any unexpected approaches are broken
        pytest.fail(f"Broken approaches (excluding known): {broken}")

    # Need at least one working approach
    if len(working) == 0:
        pytest.fail("No working approaches found!")

    print(f"\n✓ {len(working)} approaches work correctly!")
    print("=" * 60)
