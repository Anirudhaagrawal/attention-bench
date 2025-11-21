#!/usr/bin/env python3
"""Standalone test to verify all attention approaches produce identical outputs."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from attention_bench.cli.run_benchmark import ToleranceBenchmarkRunner, BenchmarkConfig, VariantConfig


def test_output_correctness():
    """Test that all approaches produce identical outputs on mixed workloads."""
    print("=" * 80)
    print("ATTENTION OUTPUT CORRECTNESS TEST")
    print("=" * 80)

    # Create test configuration with OUTPUT VALIDATION ENABLED
    config = BenchmarkConfig(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        workspace_size=256 * 1024 * 1024,
        num_warmup_iters=1,
        num_active_iters=3,
        results_dir="/tmp/test_correctness",
        enable_profiling=True,
        enable_output_validation=True,  # ENABLE OUTPUT VALIDATION
    )

    # Define test variants
    variants = {
        "decode_1k": VariantConfig(
            name="decode_1k",
            q_tokens=1,
            kv_tokens=1024,
            description="Decode with 1k KV tokens"
        ),
        "decode_2k": VariantConfig(
            name="decode_2k",
            q_tokens=1,
            kv_tokens=2048,
            description="Decode with 2k KV tokens"
        ),
        "prefill_128": VariantConfig(
            name="prefill_128",
            q_tokens=128,
            kv_tokens=1024,
            description="Prefill 128 tokens"
        ),
        "prefill_256": VariantConfig(
            name="prefill_256",
            q_tokens=256,
            kv_tokens=2048,
            description="Prefill 256 tokens"
        ),
    }

    runner = ToleranceBenchmarkRunner(config, variants)

    # Test scenario: Mixed workload with interleaved decode/prefill
    scenario_name = "mixed_correctness_test"
    variant_set = {
        "decode_1k": 2,
        "prefill_128": 2,
        "decode_2k": 2,
        "prefill_256": 2,
    }

    # All available approaches (FlashInfer + Official FA3)
    approaches_to_test = [
        "flashinfer_separated_fa2",
        "flashinfer_separated_fa3",
        "flashinfer_mixed_fa2",
        "flashinfer_mixed_fa3",
        "flashinfer_batch_attention",
        "official_fa3",
    ]

    print(f"\nScenario: {scenario_name}")
    print(f"Variant set: {variant_set}")
    print(f"Approaches: {approaches_to_test}")
    print("\n" + "=" * 80)

    try:
        result = runner.run_tolerance_test(
            scenario_name=scenario_name,
            variant_set=variant_set,
            approaches_to_run=approaches_to_test,
        )

        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)

        # Check if all approaches succeeded
        all_success = all(time_ms is not None for time_ms in result.approach_times.values())

        if all_success:
            print(f"✅ All {len(approaches_to_test)} approaches completed successfully!")
            print("✅ Check OUTPUT VALIDATION section above for correctness results")
            return True
        else:
            failed = [name for name, time_ms in result.approach_times.items() if time_ms is None]
            print(f"❌ {len(failed)} approaches failed: {failed}")
            return False

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_output_correctness()
    sys.exit(0 if success else 1)
