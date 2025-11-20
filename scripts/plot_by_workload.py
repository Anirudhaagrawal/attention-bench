#!/usr/bin/env python3
"""
Convenience script to generate filtered heatmaps by workload category.

Generates heatmaps for specific workload types (code, chat, summarization)
from existing benchmark results.
"""

import argparse
import os
from attention_bench.plotting.heatmap_decode import create_decode_heatmap
from attention_bench.plotting.heatmap_prefill import create_prefill_heatmap
from attention_bench.plotting.workload_categorizer import WORKLOAD_CATEGORIES


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered heatmaps by workload category"
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=WORKLOAD_CATEGORIES + ['all'],
        default='all',
        help="Workload category to visualize (code/chat/summarization/all)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=['decode', 'prefill', 'all'],
        default='all',
        help="Benchmark type (decode/prefill/all)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots"
    )
    args = parser.parse_args()

    # Determine which categories to process
    categories = WORKLOAD_CATEGORIES if args.workload == 'all' else [args.workload]

    # Determine which types to process
    types = ['decode', 'prefill'] if args.type == 'all' else [args.type]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Workload-Filtered Heatmap Generator")
    print("=" * 80)
    print(f"Categories: {', '.join(categories)}")
    print(f"Types: {', '.join(types)}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    print()

    # Generate heatmaps
    for category in categories:
        for bench_type in types:
            if bench_type == 'decode':
                results_file = f"{args.results_dir}/config_decode_all_combinations.json"
                output_file = f"{args.output_dir}/decode_heatmap_{category}.png"

                if not os.path.exists(results_file):
                    print(f"⚠ Skipping decode/{category}: {results_file} not found")
                    continue

                print(f"Generating decode heatmap for '{category}' workload...")
                try:
                    create_decode_heatmap(
                        results_file=results_file,
                        output_file=output_file,
                        workload_category=category
                    )
                    print(f"  ✓ Saved to: {output_file}")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

            elif bench_type == 'prefill':
                results_file = f"{args.results_dir}/config_prefill_all_combinations.json"
                output_file = f"{args.output_dir}/prefill_heatmap_{category}.png"

                if not os.path.exists(results_file):
                    print(f"⚠ Skipping prefill/{category}: {results_file} not found")
                    continue

                print(f"Generating prefill heatmap for '{category}' workload...")
                try:
                    create_prefill_heatmap(
                        results_file=results_file,
                        output_file=output_file,
                        workload_category=category
                    )
                    print(f"  ✓ Saved to: {output_file}")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
