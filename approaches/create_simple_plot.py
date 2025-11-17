#!/usr/bin/env python3
"""Simple plotting function for dynamic approaches."""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List


def create_simple_comparison_plot(results, plots_dir: str, config_name: str = ""):
    """Create a simple comparison plot for all approaches."""
    os.makedirs(plots_dir, exist_ok=True)

    if not results:
        print("No results to plot")
        return

    # Get list of all approaches
    all_approaches = set()
    for r in results:
        all_approaches.update(r.list_approaches())
    all_approaches = sorted(all_approaches)

    if not all_approaches:
        print("No approaches found in results")
        return

    # Extract data
    scenario_names = [r.scenario_name for r in results]
    n_scenarios = len(scenario_names)
    n_approaches = len(all_approaches)

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, n_scenarios * 1.5), 8))

    x = np.arange(n_scenarios)
    width = 0.8 / n_approaches if n_approaches > 0 else 0.8

    # Plot each approach
    for i, approach in enumerate(all_approaches):
        times = [r.get_time(approach) or 0 for r in results]
        offset = (i - n_approaches / 2) * width + width / 2
        ax.bar(x + offset, times, width, label=approach)

    # Formatting
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Attention Approach Comparison - {config_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    if config_name:
        output_file = os.path.join(plots_dir, f'comparison_{config_name}.png')
    else:
        output_file = os.path.join(plots_dir, 'comparison.png')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_file}")
