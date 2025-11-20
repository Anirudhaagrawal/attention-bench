#!/usr/bin/env python3
"""Setup configuration for attention-bench package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="attention-bench",
    version="0.1.0",
    description="Benchmarking framework for attention kernel implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anirudha",
    python_requires=">=3.8",

    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "ray>=2.0.0",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "flashinfer": [
            "flashinfer>=0.1.0",
        ],
    },

    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "attention-bench=attention_bench.cli.run_benchmark:main",
            "attention-bench-ray=attention_bench.cli.run_ray:main",
            "attention-config-gen=attention_bench.config.generator:main",
            "attention-heatmap=attention_bench.plotting.heatmap_decode:main",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
