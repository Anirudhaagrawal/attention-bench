#!/usr/bin/env python3
"""Attention approach implementations and registry."""

from .base import AttentionApproach, ApproachRegistry, APPROACHES, BenchmarkContext

# Import all approach implementations to trigger registration
from . import flashinfer_approaches
from . import official_fa3_approaches

__all__ = ["AttentionApproach", "ApproachRegistry", "APPROACHES", "BenchmarkContext"]
