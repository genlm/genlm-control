"""Evaluation metrics for controlled text generation.

This module provides KL divergence estimation for comparing language models
using direct log probability evaluation.
"""

from .metrics import (
    kl_divergence_direct,
    kl_divergence_potentials,
    kl_divergence_sequences,
    effective_sample_size,
    perplexity_from_kl,
)

__all__ = [
    "kl_divergence_direct",
    "kl_divergence_potentials",
    "kl_divergence_sequences",
    "effective_sample_size",
    "perplexity_from_kl",
]
