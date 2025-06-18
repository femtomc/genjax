"""
GenJAX extras - Additional functionality beyond core modules.

This module contains useful extensions and utilities that build on GenJAX core
functionality but are not part of the main inference algorithms.
"""

from .discrete_hmm import (
    discrete_hmm,
    forward_filter,
    backward_sample,
    forward_filtering_backward_sampling,
    compute_sequence_log_prob,
    sample_hmm_dataset,
    DiscreteHMMTrace,
)

__all__ = [
    "discrete_hmm",
    "forward_filter",
    "backward_sample",
    "forward_filtering_backward_sampling",
    "compute_sequence_log_prob",
    "sample_hmm_dataset",
    "DiscreteHMMTrace",
]
