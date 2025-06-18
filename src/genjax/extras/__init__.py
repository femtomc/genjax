"""
GenJAX extras - Additional functionality beyond core modules.

This module contains useful extensions and utilities that build on GenJAX core
functionality but are not part of the main inference algorithms.
"""

from .state_space import (
    # Discrete HMM functionality
    discrete_hmm,
    forward_filter,
    backward_sample,
    forward_filtering_backward_sampling,
    compute_sequence_log_prob,
    sample_hmm_dataset,
    DiscreteHMMTrace,
    # Linear Gaussian state space model functionality
    linear_gaussian,
    kalman_filter,
    kalman_smoother,
    sample_linear_gaussian_dataset,
    LinearGaussianTrace,
    # Unified testing API
    discrete_hmm_test_dataset,
    discrete_hmm_exact_log_marginal,
    linear_gaussian_test_dataset,
    linear_gaussian_exact_log_marginal,
    # Inference problem generators
    discrete_hmm_inference_problem,
    linear_gaussian_inference_problem,
)

__all__ = [
    # Discrete HMM
    "discrete_hmm",
    "forward_filter",
    "backward_sample",
    "forward_filtering_backward_sampling",
    "compute_sequence_log_prob",
    "sample_hmm_dataset",
    "DiscreteHMMTrace",
    # Linear Gaussian state space model
    "linear_gaussian",
    "kalman_filter",
    "kalman_smoother",
    "sample_linear_gaussian_dataset",
    "LinearGaussianTrace",
    # Unified testing API
    "discrete_hmm_test_dataset",
    "discrete_hmm_exact_log_marginal",
    "linear_gaussian_test_dataset",
    "linear_gaussian_exact_log_marginal",
    # Inference problem generators
    "discrete_hmm_inference_problem",
    "linear_gaussian_inference_problem",
]
