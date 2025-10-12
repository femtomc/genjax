"""
Shared JIT compilation utilities for test optimization.

Add this to your existing conftest.py or rename to conftest.py to use.
"""

import pytest
import jax
import jax.numpy as jnp
from typing import Callable


@pytest.fixture(scope="session")
def jit_cache():
    """Session-wide cache for JIT-compiled functions."""
    return {}


@pytest.fixture(scope="module")
def jit_compiler(jit_cache):
    """Provide JIT compilation with caching."""

    def compile_and_cache(name: str, fn: Callable, static_argnames=None) -> Callable:
        """Compile a function with JIT and cache it."""
        if name in jit_cache:
            return jit_cache[name]

        # Compile function
        if static_argnames:
            jitted = jax.jit(fn, static_argnames=static_argnames)
        else:
            jitted = jax.jit(fn)

        # Cache it
        jit_cache[name] = jitted

        return jitted

    return compile_and_cache


@pytest.fixture(scope="module")
def batch_tester():
    """Provide utilities for batch testing."""

    def create_batch_test(test_fn: Callable) -> Callable:
        """Convert a single-input test to a batch test."""
        return jax.vmap(test_fn)

    def batch_assert_close(actual, expected, **kwargs):
        """Assert that all elements in a batch are close."""
        return jnp.allclose(actual, expected, **kwargs)

    return {
        "create_batch": create_batch_test,
        "assert_close": batch_assert_close,
    }


@pytest.fixture(scope="module")
def warmup_jit():
    """Warm up JIT compilation with dummy data."""

    def warmup(jitted_fn: Callable, dummy_args: tuple) -> None:
        """Warm up a JIT-compiled function."""
        try:
            _ = jitted_fn(*dummy_args)
        except Exception:
            # If warmup fails, that's okay - the actual test will catch real errors
            pass

    return warmup


# Distribution-specific JIT fixtures
@pytest.fixture(scope="module")
def jitted_distributions(jit_compiler):
    """Pre-compiled distribution operations."""

    # Compile assess methods for common distributions
    distributions = {}

    # Import distributions
    from genjax.distributions import (
        normal,
        binomial,
    )

    # Normal distribution
    @jax.jit
    def normal_assess_batch(samples, mu, sigma):
        def single_assess(sample):
            logprob, _ = normal.assess(sample, mu, sigma)
            return logprob

        return jax.vmap(single_assess)(samples)

    distributions["normal_assess_batch"] = normal_assess_batch

    # Binomial distribution
    @jax.jit
    def binomial_assess_batch(samples, n, p):
        def single_assess(sample):
            logprob, _ = binomial.assess(sample, n, p)
            return logprob

        return jax.vmap(single_assess)(samples)

    distributions["binomial_assess_batch"] = binomial_assess_batch

    return distributions


# HMM-specific JIT fixtures
@pytest.fixture(scope="module")
def jitted_hmm_ops(jit_compiler):
    """Pre-compiled HMM operations."""
    from genjax.extras.state_space import forward_filter

    ops = {}

    # JIT compile forward filter
    ops["forward_filter"] = jit_compiler(
        "hmm_forward_filter",
        forward_filter,
        static_argnames=["obs_seq"],  # Sequence is often static in tests
    )

    # Batch forward filter for multiple sequences
    @jax.jit
    def batch_forward_filter(
        sequences, initial_probs, transition_matrix, emission_matrix
    ):
        """Run forward filter on multiple sequences."""

        def single_forward(seq):
            _, log_marginal = forward_filter(
                seq, initial_probs, transition_matrix, emission_matrix
            )
            return log_marginal

        return jax.vmap(single_forward)(sequences)

    ops["batch_forward_filter"] = batch_forward_filter

    return ops


# Test timing utilities
@pytest.fixture
def timer():
    """Simple timer for measuring test performance."""
    import time

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.times.append(time.time() - self.start)

        def mean(self):
            return sum(self.times) / len(self.times) if self.times else 0

        def total(self):
            return sum(self.times)

    return Timer()
