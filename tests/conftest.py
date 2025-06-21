"""
Shared fixtures and configuration for GenJAX test suite.

This module provides common fixtures, test utilities, and configuration
that can be used across all test files to reduce duplication and improve
maintainability.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
from typing import Callable
from genjax import gen, normal, exponential
from genjax.adev import expectation


# ============================================================================
# Random Key Fixtures
# ============================================================================


@pytest.fixture
def base_key():
    """Base random key for reproducible tests."""
    return jrand.key(42)


@pytest.fixture
def key_sequence(base_key):
    """Sequence of 10 split random keys."""
    return jrand.split(base_key, 10)


@pytest.fixture
def unique_key():
    """Generate a unique key for each test (non-deterministic)."""
    import time

    return jrand.key(int(time.time() * 1000) % 2**32)


# ============================================================================
# Test Tolerance Fixtures
# ============================================================================


@pytest.fixture
def strict_tolerance():
    """Strict tolerance for high-precision tests."""
    return 1e-10


@pytest.fixture
def standard_tolerance():
    """Standard tolerance for most numerical tests."""
    return 1e-6


@pytest.fixture
def loose_tolerance():
    """Loose tolerance for stochastic/convergence tests."""
    return 1e-2


@pytest.fixture
def convergence_tolerance():
    """Tolerance for convergence tests with inherent variance."""
    return 0.1


# ============================================================================
# Sample Size Fixtures
# ============================================================================


@pytest.fixture
def small_sample_size():
    """Small sample size for fast tests."""
    return 10


@pytest.fixture
def medium_sample_size():
    """Medium sample size for balanced speed/accuracy."""
    return 100


@pytest.fixture
def large_sample_size():
    """Large sample size for convergence tests."""
    return 1000


# ============================================================================
# Common Model Fixtures
# ============================================================================


@pytest.fixture
def simple_normal_model():
    """Simple normal distribution model."""

    @gen
    def model(mu, sigma):
        x = normal(mu, sigma) @ "x"
        return x

    return model


@pytest.fixture
def hierarchical_normal_model():
    """Hierarchical normal model for inference tests."""

    @gen
    def model(prior_mean, prior_std, obs_std):
        mu = normal(prior_mean, prior_std) @ "mu"
        y = normal(mu, obs_std) @ "y"
        return y

    return model


@pytest.fixture
def bivariate_normal_model():
    """Bivariate normal model with dependency."""

    @gen
    def model():
        x = normal(0.0, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return (x, y)

    return model


@pytest.fixture
def exponential_model():
    """Simple exponential model."""

    @gen
    def model(rate):
        x = exponential(rate) @ "x"
        return x

    return model


# ============================================================================
# ADEV/Expectation Fixtures
# ============================================================================


@pytest.fixture
def quadratic_expectation():
    """Quadratic expectation function for ADEV tests."""

    @expectation
    def objective(mu, sigma):
        x = normal(mu, sigma) @ "x"
        return x**2

    return objective


@pytest.fixture
def linear_expectation():
    """Linear expectation function for ADEV tests."""

    @expectation
    def objective(mu, sigma):
        x = normal(mu, sigma) @ "x"
        return x

    return objective


# ============================================================================
# HMM Parameters Fixtures
# ============================================================================


@pytest.fixture
def simple_hmm_params():
    """Simple 2-state HMM parameters."""
    initial_probs = jnp.array([0.6, 0.4])
    transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    emission_matrix = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    return {
        "initial_probs": initial_probs,
        "transition_matrix": transition_matrix,
        "emission_matrix": emission_matrix,
        "n_states": 2,
        "n_obs": 2,
    }


@pytest.fixture
def complex_hmm_params():
    """Complex 3-state, 4-observation HMM parameters."""
    initial_probs = jnp.array([0.5, 0.3, 0.2])
    transition_matrix = jnp.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    emission_matrix = jnp.array(
        [
            [0.4, 0.3, 0.2, 0.1],  # State 0
            [0.1, 0.4, 0.4, 0.1],  # State 1
            [0.1, 0.1, 0.2, 0.6],  # State 2
        ]
    )
    return {
        "initial_probs": initial_probs,
        "transition_matrix": transition_matrix,
        "emission_matrix": emission_matrix,
        "n_states": 3,
        "n_obs": 4,
    }


# ============================================================================
# VI Test Fixtures
# ============================================================================


@pytest.fixture
def vi_target_model():
    """Target model for variational inference tests."""

    @gen
    def target():
        x = normal(0.0, 1.0) @ "x"
        normal(x, 0.3) @ "y"

    return target


@pytest.fixture
def vi_simple_family():
    """Simple variational family."""

    @gen
    def family(theta):
        normal(theta, 1.0) @ "x"

    return family


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def test_observations():
    """Standard test observations for various models."""
    return {"y": jnp.array(2.0)}


@pytest.fixture
def hmm_test_sequence():
    """Test observation sequence for HMM tests."""
    return jnp.array([0, 1, 0, 1, 1, 0, 0, 1])


@pytest.fixture
def multivariate_test_data():
    """Multivariate test data."""
    return {
        "loc": jnp.array([0.0, 1.0]),
        "cov": jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        "samples": jnp.array([[0.5, 1.2], [-0.3, 0.8], [1.1, 2.1]]),
    }


# ============================================================================
# Test Utilities
# ============================================================================


class TestHelpers:
    """Collection of helper methods for tests."""

    @staticmethod
    def assert_finite_and_close(actual, expected, rtol=1e-6, msg=""):
        """Assert that values are finite and close to expected."""
        assert jnp.all(jnp.isfinite(actual)), f"Values not finite: {actual} {msg}"
        assert jnp.allclose(actual, expected, rtol=rtol), (
            f"{actual} != {expected} (rtol={rtol}) {msg}"
        )

    @staticmethod
    def assert_valid_trace(trace):
        """Assert that a trace has valid structure and finite values."""
        assert hasattr(trace, "get_choices"), "Trace missing get_choices method"
        assert hasattr(trace, "get_score"), "Trace missing get_score method"
        assert hasattr(trace, "get_retval"), "Trace missing get_retval method"

        score = trace.get_score()
        assert jnp.isfinite(score), f"Trace score not finite: {score}"

        retval = trace.get_retval()
        if isinstance(retval, (jnp.ndarray, float, int)):
            assert jnp.all(jnp.isfinite(retval)), f"Trace retval not finite: {retval}"

    @staticmethod
    def assert_valid_density(density):
        """Assert that a density value is valid (finite, possibly negative)."""
        assert jnp.isfinite(density), f"Density not finite: {density}"
        # Note: densities can be negative (log probabilities)

    @staticmethod
    def assert_gradient_finite(grad):
        """Assert that gradient estimates are finite."""
        if isinstance(grad, tuple):
            for i, g in enumerate(grad):
                assert jnp.all(jnp.isfinite(g)), f"Gradient {i} not finite: {g}"
        else:
            assert jnp.all(jnp.isfinite(grad)), f"Gradient not finite: {grad}"


@pytest.fixture
def helpers():
    """Test helper utilities."""
    return TestHelpers()


# ============================================================================
# Pytest Hooks and Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with additional settings."""
    # Add custom markers dynamically if needed
    pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add markers based on file names
        if "test_adev" in item.fspath.basename:
            item.add_marker(pytest.mark.adev)
        elif "test_smc" in item.fspath.basename:
            item.add_marker(pytest.mark.smc)
        elif "test_vi" in item.fspath.basename:
            item.add_marker(pytest.mark.vi)
        elif "test_hmm" in item.fspath.basename:
            item.add_marker(pytest.mark.hmm)
        elif "test_mcmc" in item.fspath.basename:
            item.add_marker(pytest.mark.mcmc)
        elif "test_core" in item.fspath.basename:
            item.add_marker(pytest.mark.core)

        # Add speed markers based on test names
        if "convergence" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow)
        elif "basic" in item.name or "simple" in item.name:
            item.add_marker(pytest.mark.fast)

        # Add TFP marker for tests that import tensorflow_probability
        if hasattr(item, "module") and "tensorflow_probability" in str(
            item.module.__dict__.get("__file__", "")
        ):
            item.add_marker(pytest.mark.tfp)


# ============================================================================
# JIT Compilation Fixtures for Test Optimization
# ============================================================================


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

        if static_argnames:
            jitted = jax.jit(fn, static_argnames=static_argnames)
        else:
            jitted = jax.jit(fn)

        jit_cache[name] = jitted
        return jitted

    return compile_and_cache


@pytest.fixture(scope="module")
def batch_tester():
    """Utilities for batch testing with vmap."""

    def create_batch_test(test_fn: Callable) -> Callable:
        """Convert single-input test to batch test."""
        return jax.vmap(test_fn)

    return {
        "create_batch": create_batch_test,
        "assert_close": lambda a, e, **kw: jnp.allclose(a, e, **kw),
    }


@pytest.fixture(scope="module")
def jitted_distributions(jit_compiler):
    """Pre-compiled distribution operations for fast testing."""
    from genjax.distributions import normal, binomial

    distributions = {}

    # Compile batch assess operations
    @jax.jit
    def normal_assess_batch(samples, mu, sigma):
        def single(s):
            lp, _ = normal.assess(s, mu, sigma)
            return lp

        return jax.vmap(single)(samples)

    @jax.jit
    def binomial_assess_batch(samples, n, p):
        def single(s):
            lp, _ = binomial.assess(s, n, p)
            return lp

        return jax.vmap(single)(samples)

    distributions["normal_batch"] = normal_assess_batch
    distributions["binomial_batch"] = binomial_assess_batch

    return distributions


@pytest.fixture(scope="module")
def jitted_hmm_ops(jit_compiler):
    """Pre-compiled HMM operations for fast testing."""
    from genjax.extras.state_space import forward_filter

    ops = {}
    ops["forward_filter"] = jit_compiler("hmm_forward", forward_filter)

    @jax.jit
    def batch_forward(sequences, initial_probs, transition_matrix, emission_matrix):
        def single(seq):
            _, lm = forward_filter(
                seq, initial_probs, transition_matrix, emission_matrix
            )
            return lm

        return jax.vmap(single)(sequences)

    ops["batch_forward"] = batch_forward
    return ops


# ============================================================================
# Parametrized Test Data
# ============================================================================

# Common parameter sets for reuse across tests
COMMON_TOLERANCES = [1e-6, 1e-8, 1e-10]
COMMON_SAMPLE_SIZES = [10, 50, 100]
COMMON_DISTRIBUTIONS = [
    ("normal", {"mu": 0.0, "sigma": 1.0}),
    ("exponential", {"rate": 1.0}),
]
COMMON_RANDOM_SEEDS = [42, 123, 456, 789]

# Export commonly used parametrize decorators
tolerance_params = pytest.mark.parametrize("tolerance", COMMON_TOLERANCES)
sample_size_params = pytest.mark.parametrize("sample_size", COMMON_SAMPLE_SIZES)
distribution_params = pytest.mark.parametrize(
    "dist_name,dist_params", COMMON_DISTRIBUTIONS
)
seed_params = pytest.mark.parametrize("seed", COMMON_RANDOM_SEEDS)
