"""
Test utilities and helper functions for GenJAX test suite.

This module provides common testing patterns, assertion helpers,
and reusable test components.
"""

import jax.numpy as jnp
import pytest


def assert_trace_consistency(trace1, trace2, tolerance=1e-6):
    """Assert that two traces are consistent (same choices, score, retval)."""
    assert jnp.allclose(trace1.get_score(), trace2.get_score(), rtol=tolerance)

    choices1, choices2 = trace1.get_choices(), trace2.get_choices()
    if isinstance(choices1, dict) and isinstance(choices2, dict):
        assert choices1.keys() == choices2.keys()
        for key in choices1:
            assert jnp.allclose(choices1[key], choices2[key], rtol=tolerance)
    else:
        assert jnp.allclose(choices1, choices2, rtol=tolerance)

    retval1, retval2 = trace1.get_retval(), trace2.get_retval()
    if isinstance(retval1, (tuple, list)) and isinstance(retval2, (tuple, list)):
        assert len(retval1) == len(retval2)
        for r1, r2 in zip(retval1, retval2):
            assert jnp.allclose(r1, r2, rtol=tolerance)
    else:
        assert jnp.allclose(retval1, retval2, rtol=tolerance)


def assert_simulate_assess_consistency(gen_fn, args, tolerance=1e-6):
    """Assert that simulate and assess are consistent for a generative function."""
    trace = gen_fn.simulate(args)
    choices = trace.get_choices()
    density, retval = gen_fn.assess(args, choices)

    expected_score = -density
    assert jnp.allclose(trace.get_score(), expected_score, rtol=tolerance)

    if isinstance(retval, (tuple, list)) and isinstance(
        trace.get_retval(), (tuple, list)
    ):
        assert len(retval) == len(trace.get_retval())
        for r1, r2 in zip(retval, trace.get_retval()):
            assert jnp.allclose(r1, r2, rtol=tolerance)
    else:
        assert jnp.allclose(retval, trace.get_retval(), rtol=tolerance)


def parametrized_model_test(test_func):
    """Decorator to parametrize a test across multiple models."""
    return pytest.mark.parametrize(
        "model_name",
        ["simple_normal_model", "hierarchical_normal_model", "exponential_model"],
    )(test_func)


def parametrized_tolerance_test(tolerances=None):
    """Decorator to parametrize a test across multiple tolerance values."""
    if tolerances is None:
        tolerances = [1e-6, 1e-8, 1e-10]
    return pytest.mark.parametrize("tolerance", tolerances)


def parametrized_sample_size_test(sizes=None):
    """Decorator to parametrize a test across multiple sample sizes."""
    if sizes is None:
        sizes = [10, 50, 100]
    return pytest.mark.parametrize("sample_size", sizes)


class ModelTestSuite:
    """Base class for model test suites with common test patterns."""

    @staticmethod
    def test_simulate_basic(model_fn, args=()):
        """Basic simulate test pattern."""
        trace = model_fn.simulate(args)
        assert hasattr(trace, "get_choices")
        assert hasattr(trace, "get_score")
        assert hasattr(trace, "get_retval")
        assert jnp.isfinite(trace.get_score())

    @staticmethod
    def test_assess_basic(model_fn, args=(), test_choices=None):
        """Basic assess test pattern."""
        if test_choices is None:
            trace = model_fn.simulate(args)
            test_choices = trace.get_choices()

        density, retval = model_fn.assess(args, test_choices)
        assert jnp.isfinite(density)
        return density, retval

    @staticmethod
    def test_consistency(model_fn, args=()):
        """Test simulate/assess consistency."""
        assert_simulate_assess_consistency(model_fn, args)


class DistributionTestSuite:
    """Base class for distribution test suites."""

    @staticmethod
    def test_distribution_basic(dist, args):
        """Basic distribution tests."""
        # Test simulate
        trace = dist.simulate(args)
        assert jnp.isfinite(trace.get_score())

        # Test assess
        choices = trace.get_choices()
        density, retval = dist.assess(args, choices)
        assert jnp.isfinite(density)
        assert jnp.allclose(retval, choices, rtol=1e-10)

        # Test consistency
        assert jnp.allclose(trace.get_score(), -density, rtol=1e-10)


class ConvergenceTestHelper:
    """Helper for convergence tests with proper statistical validation."""

    @staticmethod
    def test_mean_convergence(estimator_fn, true_value, n_samples=1000, tolerance=0.1):
        """Test that an estimator converges to the true value in expectation."""
        estimates = [estimator_fn() for _ in range(n_samples)]
        estimates = jnp.array(estimates)

        # Check all estimates are finite
        assert jnp.all(jnp.isfinite(estimates)), "Some estimates are not finite"

        # Check convergence
        mean_estimate = jnp.mean(estimates)
        std_estimate = jnp.std(estimates)

        # Use confidence interval for statistical test
        # For large n, sample mean is approximately normal
        margin_of_error = 1.96 * std_estimate / jnp.sqrt(n_samples)  # 95% CI

        assert jnp.abs(mean_estimate - true_value) <= tolerance, (
            f"Mean estimate {mean_estimate} not within {tolerance} of true value {true_value}. "
            f"95% CI margin: {margin_of_error}, std: {std_estimate}"
        )

        return {
            "mean": mean_estimate,
            "std": std_estimate,
            "margin_of_error": margin_of_error,
            "all_estimates": estimates,
        }

    @staticmethod
    def test_unbiasedness(estimator_fn, true_value, n_samples=1000, alpha=0.05):
        """Test unbiasedness using a t-test."""
        estimates = jnp.array([estimator_fn() for _ in range(n_samples)])

        # Remove any non-finite estimates
        finite_estimates = estimates[jnp.isfinite(estimates)]
        assert len(finite_estimates) > n_samples * 0.95, "Too many non-finite estimates"

        mean_est = jnp.mean(finite_estimates)
        std_est = jnp.std(finite_estimates)
        n = len(finite_estimates)

        # One-sample t-test: H0: mean = true_value
        t_stat = (mean_est - true_value) / (std_est / jnp.sqrt(n))

        # For large n, t-distribution approaches normal
        # Critical value for two-tailed test with alpha=0.05
        critical_value = 1.96 if n > 30 else 2.58  # Conservative

        assert jnp.abs(t_stat) < critical_value, (
            f"Estimate appears biased: t_stat={t_stat}, critical={critical_value}, "
            f"mean={mean_est}, true={true_value}, std={std_est}, n={n}"
        )


# Export commonly used test patterns
__all__ = [
    "assert_trace_consistency",
    "assert_simulate_assess_consistency",
    "parametrized_model_test",
    "parametrized_tolerance_test",
    "parametrized_sample_size_test",
    "ModelTestSuite",
    "DistributionTestSuite",
    "ConvergenceTestHelper",
]
