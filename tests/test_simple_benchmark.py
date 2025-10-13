"""
Simple benchmark test to verify pytest-benchmark setup.
"""

import pytest
import jax.numpy as jnp
from genjax.core import gen
from genjax.distributions import normal


@pytest.fixture
def simple_model():
    """Simple model for benchmarking."""

    @gen
    def model():
        x = normal(0.0, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return y

    return model


class TestSimpleBenchmarks:
    """Simple benchmarks to verify setup."""

    @pytest.mark.benchmark
    def test_jax_operation_benchmark(self, benchmark):
        """Benchmark a simple JAX operation."""

        def run_jax():
            x = jnp.array([1.0, 2.0, 3.0])
            return jnp.sum(x**2)

        result = benchmark(run_jax)
        assert result == 14.0

    @pytest.mark.benchmark
    def test_model_assess_benchmark(self, benchmark, simple_model):
        """Benchmark model assessment."""
        choices = {"x": 1.0, "y": 2.0}

        def run_assess():
            return simple_model.assess(choices)

        result = benchmark(run_assess)
        assert result is not None

    @pytest.mark.benchmark
    def test_distribution_simulate_benchmark(self, benchmark):
        """Benchmark direct distribution simulation."""

        def run_simulate():
            return normal.simulate(0.0, 1.0)

        result = benchmark(run_simulate)
        assert result is not None
