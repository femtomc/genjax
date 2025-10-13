"""
Benchmark tests for GenJAX performance analysis.

This module provides benchmark fixtures and tests to identify performance
bottlenecks in GenJAX components using pytest-benchmark.

Usage:
    pixi run benchmark                 # Run only benchmark tests
    pixi run benchmark-all             # Run all tests with benchmarking
    pixi run benchmark-save            # Save benchmark results
    pixi run benchmark-slowest         # Show slowest tests without benchmarking
"""

import pytest
import jax.numpy as jnp
from jax import random

from genjax.core import gen, const
from genjax.distributions import normal
from genjax.inference import mh, hmc, mala, init, resample
from genjax.inference.vi import elbo_vi
from genjax.pjax import seed


@pytest.fixture
def simple_model():
    """Simple Bayesian model for benchmarking."""

    @gen
    def model():
        x = normal(0.0, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return y

    return model


@pytest.fixture
def hierarchical_model():
    """More complex hierarchical model for benchmarking."""

    @gen
    def model(n_groups=5, n_obs_per_group=10):
        # Global parameters
        mu_global = normal(0.0, 10.0) @ "mu_global"
        sigma_global = normal(0.0, 1.0) @ "sigma_global"

        # Group-specific parameters
        group_means = []
        for i in range(n_groups):
            group_mu = normal(mu_global, sigma_global) @ f"group_{i}/mu"
            group_means.append(group_mu)

            # Observations within group
            for j in range(n_obs_per_group):
                obs = normal(group_mu, 0.5) @ f"group_{i}/obs_{j}"

        return group_means

    return model


@pytest.fixture
def smc_model():
    """Sequential model for SMC benchmarking."""

    @gen
    def transition(prev_state):
        new_state = normal(prev_state, 0.1) @ "state"
        obs = normal(new_state, 0.05) @ "obs"
        return new_state

    return transition


class TestCorePerformance:
    """Benchmark core GenJAX operations."""

    @pytest.mark.benchmark
    def test_simple_simulate_benchmark(self, benchmark, simple_model):
        """Benchmark basic model simulation."""

        @seed
        def run_simulate():
            return simple_model.simulate()

        key = random.PRNGKey(42)
        result = benchmark(run_simulate, key)
        assert result is not None

    @pytest.mark.benchmark
    def test_simple_assess_benchmark(self, benchmark, simple_model):
        """Benchmark basic model assessment."""
        choices = {"x": 1.0, "y": 2.0}

        def run_assess():
            return simple_model.assess(choices)

        result = benchmark(run_assess)
        assert result is not None

    @pytest.mark.benchmark
    def test_hierarchical_simulate_benchmark(self, benchmark, hierarchical_model):
        """Benchmark hierarchical model simulation."""

        @seed
        def run_simulate():
            return hierarchical_model.simulate(5, 10)

        key = random.PRNGKey(42)
        result = benchmark(run_simulate, key)
        assert result is not None


class TestMCMCPerformance:
    """Benchmark MCMC algorithms."""

    @pytest.mark.benchmark
    def test_mh_benchmark(self, benchmark, simple_model):
        """Benchmark Metropolis-Hastings performance."""

        @seed
        def setup_and_run():
            # Setup
            trace = simple_model.simulate()

            # Single MH step
            from genjax.core import sel

            selection = sel("x")
            return mh(trace, selection)

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None

    @pytest.mark.benchmark
    def test_hmc_benchmark(self, benchmark, simple_model):
        """Benchmark HMC performance."""

        @seed
        def setup_and_run():
            # Setup
            trace = simple_model.simulate()

            # Single HMC step
            from genjax.core import sel

            selection = sel("x")
            step_size = 0.1
            n_steps = 10
            return hmc(trace, selection, step_size, n_steps)

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None

    @pytest.mark.benchmark
    def test_mala_benchmark(self, benchmark, simple_model):
        """Benchmark MALA performance."""

        @seed
        def setup_and_run():
            # Setup
            trace = simple_model.simulate()

            # Single MALA step
            from genjax.core import sel

            selection = sel("x")
            step_size = 0.01
            return mala(trace, selection, step_size)

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None


class TestSMCPerformance:
    """Benchmark SMC algorithms."""

    @pytest.mark.benchmark
    def test_smc_init_benchmark(self, benchmark, simple_model):
        """Benchmark SMC initialization."""

        @seed
        def setup_and_run():
            constraints = {"y": 1.5}
            n_particles = const(100)
            return init(simple_model, (), n_particles, constraints)

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None

    @pytest.mark.benchmark
    def test_smc_resample_benchmark(self, benchmark, simple_model):
        """Benchmark SMC resampling."""

        @seed
        def setup_and_run():
            # Initialize particles with varying weights
            constraints = {"y": 1.5}
            n_particles = const(100)
            particles = init(simple_model, (), n_particles, constraints)

            return resample(particles)

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None


class TestVIPerformance:
    """Benchmark Variational Inference."""

    @pytest.mark.benchmark
    def test_vi_elbo_benchmark(self, benchmark, simple_model):
        """Benchmark ELBO computation."""

        # Create a simple variational family matching the working tests
        @gen
        def variational_family(constraint, theta):
            from genjax.adev import normal_reinforce

            normal_reinforce(theta, 1.0) @ "x"

        @seed
        def setup_and_run():
            # Setup VI matching the working tests
            constraints = {"y": 1.5}
            init_params = jnp.array(0.1)  # Single scalar parameter

            # Run VI optimization for benchmarking
            return elbo_vi(
                simple_model,
                variational_family,
                init_params,
                constraints,
                target_args=(),
                learning_rate=1e-3,
                n_iterations=50,  # Minimal iterations for benchmarking
            )

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None


class TestVectorizationPerformance:
    """Benchmark vectorization performance."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [10, 100, 1000])
    def test_vmap_simulation_benchmark(self, benchmark, simple_model, batch_size):
        """Benchmark vectorized simulation with different batch sizes."""

        @seed
        def setup_and_run():
            from genjax.pjax import modular_vmap

            # Vectorize simulation
            def single_sim(_):
                return simple_model.simulate()

            vectorized_sim = modular_vmap(single_sim, in_axes=(0,))

            # Run vectorized simulation
            return vectorized_sim(jnp.arange(batch_size))

        key = random.PRNGKey(42)
        result = benchmark(setup_and_run, key)
        assert result is not None


# Test fixtures for common benchmark patterns
@pytest.fixture
def benchmark_model_suite():
    """Collection of models for comprehensive benchmarking."""
    models = {}

    @gen
    def simple():
        x = normal(0.0, 1.0) @ "x"
        return x

    models["simple"] = simple

    @gen
    def medium():
        params = []
        for i in range(10):
            param = normal(0.0, 1.0) @ f"param_{i}"
            params.append(param)
        return jnp.array(params)

    models["medium"] = medium

    @gen
    def complex():
        # Hierarchical model with 50 parameters
        global_mean = normal(0.0, 10.0) @ "global_mean"
        global_std = normal(0.0, 1.0) @ "global_std"

        local_params = []
        for i in range(10):
            local_mean = normal(global_mean, global_std) @ f"local_{i}/mean"
            for j in range(5):
                param = normal(local_mean, 0.5) @ f"local_{i}/param_{j}"
                local_params.append(param)

        return jnp.array(local_params)

    models["complex"] = complex

    return models


class TestComprehensiveBenchmarks:
    """Comprehensive benchmarks across model complexity."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("model_name", ["simple", "medium", "complex"])
    def test_simulation_scaling(self, benchmark, benchmark_model_suite, model_name):
        """Benchmark simulation across model complexity."""
        model = benchmark_model_suite[model_name]

        @seed
        def run_simulation():
            return model.simulate()

        key = random.PRNGKey(42)
        result = benchmark(run_simulation, key)
        assert result is not None

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_particles", [10, 100, 1000])
    def test_smc_scaling(self, benchmark, simple_model, n_particles):
        """Benchmark SMC performance scaling with particle count."""

        @seed
        def run_smc():
            constraints = {"x": 0.5}
            return init(simple_model, (), const(n_particles), constraints)

        key = random.PRNGKey(42)
        result = benchmark(run_smc, key)
        assert result is not None


# Utility functions for benchmark analysis
def analyze_benchmark_results(benchmark_file=".benchmarks/benchmarks.json"):
    """
    Analyze saved benchmark results to identify performance patterns.

    This function can be called after running benchmarks to get insights
    into which operations are slowest.
    """
    try:
        import json

        with open(benchmark_file, "r") as f:
            data = json.load(f)

        # Extract and sort benchmark results
        benchmarks = data.get("benchmarks", [])
        sorted_benchmarks = sorted(
            benchmarks, key=lambda x: x["stats"]["mean"], reverse=True
        )

        print("=== Top 10 Slowest Operations ===")
        for i, bench in enumerate(sorted_benchmarks[:10]):
            name = bench["name"]
            mean_time = bench["stats"]["mean"]
            print(f"{i + 1:2d}. {name}: {mean_time:.4f}s")

        return sorted_benchmarks
    except FileNotFoundError:
        print(f"Benchmark file {benchmark_file} not found. Run benchmarks first.")
        return []


if __name__ == "__main__":
    # Example usage when run directly
    print("This module contains benchmark tests for GenJAX.")
    print("Run with: pixi run benchmark")
    print("")
    print("Available benchmark commands:")
    print("  pixi run benchmark        - Run only benchmark tests")
    print("  pixi run benchmark-all    - Run all tests with benchmarking")
    print("  pixi run benchmark-save   - Save benchmark results")
    print("  pixi run benchmark-slowest - Show slowest tests")
