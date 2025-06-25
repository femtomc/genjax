"""
Test cases for GenJAX performance and scalability.

These tests validate performance characteristics including:
- Compilation times and caching
- Memory usage patterns
- Scalability with problem size
- JAX transformation efficiency
- Inference algorithm performance
"""

import time
import jax.numpy as jnp
import jax.random as jrand
import pytest
from jax import jit, vmap
from jax.lax import scan

from genjax import gen
from genjax.core import sel, const
from genjax.pjax import seed, modular_vmap
from genjax.distributions import normal, exponential
from genjax.adev import expectation
from genjax import normal_reparam, normal_reinforce

# GP module removed - GP performance tests disabled
from genjax.inference.mcmc import mh


# =============================================================================
# COMPILATION AND CACHING TESTS
# =============================================================================


class TestCompilationPerformance:
    """Test JAX compilation and caching performance."""

    def test_generative_function_compilation_caching(self):
        """Test that generative functions compile and cache efficiently."""

        @gen
        def test_model(x, y):
            z = normal(x + y, 1.0) @ "z"
            return z

        # First compilation
        key = jrand.key(42)
        start_time = time.time()

        jitted_simulate = jit(seed(test_model.simulate))
        first_result = jitted_simulate(key, 1.0, 2.0)

        first_compile_time = time.time() - start_time

        # Second call (should be cached)
        start_time = time.time()
        second_result = jitted_simulate(key, 1.0, 2.0)
        cached_time = time.time() - start_time

        # Cached call should be much faster
        assert cached_time < first_compile_time / 10

        # Results should be identical (same key)
        assert jnp.allclose(first_result.get_retval(), second_result.get_retval())

    def test_adev_compilation_performance(self):
        """Test ADEV gradient estimation compilation."""

        @expectation
        def adev_objective(params):
            x = normal_reparam(params[0], params[1])
            y = normal_reinforce(x, params[2])
            return x + y

        params = jnp.array([0.5, 1.0, 0.5])

        # First gradient computation (includes compilation)
        start_time = time.time()
        first_grad = adev_objective.grad_estimate(params)
        first_time = time.time() - start_time

        # Second computation (should be cached)
        start_time = time.time()
        second_grad = adev_objective.grad_estimate(params)
        cached_time = time.time() - start_time

        # Both should be finite
        assert jnp.all(jnp.isfinite(first_grad))
        assert jnp.all(jnp.isfinite(second_grad))

        # Cached should be faster (though results may differ due to randomness)
        assert cached_time < first_time / 2

    def test_gp_compilation_scaling_DISABLED(self):
        """Test GP compilation scaling with data size. DISABLED: GP module removed."""
        pytest.skip("GP module removed from GenJAX")

    def test_nested_transformation_compilation(self):
        """Test compilation of nested JAX transformations."""

        @gen
        def simple_model(x):
            y = normal(x, 1.0) @ "y"
            return y

        # Nested vmap and jit
        def nested_operation(x_batch):
            vmapped_simulate = modular_vmap(seed(simple_model.simulate))
            return vmapped_simulate(jrand.split(jrand.key(42), len(x_batch)), x_batch)

        jitted_nested = jit(nested_operation)

        x_batch = jnp.array([1.0, 2.0, 3.0])

        # Should compile and run without errors
        start_time = time.time()
        result = jitted_nested(x_batch)
        compile_time = time.time() - start_time

        # Should be reasonable compilation time (< 10 seconds)
        assert compile_time < 10.0

        # Should produce correct output shape
        assert result.get_retval().shape == (3,)


# =============================================================================
# MEMORY USAGE TESTS
# =============================================================================


class TestMemoryUsage:
    """Test memory usage patterns and efficiency."""

    def test_large_trace_memory_efficiency(self):
        """Test memory efficiency with large traces."""

        @gen
        def large_model():
            # Create model with many variables
            total = 0.0
            for i in range(1000):
                x = normal(0.0, 1.0) @ f"x_{i}"
                total += x
            return total / 1000.0

        key = jrand.key(42)

        # Should handle large traces without excessive memory
        trace = seed(large_model.simulate)(key)
        choices = trace.get_choices()

        # Should have all 1000 variables
        assert len(choices) == 1000
        for i in range(1000):
            assert f"x_{i}" in choices

        # Return value should be finite
        assert jnp.isfinite(trace.get_retval())

    def test_sequential_memory_usage(self):
        """Test memory usage in sequential operations."""

        from genjax.core import Scan

        @gen
        def step_model(carry, x):
            noise = normal(0.0, 0.1) @ "step"  # Center noise at 0
            new_carry = carry * 0.9 + noise + x  # Damping factor to prevent explosion
            return new_carry, noise

        @gen
        def sequential_model(init, xs):
            scan_gf = Scan(step_model, length=const(len(xs)))
            final, outputs = scan_gf(init, xs) @ "scan_steps"
            return final

        # Test with increasing sequence lengths
        for n_steps in [10, 100, 500]:
            key = jrand.key(123)
            init = 0.0
            xs = jnp.zeros(n_steps)  # Create outside the function
            trace = seed(sequential_model.simulate)(key, init, xs)

            # Should complete without memory issues
            assert jnp.isfinite(trace.get_retval())

            choices = trace.get_choices()
            # Should have the scan_steps containing the step samples
            assert "scan_steps" in choices
            # The step samples should be an array of length n_steps
            step_samples = choices["scan_steps"]["step"]
            assert len(step_samples) == n_steps

    def test_gp_memory_scaling_DISABLED(self):
        """Test GP memory usage scaling. DISABLED: GP module removed."""
        pytest.skip("GP module removed from GenJAX")

    def test_inference_memory_patterns(self):
        """Test memory usage in inference algorithms."""

        @gen
        def target_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        # MCMC chain
        key = jrand.key(789)
        initial_trace = seed(target_model.simulate)(key)

        def mcmc_step(trace, key):
            # Use MH with selection - it will use the model's internal proposal
            return mh(trace, sel("x"))

        # Run MCMC chain
        keys = jrand.split(key, 100)

        def scan_mcmc(trace, key):
            new_trace, accepted = mcmc_step(trace, key)
            return new_trace, new_trace

        final_trace, all_traces = scan(scan_mcmc, initial_trace, keys)

        # Should complete without memory issues
        assert hasattr(final_trace, "get_choices")
        assert all_traces.get_retval().shape == (100,)


# =============================================================================
# SCALABILITY TESTS
# =============================================================================


class TestScalabilityPerformance:
    """Test performance scaling with problem size."""

    def test_model_complexity_scaling(self):
        """Test performance scaling with model complexity."""

        def create_hierarchical_model(n_groups, n_per_group):
            @gen
            def hierarchical_model():
                # Population parameters
                mu_global = normal(0.0, 1.0) @ "mu_global"
                sigma_global = exponential(1.0) @ "sigma_global"

                # Group parameters
                for g in range(n_groups):
                    mu_group = normal(mu_global, sigma_global) @ f"mu_group_{g}"
                    sigma_group = exponential(1.0) @ f"sigma_group_{g}"

                    # Individual observations
                    for i in range(n_per_group):
                        normal(mu_group, sigma_group) @ f"obs_{g}_{i}"

                return mu_global

            return hierarchical_model

        # Test scaling with complexity
        complexity_configs = [
            (2, 5),  # 2 groups, 5 per group
            (3, 10),  # 3 groups, 10 per group
            (5, 20),  # 5 groups, 20 per group
        ]

        simulation_times = []

        for n_groups, n_per_group in complexity_configs:
            model = create_hierarchical_model(n_groups, n_per_group)
            key = jrand.key(42)

            # Time simulation
            start_time = time.time()
            trace = seed(model.simulate)(key)
            sim_time = time.time() - start_time

            simulation_times.append(sim_time)

            # Should produce valid trace
            assert jnp.isfinite(trace.get_retval())

            # Should have correct number of variables
            choices = trace.get_choices()
            expected_vars = 2 + 2 * n_groups + n_groups * n_per_group
            assert len(choices) == expected_vars

        # Time should scale reasonably (not exponentially)
        assert simulation_times[-1] < simulation_times[0] * 10

    def test_adev_parameter_scaling(self):
        """Test ADEV performance scaling with parameter dimension."""

        def create_high_dim_objective(n_params):
            @expectation
            def objective(params):
                assert params.shape == (n_params,)
                total = 0.0
                for i in range(n_params):
                    x = normal_reparam(params[i], 1.0)
                    total += x**2
                return total / n_params

            return objective

        param_dimensions = [10, 50, 100]
        gradient_times = []

        for n_params in param_dimensions:
            objective = create_high_dim_objective(n_params)
            params = jnp.zeros(n_params)

            # Time gradient computation
            start_time = time.time()
            grad = objective.grad_estimate(params)
            grad_time = time.time() - start_time

            gradient_times.append(grad_time)

            # Should produce correct gradient
            assert grad.shape == (n_params,)
            assert jnp.all(jnp.isfinite(grad))

        # Gradient time should scale reasonably
        assert gradient_times[-1] < gradient_times[0] * 20

    def test_gp_data_size_scaling_DISABLED(self):
        """Test GP performance scaling with data size. DISABLED: GP module removed."""
        pytest.skip("GP module removed from GenJAX")

    def test_inference_chain_length_scaling(self):
        """Test inference performance scaling with chain length."""

        @gen
        def simple_target():
            x = normal(0.0, 1.0) @ "x"
            normal(x, 0.5) @ "y"

        @gen
        def simple_proposal(trace):
            old_x = trace.get_choices()["x"]
            x = normal(old_x, 0.1) @ "x"
            normal(x, 0.5) @ "y"

        # Test different chain lengths
        chain_lengths = [10, 50, 100]
        chain_times = []

        for n_steps in chain_lengths:
            key = jrand.key(456)
            initial_trace = seed(simple_target.simulate)(key)

            # Constrain to observed data
            data = {"y": 1.0}
            initial_trace = simple_target.generate(data)[0]

            def mcmc_step(trace, key):
                return mh(trace, sel("y"))

            keys = jrand.split(key, n_steps)

            # Time MCMC chain
            start_time = time.time()

            def scan_mcmc(trace, key):
                new_trace, accepted = mcmc_step(trace, key)
                return new_trace, accepted

            final_trace, accepts = scan(scan_mcmc, initial_trace, keys)
            chain_time = time.time() - start_time

            chain_times.append(chain_time)

            # Should produce valid results
            assert hasattr(final_trace, "get_choices")
            assert accepts.shape == (n_steps,)

        # Chain time should scale linearly
        assert chain_times[-1] < chain_times[0] * 15


# =============================================================================
# JAX TRANSFORMATION EFFICIENCY TESTS
# =============================================================================


class TestJAXTransformationEfficiency:
    """Test efficiency of JAX transformations with GenJAX."""

    def test_vmap_efficiency(self):
        """Test vmap efficiency compared to manual batching."""

        @gen
        def single_model(x):
            y = normal(x, 1.0) @ "y"
            return y

        # Vectorized version
        vmapped_model = single_model.vmap(in_axes=(0,))

        x_batch = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        keys = jrand.split(jrand.key(42), len(x_batch))

        # Time vectorized version
        start_time = time.time()
        vmapped_result = vmapped_model.simulate(x_batch)
        vmap_time = time.time() - start_time

        # Time manual batching
        start_time = time.time()
        manual_results = []
        for i, x in enumerate(x_batch):
            result = seed(single_model.simulate)(keys[i], x)
            manual_results.append(result.get_retval())
        manual_time = time.time() - start_time

        # Vectorized should be faster for this size
        # (Though compilation overhead might make this variable)
        assert vmap_time < manual_time * 2  # Allow some overhead

        # Results should have correct shapes
        assert vmapped_result.get_retval().shape == (5,)

    def test_scan_vs_python_loop_efficiency(self):
        """Test scan efficiency compared to Python loops."""

        @gen
        def step_model(carry, x):
            new_carry = normal(carry + x, 0.1) @ "step"
            return new_carry, new_carry

        # Scan version
        @gen
        def scan_model(init, xs):
            final, outputs = scan(step_model, init, xs)
            return outputs

        init = 0.0
        xs = jnp.array([0.1, -0.1, 0.2, -0.2, 0.1])
        key = jrand.key(123)

        # Time scan version
        start_time = time.time()
        scan_result = seed(scan_model.simulate)(key, init, xs)
        scan_time = time.time() - start_time

        # Scan should produce valid results
        assert scan_result.get_retval().shape == (5,)
        assert jnp.all(jnp.isfinite(scan_result.get_retval()))

        # Should complete in reasonable time
        assert scan_time < 5.0  # Should be fast

    def test_nested_transformation_efficiency(self):
        """Test efficiency of nested transformations."""

        @gen
        def base_model(x, y):
            z = normal(x + y, 1.0) @ "z"
            return z

        # Create nested vmap + scan
        def nested_operation(x_matrix, y_vector):
            # x_matrix: (n_batch, n_steps)
            # y_vector: (n_steps,)

            def single_batch(x_row):
                def step(carry, inputs):
                    x_val, y_val = inputs
                    result = seed(base_model.simulate)(carry, x_val, y_val)
                    return carry, result.get_retval()

                keys = jrand.split(jrand.key(42), len(x_row))
                _, outputs = scan(step, keys[0], (x_row, y_vector))
                return outputs

            return vmap(single_batch)(x_matrix)

        x_matrix = jnp.ones((3, 4))  # 3 batches, 4 steps each
        y_vector = jnp.array([0.1, 0.2, 0.3, 0.4])

        # Should handle nested transformations efficiently
        start_time = time.time()
        result = nested_operation(x_matrix, y_vector)
        nested_time = time.time() - start_time

        # Should produce correct output shape
        assert result.shape == (3, 4)
        assert jnp.all(jnp.isfinite(result))

        # Should complete in reasonable time
        assert nested_time < 10.0

    def test_jit_compilation_benefits(self):
        """Test JIT compilation performance benefits."""

        @gen
        def compute_heavy_model(n_vars):
            total = 0.0
            for i in range(n_vars):
                x = normal(0.0, 1.0) @ f"x_{i}"
                # Some computation
                total += jnp.sin(x) * jnp.cos(x**2)
            return total / n_vars

        n_vars = 50
        key = jrand.key(789)

        # Non-JIT version
        start_time = time.time()
        result_no_jit = seed(compute_heavy_model.simulate)(key, n_vars)
        no_jit_time = time.time() - start_time

        # JIT version
        jitted_simulate = jit(seed(compute_heavy_model.simulate))

        # First call (includes compilation)
        start_time = time.time()
        result_jit_first = jitted_simulate(key, n_vars)
        jit_first_time = time.time() - start_time

        # Second call (cached)
        start_time = time.time()
        result_jit_cached = jitted_simulate(key, n_vars)
        jit_cached_time = time.time() - start_time

        # Results should be identical
        assert jnp.allclose(result_no_jit.get_retval(), result_jit_first.get_retval())
        assert jnp.allclose(
            result_jit_first.get_retval(), result_jit_cached.get_retval()
        )

        # Cached JIT should be fastest
        assert jit_cached_time < no_jit_time
        assert jit_cached_time < jit_first_time / 5


# =============================================================================
# INFERENCE ALGORITHM PERFORMANCE TESTS
# =============================================================================


class TestInferenceAlgorithmPerformance:
    """Test performance characteristics of inference algorithms."""

    def test_mcmc_convergence_rate(self):
        """Test MCMC convergence rate and efficiency."""

        # Simple 2D Gaussian target
        @gen
        def bivariate_target():
            x = normal(0.0, 1.0) @ "x"
            y = normal(0.5 * x, 0.8) @ "y"
            return jnp.array([x, y])

        @gen
        def proposal(trace):
            old_choices = trace.get_choices()
            x = normal(old_choices["x"], 0.2) @ "x"
            y = normal(old_choices["y"], 0.2) @ "y"

        # Run MCMC chain
        key = jrand.key(101112)
        initial_trace = seed(bivariate_target.simulate)(key)

        def mcmc_step(trace, key):
            return mh(trace, sel("x") | sel("y"))

        n_steps = 1000
        keys = jrand.split(key, n_steps)

        # Time MCMC run
        start_time = time.time()

        def scan_mcmc(trace, key):
            new_trace, accepted = mcmc_step(trace, key)
            return new_trace, (new_trace.get_retval(), accepted)

        final_trace, (samples, accepts) = scan(scan_mcmc, initial_trace, keys)
        mcmc_time = time.time() - start_time

        # Should have reasonable acceptance rate
        accept_rate = jnp.mean(accepts)
        assert 0.2 < accept_rate < 0.8  # Reasonable acceptance rate

        # Should have reasonable timing
        assert mcmc_time < 30.0  # Should complete in reasonable time

        # Samples should have correct shape
        assert samples.shape == (1000, 2)

    def test_smc_particle_scaling(self):
        """Test SMC performance scaling with particle count."""

        @gen
        def smc_target(t):
            x = normal(0.0, 1.0) @ "x"
            # Likelihood becomes more concentrated over time
            normal(x, 1.0 / (1.0 + t)) @ "obs"

        particle_counts = [100, 500, 1000]
        smc_times = []

        for n_particles in particle_counts:
            key = jrand.key(131415)

            # Generate data
            time_steps = jnp.arange(5.0)
            observations = {"obs": jnp.array([0.5, 0.3, 0.1, 0.0, -0.1])}

            # Time SMC
            start_time = time.time()

            # Initial particles
            def init_particles(key):
                keys = jrand.split(key, n_particles)
                traces = modular_vmap(seed(smc_target.simulate))(keys, 0.0)
                return traces

            particles = init_particles(key)

            # SMC steps
            for t, obs in zip(time_steps[1:], observations["obs"][1:]):

                def update_particle(trace, key):
                    # Simple importance sampling update
                    new_trace = seed(smc_target.simulate)(key, t)
                    weight = normal.logpdf(
                        obs, new_trace.get_choices()["x"], 1.0 / (1.0 + t)
                    )
                    return new_trace, weight

                keys = jrand.split(key, n_particles)
                new_particles = modular_vmap(update_particle)(particles, keys)
                particles = new_particles[0]  # For simplicity, just take traces

            smc_time = time.time() - start_time
            smc_times.append(smc_time)

            # Should produce valid particles
            assert particles.get_retval().shape == (n_particles,)

        # SMC time should scale reasonably with particle count
        assert smc_times[-1] < smc_times[0] * 15

    def test_variational_inference_convergence_speed(self):
        """Test VI convergence speed and efficiency."""

        @gen
        def target_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            normal(y, 0.1) @ "obs"

        @gen
        def variational_family(params):
            mu, log_sigma = params
            sigma = jnp.exp(log_sigma)
            x = normal_reparam(mu, sigma) @ "x"
            y = normal_reparam(x, sigma) @ "y"

        @expectation
        def elbo(data, params):
            tr = variational_family.simulate(params)
            q_score = tr.get_score()
            p_score, _ = target_model.assess({**data, **tr.get_choices()})
            return p_score + q_score

        # VI optimization
        data = {"obs": 1.0}
        init_params = jnp.array([0.0, 0.0])

        def vi_step(params, step):
            _, grad = elbo.grad_estimate(data, params)
            return params + 0.01 * grad, elbo.estimate(data, params)

        n_steps = 100

        # Time VI optimization
        start_time = time.time()
        final_params, elbos = scan(vi_step, init_params, jnp.arange(n_steps))
        vi_time = time.time() - start_time

        # Should converge (ELBO should generally improve)
        elbo_improvement = elbos[-10:].mean() - elbos[:10].mean()
        assert elbo_improvement > 0  # Should show improvement

        # Should complete in reasonable time
        assert vi_time < 20.0

        # Final parameters should be reasonable
        assert jnp.all(jnp.isfinite(final_params))

    def test_gp_inference_efficiency_DISABLED(self):
        """Test GP inference efficiency with exact conditioning. DISABLED: GP module removed."""
        pytest.skip("GP module removed from GenJAX")
