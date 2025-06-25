"""
Comprehensive test expansion for GenJAX inference modules.

This test script focuses on expanding coverage for the inference algorithms:
- MCMC (currently 17% coverage)
- SMC (currently 23% coverage)
- VI (currently 35% coverage)

The tests follow GenJAX patterns and focus on functionality not covered
by existing tests.
"""

import jax.numpy as jnp
import jax.random as jrand
import pytest

from genjax.core import gen, sel, const
from genjax.pjax import seed
from genjax.distributions import normal, beta
from genjax.inference import (
    mh,
    mala,
    hmc,
    chain,
    MCMCResult,
    init,
    extend,
    resample,
    ParticleCollection,
    mean_field_normal_family,
    full_covariance_normal_family,
)


# =============================================================================
# MCMC EXPANDED TESTS (Target: MCMC 17% -> 60%+)
# =============================================================================


class TestMCMCExpanded:
    """Expand MCMC test coverage with edge cases and advanced functionality."""

    def test_mh_with_empty_selection(self):
        """Test MH with empty selection (should return unchanged trace)."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        key = jrand.key(42)
        trace = seed(simple_model.simulate)(key)

        # Empty selection should return unchanged trace
        empty_selection = sel()  # Empty selection
        result_trace = mh(trace, empty_selection)

        # Should be identical (no addresses selected)
        original_x = trace.get_choices()["x"]
        result_x = result_trace.get_choices()["x"]
        assert jnp.allclose(original_x, result_x)

    def test_mh_with_all_selection(self):
        """Test MH with all addresses selected."""

        @gen
        def multi_param_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            z = beta(2.0, 2.0) @ "z"
            return x + y + z

        key = jrand.key(42)
        trace = seed(multi_param_model.simulate)(key)

        # Select all addresses
        all_selection = sel("x") | sel("y") | sel("z")
        result_trace = mh(trace, all_selection)

        # Should produce valid trace
        assert hasattr(result_trace, "get_choices")
        assert "x" in result_trace.get_choices()
        assert "y" in result_trace.get_choices()
        assert "z" in result_trace.get_choices()

    def test_mala_step_size_boundary_cases(self):
        """Test MALA with extreme step sizes."""

        @gen
        def normal_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        key = jrand.key(42)
        trace = seed(normal_model.simulate)(key)
        selection = sel("x")

        # Very small step size (should have high acceptance)
        tiny_step = 1e-6
        result1 = mala(trace, selection, tiny_step)

        # Very large step size (should have low acceptance but still work)
        large_step = 10.0
        result2 = mala(trace, selection, large_step)

        # Both should produce valid traces
        assert hasattr(result1, "get_choices")
        assert hasattr(result2, "get_choices")

    def test_hmc_with_single_step(self):
        """Test HMC with n_steps=1 (equivalent to Langevin)."""

        @gen
        def normal_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        key = jrand.key(42)
        trace = seed(normal_model.simulate)(key)
        selection = sel("x")

        # Single leapfrog step
        result = hmc(trace, selection, step_size=0.1, n_steps=1)

        assert hasattr(result, "get_choices")
        assert "x" in result.get_choices()

    def test_hmc_with_many_steps(self):
        """Test HMC with many leapfrog steps."""

        @gen
        def bivariate_normal():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        key = jrand.key(42)
        trace = seed(bivariate_normal.simulate)(key)
        selection = sel("x") | sel("y")

        # Many leapfrog steps
        result = hmc(trace, selection, step_size=0.01, n_steps=50)

        assert hasattr(result, "get_choices")
        assert "x" in result.get_choices()
        assert "y" in result.get_choices()

    def test_chain_with_burn_in_and_thinning(self):
        """Test chain function with burn-in and thinning parameters."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        def mh_kernel(trace):
            return mh(trace, sel("x"))

        key = jrand.key(42)
        trace = seed(simple_model.simulate)(key)

        mcmc_algorithm = chain(mh_kernel)
        seeded_algorithm = seed(mcmc_algorithm)

        # Test with burn-in and thinning
        result = seeded_algorithm(
            key,
            trace,
            n_steps=const(100),
            burn_in=const(20),
            autocorrelation_resampling=const(2),
        )

        assert isinstance(result, MCMCResult)
        # Should have (100 - 20) / 2 = 40 samples
        expected_samples = (100 - 20) // 2
        assert result.traces.get_choices()["x"].shape[0] == expected_samples

    def test_mcmc_with_multiple_chains(self):
        """Test MCMC with multiple parallel chains."""

        @gen
        def beta_model():
            p = beta(2.0, 2.0) @ "p"
            return p

        def mh_kernel(trace):
            return mh(trace, sel("p"))

        key = jrand.key(42)
        trace = seed(beta_model.simulate)(key)

        mcmc_algorithm = chain(mh_kernel)
        seeded_algorithm = seed(mcmc_algorithm)

        # Multiple chains
        result = seeded_algorithm(key, trace, n_steps=const(50), n_chains=const(3))

        assert isinstance(result, MCMCResult)
        assert result.n_chains == 3
        # Traces should have shape (n_chains, n_samples)
        assert result.traces.get_choices()["p"].shape == (3, 50)

        # Should have convergence diagnostics
        assert hasattr(result, "rhat")
        assert hasattr(result, "ess_bulk")
        assert hasattr(result, "ess_tail")


# =============================================================================
# SMC EXPANDED TESTS (Target: SMC 23% -> 60%+)
# =============================================================================


class TestSMCExpanded:
    """Expand SMC test coverage with advanced scenarios."""

    def test_particle_collection_edge_cases(self):
        """Test ParticleCollection with edge cases."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        # Very few particles
        particles_few = init(
            target_gf=simple_model, target_args=(), n_samples=const(2), constraints={}
        )

        assert isinstance(particles_few, ParticleCollection)
        assert particles_few.n_samples.value == 2

        # Check effective sample size with few particles
        ess = particles_few.effective_sample_size()
        assert ess > 0
        assert ess <= 2

    def test_extend_with_empty_constraints(self):
        """Test extend operation with no new constraints."""

        @gen
        def base_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        @gen
        def extended_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        # Initialize with base model
        particles = init(
            target_gf=base_model, target_args=(), n_samples=const(100), constraints={}
        )

        # Extend with no new constraints
        extended_particles = extend(
            particles=particles,
            extended_target_gf=extended_model,
            extended_target_args=(),
            constraints={},
        )

        assert isinstance(extended_particles, ParticleCollection)
        assert "y" in extended_particles.traces.get_choices()

    def test_resample_with_different_methods(self):
        """Test different resampling methods."""

        @gen
        def weighted_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        particles = init(
            target_gf=weighted_model,
            target_args=(),
            n_samples=const(100),
            constraints={},
        )

        # Categorical resampling (default)
        resampled_categorical = resample(particles, method="categorical")
        assert isinstance(resampled_categorical, ParticleCollection)

        # Systematic resampling
        resampled_systematic = resample(particles, method="systematic")
        assert isinstance(resampled_systematic, ParticleCollection)

        # After resampling, weights should be uniform (log 0)
        cat_weights = resampled_categorical.log_weights
        sys_weights = resampled_systematic.log_weights

        # All weights should be zero in log space (uniform weights)
        assert jnp.allclose(cat_weights, 0.0, atol=1e-6)
        assert jnp.allclose(sys_weights, 0.0, atol=1e-6)


# =============================================================================
# VI EXPANDED TESTS (Target: VI 35% -> 70%+)
# =============================================================================


class TestVIExpanded:
    """Expand VI test coverage with simpler working tests."""

    def test_mean_field_normal_family_basic(self):
        """Test mean field family basic functionality."""
        n_dims = 3
        family = mean_field_normal_family(n_dims)

        # Should return a working variational family
        assert hasattr(family, "simulate")

        # Test basic functionality with proper parameters
        params = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.0]),  # means
                jnp.array([0.0, 0.0, 0.0]),  # log_stds
            ]
        )

        constraint = {}
        trace = family.simulate(constraint, params)
        assert hasattr(trace, "get_retval")
        assert trace.get_retval().shape == (n_dims,)

    def test_full_covariance_normal_family_basic(self):
        """Test full covariance family basic functionality."""
        n_dims = 2
        family = full_covariance_normal_family(n_dims)

        # Should return a working variational family
        assert hasattr(family, "simulate")

        # Test basic functionality
        mean = jnp.array([0.0, 0.0])
        chol_cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Identity
        params = {"mean": mean, "chol_cov": chol_cov}

        constraint = {}
        trace = family.simulate(constraint, params)
        assert hasattr(trace, "get_retval")
        assert trace.get_retval().shape == (n_dims,)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestInferenceIntegration:
    """Test integration between different inference methods."""

    def test_mcmc_to_smc_basic(self):
        """Test basic compatibility between MCMC and SMC."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            return x

        # Test that both methods work on same model
        def mh_kernel(trace):
            return mh(trace, sel("x"))

        key = jrand.key(42)
        initial_trace = seed(simple_model.simulate)(key)

        mcmc_chain = chain(mh_kernel)
        seeded_chain = seed(mcmc_chain)
        mcmc_result = seeded_chain(key, initial_trace, const(10))

        # SMC on same model
        smc_particles = init(
            target_gf=simple_model, target_args=(), n_samples=const(50), constraints={}
        )

        assert isinstance(mcmc_result, MCMCResult)
        assert isinstance(smc_particles, ParticleCollection)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
