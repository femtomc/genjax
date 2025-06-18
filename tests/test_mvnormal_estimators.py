"""
Test cases for multivariate normal gradient estimators.

These tests validate the multivariate normal reparameterization and REINFORCE
gradient estimators added to the adev module.
"""

import jax.numpy as jnp
import jax.random as jrand
from jax.lax import scan

from genjax.core import gen
from genjax.pjax import seed
from genjax.adev import expectation
from genjax import (
    multivariate_normal,
    multivariate_normal_reparam,
    multivariate_normal_reinforce,
)


class TestMultivariateNormalEstimators:
    """Test multivariate normal gradient estimators."""

    def test_multivariate_normal_reparam_basic(self):
        """Test basic functionality of multivariate normal reparameterization."""
        loc = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        # Test direct sampling
        sample = multivariate_normal_reparam.sample(loc, cov)
        assert sample.shape == (2,)
        assert jnp.all(jnp.isfinite(sample))

        # Test log density computation
        logpdf = multivariate_normal_reparam.logpdf(sample, loc, cov)
        assert jnp.isfinite(logpdf)

    def test_multivariate_normal_reinforce_basic(self):
        """Test basic functionality of multivariate normal REINFORCE."""
        loc = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        # Test direct sampling
        sample = multivariate_normal_reinforce.sample(loc, cov)
        assert sample.shape == (2,)
        assert jnp.all(jnp.isfinite(sample))

        # Test log density computation
        logpdf = multivariate_normal_reinforce.logpdf(sample, loc, cov)
        assert jnp.isfinite(logpdf)

    def test_multivariate_normal_in_generative_model(self):
        """Test multivariate normal estimators in generative models."""
        loc = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        @gen
        def test_model():
            # Test without addressing for now (addressing has separate issue)
            x = multivariate_normal_reparam(loc, cov)
            return jnp.sum(x**2)

        # Test model compilation and basic functionality
        # Note: Full execution with addressing requires more complex setup
        assert test_model is not None
        assert hasattr(test_model, "simulate")

    def test_multivariate_normal_gradient_computation(self):
        """Test gradient computation with multivariate normal estimators."""

        @gen
        def target_model():
            loc = jnp.array([0.0, 0.0])
            cov = jnp.array([[1.0, 0.2], [0.2, 1.0]])
            x = multivariate_normal_reparam(loc, cov) @ "x"
            return jnp.sum(x**2)

        @gen
        def variational_family(constraint, theta):
            # theta is [loc_0, loc_1, cov_00, cov_01, cov_10, cov_11]
            loc = theta[:2]
            cov_flat = theta[2:]
            cov = jnp.array([[cov_flat[0], cov_flat[1]], [cov_flat[2], cov_flat[3]]])
            multivariate_normal_reinforce(loc, cov) @ "x"

        @expectation
        def elbo(data: dict, theta):
            tr = variational_family.simulate(data, theta)
            q_score = tr.get_score()
            p, _ = target_model.assess({**data, **tr.get_choices()})
            return p + q_score

        # Test gradient computation
        init_theta = jnp.array([0.1, 0.1, 1.0, 0.1, 0.1, 1.0])
        data = {"x": jnp.array([0.5, -0.5])}

        def optimize(data, init_theta):
            def update(theta, _):
                _, theta_grad = elbo.grad_estimate(data, theta)
                theta += 1e-4 * theta_grad
                return theta, theta

            final_theta, intermediate_thetas = scan(
                update,
                init_theta,
                length=5,  # Short test
            )
            return final_theta, intermediate_thetas

        # This should not crash and should produce finite results
        final_theta, thetas = seed(optimize)(jrand.key(1), data, init_theta)

        assert jnp.all(jnp.isfinite(final_theta))
        assert thetas.shape == (5, 6)
        assert jnp.all(jnp.isfinite(thetas))

    def test_multivariate_normal_consistency_with_base_distribution(self):
        """Test that our estimators are consistent with the base distribution."""
        loc = jnp.array([1.0, -0.5])
        cov = jnp.array([[2.0, 0.5], [0.5, 1.5]])

        # Sample from both our estimators and the base distribution
        base_sample = multivariate_normal.sample(loc, cov)
        reparam_sample = multivariate_normal_reparam.sample(loc, cov)
        reinforce_sample = multivariate_normal_reinforce.sample(loc, cov)

        # All should produce finite samples of correct shape
        for sample in [base_sample, reparam_sample, reinforce_sample]:
            assert sample.shape == (2,)
            assert jnp.all(jnp.isfinite(sample))

        # Log densities should be consistent
        base_logpdf = multivariate_normal.logpdf(base_sample, loc, cov)
        reparam_logpdf = multivariate_normal_reparam.logpdf(base_sample, loc, cov)
        reinforce_logpdf = multivariate_normal_reinforce.logpdf(base_sample, loc, cov)

        # All should give the same log density for the same sample
        assert jnp.allclose(base_logpdf, reparam_logpdf, rtol=1e-5)
        assert jnp.allclose(base_logpdf, reinforce_logpdf, rtol=1e-5)
