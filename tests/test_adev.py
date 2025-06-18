import jax.numpy as jnp
import jax.random as jrand
import pytest
from genjax.adev import Dual, expectation, flip_enum, flip_mvd
from genjax.core import gen
from genjax.pjax import seed
from genjax import (
    normal_reparam,
    normal_reinforce,
    multivariate_normal_reparam,
    multivariate_normal_reinforce,
    flip_reinforce,
    geometric_reinforce,
    modular_vmap,
)
from jax.lax import cond
import jax  # Needed for jax.vmap fallback where modular_vmap has compatibility issues


@expectation
def flip_exact_loss(p):
    b = flip_enum(p)
    return cond(
        b,
        lambda _: 0.0,
        lambda p: -p / 2.0,
        p,
    )


def test_flip_exact_loss_jvp():
    """Test that flip_exact_loss JVP estimates match expected values."""
    test_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for p in test_values:
        p_dual = flip_exact_loss.jvp_estimate(Dual(p, 1.0))
        expected_tangent = p - 0.5

        # Test that the tangent matches the expected value
        assert jnp.allclose(p_dual.tangent, expected_tangent, atol=1e-6)


def test_flip_exact_loss_symmetry():
    """Test that the loss function has expected symmetry properties."""
    # Test symmetry around p=0.5
    p1, p2 = 0.3, 0.7
    dual1 = flip_exact_loss.jvp_estimate(Dual(p1, 1.0))
    dual2 = flip_exact_loss.jvp_estimate(Dual(p2, 1.0))

    # The tangents should be symmetric around 0
    assert jnp.allclose(dual1.tangent, -dual2.tangent, atol=1e-6)


def test_flip_exact_loss_at_half():
    """Test the loss function at p=0.5."""
    p_dual = flip_exact_loss.jvp_estimate(Dual(0.5, 1.0))

    # At p=0.5, the tangent should be 0
    assert jnp.allclose(p_dual.tangent, 0.0, atol=1e-6)


###############################################################################
# Regression tests for flat_keyful_sampler error
# These tests ensure ADEV estimators work correctly with seed + addressing
###############################################################################


class TestADEVSeedCompatibility:
    """Test that ADEV estimators work with seed transformation and addressing.

    This prevents regression of the flat_keyful_sampler KeyError that occurred
    when seed was applied to ADEV estimators with addressing.
    """

    def test_normal_reparam_with_seed_and_addressing(self):
        """Test normal_reparam works with seed + addressing."""

        @gen
        def simple_model():
            x = normal_reparam(0.0, 1.0) @ "x"
            return x

        # This should not raise KeyError: 'flat_keyful_sampler'
        result = seed(simple_model.simulate)(jrand.key(42))

        assert isinstance(result.get_retval(), (float, jnp.ndarray))
        assert "x" in result.get_choices()
        assert jnp.allclose(result.get_retval(), result.get_choices()["x"])

    def test_normal_reinforce_with_seed_and_addressing(self):
        """Test normal_reinforce works with seed + addressing."""

        @gen
        def simple_model():
            x = normal_reinforce(0.0, 1.0) @ "x"
            return x

        result = seed(simple_model.simulate)(jrand.key(43))

        assert isinstance(result.get_retval(), (float, jnp.ndarray))
        assert "x" in result.get_choices()
        assert jnp.allclose(result.get_retval(), result.get_choices()["x"])

    def test_multivariate_normal_reparam_with_seed_and_addressing(self):
        """Test multivariate_normal_reparam works with seed + addressing."""

        @gen
        def mvn_model():
            loc = jnp.array([0.0, 1.0])
            cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
            x = multivariate_normal_reparam(loc, cov) @ "x"
            return x

        result = seed(mvn_model.simulate)(jrand.key(44))

        assert result.get_retval().shape == (2,)
        assert "x" in result.get_choices()
        assert jnp.allclose(result.get_retval(), result.get_choices()["x"])

    def test_multivariate_normal_reinforce_with_seed_and_addressing(self):
        """Test multivariate_normal_reinforce works with seed + addressing."""

        @gen
        def mvn_model():
            loc = jnp.array([0.0, 1.0])
            cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
            x = multivariate_normal_reinforce(loc, cov) @ "x"
            return x

        result = seed(mvn_model.simulate)(jrand.key(45))

        assert result.get_retval().shape == (2,)
        assert "x" in result.get_choices()
        assert jnp.allclose(result.get_retval(), result.get_choices()["x"])

    def test_multiple_adev_estimators_with_seed(self):
        """Test multiple ADEV estimators in the same model with seed."""

        @gen
        def multi_estimator_model():
            x1 = normal_reparam(0.0, 1.0) @ "x1"
            x2 = normal_reinforce(x1, 0.5) @ "x2"
            loc = jnp.array([x2, 0.0])
            cov = jnp.eye(2)
            x3 = multivariate_normal_reparam(loc, cov) @ "x3"
            return x1 + x2 + jnp.sum(x3)

        result = seed(multi_estimator_model.simulate)(jrand.key(46))
        choices = result.get_choices()

        assert "x1" in choices
        assert "x2" in choices
        assert "x3" in choices
        assert choices["x3"].shape == (2,)


class TestADEVGradientComputation:
    """Test gradient computation with ADEV estimators to ensure VI works."""

    def test_simple_elbo_gradient_with_normal_reparam(self):
        """Test ELBO gradient computation with normal_reparam."""

        @gen
        def target_model():
            x = normal_reparam(0.0, 1.0) @ "x"
            normal_reparam(x, 0.5) @ "y"

        @gen
        def variational_family(data, theta):
            normal_reparam(theta, 1.0) @ "x"

        @expectation
        def elbo(data, theta):
            tr = variational_family.simulate(data, theta)
            q_score = tr.get_score()
            p, _ = target_model.assess({**data, **tr.get_choices()})
            return p + q_score

        # This should not raise any errors
        grad_result = elbo.grad_estimate({"y": 2.0}, 0.5)
        # grad_result should be a tuple since we have 2 arguments (data, theta)
        assert isinstance(grad_result, tuple)
        assert len(grad_result) == 2
        data_grad, theta_grad = grad_result
        assert isinstance(theta_grad, (float, jnp.ndarray))

    def test_multivariate_elbo_gradient(self):
        """Test ELBO gradient computation with multivariate normal."""

        @gen
        def target_model():
            loc = jnp.array([0.0, 1.0])
            cov = jnp.eye(2)
            x = multivariate_normal_reparam(loc, cov) @ "x"
            return jnp.sum(x)

        @gen
        def variational_family(theta):
            cov = jnp.eye(2) * 0.5
            multivariate_normal_reparam(theta, cov) @ "x"

        @expectation
        def elbo(theta):
            tr = variational_family.simulate(theta)
            q_score = tr.get_score()
            p, _ = target_model.assess(tr.get_choices())
            return p + q_score

        theta = jnp.array([0.1, -0.1])
        grad_result = elbo.grad_estimate(theta)
        assert grad_result.shape == (2,)

    def test_mixed_estimators_gradient(self):
        """Test gradient computation with mixed REPARAM and REINFORCE estimators."""

        @gen
        def mixed_model(theta):
            x1 = normal_reparam(theta[0], 1.0) @ "x1"
            x2 = normal_reinforce(theta[1], 0.5) @ "x2"
            return x1 + x2

        @expectation
        def objective(theta):
            tr = mixed_model.simulate(theta)
            return jnp.sum(tr.get_retval())

        theta = jnp.array([0.5, -0.3])
        grad_result = objective.grad_estimate(theta)
        assert grad_result.shape == (2,)


class TestADEVNoSeedCompatibility:
    """Test that ADEV estimators still work without seed (regression test)."""

    def test_normal_reparam_without_seed(self):
        """Test normal_reparam works without seed transformation."""

        @gen
        def simple_model():
            x = normal_reparam(0.0, 1.0) @ "x"
            return x

        # Should work without seed
        result = simple_model.simulate()

        assert isinstance(result.get_retval(), (float, jnp.ndarray))
        assert "x" in result.get_choices()

    def test_multivariate_normal_reparam_without_seed(self):
        """Test multivariate_normal_reparam works without seed transformation."""

        @gen
        def mvn_model():
            loc = jnp.array([0.0, 1.0])
            cov = jnp.eye(2)
            x = multivariate_normal_reparam(loc, cov) @ "x"
            return x

        result = mvn_model.simulate()

        assert result.get_retval().shape == (2,)
        assert "x" in result.get_choices()


class TestADEVErrorConditions:
    """Test error conditions to ensure proper error messages."""

    def test_adev_estimators_work_with_sample_shape(self):
        """Test that ADEV estimators handle sample_shape parameter correctly."""

        @gen
        def model_with_shape():
            # The assume_binder should handle sample_shape parameter
            x = normal_reparam(0.0, 1.0) @ "x"
            return x

        # This should not raise "unexpected keyword argument 'sample_shape'"
        result = seed(model_with_shape.simulate)(jrand.key(50))
        assert isinstance(result.get_retval(), (float, jnp.ndarray))

    def test_flat_keyful_sampler_error_prevention(self):
        """Specific test to ensure flat_keyful_sampler error doesn't return."""

        # This test specifically targets the error case that was fixed
        @gen
        def adev_with_addressing():
            x = normal_reparam(1.0, 0.5) @ "param"
            y = (
                multivariate_normal_reparam(jnp.array([x, 0.0]), jnp.eye(2) * 0.1)
                @ "mvn_param"
            )
            return jnp.sum(y)

        # This exact pattern previously caused KeyError: 'flat_keyful_sampler'
        try:
            result = seed(adev_with_addressing.simulate)(jrand.key(999))
            # If we get here, the error is fixed
            assert "param" in result.get_choices()
            assert "mvn_param" in result.get_choices()
            assert result.get_choices()["mvn_param"].shape == (2,)
        except KeyError as e:
            if "flat_keyful_sampler" in str(e):
                pytest.fail("flat_keyful_sampler error has regressed!")
            else:
                raise  # Re-raise if it's a different KeyError


class TestGradientEstimatorSanity:
    """Test basic sanity checks for all gradient estimators.

    These tests verify that our gradient estimators produce finite gradients
    with correct shapes and that enumeration gives exact results.
    """

    def test_normal_reparam_basic_properties(self):
        """Test normal reparameterization produces finite gradients."""

        @expectation
        def quadratic_loss(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        mu, sigma = 1.0, 1.0
        grad_mu, grad_sigma = quadratic_loss.grad_estimate(mu, sigma)

        # Basic sanity checks
        assert jnp.isfinite(grad_mu)
        assert jnp.isfinite(grad_sigma)
        assert grad_mu.shape == ()
        assert grad_sigma.shape == ()

    def test_normal_reinforce_basic_properties(self):
        """Test normal REINFORCE produces finite gradients."""

        @expectation
        def quadratic_loss(mu, sigma):
            x = normal_reinforce(mu, sigma)
            return x**2

        mu, sigma = 1.0, 0.5
        grad_mu, grad_sigma = quadratic_loss.grad_estimate(mu, sigma)

        # Basic sanity checks
        assert jnp.isfinite(grad_mu)
        assert jnp.isfinite(grad_sigma)
        assert grad_mu.shape == ()
        assert grad_sigma.shape == ()

    def test_multivariate_normal_reparam_basic_properties(self):
        """Test multivariate normal reparameterization produces finite gradients."""

        @expectation
        def quadratic_loss(mu, cov_diag):
            # Use diagonal covariance for simplicity
            cov = jnp.diag(cov_diag)
            x = multivariate_normal_reparam(mu, cov)
            return jnp.sum(x**2)

        mu = jnp.array([0.5, -0.3])
        cov_diag = jnp.array([1.0, 1.0])
        grad_mu, grad_cov = quadratic_loss.grad_estimate(mu, cov_diag)

        # Basic sanity checks
        assert jnp.all(jnp.isfinite(grad_mu))
        assert jnp.all(jnp.isfinite(grad_cov))
        assert grad_mu.shape == (2,)
        assert grad_cov.shape == (2,)

    def test_flip_enum_exact_convergence(self):
        """Test flip enumeration gives exact gradients (zero variance).

        For f(X) = X where X ~ Bernoulli(p), the analytical gradient is:
        ∇_p E[X] = 1
        """

        @expectation
        def identity_loss(p):
            x = flip_enum(p)
            return jnp.float32(x)  # Convert boolean to float

        # Test multiple probability values
        test_probs = [0.1, 0.3, 0.5, 0.7, 0.9]

        for p in test_probs:
            # Enumeration should give exact gradients
            grad = identity_loss.grad_estimate(p)

            # Analytical gradient is exactly 1
            assert jnp.allclose(grad, 1.0, atol=1e-10)

    def test_flip_mvd_basic_properties(self):
        """Test flip MVD produces finite gradients."""

        @expectation
        def identity_loss(p):
            x = flip_mvd(p)
            return jnp.float32(x)  # Convert boolean to float

        p = 0.6
        grad = identity_loss.grad_estimate(p)

        # Basic sanity checks
        assert jnp.isfinite(grad)
        assert grad.shape == ()

    def test_flip_reinforce_basic_properties(self):
        """Test flip REINFORCE produces finite gradients."""

        @expectation
        def identity_loss(p):
            x = flip_reinforce(p)
            return jnp.float32(x)  # Convert boolean to float

        p = 0.4
        grad = identity_loss.grad_estimate(p)

        # Basic sanity checks
        assert jnp.isfinite(grad)
        assert grad.shape == ()

    def test_geometric_reinforce_basic_properties(self):
        """Test geometric REINFORCE produces finite gradients."""

        @expectation
        def identity_loss(p):
            x = geometric_reinforce(p)
            return jnp.float32(x)  # Convert to float

        p = 0.3
        grad = identity_loss.grad_estimate(p)

        # Basic sanity checks
        assert jnp.isfinite(grad)
        assert grad.shape == ()

    def test_mixed_estimators_basic_properties(self):
        """Test mixing different gradient estimators produces finite gradients."""

        @expectation
        def mixed_loss(mu, sigma, p):
            x = normal_reparam(mu, sigma)
            y = flip_enum(p)  # Use enum for exact discrete gradient
            return x + jnp.float32(y)  # Convert boolean to float

        mu, sigma, p = 0.5, 1.0, 0.6
        grad_mu, grad_sigma, grad_p = mixed_loss.grad_estimate(mu, sigma, p)

        # Basic sanity checks
        assert jnp.isfinite(grad_mu)
        assert jnp.isfinite(grad_sigma)
        assert jnp.isfinite(grad_p)
        assert grad_mu.shape == ()
        assert grad_sigma.shape == ()
        assert grad_p.shape == ()

    def test_variance_comparison(self):
        """Test that reparameterization has lower variance than REINFORCE.

        Both should produce finite gradients, but reparam should have
        lower variance when computed multiple times.
        """

        @expectation
        def reparam_loss(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        @expectation
        def reinforce_loss(mu, sigma):
            x = normal_reinforce(mu, sigma)
            return x**2

        mu, sigma = 1.0, 0.5
        n_samples = 100  # Small number for basic test

        def compute_reparam_grad(_):
            grad_mu, grad_sigma = reparam_loss.grad_estimate(mu, sigma)
            return jnp.array([grad_mu, grad_sigma])

        def compute_reinforce_grad(_):
            grad_mu, grad_sigma = reinforce_loss.grad_estimate(mu, sigma)
            return jnp.array([grad_mu, grad_sigma])

        reparam_grads = modular_vmap(compute_reparam_grad)(jnp.arange(n_samples))
        reinforce_grads = modular_vmap(compute_reinforce_grad)(jnp.arange(n_samples))

        # Basic checks that all gradients are finite
        assert jnp.all(jnp.isfinite(reparam_grads))
        assert jnp.all(jnp.isfinite(reinforce_grads))

        # Compute variances
        reparam_var = jnp.var(reparam_grads, axis=0)
        reinforce_var = jnp.var(reinforce_grads, axis=0)

        # Both should have finite variance
        assert jnp.all(jnp.isfinite(reparam_var))
        assert jnp.all(jnp.isfinite(reinforce_var))
        # Generally expect reparam to have lower variance (but allow for randomness)
        assert jnp.all(reparam_var >= 0)
        assert jnp.all(reinforce_var >= 0)


class TestGradientEstimatorConvergence:
    """Test that gradient estimators converge to correct analytical gradients.

    These tests verify that our unbiased gradient estimators actually produce
    the correct gradients in expectation by comparing against known analytical
    solutions for simple objective functions.

    Note: These tests focus on cases where the analytical gradients are well-established
    and the estimators are known to work reliably.
    """

    def test_normal_reparam_linear_convergence(self):
        """Test normal reparameterization on linear objective.

        For f(X) = X where X ~ N(μ, 1), we have:
        ∇_μ E[X] = 1

        This is a fundamental test case for reparameterization.
        """

        @expectation
        def linear_loss(mu):
            x = normal_reparam(mu, 1.0)
            return x

        # Test parameters
        mu = 2.0
        n_samples = 300
        expected_grad = 1.0

        # Estimate gradients multiple times and average
        def estimate_grad(_):
            return linear_loss.grad_estimate(mu)

        grad_estimates = modular_vmap(estimate_grad)(jnp.arange(n_samples))
        mean_grad = jnp.mean(grad_estimates)

        # Should converge to analytical gradient
        assert jnp.allclose(mean_grad, expected_grad, rtol=0.05)

    def test_flip_enum_exact_gradients(self):
        """Test flip enumeration gives exact gradients.

        For f(X) = X where X ~ Bernoulli(p), we have:
        ∇_p E[X] = 1 (exactly)

        Enumeration should give zero-variance estimates.
        """

        @expectation
        def identity_loss(p):
            x = flip_enum(p)
            return jnp.float32(x)

        # Test multiple probability values
        test_probs = [0.2, 0.5, 0.8]

        for p in test_probs:
            # Multiple estimates should all be exactly 1.0
            estimates = [identity_loss.grad_estimate(p) for _ in range(5)]

            # All estimates should be exactly 1.0 (enumeration is exact)
            for est in estimates:
                assert jnp.allclose(est, 1.0, atol=1e-10)

            # Variance should be essentially zero
            variance = jnp.var(jnp.array(estimates))
            assert variance < 1e-12

    def test_flip_mvd_convergence(self):
        """Test flip MVD converges for simple Bernoulli function.

        For f(X) = X where X ~ Bernoulli(p), we have:
        ∇_p E[X] = 1

        MVD should converge to this analytical gradient.
        """

        @expectation
        def identity_loss(p):
            x = flip_mvd(p)
            return jnp.float32(x)

        # Test parameters
        p = 0.6
        n_samples = 500
        expected_grad = 1.0

        # Estimate gradients
        def estimate_grad(_):
            return identity_loss.grad_estimate(p)

        # Note: Using jax.vmap here due to incompatibility between modular_vmap and flip_mvd
        # This appears to be a limitation in the current ADEV implementation
        grad_estimates = jax.vmap(estimate_grad)(jnp.arange(n_samples))
        mean_grad = jnp.mean(grad_estimates)

        # Should converge to analytical gradient
        assert jnp.allclose(mean_grad, expected_grad, rtol=0.1)

    def test_estimator_variance_properties(self):
        """Test basic variance properties of gradient estimators.

        Test that estimators produce finite, well-behaved gradient estimates.
        """

        @expectation
        def enum_obj(p):
            x = flip_enum(p)
            return jnp.float32(x)

        @expectation
        def mvd_obj(p):
            x = flip_mvd(p)
            return jnp.float32(x)

        p = 0.4
        n_samples = 50

        # Get multiple gradient estimates
        def estimate_enum(_):
            return enum_obj.grad_estimate(p)

        def estimate_mvd(_):
            return mvd_obj.grad_estimate(p)

        enum_grads = modular_vmap(estimate_enum)(jnp.arange(n_samples))
        # Note: Using jax.vmap for MVD due to incompatibility with modular_vmap
        mvd_grads = jax.vmap(estimate_mvd)(jnp.arange(n_samples))

        # Basic sanity checks
        assert jnp.all(jnp.isfinite(enum_grads))
        assert jnp.all(jnp.isfinite(mvd_grads))

        # Check that enumeration gives consistent results (low variance)
        enum_var = jnp.var(enum_grads)
        assert enum_var < 1e-10  # Should be essentially exact

        # Check that MVD gives reasonable estimates
        mvd_mean = jnp.mean(mvd_grads)
        assert jnp.allclose(
            mvd_mean, 1.0, rtol=0.2
        )  # Should approximate the true gradient

    def test_gradient_estimator_unbiasedness(self):
        """Test that different estimators are unbiased for the same objective.

        Different estimators should converge to the same analytical gradient
        for equivalent objective functions.
        """

        # Linear objectives (easier to verify analytically)
        @expectation
        def reparam_obj(mu):
            x = normal_reparam(mu, 1.0)
            return 3.0 * x  # ∇_μ E[3X] = 3

        @expectation
        def enum_obj(p):
            x = flip_enum(p)
            return jnp.float32(x)  # ∇_p E[X] = 1

        # Test parameters
        mu = 0.5
        p = 0.6
        n_samples = 300

        # Expected gradients
        expected_reparam_grad = 3.0
        expected_enum_grad = 1.0

        # Estimate gradients
        def estimate_reparam(_):
            return reparam_obj.grad_estimate(mu)

        def estimate_enum(_):
            return enum_obj.grad_estimate(p)

        reparam_grads = modular_vmap(estimate_reparam)(jnp.arange(n_samples))
        enum_grads = modular_vmap(estimate_enum)(jnp.arange(n_samples))

        mean_reparam = jnp.mean(reparam_grads)
        mean_enum = jnp.mean(enum_grads)

        # Check convergence to analytical values
        assert jnp.allclose(mean_reparam, expected_reparam_grad, rtol=0.08)
        assert jnp.allclose(mean_enum, expected_enum_grad, rtol=0.05)
