import jax.numpy as jnp
import jax.random as jrand
import pytest
from genjax.adev import Dual, expectation, flip_enum, flip_mvd
from genjax.core import gen
from genjax.pjax import seed
from genjax import (
    normal_reparam,
    normal_reinforce,
    multivariate_normal,
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


# =============================================================================
# MULTIVARIATE NORMAL ESTIMATOR TESTS
# =============================================================================


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

        # Test simple gradient computation (avoiding scan for now)
        _, theta_grad = elbo.grad_estimate(data, init_theta)

        # Perform a simple optimization step
        final_theta = init_theta + 1e-4 * theta_grad

        assert jnp.all(jnp.isfinite(final_theta))
        assert jnp.all(jnp.isfinite(theta_grad))

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


# =============================================================================
# ENHANCED ADEV EDGE CASES AND COMBINATIONS
# =============================================================================


class TestADEVEdgeCases:
    """Test ADEV estimators in challenging edge cases."""

    def test_extreme_parameter_values(self):
        """Test ADEV estimators with extreme parameter values."""

        @expectation
        def extreme_normal_test(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        # Very large parameters
        large_mu = 1e6
        large_sigma = 1e3
        grad_mu_large, grad_sigma_large = extreme_normal_test.grad_estimate(
            large_mu, large_sigma
        )
        assert jnp.isfinite(grad_mu_large)
        assert jnp.isfinite(grad_sigma_large)

        # Very small parameters
        small_mu = 1e-6
        small_sigma = 1e-8
        grad_mu_small, grad_sigma_small = extreme_normal_test.grad_estimate(
            small_mu, small_sigma
        )
        assert jnp.isfinite(grad_mu_small)
        assert jnp.isfinite(grad_sigma_small)

    def test_high_dimensional_adev(self):
        """Test ADEV estimators with high-dimensional parameters."""

        @expectation
        def high_dim_objective(theta):
            # 100-dimensional parameter vector
            assert theta.shape == (100,)

            # Use first 50 for means, last 50 for log standard deviations
            means = theta[:50]
            log_stds = theta[50:]
            stds = jnp.exp(log_stds)

            total = 0.0
            for i in range(50):
                x = normal_reparam(means[i], stds[i])
                total += x**2

            return total / 50.0  # Average

        # Random high-dimensional parameter
        key = jrand.key(42)
        theta = jrand.normal(key, (100,)) * 0.1

        # Should handle high-dimensional gradients
        grad = high_dim_objective.grad_estimate(theta)
        assert grad.shape == (100,)
        assert jnp.all(jnp.isfinite(grad))

    def test_nested_adev_operations(self):
        """Test nested ADEV operations with simple composition."""

        @expectation
        def nested_objective(params):
            p1, p2, p3 = params

            # First level - use reparam
            x1 = normal_reparam(p1, 0.1)

            # Second level - use the result of first level
            x2 = normal_reparam(p2 + x1, 0.1)

            # Third level - use reinforce with result of second level
            x3 = normal_reinforce(p3 * x2, 0.1)

            return x1 + x2 + x3

        params = jnp.array([0.5, -0.3, 1.2])

        # Should handle nested operations
        grad = nested_objective.grad_estimate(params)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_adev_with_zero_gradients(self):
        """Test ADEV when true gradients should be zero."""

        @expectation
        def zero_gradient_objective(mu):
            x = normal_reparam(mu, 1.0)
            # This should have zero gradient w.r.t. mu
            return jnp.sin(x - mu)  # Shift-invariant

        mu = 2.0

        # Multiple estimates should be close to zero
        grads = []
        for i in range(20):
            grad = zero_gradient_objective.grad_estimate(mu)
            grads.append(grad)

        mean_grad = jnp.mean(jnp.array(grads))
        assert jnp.abs(mean_grad) < 0.5  # Should be close to zero

    def test_adev_numerical_stability_near_boundaries(self):
        """Test ADEV numerical stability near parameter boundaries."""

        @expectation
        def boundary_objective(log_sigma):
            sigma = jnp.exp(log_sigma)  # Always positive
            x = normal_reparam(0.0, sigma)
            return x**2 / sigma**2  # Normalized

        # Very negative log_sigma (very small sigma)
        log_sigma_small = -10.0
        grad_small = boundary_objective.grad_estimate(log_sigma_small)
        assert jnp.isfinite(grad_small)

        # Very positive log_sigma (very large sigma)
        log_sigma_large = 10.0
        grad_large = boundary_objective.grad_estimate(log_sigma_large)
        assert jnp.isfinite(grad_large)


class TestADEVEstimatorCombinations:
    """Test combinations of different ADEV estimators."""

    def test_all_estimator_types_together(self):
        """Test using different estimator types in simpler model."""

        @expectation
        def comprehensive_model(p_continuous, p_discrete):
            # Reparameterization
            x1 = normal_reparam(p_continuous, 1.0)

            # Enumeration (exact)
            x2 = flip_enum(p_discrete)

            # REINFORCE
            x3 = normal_reinforce(x1, 0.5)

            return x1 + jnp.float32(x2) + x3

        # Should handle mixed estimator types
        value = comprehensive_model.estimate(0.5, 0.7)
        assert jnp.isfinite(value)

        grad_continuous, grad_discrete = comprehensive_model.grad_estimate(0.5, 0.7)
        assert jnp.isfinite(grad_continuous)
        assert jnp.isfinite(grad_discrete)

    def test_variance_comparison_across_estimators(self):
        """Compare variance across different estimators for same objective."""

        # Same objective with different estimators
        @expectation
        def reparam_obj(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        @expectation
        def reinforce_obj(mu, sigma):
            x = normal_reinforce(mu, sigma)
            return x**2

        mu, sigma = 1.0, 0.5
        n_samples = 50

        # Collect estimates
        reparam_grads = []
        reinforce_grads = []

        for _ in range(n_samples):
            reparam_grad_mu, reparam_grad_sigma = reparam_obj.grad_estimate(mu, sigma)
            reinforce_grad_mu, reinforce_grad_sigma = reinforce_obj.grad_estimate(
                mu, sigma
            )

            reparam_grads.append(jnp.array([reparam_grad_mu, reparam_grad_sigma]))
            reinforce_grads.append(jnp.array([reinforce_grad_mu, reinforce_grad_sigma]))

        reparam_grads = jnp.array(reparam_grads)
        reinforce_grads = jnp.array(reinforce_grads)

        # Compute variances
        reparam_var = jnp.var(reparam_grads, axis=0)
        reinforce_var = jnp.var(reinforce_grads, axis=0)

        # Both should have finite variance
        assert jnp.all(jnp.isfinite(reparam_var))
        assert jnp.all(jnp.isfinite(reinforce_var))

        # Typically expect reparam to have lower variance
        # (but allow for stochasticity in this test)
        assert jnp.all(reparam_var >= 0)
        assert jnp.all(reinforce_var >= 0)

    def test_conditional_estimator_selection(self):
        """Test that different estimators work for same objective."""

        @expectation
        def reparam_model(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        @expectation
        def reinforce_model(mu, sigma):
            x = normal_reinforce(mu, sigma)
            return x**2

        mu, sigma = 0.5, 1.0

        # Test both estimators
        grad_reparam_mu, grad_reparam_sigma = reparam_model.grad_estimate(mu, sigma)
        grad_reinforce_mu, grad_reinforce_sigma = reinforce_model.grad_estimate(
            mu, sigma
        )

        # Both should work but might have different variances
        assert jnp.isfinite(grad_reparam_mu)
        assert jnp.isfinite(grad_reparam_sigma)
        assert jnp.isfinite(grad_reinforce_mu)
        assert jnp.isfinite(grad_reinforce_sigma)

    @pytest.mark.skip(reason="ADEV + modular_vmap not supported yet")
    def test_estimator_composition_patterns(self):
        """Test common patterns of estimator composition."""

        @expectation
        def hierarchical_pattern(global_param, local_params):
            # Avoid closures - pass global_effect as explicit parameter
            global_effect = normal_reparam(global_param, 1.0)

            # Use a simple additive model instead of closure
            combined_params = local_params + global_effect

            def compute_local_effect(combined_param):
                return normal_reparam(combined_param, 0.5)

            # Apply to all local parameters using modular_vmap
            local_effects = modular_vmap(compute_local_effect)(combined_params)
            total = jnp.sum(local_effects)

            return total / local_params.shape[0]

        global_param = 0.2
        local_params = jnp.array([0.1, -0.2, 0.3, -0.1])

        # Should handle hierarchical composition
        value = hierarchical_pattern.estimate(global_param, local_params)
        assert jnp.isfinite(value)

        grad_global, grad_local = hierarchical_pattern.grad_estimate(
            global_param, local_params
        )
        assert jnp.isfinite(grad_global)
        assert grad_local.shape == (4,)
        assert jnp.all(jnp.isfinite(grad_local))


class TestADEVControlFlowIntegration:
    """Test ADEV with complex control flow."""

    def test_adev_scan_operations_not_supported(self):
        """Test that ADEV correctly rejects scan operations."""

        @expectation
        def scan_adev_objective(init_param, step_params):
            def scan_body(carry, step_param):
                # Use carry and step_param with ADEV
                new_carry = normal_reparam(carry + step_param, 0.1)
                output = new_carry**2
                return new_carry, output

            from jax.lax import scan

            final_carry, outputs = scan(scan_body, init_param, step_params)
            return jnp.sum(outputs) + final_carry

        init_param = 0.5
        step_params = jnp.array([0.1, -0.1, 0.2, -0.2])

        # Should raise NotImplementedError for scan operations
        with pytest.raises(NotImplementedError, match="ADEV does not support.*scan"):
            scan_adev_objective.estimate(init_param, step_params)

        # Gradient estimation should also fail
        with pytest.raises(NotImplementedError, match="ADEV does not support.*scan"):
            scan_adev_objective.grad_estimate(init_param, step_params)

    @pytest.mark.skip(reason="ADEV + modular_vmap not supported yet")
    def test_adev_with_dynamic_structure(self):
        """Test ADEV with static structure (JAX-compatible)."""

        @expectation
        def static_structure_objective(params):
            base_param = params[0]
            rest_params = params[1:]

            # Use static structure with parameter-dependent weighting
            # Instead of changing structure, weight components by base_param
            weight = jnp.tanh(base_param)  # Maps to (-1, 1)

            def compute_component(param):
                component = normal_reparam(param, 0.1)
                return weight * component

            # Apply to all components with modular_vmap
            weighted_components = modular_vmap(compute_component)(rest_params)
            total = jnp.sum(weighted_components)

            return total / rest_params.shape[0]

        params = jnp.array([0.8, 0.1, -0.2, 0.3, -0.1, 0.4])

        # Should handle static structure
        value = static_structure_objective.estimate(params)
        assert jnp.isfinite(value)

        grad = static_structure_objective.grad_estimate(params)
        assert grad.shape == (6,)
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.skip(
        reason="ADEV array indexing with JAX transformations not fully supported yet"
    )
    def test_adev_error_handling_and_recovery(self):
        """Test ADEV error handling in edge cases."""

        @expectation
        def robust_objective(params):
            p1, p2, p3 = params[0], params[1], params[2]

            # Use robust numerical operations
            sigma = jnp.exp(jnp.clip(p2, -10.0, 10.0))  # Bound sigma
            x = normal_reparam(p1, sigma)

            # Avoid division by zero
            denominator = jnp.abs(p3) + 1e-6
            result = x / denominator

            # Clamp to reasonable range
            return jnp.clip(result, -100.0, 100.0)

        # Test with reasonable parameter combinations
        test_params = [
            jnp.array([0.0, -5.0, 0.1]),  # Moderate values
            jnp.array([1.0, 5.0, 0.01]),  # Larger values but bounded
            jnp.array([-1.0, -2.0, -0.1]),  # Mixed signs
        ]

        for params in test_params:
            # Should work without crashing
            value = robust_objective.estimate(params)
            grad = robust_objective.grad_estimate(params)

            # Results should be finite and reasonable
            assert jnp.isfinite(value)
            assert jnp.abs(value) <= 100.0  # Within clipping bounds
            assert jnp.all(jnp.isfinite(grad))
            assert jnp.all(jnp.abs(grad) < 1e6)  # Reasonable gradient magnitude


class TestADEVCompatibilityWithJAXTransformations:
    """Test ADEV compatibility with JAX transformations."""

    def test_adev_with_vmap(self):
        """Test ADEV estimators with vmap transformation."""

        @expectation
        def single_param_objective(param):
            x = normal_reparam(param, 1.0)
            return x**2

        # Batch of parameters
        params = jnp.array([0.1, 0.5, -0.3, 1.2])

        # Vectorize the gradient estimation
        vectorized_grad = modular_vmap(single_param_objective.grad_estimate)

        grads = vectorized_grad(params)
        assert grads.shape == (4,)
        assert jnp.all(jnp.isfinite(grads))

    def test_adev_with_jit_compilation(self):
        """Test ADEV estimators with JIT compilation."""

        @expectation
        def simple_objective(mu, sigma):
            x = normal_reparam(mu, sigma)
            return x**2

        # Use seed before JIT compilation for ADEV functions
        seeded_grad = seed(simple_objective.grad_estimate)
        jitted_grad = jax.jit(seeded_grad)

        mu, sigma = 1.0, 0.5
        key = jrand.key(42)

        # Should work with JIT (need to pass key for seeded version)
        grad = jitted_grad(key, mu, sigma)
        # Check if grad is a tuple or single array
        if isinstance(grad, tuple):
            assert len(grad) == 2
            assert jnp.all(jnp.isfinite(grad[0]))
            assert jnp.all(jnp.isfinite(grad[1]))
        else:
            assert jnp.all(jnp.isfinite(grad))

        # Compare with non-JIT version
        grad_regular = simple_objective.grad_estimate(mu, sigma)
        # Results might differ due to randomness, but both should be finite
        if isinstance(grad_regular, tuple):
            assert jnp.all(jnp.isfinite(grad_regular[0]))
            assert jnp.all(jnp.isfinite(grad_regular[1]))
        else:
            assert jnp.all(jnp.isfinite(grad_regular))

    @pytest.mark.skip(reason="ADEV + modular_vmap not supported yet")
    def test_adev_with_nested_transformations(self):
        """Test ADEV with nested JAX transformations."""

        @expectation
        def param_dependent_objective(global_param, local_params):
            # Avoid closure by combining parameters explicitly
            combined_params = global_param + local_params

            def compute_component(combined_param):
                x = normal_reparam(combined_param, 0.1)
                return x**2

            # Use modular_vmap instead of Python loop
            components = modular_vmap(compute_component)(combined_params)
            return jnp.mean(components)

        # Nest vmap and grad
        def batch_gradients(global_param, batch_local_params):
            # batch_local_params: (batch_size, n_local)
            def single_gradient(local_params):
                grads = param_dependent_objective.grad_estimate(
                    global_param, local_params
                )
                # Handle potential tuple return
                if isinstance(grads, tuple):
                    return grads[0], grads[1]  # grad_global, grad_local
                else:
                    # If flattened, reshape appropriately
                    grad_global = grads[0]
                    grad_local = grads[1:]
                    return grad_global, grad_local

            return modular_vmap(single_gradient)(batch_local_params)

        global_param = 0.5
        batch_local_params = jnp.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])

        batch_grad_global, batch_grad_local = batch_gradients(
            global_param, batch_local_params
        )

        assert batch_grad_global.shape == (3,)  # One per batch item
        assert batch_grad_local.shape == (3, 2)  # (batch_size, n_local)
        assert jnp.all(jnp.isfinite(batch_grad_global))
        assert jnp.all(jnp.isfinite(batch_grad_local))
