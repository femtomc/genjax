import jax.numpy as jnp
import jax.random as jrand
import pytest
from genjax.adev import Dual, expectation, flip_enum
from genjax.core import gen, seed
from genjax import (
    normal_reparam,
    normal_reinforce,
    multivariate_normal_reparam,
    multivariate_normal_reinforce,
)
from jax.lax import cond


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
        result = seed(simple_model.simulate)(jrand.key(42), ())

        assert isinstance(result.get_retval(), (float, jnp.ndarray))
        assert "x" in result.get_choices()
        assert jnp.allclose(result.get_retval(), result.get_choices()["x"])

    def test_normal_reinforce_with_seed_and_addressing(self):
        """Test normal_reinforce works with seed + addressing."""

        @gen
        def simple_model():
            x = normal_reinforce(0.0, 1.0) @ "x"
            return x

        result = seed(simple_model.simulate)(jrand.key(43), ())

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

        result = seed(mvn_model.simulate)(jrand.key(44), ())

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

        result = seed(mvn_model.simulate)(jrand.key(45), ())

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

        result = seed(multi_estimator_model.simulate)(jrand.key(46), ())
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
            tr = variational_family.simulate((data, theta))
            q_score = tr.get_score()
            p = target_model.log_density((), {**data, **tr.get_choices()})
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
            tr = variational_family.simulate((theta,))
            q_score = tr.get_score()
            p = target_model.log_density((), tr.get_choices())
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
            tr = mixed_model.simulate((theta,))
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
        result = simple_model.simulate(())

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

        result = mvn_model.simulate(())

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
        result = seed(model_with_shape.simulate)(jrand.key(50), ())
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
            result = seed(adev_with_addressing.simulate)(jrand.key(999), ())
            # If we get here, the error is fixed
            assert "param" in result.get_choices()
            assert "mvn_param" in result.get_choices()
            assert result.get_choices()["mvn_param"].shape == (2,)
        except KeyError as e:
            if "flat_keyful_sampler" in str(e):
                pytest.fail("flat_keyful_sampler error has regressed!")
            else:
                raise  # Re-raise if it's a different KeyError
