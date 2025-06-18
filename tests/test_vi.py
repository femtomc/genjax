"""
Test cases for GenJAX variational inference algorithms.

These tests validate the VI module functionality including ELBO optimization,
variational families, and complete VI pipelines.
"""

import jax.numpy as jnp
import jax.random as jrand
from jax.lax import scan

from genjax.core import gen, const
from genjax.pjax import seed
from genjax.adev import (
    expectation,
    normal_reinforce,
    normal_reparam,
    multivariate_normal_reparam,
)
from genjax.inference import (
    VariationalApproximation,
    elbo_factory,
    mean_field_normal_family,
    full_covariance_normal_family,
    elbo_vi,
)


def create_simple_variational_model():
    """Create a simple variational model for demonstration."""

    @gen
    def variational_model():
        x = normal_reparam(0.0, 1.0) @ "x"
        normal_reparam(x, 0.3) @ "y"

    return variational_model


def create_simple_variational_family():
    """Create a simple variational family for demonstration."""

    @gen
    def variational_family(constraint, theta):
        # Variational family now has access to constraints but doesn't need to use them
        normal_reinforce(theta, 1.0) @ "x"

    return variational_family


class TestBasicVIFunctionality:
    def test_simple_vi_example_reproduction(self):
        @gen
        def variational_model():
            x = normal_reparam(0.0, 1.0) @ "x"
            normal_reparam(x, 0.3) @ "y"

        @gen
        def variational_family(constraint, theta):
            normal_reinforce(theta, 1.0) @ "x"

        @expectation
        def elbo(data: dict, theta):
            tr = variational_family.simulate(data, theta)
            q_score = tr.get_score()  # log(1/q) = -log(q)
            p, _ = variational_model.assess({**data, **tr.get_choices()})
            # ELBO = log(p) - log(q) = log(p) + q_score
            return p + q_score

        def optimize(data, init_theta):
            def update(theta, _):
                _, theta_grad = elbo.grad_estimate(data, theta)
                theta += 1e-3 * theta_grad
                return theta, theta

            final_theta, intermediate_thetas = scan(
                update,
                init_theta,
                length=100,  # Fewer iterations for faster testing
            )
            return final_theta, intermediate_thetas

        # Test the optimization
        final_theta, thetas = seed(optimize)(jrand.key(1), {"y": 3.0}, 0.01)

        # Verify convergence characteristics
        assert thetas.shape == (100,)
        assert jnp.isfinite(final_theta)
        assert (
            jnp.abs(final_theta - 2.5) < 1.0
        )  # Should converge near the posterior mean

    def test_variational_approximation_dataclass(self):
        """Test VariationalApproximation Pytree dataclass."""
        # Create some dummy data
        final_params = jnp.array(1.5)
        param_history = jnp.linspace(0.0, 1.5, 50)
        loss_history = jnp.exp(-jnp.linspace(0, 5, 50))  # Decreasing loss
        n_iterations = const(50)

        result = VariationalApproximation(
            final_params=final_params,
            param_history=param_history,
            loss_history=loss_history,
            n_iterations=n_iterations,
        )

        # Test getter methods
        assert jnp.allclose(result.get_final_params(), final_params)
        assert jnp.allclose(result.get_param_history(), param_history)
        assert jnp.allclose(result.get_loss_history(), loss_history)
        assert result.n_iterations == n_iterations


class TestELBOFactoryAndOptimization:
    """Test ELBO factory and optimization functions."""

    def test_elbo_factory_basic(self):
        """Test basic ELBO factory functionality."""
        # Create simple models
        target_model = create_simple_variational_model()
        variational_family = create_simple_variational_family()

        # Test that we can evaluate and differentiate
        constraint = {"y": 2.0}

        # Create ELBO function with constraint bound
        elbo_fn = elbo_factory(target_model, variational_family, constraint)

        theta = 1.0
        grad_val = elbo_fn.grad_estimate(theta)

        assert jnp.isfinite(grad_val)
        assert isinstance(grad_val, (float, jnp.ndarray))


class TestVariationalFamilies:
    """Test variational family constructors."""

    def test_mean_field_normal_family_reparam(self):
        """Test mean-field normal family with reparameterization."""
        n_dims = 3
        family = mean_field_normal_family(n_dims, gradient_estimator="reparam")

        # Test sampling
        params = jnp.concatenate(
            [
                jnp.array([0.0, 1.0, -0.5]),  # means
                jnp.array([0.0, -0.5, 0.2]),  # log_stds
            ]
        )

        constraint = {}  # Dummy constraint for testing
        trace = family.simulate(constraint, params)
        samples = trace.get_retval()

        assert samples.shape == (n_dims,)
        assert jnp.all(jnp.isfinite(samples))

    def test_mean_field_normal_family_reinforce(self):
        """Test mean-field normal family with REINFORCE."""
        n_dims = 2
        family = mean_field_normal_family(n_dims, gradient_estimator="reinforce")

        params = jnp.concatenate(
            [
                jnp.array([1.0, -1.0]),  # means
                jnp.array([0.1, 0.3]),  # log_stds
            ]
        )

        constraint = {}  # Dummy constraint for testing
        trace = family.simulate(constraint, params)
        samples = trace.get_retval()

        assert samples.shape == (n_dims,)
        assert jnp.all(jnp.isfinite(samples))

    def test_mean_field_normal_family_invalid_estimator(self):
        """Test that invalid gradient estimator raises error."""
        try:
            mean_field_normal_family(2, gradient_estimator="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown gradient estimator" in str(e)

    def test_full_covariance_normal_family(self):
        """Test full-covariance multivariate normal family."""
        n_dims = 2
        family = full_covariance_normal_family(n_dims, gradient_estimator="reparam")

        # Create parameters
        mean = jnp.array([1.0, -0.5])
        chol_cov = jnp.array([[1.0, 0.0], [0.5, 0.8]])  # Lower triangular
        params = {"mean": mean, "chol_cov": chol_cov}

        constraint = {}  # Dummy constraint for testing
        trace = family.simulate(constraint, params)
        samples = trace.get_retval()

        assert samples.shape == (n_dims,)
        assert jnp.all(jnp.isfinite(samples))


class TestCompleteVIPipeline:
    """Test the complete variational inference pipeline."""

    def test_variational_inference_pipeline(self):
        """Test the complete VI pipeline function."""
        # Create models
        target_model = create_simple_variational_model()
        variational_family = create_simple_variational_family()

        # Set up parameters
        init_params = jnp.array(0.05)
        data = {"y": 2.5}

        def run_vi():
            return elbo_vi(
                target_gf=target_model,
                variational_family=variational_family,
                init_params=init_params,
                constraint=data,
                learning_rate=2e-3,
                n_iterations=150,
                track_history=True,
            )

        result = seed(run_vi)(jrand.key(999))

        # Verify result structure
        assert isinstance(result, VariationalApproximation)
        assert result.n_iterations == 150
        assert len(result.param_history) == 150
        assert len(result.loss_history) == 150

        # Check convergence to reasonable value
        assert jnp.isfinite(result.final_params)

        # For this simple model with y=2.5, should converge near 2.1-2.2
        assert jnp.abs(result.final_params - 2.1) < 0.8

    def test_variational_inference_with_target_args(self):
        """Test VI pipeline with target function arguments."""

        # Create a target model that takes arguments
        @gen
        def parameterized_target_model(prior_mean, prior_std):
            x = normal_reparam(prior_mean, prior_std) @ "x"
            normal_reparam(x, 0.3) @ "y"

        variational_family = create_simple_variational_family()

        def run_vi():
            return elbo_vi(
                target_gf=parameterized_target_model,
                variational_family=variational_family,
                init_params=jnp.array(0.1),
                constraint={"y": 1.0},
                target_args=(0.5, 2.0),  # prior_mean=0.5, prior_std=2.0
                learning_rate=1e-3,
                n_iterations=100,
            )

        result = seed(run_vi)(jrand.key(42))

        assert isinstance(result, VariationalApproximation)
        assert jnp.isfinite(result.final_params)

    def test_multidimensional_vi(self):
        """Test VI with multidimensional variational family."""

        # Create a 2D target model
        @gen
        def target_2d():
            # Use multivariate normal with same addressing as variational family
            loc = jnp.array([0.0, 0.0])
            cov = jnp.eye(2)
            x = multivariate_normal_reparam(loc, cov) @ "x"
            normal_reparam(jnp.sum(x), 0.1) @ "y"

        # Use mean-field family
        variational_family = mean_field_normal_family(2, "reparam")

        # Initial parameters: [mean1, mean2, log_std1, log_std2]
        init_params = jnp.array([0.1, 0.1, -1.0, -1.0])
        data = {"y": 2.0}

        def run_vi():
            return elbo_vi(
                target_gf=target_2d,
                variational_family=variational_family,
                init_params=init_params,
                constraint=data,
                learning_rate=5e-3,
                n_iterations=200,
            )

        result = seed(run_vi)(jrand.key(555))

        assert result.final_params.shape == (4,)
        assert jnp.all(jnp.isfinite(result.final_params))

        # The means should converge to approximately 1.0 each (since x1 + x2 ≈ 2.0)
        final_means = result.final_params[:2]
        assert jnp.abs(jnp.sum(final_means) - 2.0) < 0.5


class TestRobustness:
    """Test robustness and edge cases."""

    def test_different_learning_rates(self):
        """Test VI with different learning rates."""
        target_model = create_simple_variational_model()
        variational_family = create_simple_variational_family()
        data = {"y": 1.5}

        learning_rates = [1e-4, 1e-3, 1e-2]
        results = []

        for lr in learning_rates:

            def run_vi():
                return elbo_vi(
                    target_gf=target_model,
                    variational_family=variational_family,
                    init_params=jnp.array(0.1),
                    constraint=data,
                    learning_rate=lr,
                    n_iterations=100,
                )

            result = seed(run_vi)(jrand.key(42))
            results.append(result)

        # All should converge to finite values
        for result in results:
            assert jnp.isfinite(result.final_params)

        # Higher learning rates should generally converge faster (but may be less stable)
        # Check that all converge to reasonable region (relaxed tolerance)
        for result in results:
            assert jnp.abs(result.final_params - 1.3) < 1.5

    def test_different_initializations(self):
        """Test VI with different initializations."""
        target_model = create_simple_variational_model()
        variational_family = create_simple_variational_family()
        data = {"y": 2.0}

        init_values = [-2.0, 0.0, 2.0, 5.0]

        for init_val in init_values:

            def run_vi():
                return elbo_vi(
                    target_gf=target_model,
                    variational_family=variational_family,
                    init_params=jnp.array(init_val),
                    constraint=data,
                    learning_rate=2e-3,
                    n_iterations=200,
                )

            result = seed(run_vi)(jrand.key(123))

            # Should converge regardless of initialization
            assert jnp.isfinite(result.final_params)
            # For y=2.0, should converge near 1.7
            assert jnp.abs(result.final_params - 1.7) < 1.0

    def test_small_datasets(self):
        """Test behavior with edge case data values."""
        target_model = create_simple_variational_model()
        variational_family = create_simple_variational_family()

        # Test with extreme values
        extreme_data_values = [{"y": -10.0}, {"y": 10.0}, {"y": 0.0}]

        for data in extreme_data_values:

            def run_vi():
                return elbo_vi(
                    target_gf=target_model,
                    variational_family=variational_family,
                    init_params=jnp.array(0.1),
                    constraint=data,
                    learning_rate=1e-3,
                    n_iterations=150,
                )

            result = seed(run_vi)(jrand.key(789))

            # Should not crash and should produce finite results
            assert jnp.isfinite(result.final_params)
            assert jnp.all(jnp.isfinite(result.loss_history))
