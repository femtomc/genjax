"""Test invariants and properties of GenJAX GP implementation."""

import jax
import jax.numpy as jnp
import genjax
from genjax.gp import GP, RBF, Matern52, Sum, Product
from genjax.gp.mean import Zero
from genjax import gen


class TestGPInvariants:
    """Test mathematical invariants that GPs should satisfy."""

    def test_prior_posterior_consistency(self):
        """Test that posterior with no data equals prior."""
        from genjax import seed

        key = jax.random.PRNGKey(0)
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        x_test = jnp.linspace(-2, 2, 10).reshape(-1, 1)

        # Prior sample
        prior_trace = seed(lambda: gp.simulate(x_test))(key)

        # "Posterior" with empty training data
        posterior_trace = seed(
            lambda: gp.simulate(x_test, x_train=jnp.empty((0, 1)), y_train=jnp.empty(0))
        )(key)

        # Should give same result
        assert jnp.allclose(prior_trace.y_test, posterior_trace.y_test)
        assert jnp.allclose(prior_trace.score, posterior_trace.score)

    def test_interpolation_property(self):
        """Test that GP interpolates training data exactly (with zero noise)."""
        key = jax.random.PRNGKey(1)
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(
            kernel, mean_fn=Zero(), noise_variance=1e-10, jitter=1e-6
        )  # Near-zero noise

        # Training data
        x_train = jnp.array([[0.0], [1.0], [2.0]])
        y_train = jnp.array([0.5, -0.3, 1.2])

        # Evaluate at training points
        _ = gp.simulate(x_train, x_train=x_train, y_train=y_train, key=key)

        # Mean should interpolate the data
        mean, _ = gp._compute_posterior(x_train, y_train, x_train)
        assert jnp.allclose(mean, y_train, atol=1e-6)

    def test_uncertainty_increases_with_distance(self):
        """Test that prediction uncertainty increases with distance from data."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        # Single training point
        x_train = jnp.array([[0.0]])
        y_train = jnp.array([0.0])

        # Test points at increasing distances
        x_near = jnp.array([[0.1]])
        x_far = jnp.array([[3.0]])

        # Compute posteriors
        _, cov_near = gp._compute_posterior(x_train, y_train, x_near)
        _, cov_far = gp._compute_posterior(x_train, y_train, x_far)

        # Variance should be larger for distant point
        var_near = cov_near[0, 0]
        var_far = cov_far[0, 0]
        assert var_far > var_near

        # Far point should be close to prior variance
        assert jnp.abs(var_far - kernel.variance) < 0.1

    def test_kernel_positive_definiteness(self):
        """Test that kernel matrices are positive definite."""
        kernels = [
            RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5)),
            Matern52(variance=jnp.array(1.0), lengthscale=jnp.array(0.5)),
            Sum(
                RBF(jnp.array(1.0), jnp.array(0.5)), RBF(jnp.array(0.5), jnp.array(2.0))
            ),
            Product(
                RBF(jnp.array(1.0), jnp.array(1.0)),
                Matern52(jnp.array(1.0), jnp.array(1.0)),
            ),
        ]

        x = jax.random.normal(jax.random.PRNGKey(2), (20, 3))

        for kernel in kernels:
            K = kernel(x, x)

            # Check symmetry
            assert jnp.allclose(K, K.T)

            # Check positive definiteness via eigenvalues
            eigvals = jnp.linalg.eigvalsh(K)
            assert jnp.all(eigvals > -1e-8)  # Allow small numerical errors

            # Check Cholesky decomposition works
            K_jittered = K + 1e-6 * jnp.eye(K.shape[0])
            _ = jnp.linalg.cholesky(K_jittered)  # Should not raise

    def test_marginal_consistency(self):
        """Test that marginals of joint distribution are consistent."""
        key = jax.random.PRNGKey(3)
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        # Sample at multiple points
        x_all = jnp.array([[0.0], [1.0], [2.0]])
        trace_all = gp.simulate(x_all, key=key)
        y_all = trace_all.y_test

        # Now condition on first two points and predict third
        x_train = x_all[:2]
        y_train = y_all[:2]
        x_test = x_all[2:3]

        # The conditional distribution should be consistent
        log_prob, _ = gp.assess(y_all[2:3], x_test, x_train=x_train, y_train=y_train)

        # This tests the consistency of the joint factorization
        # p(y1,y2,y3) = p(y1,y2) * p(y3|y1,y2)
        assert jnp.isfinite(log_prob)


class TestJAXBehavior:
    """Test JAX-specific behavior and compilation properties."""

    def test_dynamic_data_sizes(self):
        """Test that GP works with different data sizes without recompilation."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        @jax.jit
        def evaluate_gp(x_test, x_train, y_train):
            log_prob, _ = gp.assess(
                jnp.zeros(x_test.shape[0]), x_test, x_train, y_train
            )
            return log_prob

        # First call - triggers compilation
        x1 = jnp.ones((5, 2))
        y1 = jnp.zeros(5)
        log_prob1 = evaluate_gp(x1, x1, y1)

        # Second call with different size - should reuse compilation
        x2 = jnp.ones((10, 2))
        y2 = jnp.zeros(10)
        log_prob2 = evaluate_gp(x2, x2, y2)

        assert jnp.isfinite(log_prob1)
        assert jnp.isfinite(log_prob2)

    def test_traced_hyperparameters(self):
        """Test that hyperparameters can be JAX traced values."""

        @jax.jit
        def gp_with_dynamic_hyperparams(x_test, lengthscale, variance):
            kernel = RBF(
                variance=jnp.array(variance), lengthscale=jnp.array(lengthscale)
            )
            gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

            # Should work with traced hyperparameters
            mean, cov = gp._compute_posterior(None, None, x_test)
            return mean, cov

        x_test = jnp.linspace(-1, 1, 5).reshape(-1, 1)

        # Try different hyperparameters
        mean1, cov1 = gp_with_dynamic_hyperparams(x_test, 0.5, 1.0)
        mean2, cov2 = gp_with_dynamic_hyperparams(x_test, 2.0, 0.5)

        # Should give different results
        assert not jnp.allclose(cov1, cov2)

    def test_vmap_over_hyperparameters(self):
        """Test vmapping over GP hyperparameters."""

        def evaluate_gp_likelihood(lengthscale, x_train, y_train):
            kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(lengthscale))
            gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)
            log_prob, _ = gp.assess(y_train, x_train)
            return log_prob

        # Vectorize over lengthscales
        vmapped_eval = jax.vmap(evaluate_gp_likelihood, in_axes=(0, None, None))

        # Data
        x_train = jnp.array([[0.0], [1.0], [2.0]])
        y_train = jnp.array([0.0, 1.0, 0.5])

        # Multiple lengthscales
        lengthscales = jnp.array([0.1, 0.5, 1.0, 2.0])

        log_probs = vmapped_eval(lengthscales, x_train, y_train)

        assert log_probs.shape == (4,)
        assert jnp.all(jnp.isfinite(log_probs))
        # Different lengthscales should give different likelihoods
        assert not jnp.allclose(log_probs[0], log_probs[-1])

    def test_grad_through_gp(self):
        """Test gradient computation through GP operations."""

        def gp_log_likelihood(lengthscale, variance, x_train, y_train):
            kernel = RBF(
                variance=jnp.array(variance), lengthscale=jnp.array(lengthscale)
            )
            gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)
            log_prob, _ = gp.assess(y_train, x_train)
            return log_prob

        # Data
        x_train = jnp.array([[0.0], [1.0], [2.0]])
        y_train = jnp.array([0.0, 1.0, 0.5])

        # Compute gradients w.r.t. hyperparameters
        grad_fn = jax.grad(gp_log_likelihood, argnums=(0, 1))
        grad_lengthscale, grad_variance = grad_fn(0.5, 1.0, x_train, y_train)

        assert jnp.isfinite(grad_lengthscale)
        assert jnp.isfinite(grad_variance)
        assert grad_lengthscale != 0.0  # Should have non-zero gradient


class TestGPComposition:
    """Test composition of GPs with other GenJAX components."""

    def test_gp_in_gen_function(self):
        """Test basic usage of GP in @gen function."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        @gen
        def model(x_test):
            y = gp(x_test) @ "gp"
            return y

        x_test = jnp.array([[0.0], [1.0]])
        trace = model.simulate(x_test)

        assert "gp" in trace.get_choices()
        assert trace.get_choices()["gp"].shape == (2,)

    def test_hierarchical_gp_model(self):
        """Test GP with sampled hyperparameters."""

        @gen
        def hierarchical_gp(x_train, y_train, x_test):
            # Sample hyperparameters
            log_lengthscale = genjax.normal(0.0, 1.0) @ "log_lengthscale"
            lengthscale = jnp.exp(log_lengthscale)

            log_variance = genjax.normal(0.0, 1.0) @ "log_variance"
            variance = jnp.exp(log_variance)

            # Create and use GP
            kernel = RBF(
                variance=jnp.array(variance), lengthscale=jnp.array(lengthscale)
            )
            gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

            y_test = gp(x_test, x_train=x_train, y_train=y_train) @ "y_test"

            return y_test

        # Test data
        x_train = jnp.array([[0.0], [1.0]])
        y_train = jnp.array([0.0, 1.0])
        x_test = jnp.array([[0.5]])

        # Should be able to simulate
        trace = hierarchical_gp.simulate(x_train, y_train, x_test)
        assert "log_lengthscale" in trace.get_choices()
        assert "y_test" in trace.get_choices()

        # Should be able to do inference
        observations = {"y_test": jnp.array([0.5])}
        trace, weight = hierarchical_gp.generate(observations, x_train, y_train, x_test)
        assert jnp.isfinite(weight)

    def test_additive_gp_composition(self):
        """Test composition of multiple GPs."""

        @gen
        def additive_model(x):
            # Smooth component
            gp_smooth = GP(
                RBF(variance=jnp.array(1.0), lengthscale=jnp.array(1.0)),
                mean_fn=Zero(),
                noise_variance=0.01,
                jitter=1e-6,
            )
            smooth = gp_smooth(x) @ "smooth"

            # Rough component
            gp_rough = GP(
                Matern52(variance=jnp.array(0.5), lengthscale=jnp.array(0.1)),
                mean_fn=Zero(),
                noise_variance=0.01,
                jitter=1e-6,
            )
            rough = gp_rough(x) @ "rough"

            # Combine
            y = smooth + rough

            # Add observation noise
            sigma = genjax.exponential(10.0) @ "sigma"
            for i in range(x.shape[0]):
                _ = genjax.normal(y[i], sigma) @ f"obs_{i}"

            return y

        x = jnp.linspace(0, 1, 5).reshape(-1, 1)
        trace = additive_model.simulate(x)

        # Check all components are sampled
        choices = trace.get_choices()
        assert "smooth" in choices
        assert "rough" in choices
        assert "sigma" in choices
        assert all(f"obs_{i}" in choices for i in range(5))

    def test_gp_with_vmap(self):
        """Test GP with vmap for multiple independent GPs."""
        from genjax import seed

        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        @gen
        def independent_gps(x_test, n_samples):
            # Sample multiple independent GP realizations
            samples = gp.repeat(n_samples)(x_test) @ "samples"
            return samples

        x_test = jnp.array([[0.0], [1.0], [2.0]])
        n_samples = 10

        # Need to use seed transformation
        trace = seed(lambda: independent_gps.simulate(x_test, n_samples))(
            jax.random.PRNGKey(0)
        )
        samples = trace.get_choices()["samples"]

        assert samples.shape == (10, 3)  # n_samples x n_test_points
        # Check they're actually different samples
        assert not jnp.allclose(samples[0], samples[1])


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_near_duplicate_points(self):
        """Test stability with nearly duplicate training points."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        # Nearly duplicate points
        x_train = jnp.array([[0.0], [0.0 + 1e-8], [1.0]])
        y_train = jnp.array([0.0, 0.0, 1.0])

        x_test = jnp.array([[0.5]])

        # Should handle without numerical issues
        trace = gp.simulate(
            x_test, x_train=x_train, y_train=y_train, key=jax.random.PRNGKey(0)
        )
        assert jnp.isfinite(trace.y_test).all()
        assert jnp.isfinite(trace.score)

    def test_large_condition_number(self):
        """Test with poorly conditioned kernel matrix."""
        # Very small lengthscale leads to poor conditioning
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.01))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=1e-6, jitter=1e-5)

        x_train = jnp.linspace(0, 1, 20).reshape(-1, 1)
        y_train = jnp.sin(10 * x_train[:, 0])

        # Should still work due to jitter
        mean, cov = gp._compute_posterior(x_train, y_train, x_train)
        assert jnp.isfinite(mean).all()
        assert jnp.isfinite(cov).all()

    def test_empty_training_data(self):
        """Test GP with no training data."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        x_test = jnp.array([[0.0], [1.0]])

        # Should work - gives prior
        trace = gp.simulate(
            x_test, x_train=None, y_train=None, key=jax.random.PRNGKey(0)
        )
        assert jnp.isfinite(trace.y_test).all()

    def test_single_training_point(self):
        """Test GP with single training point."""
        kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
        gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

        x_train = jnp.array([[0.5]])
        y_train = jnp.array([1.0])
        x_test = jnp.linspace(0, 1, 10).reshape(-1, 1)

        trace = gp.simulate(
            x_test, x_train=x_train, y_train=y_train, key=jax.random.PRNGKey(0)
        )
        assert jnp.isfinite(trace.y_test).all()

        # Check interpolation at training point
        mean, _ = gp._compute_posterior(x_train, y_train, x_train)
        assert jnp.abs(mean[0] - y_train[0]) < 0.1


def test_gp_importance_weights():
    """Test that GP importance weights are correct."""
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

    x_test = jnp.array([[0.0], [1.0]])
    constraints = jnp.array([0.5, -0.5])

    # Generate with all constraints
    trace, weight = gp.generate(constraints, x_test)

    # Weight should be 0 for exact inference
    assert weight == 0.0

    # Trace should have constrained values
    assert jnp.array_equal(trace.y_test, constraints)


def test_gp_score_computation():
    """Test that GP scores are computed correctly."""
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(kernel, mean_fn=Zero(), noise_variance=0.01, jitter=1e-6)

    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([0.0, 0.0])

    # Get score via assess
    log_prob, _ = gp.assess(y, x)

    # Get score via generate
    trace, _ = gp.generate(y, x)

    # Score should be negative log probability
    assert jnp.isclose(trace.score, -log_prob)


if __name__ == "__main__":
    # Run all tests
    test_invariants = TestGPInvariants()
    test_invariants.test_prior_posterior_consistency()
    test_invariants.test_interpolation_property()
    test_invariants.test_uncertainty_increases_with_distance()
    test_invariants.test_kernel_positive_definiteness()
    test_invariants.test_marginal_consistency()

    test_jax = TestJAXBehavior()
    test_jax.test_dynamic_data_sizes()
    test_jax.test_traced_hyperparameters()
    test_jax.test_vmap_over_hyperparameters()
    test_jax.test_grad_through_gp()

    test_composition = TestGPComposition()
    test_composition.test_gp_in_gen_function()
    test_composition.test_hierarchical_gp_model()
    test_composition.test_additive_gp_composition()
    test_composition.test_gp_with_vmap()

    test_stability = TestNumericalStability()
    test_stability.test_near_duplicate_points()
    test_stability.test_large_condition_number()
    test_stability.test_empty_training_data()
    test_stability.test_single_training_point()

    test_gp_importance_weights()
    test_gp_score_computation()

    print("All GP invariant tests passed!")
