"""Tests for GenJAX Gaussian Process module."""

import jax
import jax.numpy as jnp
import genjax
from genjax.gp import GP, RBF, Matern52
from genjax.gp.mean import Zero, Constant as ConstantMean
from genjax import gen, seed


def test_gp_kernels():
    """Test basic kernel functionality."""
    x = jnp.array([[0.0], [1.0], [2.0]])

    # Test RBF kernel
    rbf = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(1.0))
    K = rbf(x, x)
    assert K.shape == (3, 3)
    assert jnp.allclose(jnp.diag(K), 1.0)  # Variance on diagonal

    # Test Matern52 kernel
    matern = Matern52(variance=jnp.array(2.0), lengthscale=jnp.array(0.5))
    K = matern(x, x)
    assert K.shape == (3, 3)
    assert jnp.allclose(jnp.diag(K), 2.0)  # Variance on diagonal


def test_gp_mean_functions():
    """Test mean function implementations."""
    x = jnp.array([[0.0], [1.0], [2.0]])

    # Zero mean
    zero_mean = Zero()
    m = zero_mean(x)
    assert jnp.allclose(m, 0.0)

    # Constant mean
    const_mean = ConstantMean(value=jnp.array(3.14))
    m = const_mean(x)
    assert jnp.allclose(m, 3.14)


def test_gp_simulate():
    """Test GP forward simulation."""
    key = jax.random.PRNGKey(42)

    # Create GP with RBF kernel
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(
        kernel=kernel,
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    # Test points
    x_test = jnp.linspace(-2, 2, 10).reshape(-1, 1)

    # Simulate from prior using seed transformation
    simulate_fn = seed(gp.simulate)
    trace = simulate_fn(key, x_test)

    assert trace.y_test.shape == (10,)
    assert trace.score > 0  # Score is -log_prob, so positive


def test_gp_conditioning():
    """Test GP conditioning on observations."""
    key = jax.random.PRNGKey(42)

    # Create GP
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(
        kernel=kernel,
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    # Training data
    x_train = jnp.array([[0.0], [1.0]])
    y_train = jnp.array([0.0, 1.0])

    # Test points
    x_test = jnp.linspace(-1, 2, 20).reshape(-1, 1)

    # Simulate from posterior
    simulate_fn = seed(gp.simulate)
    trace = simulate_fn(key, x_test, x_train=x_train, y_train=y_train)

    assert trace.y_test.shape == (20,)
    # GP should interpolate between training points
    # At x≈0.58, we expect a value between 0 and 1
    assert -0.5 < trace.y_test[10] < 1.5  # Reasonable range
    assert 0.0 < trace.y_test[15] < 2.0  # Reasonable range


def test_gp_assess():
    """Test GP density evaluation."""
    # Create GP
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(
        kernel=kernel,
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    # Test data
    x_test = jnp.array([[0.0], [1.0]])
    y_test = jnp.array([0.5, -0.5])

    # Assess density
    log_prob, retval = gp.assess(y_test, x_test)

    assert log_prob < 0  # Log probability should be negative
    assert jnp.array_equal(retval, y_test)


def test_gp_in_gen_function():
    """Test using GP within a @gen function."""
    # Create GP
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(
        kernel=kernel,
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    @gen
    def model(x_train, y_train, x_test):
        # Sample from GP posterior
        y_test = gp(x_test, x_train=x_train, y_train=y_train) @ "gp"

        # Could compose with other generative functions here
        # For example, sample observation noise
        noise = genjax.normal(0.0, 0.1) @ "noise"

        return y_test + noise

    # Data
    x_train = jnp.array([[0.0], [1.0]])
    y_train = jnp.array([0.0, 1.0])
    x_test = jnp.array([[0.5]])

    # Run inference
    trace = model.simulate(x_train, y_train, x_test)

    # Check we got reasonable values
    # When GP is used in a @gen function, the trace stores the GP's output
    choices = trace.get_choices()
    assert "gp" in choices
    gp_sample = choices["gp"]
    # The GP returns a 1D array for test points
    assert gp_sample.shape == (1,)
    assert -2.0 < gp_sample[0] < 2.0  # Should be reasonable


def test_gp_generate_with_constraints():
    """Test GP generation with constraints."""
    # Create GP
    kernel = RBF(variance=jnp.array(1.0), lengthscale=jnp.array(0.5))
    gp = GP(
        kernel=kernel,
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    # Test points
    x_test = jnp.array([[0.0], [1.0], [2.0]])
    constraints = jnp.array([0.0, 1.0, 0.5])

    # Generate with constraints
    trace, weight = gp.generate(constraints, x_test)

    assert jnp.array_equal(trace.y_test, constraints)
    assert weight == 0.0  # Exact inference has zero weight


def test_gp_composition_example():
    """Example of composing GPs with other models."""
    # Create two GPs with different kernels
    gp1 = GP(
        kernel=RBF(variance=jnp.array(1.0), lengthscale=jnp.array(1.0)),
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )
    gp2 = GP(
        kernel=Matern52(variance=jnp.array(0.5), lengthscale=jnp.array(0.5)),
        mean_fn=Zero(),
        noise_variance=0.01,
        jitter=1e-6,
    )

    @gen
    def hierarchical_gp_model(x_train, y_train, x_test):
        # Sample hyperparameters
        lengthscale = genjax.exponential(1.0) @ "lengthscale"

        # Create GP with sampled hyperparameters
        kernel = RBF(variance=jnp.array(1.0), lengthscale=lengthscale)
        _ = GP(
            kernel=kernel,
            mean_fn=Zero(),
            noise_variance=0.01,
            jitter=1e-6,
        )

        # Sample base function from first GP
        f1 = gp1(x_test, x_train, y_train) @ "f1"

        # Sample residual from second GP
        f2 = gp2(x_test) @ "f2"

        # Combine
        y = f1 + f2

        # Add observation noise
        for i in range(x_test.shape[0]):
            _ = genjax.normal(y[i], 0.1) @ f"obs_{i}"

        return y

    # This demonstrates how GPs can be composed with other
    # probabilistic components in a larger model


if __name__ == "__main__":
    test_gp_kernels()
    test_gp_mean_functions()
    test_gp_simulate()
    test_gp_conditioning()
    test_gp_assess()
    test_gp_in_gen_function()
    test_gp_generate_with_constraints()
    test_gp_composition_example()
    print("All tests passed!")
