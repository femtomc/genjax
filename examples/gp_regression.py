"""Example: Gaussian Process Regression with GenJAX.

This example demonstrates how to use GPs as generative functions
for Bayesian regression with uncertainty quantification.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import genjax
from genjax.gp import GP, RBF, Matern52, LinearMean
from genjax import gen


def generate_synthetic_data(key, n_points=20):
    """Generate synthetic regression data."""
    key1, key2 = jax.random.split(key)

    # Generate random x values
    x = jax.random.uniform(key1, (n_points,), minval=-3, maxval=3)
    x = jnp.sort(x).reshape(-1, 1)

    # True function: sin(x) + small linear trend
    y_true = jnp.sin(x[:, 0]) + 0.1 * x[:, 0]

    # Add noise
    noise = jax.random.normal(key2, (n_points,)) * 0.1
    y = y_true + noise

    return x, y, y_true


@gen
def gp_regression_model(x_train, x_test, noise_std):
    """GP regression model with hyperparameter inference."""
    # Sample hyperparameters
    variance = genjax.gamma(2.0, 2.0) @ "variance"
    lengthscale = genjax.gamma(2.0, 2.0) @ "lengthscale"

    # Create GP with sampled hyperparameters
    kernel = RBF(variance=variance, lengthscale=lengthscale)
    gp = GP(kernel, noise_variance=noise_std**2)

    # Sample function values at test points
    f_test = gp(x_test) @ "f_test"

    # Sample function values at training points
    f_train = gp(x_train) @ "f_train"

    # Observe training data with noise
    for i in range(x_train.shape[0]):
        y_i = genjax.normal(f_train[i], noise_std) @ f"y_{i}"

    return f_test, f_train


def fit_gp_direct(x_train, y_train, x_test):
    """Fit GP with fixed hyperparameters using exact inference."""
    # Use reasonable hyperparameters
    kernel = RBF(variance=1.0, lengthscale=0.5)
    gp = GP(kernel, noise_variance=0.01)

    # Get posterior mean and samples
    key = jax.random.PRNGKey(0)
    samples = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        trace = gp.simulate(x_test, x_train=x_train, y_train=y_train, key=subkey)
        samples.append(trace.y_test)

    samples = jnp.stack(samples)
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)

    return mean, std, samples


def plot_gp_results(
    x_train, y_train, x_test, mean, std, samples=None, y_true_test=None
):
    """Plot GP regression results."""
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.scatter(x_train[:, 0], y_train, c="red", s=50, label="Training data", zorder=3)

    # Plot true function if available
    if y_true_test is not None:
        plt.plot(x_test[:, 0], y_true_test, "k--", label="True function", alpha=0.5)

    # Plot GP mean
    plt.plot(x_test[:, 0], mean, "b-", label="GP mean", linewidth=2)

    # Plot uncertainty bands
    plt.fill_between(
        x_test[:, 0],
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.3,
        color="blue",
        label="95% confidence",
    )

    # Plot some samples if provided
    if samples is not None:
        for i in range(min(5, samples.shape[0])):
            plt.plot(x_test[:, 0], samples[i], "b-", alpha=0.1, linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian Process Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def compare_kernels(x_train, y_train, x_test):
    """Compare different kernel functions."""
    kernels = {
        "RBF": RBF(variance=1.0, lengthscale=0.5),
        "Matern52": Matern52(variance=1.0, lengthscale=0.5),
        "RBF (long)": RBF(variance=1.0, lengthscale=2.0),
        "RBF (short)": RBF(variance=1.0, lengthscale=0.1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, kernel) in enumerate(kernels.items()):
        ax = axes[idx]

        # Fit GP
        gp = GP(kernel, noise_variance=0.01)

        # Get samples
        key = jax.random.PRNGKey(idx)
        samples = []
        for i in range(50):
            key, subkey = jax.random.split(key)
            trace = gp.simulate(x_test, x_train=x_train, y_train=y_train, key=subkey)
            samples.append(trace.y_test)

        samples = jnp.stack(samples)
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        # Plot
        ax.scatter(x_train[:, 0], y_train, c="red", s=30)
        ax.plot(x_test[:, 0], mean, "b-", linewidth=2)
        ax.fill_between(
            x_test[:, 0], mean - 2 * std, mean + 2 * std, alpha=0.3, color="blue"
        )

        # Plot some samples
        for i in range(5):
            ax.plot(x_test[:, 0], samples[i], "b-", alpha=0.1, linewidth=1)

        ax.set_title(f"{name} Kernel")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run GP regression examples."""
    # Generate data
    key = jax.random.PRNGKey(42)
    x_train, y_train, y_true_train = generate_synthetic_data(key, n_points=15)

    # Test points
    x_test = jnp.linspace(-4, 4, 200).reshape(-1, 1)
    y_true_test = jnp.sin(x_test[:, 0]) + 0.1 * x_test[:, 0]

    # Fit GP with direct method
    print("Fitting GP with exact inference...")
    mean, std, samples = fit_gp_direct(x_train, y_train, x_test)

    # Plot results
    fig1 = plot_gp_results(
        x_train, y_train, x_test, mean, std, samples=samples, y_true_test=y_true_test
    )

    # Compare different kernels
    print("Comparing different kernel functions...")
    fig2 = compare_kernels(x_train, y_train, x_test)

    # Save figures
    fig1.savefig("gp_regression_result.png", dpi=150)
    fig2.savefig("gp_kernel_comparison.png", dpi=150)
    print("Saved figures: gp_regression_result.png, gp_kernel_comparison.png")

    # Example: Using GP in a hierarchical model
    print("\nExample of GP in hierarchical model:")

    @gen
    def hierarchical_model(x_train, y_train, x_test):
        # Global trend with linear mean function
        slope = genjax.normal(0.0, 1.0) @ "slope"
        intercept = genjax.normal(0.0, 1.0) @ "intercept"

        # Local variations with GP
        kernel = RBF(variance=0.5, lengthscale=0.5)
        mean_fn = LinearMean(weights=jnp.array([slope]), bias=intercept)
        gp = GP(kernel, mean_fn=mean_fn, noise_variance=0.01)

        # Generate predictions
        y_pred = gp(x_test, x_train=x_train, y_train=y_train) @ "predictions"

        return y_pred

    # This shows how GPs can be integrated into larger probabilistic models
    trace = hierarchical_model.simulate(x_train, y_train, x_test[:5])
    print(f"Hierarchical model predictions (first 5): {trace.get_retval()}")


if __name__ == "__main__":
    main()
