from genjax import gen, uniform, normal
from genjax import seed
import jax.numpy as jnp
import jax.random as jrand

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
from genjax import Pytree
import jax
import matplotlib.pyplot as plt
import numpy as np

pi = jnp.pi
tfd = tfp.distributions

exponential = tfp_distribution(tfd.Exponential)


@Pytree.dataclass
class Lambda(Pytree):
    f: any = Pytree.static()
    dynamic_vals: jnp.ndarray
    static_vals: tuple = Pytree.static(default=())

    def __call__(self, *x):
        return self.f(*x, *self.static_vals, self.dynamic_vals)


### Model + inference code ###
@gen
def point(x, curve):
    y_det = curve(x)
    y_observed = normal(y_det, 0.3) @ "obs"
    return y_observed


def sinfn(x, a):
    return jnp.sin(2.0 * pi * a[0] * x + a[1])


@gen
def sine():
    freq = exponential(10.0) @ "freq"
    offset = uniform(0.0, 2.0 * pi) @ "off"
    return Lambda(sinfn, jnp.array([freq, offset]))


@gen
def onepoint_curve(x):
    curve = sine() @ "curve"
    y = point(x, curve) @ "y"
    return curve, (x, y)


def npoint_curve_factory(n: int):
    """Factory function to create npoint_curve with static n parameter."""

    @gen
    def npoint_curve():
        curve = sine() @ "curve"
        xs = jnp.linspace(0, n / 4, n)  # n is now static from factory closure
        ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
        return curve, (xs, ys)

    return npoint_curve


def _infer_latents(key, ys, n_samples):
    """
    Infer latent curve parameters using genjax.smc.default_importance_sampling.

    Uses factory pattern for npoint_curve to handle static n parameter, and proper
    closure pattern from test_smc.py for default_importance_sampling with seed.
    """
    from genjax.smc import default_importance_sampling

    constraints = {"ys": {"obs": ys}}
    n_points = len(ys)

    # Create model with static n using factory pattern
    npoint_curve_model = npoint_curve_factory(n_points)

    # Create closure for default_importance_sampling that captures static arguments
    # This pattern follows test_smc.py lines 242-248
    def default_importance_sampling_closure(target_gf, target_args, constraints):
        return default_importance_sampling(
            target_gf,
            target_args,
            n_samples,  # n_samples captured as static
            constraints,
        )

    # Apply seed to the closure - pattern from test_smc.py lines 251-256
    result = seed(default_importance_sampling_closure)(
        key,
        npoint_curve_model,  # target generative function (from factory)
        (),  # target args (empty since n is captured in factory)
        constraints,  # constraints
    )

    # Extract samples (traces) and weights for compatibility
    return result.traces, result.log_weights


# For backward compatibility and JIT compilation
infer_latents = jax.jit(_infer_latents, static_argnums=(2,))


def get_points_for_inference(n_points=20):
    npoint_curve_model = npoint_curve_factory(n_points)
    trace = npoint_curve_model.simulate(())
    return trace.get_retval()


def plot_inference_diagnostics(curve, xs, ys, samples, log_weights):
    """
    Create diagnostic plots for GenJAX inference results.

    Args:
        curve: Curve object with parameters
        xs: Input locations
        ys: Observed data
        samples: Trace samples from inference
        log_weights: Log importance weights
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GenJAX Inference Diagnostics", fontsize=16)

    # Convert to numpy for plotting
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    log_weights_np = np.array(log_weights)

    # Plot 1: Data and posterior predictive samples
    ax1 = axes[0, 0]
    ax1.scatter(xs_np, ys_np, c="red", s=50, alpha=0.7, label="Observed data", zorder=3)

    # Plot true curve for reference
    x_fine = np.linspace(xs_np.min(), xs_np.max(), 100)
    true_freq = 0.3
    true_offset = 1.5
    y_true = np.sin(2 * np.pi * true_freq * x_fine + true_offset)
    ax1.plot(x_fine, y_true, "g-", linewidth=3, alpha=0.8, label="True curve", zorder=2)

    # Importance resample to get proper posterior samples using GenJAX categorical
    n_resample = min(500, len(samples.get_choices()["curve"]["freq"]))

    # Use GenJAX categorical with log weights directly
    # Generate a key for resampling
    resample_key = jrand.key(123)  # Use a fixed seed for reproducibility

    # Sample from categorical distribution with log weights as logits
    resampled_indices = jrand.categorical(
        resample_key, jnp.array(log_weights), shape=(n_resample,)
    )

    # Sample posterior curves using properly resampled indices
    for idx in resampled_indices:
        freq = float(samples.get_choices()["curve"]["freq"][idx])
        offset = float(samples.get_choices()["curve"]["off"][idx])
        y_curve = np.sin(2 * np.pi * freq * x_fine + offset)
        ax1.plot(
            x_fine, y_curve, "b-", alpha=0.05, linewidth=0.3
        )  # Lower alpha, thinner lines

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Data + Posterior Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter posterior distributions
    ax2 = axes[0, 1]
    freqs = np.array(samples.get_choices()["curve"]["freq"])
    offsets = np.array(samples.get_choices()["curve"]["off"])

    # Use resampled indices for proper posterior representation
    resampled_freqs = freqs[resampled_indices]
    resampled_offsets = offsets[resampled_indices]

    ax2.scatter(
        resampled_freqs, resampled_offsets, alpha=0.3, s=10, label="Resampled posterior"
    )
    ax2.scatter(
        true_freq,
        true_offset,
        c="green",
        s=100,
        marker="*",
        label="True parameters",
        zorder=3,
    )
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Offset")
    ax2.set_title("Parameter Posterior")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter traces
    ax3 = axes[1, 0]
    ax3.plot(freqs, alpha=0.7, label="Frequency")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Value")
    ax3.set_title("Parameter Trace (Frequency)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Log weights distribution
    ax4 = axes[1, 1]
    ax4.hist(log_weights_np, bins=30, alpha=0.7, density=True, color="blue")
    ax4.axvline(
        log_weights_np.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {log_weights_np.mean():.2f}",
    )
    ax4.set_xlabel("Log Weight")
    ax4.set_ylabel("Density")
    ax4.set_title("Importance Weights Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/curvefit/figs/genjax_diagnostics.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved diagnostic plot as 'examples/curvefit/figs/genjax_diagnostics.pdf'")


def log_marginal_likelihood(log_weights):
    """Estimate log marginal likelihood from importance weights."""
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(len(log_weights))


def effective_sample_size(log_weights):
    """Compute effective sample size from log importance weights."""
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    return 1.0 / jnp.sum(weights_normalized**2)


# Example usage and testing
if __name__ == "__main__":
    import jax.random as jrand
    from data import generate_test_dataset, print_dataset_summary

    # Generate common test data
    data = generate_test_dataset(seed=42, n_points=20)
    xs, ys = data["xs"], data["ys"]
    print_dataset_summary(data, "GenJAX Test Dataset")

    print("\n=== Importance Sampling (SMC) ===")
    # Run inference
    key = jrand.key(42)
    samples, log_weights = infer_latents(key, ys, 5000)

    print("Inference complete:")
    print(f"  Number of samples: {len(log_weights)}")
    print(f"  Log marginal likelihood: {log_marginal_likelihood(log_weights):.4f}")
    print(f"  Effective sample size: {effective_sample_size(log_weights):.1f}")

    # Extract parameter statistics
    freqs = samples.get_choices()["curve"]["freq"]
    offsets = samples.get_choices()["curve"]["off"]

    print(f"  Frequency range: [{freqs.min():.3f}, {freqs.max():.3f}]")
    print(f"  Offset range: [{offsets.min():.3f}, {offsets.max():.3f}]")

    print("\n=== Generating Diagnostic Plots ===")
    plot_inference_diagnostics(None, xs, ys, samples, log_weights)
