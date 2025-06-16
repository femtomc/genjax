"""
NumPyro implementation of the curve fitting model for benchmark comparisons.

This module provides equivalent NumPyro implementations of the GenJAX curve fitting model,
including the outlier-robust sine wave inference with importance sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import replay, seed
from numpyro.infer.util import log_density
from numpyro.infer import MCMC, HMC
import matplotlib.pyplot as plt
import numpy as np


def sinfn(x, freq, offset):
    """Sine function with frequency and offset parameters."""
    return jnp.sin(2.0 * jnp.pi * freq * x + offset)


def point_model(x, freq, offset, obs=None):
    """
    NumPyro model for a single data point with Gaussian noise.

    Args:
        x: Input location (scalar)
        freq: Frequency parameter
        offset: Offset parameter
        obs: Observed value (for conditioning)
    """
    # Deterministic curve value
    y_det = sinfn(x, freq, offset)

    # Observation with noise
    y_observed = numpyro.sample("obs", dist.Normal(y_det, 0.3), obs=obs)

    return y_observed


def npoint_model(xs, obs_dict=None):
    """
    NumPyro model for multiple data points with shared sine wave parameters.

    Args:
        xs: Input locations (jax array)
        obs_dict: Dictionary with observations for conditioning
    """
    # Sample sine wave parameters
    freq = numpyro.sample("freq", dist.Exponential(10.0))
    offset = numpyro.sample("offset", dist.Uniform(0.0, 2.0 * jnp.pi))

    # Sample points independently using plates
    with numpyro.plate("data", len(xs)):
        # Extract observations if provided
        obs_vals = None
        if obs_dict is not None and "obs" in obs_dict:
            obs_vals = obs_dict["obs"]

        # Vectorized computation for deterministic values
        y_det = sinfn(xs, freq, offset)

        # Observations with noise
        y_observed = numpyro.sample("obs", dist.Normal(y_det, 0.3), obs=obs_vals)

    return y_observed


def guide_npoint(xs, obs_dict=None):
    """
    Guide (proposal) for importance sampling that samples from the prior.

    This is equivalent to using the model's internal proposal in GenJAX.
    """
    # Sample parameters from prior (same as model)
    numpyro.sample("freq", dist.Exponential(10.0))
    numpyro.sample("offset", dist.Uniform(0.0, 2.0 * jnp.pi))


def single_importance_sample(key, xs, obs_dict):
    """
    Single importance sampling step.

    Args:
        key: JAX random key
        xs: Input locations
        obs_dict: Observations dictionary

    Returns:
        Tuple of (sample_dict, log_weight)
    """
    key1, key2 = jrand.split(key)

    # Sample from guide
    seeded_guide = seed(guide_npoint, key1)
    guide_log_density, guide_trace = log_density(seeded_guide, (xs, None), {}, {})

    # Replay model with guide trace
    seeded_model = seed(npoint_model, key2)
    replay_model = replay(seeded_model, guide_trace)
    model_log_density, model_trace = log_density(replay_model, (xs, obs_dict), {}, {})

    # Compute importance weight
    log_weight = model_log_density - guide_log_density

    # Extract sample
    sample = {
        "freq": model_trace["freq"]["value"],
        "offset": model_trace["offset"]["value"],
    }

    return sample, log_weight


def run_importance_sampling(key, xs, ys, num_samples=1000):
    """
    Run importance sampling inference for the curve fitting model.

    Args:
        key: JAX random key
        xs: Input locations (jax array)
        ys: Observed values (jax array)
        num_samples: Number of importance samples

    Returns:
        Dictionary with samples and log weights
    """
    # Prepare observations
    obs_dict = {"obs": ys}

    # Generate random keys for sampling
    keys = jrand.split(key, num_samples)

    # Vectorized importance sampling
    vectorized_sample = jax.vmap(
        lambda k: single_importance_sample(k, xs, obs_dict), in_axes=0
    )

    samples, log_weights = vectorized_sample(keys)

    return {"samples": samples, "log_weights": log_weights, "num_samples": num_samples}


def log_marginal_likelihood(log_weights):
    """
    Estimate log marginal likelihood from importance weights.

    Args:
        log_weights: Array of log importance weights

    Returns:
        Log marginal likelihood estimate
    """
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(len(log_weights))


def effective_sample_size(log_weights):
    """
    Compute effective sample size from log importance weights.

    Args:
        log_weights: Array of log importance weights

    Returns:
        Effective sample size
    """
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    return 1.0 / jnp.sum(weights_normalized**2)


def run_hmc_inference(key, xs, ys, num_samples=1000, num_warmup=500):
    """
    Run Hamiltonian Monte Carlo inference for the curve fitting model.

    Args:
        key: JAX random key
        xs: Input locations (jax array)
        ys: Observed values (jax array)
        num_samples: Number of HMC samples
        num_warmup: Number of warmup samples

    Returns:
        Dictionary with posterior samples and diagnostics
    """
    # Prepare observations
    obs_dict = {"obs": ys}

    # Create conditioned model for HMC
    def conditioned_model():
        return npoint_model(xs, obs_dict)

    # Setup HMC sampler (remove num_steps to allow adaptive step size)
    hmc_kernel = HMC(conditioned_model, step_size=0.01)
    mcmc = MCMC(hmc_kernel, num_warmup=num_warmup, num_samples=num_samples)

    # Run MCMC
    mcmc.run(key)

    # Extract samples
    samples = mcmc.get_samples()

    # Get diagnostics
    diagnostics = {
        "divergences": mcmc.get_extra_fields().get("diverging", jnp.array([])),
        "accept_probs": mcmc.get_extra_fields().get("accept_prob", jnp.array([])),
        "step_size": mcmc.get_extra_fields().get("step_size", None),
    }

    return {
        "samples": {
            "freq": samples["freq"],
            "offset": samples["offset"],
        },
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
    }


def hmc_summary_statistics(hmc_result):
    """
    Compute summary statistics for HMC results.

    Args:
        hmc_result: Dictionary returned by run_hmc_inference

    Returns:
        Dictionary with summary statistics
    """
    freq_samples = hmc_result["samples"]["freq"]
    offset_samples = hmc_result["samples"]["offset"]

    # Compute effective sample size and R-hat (simplified)
    def eff_sample_size(x):
        # Simplified ESS calculation
        n = len(x)
        # Autocorrelation-based ESS (simplified)
        return n / (
            1
            + 2
            * jnp.sum(
                jnp.abs(
                    jnp.correlate(x - jnp.mean(x), x - jnp.mean(x), mode="full")[
                        n - 1 : n + 10
                    ]
                )
            )
        )

    summary = {
        "freq": {
            "mean": jnp.mean(freq_samples),
            "std": jnp.std(freq_samples),
            "quantiles": jnp.percentile(
                freq_samples, jnp.array([2.5, 25, 50, 75, 97.5])
            ),
            "ess": eff_sample_size(freq_samples),
        },
        "offset": {
            "mean": jnp.mean(offset_samples),
            "std": jnp.std(offset_samples),
            "quantiles": jnp.percentile(
                offset_samples, jnp.array([2.5, 25, 50, 75, 97.5])
            ),
            "ess": eff_sample_size(offset_samples),
        },
        "num_divergences": jnp.sum(hmc_result["diagnostics"]["divergences"])
        if len(hmc_result["diagnostics"]["divergences"]) > 0
        else 0,
        "mean_accept_prob": jnp.mean(hmc_result["diagnostics"]["accept_probs"])
        if len(hmc_result["diagnostics"]["accept_probs"]) > 0
        else 0.0,
    }

    return summary


# JIT compile the HMC inference function for performance
run_hmc_inference_jit = jax.jit(
    run_hmc_inference,
    static_argnums=(3, 4),  # num_samples and num_warmup are static
)


def generate_test_data(key, n_points=10):
    """
    Generate test data from the model for benchmarking.

    Args:
        key: JAX random key
        n_points: Number of data points

    Returns:
        Tuple of (xs, ys) as jax arrays
    """
    xs = jnp.arange(0, n_points, dtype=jnp.float32)

    # Generate data using the model
    seeded_model = seed(npoint_model, key)
    trace = numpyro.handlers.trace(seeded_model).get_trace(xs, None)
    ys = trace["obs"]["value"]

    return xs, ys


def plot_inference_diagnostics(xs, ys, is_result, hmc_result=None):
    """
    Create diagnostic plots for NumPyro inference results.

    Args:
        xs: Input locations
        ys: Observed data
        is_result: Importance sampling results
        hmc_result: HMC results (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("NumPyro Inference Diagnostics", fontsize=16)

    # Convert to numpy for plotting
    xs_np = np.array(xs)
    ys_np = np.array(ys)

    # Plot 1: Data and posterior predictive samples
    ax1 = axes[0, 0]
    ax1.scatter(xs_np, ys_np, c="red", s=50, alpha=0.7, label="Observed data", zorder=3)

    # Plot true curve for reference
    x_fine = np.linspace(xs_np.min(), xs_np.max(), 100)
    true_freq = 0.3
    true_offset = 1.5
    y_true = np.sin(2 * np.pi * true_freq * x_fine + true_offset)
    ax1.plot(x_fine, y_true, "g-", linewidth=3, alpha=0.8, label="True curve", zorder=2)

    # Importance resample to get proper posterior samples using JAX categorical
    log_weights = jnp.array(is_result["log_weights"])

    # Get all samples
    is_freqs_all = np.array(is_result["samples"]["freq"])
    is_offsets_all = np.array(is_result["samples"]["offset"])

    # Resample indices using JAX categorical with log weights
    n_resample = min(500, len(is_freqs_all))
    resample_key = jrand.key(123)  # Use a fixed seed for reproducibility
    resampled_indices = jrand.categorical(
        resample_key, log_weights, shape=(n_resample,)
    )

    # Sample posterior curves using properly resampled indices
    for idx in resampled_indices:
        freq = is_freqs_all[idx]
        offset = is_offsets_all[idx]
        y_curve = np.sin(2 * np.pi * freq * x_fine + offset)
        ax1.plot(
            x_fine, y_curve, "b-", alpha=0.05, linewidth=0.3
        )  # Lower alpha, thinner lines

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Data + Posterior Curves (IS)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter posterior distributions
    ax2 = axes[0, 1]

    # Use resampled indices for proper posterior representation
    resampled_freqs = is_freqs_all[resampled_indices]
    resampled_offsets = is_offsets_all[resampled_indices]

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

    # Add HMC samples if available
    if hmc_result:
        hmc_freqs = np.array(hmc_result["samples"]["freq"])
        hmc_offsets = np.array(hmc_result["samples"]["offset"])
        ax2.scatter(
            hmc_freqs, hmc_offsets, alpha=0.3, s=10, c="orange", label="HMC samples"
        )

    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Offset")
    ax2.set_title("Parameter Posterior")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: HMC trace plots
    ax3 = axes[1, 0]
    if hmc_result:
        hmc_freqs = np.array(hmc_result["samples"]["freq"])
        ax3.plot(hmc_freqs, alpha=0.7, label="Frequency")
        ax3.set_xlabel("Sample")
        ax3.set_ylabel("Value")
        ax3.set_title("HMC Trace (Frequency)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No HMC results", transform=ax3.transAxes, ha="center")

    # Plot 4: Log weights distribution (importance sampling)
    ax4 = axes[1, 1]
    log_weights = np.array(is_result["log_weights"])
    ax4.hist(log_weights, bins=30, alpha=0.7, density=True, color="blue")
    ax4.axvline(
        log_weights.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {log_weights.mean():.2f}",
    )
    ax4.set_xlabel("Log Weight")
    ax4.set_ylabel("Density")
    ax4.set_title("Importance Weights Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/curvefit/figs/numpyro_diagnostics.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved diagnostic plot as 'examples/curvefit/figs/numpyro_diagnostics.pdf'")


# JIT compile the inference function for performance
run_importance_sampling_jit = jax.jit(
    run_importance_sampling,
    static_argnums=(3,),  # num_samples is static
)


# Example usage and testing
if __name__ == "__main__":
    from data import generate_test_dataset, print_dataset_summary

    # Generate common test data
    data = generate_test_dataset(seed=42, n_points=20)
    xs, ys = data["xs"], data["ys"]
    print_dataset_summary(data, "NumPyro Test Dataset")

    # Generate keys for inference
    key = jrand.key(42)
    key1, key2 = jrand.split(key, 2)

    print("\n=== Importance Sampling ===")
    # Run importance sampling
    result = run_importance_sampling_jit(key2, xs, ys, num_samples=5000)

    print("Inference complete:")
    print(f"  Number of samples: {result['num_samples']}")
    print(
        f"  Log marginal likelihood: {log_marginal_likelihood(result['log_weights']):.4f}"
    )
    print(
        f"  Effective sample size: {effective_sample_size(result['log_weights']):.1f}"
    )

    # Extract parameter statistics
    freqs = result["samples"]["freq"]
    offsets = result["samples"]["offset"]

    print(f"  Frequency range: [{freqs.min():.3f}, {freqs.max():.3f}]")
    print(f"  Offset range: [{offsets.min():.3f}, {offsets.max():.3f}]")

    print("\n=== Hamiltonian Monte Carlo ===")
    # Run HMC inference (don't use JIT due to diagnostics formatting issues)
    hmc_result = run_hmc_inference(key1, xs, ys, num_samples=2000, num_warmup=1000)

    print("HMC inference complete:")
    print(f"  Number of samples: {hmc_result['num_samples']}")
    print(f"  Number of warmup: {hmc_result['num_warmup']}")

    # Compute summary statistics
    summary = hmc_summary_statistics(hmc_result)

    print(
        f"  Frequency posterior: μ={summary['freq']['mean']:.3f}, σ={summary['freq']['std']:.3f}"
    )
    print(
        f"  Offset posterior: μ={summary['offset']['mean']:.3f}, σ={summary['offset']['std']:.3f}"
    )
    print(f"  Number of divergences: {summary['num_divergences']}")
    print(f"  Mean acceptance probability: {summary['mean_accept_prob']:.3f}")
    print(f"  Frequency ESS: {summary['freq']['ess']:.1f}")
    print(f"  Offset ESS: {summary['offset']['ess']:.1f}")

    print("\n=== Generating Diagnostic Plots ===")
    plot_inference_diagnostics(xs, ys, result, hmc_result)
