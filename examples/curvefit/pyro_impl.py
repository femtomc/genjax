"""
Pyro implementation of the curve fitting model for benchmark comparisons.

This module provides equivalent Pyro implementations of the GenJAX curve fitting model,
including the outlier-robust sine wave inference with importance sampling.
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import math
import numpy as np
import matplotlib.pyplot as plt


def sinfn(x, freq, offset):
    """Sine function with frequency and offset parameters."""
    return torch.sin(2.0 * math.pi * freq * x + offset)


def point_model(x, freq, offset, idx, obs=None):
    """
    Pyro model for a single data point with Gaussian noise.

    Args:
        x: Input location (scalar)
        freq: Frequency parameter
        offset: Offset parameter
        idx: Index for unique sample site names
        obs: Observed value (for conditioning)
    """
    # Deterministic curve value
    y_det = sinfn(x, freq, offset)

    # Observation with noise (unique name per index)
    y_observed = pyro.sample(f"obs_{idx}", dist.Normal(y_det, 0.3), obs=obs)

    return y_observed


def npoint_model(xs, obs_dict=None):
    """
    Pyro model for multiple data points with shared sine wave parameters.

    Args:
        xs: Input locations (tensor)
        obs_dict: Dictionary with observations for conditioning
    """
    # Sample sine wave parameters
    freq = pyro.sample("freq", dist.Exponential(10.0))
    offset = pyro.sample("offset", dist.Uniform(0.0, 2.0 * math.pi))

    # Sample points independently
    ys = []
    for i, x in enumerate(xs):
        with pyro.plate(f"data_{i}", 1):
            # Extract observation if provided
            obs_val = None
            if obs_dict is not None and "obs" in obs_dict:
                obs_val = obs_dict["obs"][i] if i < len(obs_dict["obs"]) else None

            y = point_model(x, freq, offset, i, obs=obs_val)
            ys.append(y)

    return torch.stack(ys)


def guide_npoint(xs, obs_dict=None):
    """
    Guide (proposal) for importance sampling that samples from the prior.

    This is equivalent to using the model's internal proposal in GenJAX.
    """
    # Sample parameters from prior (same as model)
    pyro.sample("freq", dist.Exponential(10.0))
    pyro.sample("offset", dist.Uniform(0.0, 2.0 * math.pi))


def run_importance_sampling(xs, ys, num_samples=1000):
    """
    Run importance sampling inference for the curve fitting model.

    Args:
        xs: Input locations (numpy array or list)
        ys: Observed values (numpy array or list)
        num_samples: Number of importance samples

    Returns:
        Dictionary with samples and log weights
    """
    # Convert to torch tensors
    if isinstance(xs, (list, np.ndarray)):
        xs = torch.tensor(xs, dtype=torch.float32)
    if isinstance(ys, (list, np.ndarray)):
        ys = torch.tensor(ys, dtype=torch.float32)

    # Clear any existing parameters
    pyro.clear_param_store()

    # Manual importance sampling
    samples = []
    log_weights = []

    for _ in range(num_samples):
        # Sample from guide (prior)
        guide_trace = pyro.poutine.trace(guide_npoint).get_trace(xs)

        # Compute log probability under guide
        guide_log_prob = guide_trace.log_prob_sum()

        # Replay model with guide samples and condition on observations
        conditions = {}
        for i, y in enumerate(ys):
            conditions[f"obs_{i}"] = y

        conditioned_model = pyro.poutine.condition(npoint_model, data=conditions)
        replayed_model = pyro.poutine.replay(conditioned_model, trace=guide_trace)
        model_trace = pyro.poutine.trace(replayed_model).get_trace(xs)

        # Compute log probability under model
        model_log_prob = model_trace.log_prob_sum()

        # Importance weight is model_prob / guide_prob
        log_weight = model_log_prob - guide_log_prob

        # Extract sample
        sample = {
            "freq": guide_trace.nodes["freq"]["value"].item(),
            "offset": guide_trace.nodes["offset"]["value"].item(),
        }

        samples.append(sample)
        log_weights.append(log_weight.item())

    return {
        "samples": samples,
        "log_weights": torch.tensor(log_weights),
        "num_samples": num_samples,
    }


def log_marginal_likelihood(log_weights):
    """
    Estimate log marginal likelihood from importance weights.

    Args:
        log_weights: Tensor of log importance weights

    Returns:
        Log marginal likelihood estimate
    """
    return torch.logsumexp(log_weights, dim=0) - math.log(len(log_weights))


def effective_sample_size(log_weights):
    """
    Compute effective sample size from log importance weights.

    Args:
        log_weights: Tensor of log importance weights

    Returns:
        Effective sample size
    """
    log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
    weights_normalized = torch.exp(log_weights_normalized)
    return 1.0 / torch.sum(weights_normalized**2)


def variational_guide(xs, obs_dict=None):
    """
    Variational guide for SVI that learns the main parameters.
    """
    # Learnable parameters for frequency (exponential distribution approximated by LogNormal)
    freq_loc = pyro.param("freq_loc", torch.tensor(-1.0))  # log-space
    freq_scale = pyro.param(
        "freq_scale", torch.tensor(0.5), constraint=dist.constraints.positive
    )

    # Learnable parameters for offset (use Beta distribution scaled to [0, 2π])
    offset_alpha = pyro.param(
        "offset_alpha", torch.tensor(2.0), constraint=dist.constraints.positive
    )
    offset_beta = pyro.param(
        "offset_beta", torch.tensor(2.0), constraint=dist.constraints.positive
    )

    # Sample from variational distributions
    # Frequency: use LogNormal to ensure positivity
    pyro.sample("freq", dist.LogNormal(freq_loc, freq_scale))

    # Offset: use Beta distribution scaled to [0, 2π] to match model's Uniform
    beta_sample = pyro.sample("beta_sample", dist.Beta(offset_alpha, offset_beta))
    offset = beta_sample * (2.0 * math.pi)  # Scale [0,1] to [0, 2π]
    pyro.sample("offset", dist.Delta(offset))


def run_variational_inference(xs, ys, num_iterations=1000, learning_rate=0.01):
    """
    Run stochastic variational inference for the curve fitting model.

    Args:
        xs: Input locations (numpy array or list)
        ys: Observed values (numpy array or list)
        num_iterations: Number of SVI iterations
        learning_rate: Learning rate for optimization

    Returns:
        Dictionary with final guide parameters and ELBO trace
    """
    # Convert to torch tensors
    if isinstance(xs, (list, np.ndarray)):
        xs = torch.tensor(xs, dtype=torch.float32)
    if isinstance(ys, (list, np.ndarray)):
        ys = torch.tensor(ys, dtype=torch.float32)

    # Clear parameters and setup model conditioning
    pyro.clear_param_store()

    # Create conditioned model
    conditions = {}
    for i, y in enumerate(ys):
        conditions[f"obs_{i}"] = y
    conditioned_model = pyro.poutine.condition(npoint_model, data=conditions)

    # Setup SVI
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(conditioned_model, variational_guide, optimizer, loss=Trace_ELBO())

    # Run optimization
    elbo_trace = []
    for step in range(num_iterations):
        loss = svi.step(xs)
        elbo_trace.append(-loss)  # Convert loss to ELBO

        if step % 100 == 0:
            print(f"  Step {step}, ELBO: {-loss:.3f}")

    # Extract final parameters
    final_params = {}
    for name, param in pyro.get_param_store().items():
        final_params[name] = param.detach().clone()

    return {
        "final_params": final_params,
        "elbo_trace": torch.tensor(elbo_trace),
        "num_iterations": num_iterations,
    }


def sample_from_variational_posterior(xs, num_samples=1000):
    """
    Sample from the fitted variational posterior.

    Args:
        xs: Input locations for the guide
        num_samples: Number of samples to draw

    Returns:
        Dictionary with posterior samples
    """
    samples = []

    for _ in range(num_samples):
        # Sample from variational guide
        trace = pyro.poutine.trace(variational_guide).get_trace(xs)

        sample = {
            "freq": trace.nodes["freq"]["value"].item(),
            "offset": trace.nodes["offset"]["value"].item(),
        }
        samples.append(sample)

    return {"samples": samples, "num_samples": num_samples}


def generate_test_data(n_points=10, seed=42):
    """
    Generate test data from the model for benchmarking.

    Args:
        n_points: Number of data points
        seed: Random seed

    Returns:
        Tuple of (xs, ys) as torch tensors
    """
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    xs = torch.arange(0, n_points, dtype=torch.float32)

    # Generate data using the model
    pyro.clear_param_store()
    trace = pyro.poutine.trace(npoint_model).get_trace(xs)

    # Collect indexed observation values
    ys = []
    for i in range(n_points):
        ys.append(trace.nodes[f"obs_{i}"]["value"])
    ys = torch.stack(ys)

    return xs, ys


def plot_inference_diagnostics(xs, ys, is_result, vi_result, vi_samples):
    """
    Create diagnostic plots for Pyro inference results.

    Args:
        xs: Input locations
        ys: Observed data
        is_result: Importance sampling results
        vi_result: Variational inference results
        vi_samples: Samples from variational posterior
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pyro Inference Diagnostics", fontsize=16)

    # Convert to numpy for plotting
    xs_np = xs.detach().numpy() if torch.is_tensor(xs) else np.array(xs)
    ys_np = (
        ys.detach().numpy().flatten() if torch.is_tensor(ys) else np.array(ys).flatten()
    )

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
    import jax.numpy as jnp
    import jax.random as jrand

    log_weights = jnp.array(is_result["log_weights"])

    # Get all samples
    is_freqs_all = np.array([s["freq"] for s in is_result["samples"]])
    is_offsets_all = np.array([s["offset"] for s in is_result["samples"]])

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

    # Add VI samples if available
    if vi_samples:
        vi_freqs = [s["freq"] for s in vi_samples["samples"]]
        vi_offsets = [s["offset"] for s in vi_samples["samples"]]
        ax2.scatter(
            vi_freqs, vi_offsets, alpha=0.3, s=10, c="orange", label="VI samples"
        )

    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Offset")
    ax2.set_title("Parameter Posterior")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: ELBO trace
    ax3 = axes[1, 0]
    if vi_result:
        elbo_trace = vi_result["elbo_trace"].detach().numpy()
        ax3.plot(elbo_trace, "g-", linewidth=2)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("ELBO")
        ax3.set_title("Variational Inference Convergence")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No VI results", transform=ax3.transAxes, ha="center")

    # Plot 4: Log weights distribution (importance sampling)
    ax4 = axes[1, 1]
    log_weights = is_result["log_weights"].detach().numpy()
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
        "examples/curvefit/figs/pyro_diagnostics.pdf", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved diagnostic plot as 'examples/curvefit/figs/pyro_diagnostics.pdf'")


# Example usage and testing
if __name__ == "__main__":
    from data import generate_test_dataset, print_dataset_summary, convert_to_torch

    # Generate common test data
    data = generate_test_dataset(seed=42, n_points=20)
    torch_data = convert_to_torch(data)
    xs, ys = (
        torch_data["xs"],
        torch_data["ys"].unsqueeze(-1),
    )  # Add batch dimension for consistency
    print_dataset_summary(data, "Pyro Test Dataset")

    print("\n=== Importance Sampling ===")
    # Run importance sampling
    result = run_importance_sampling(xs, ys, num_samples=5000)

    print("Inference complete:")
    print(f"  Number of samples: {result['num_samples']}")
    print(
        f"  Log marginal likelihood: {log_marginal_likelihood(result['log_weights']):.4f}"
    )
    print(
        f"  Effective sample size: {effective_sample_size(result['log_weights']):.1f}"
    )

    # Extract parameter statistics
    freqs = [s["freq"] for s in result["samples"]]
    offsets = [s["offset"] for s in result["samples"]]

    print(f"  Frequency range: [{min(freqs):.3f}, {max(freqs):.3f}]")
    print(f"  Offset range: [{min(offsets):.3f}, {max(offsets):.3f}]")

    print("\n=== Variational Inference ===")
    # Run variational inference with simplified guide
    vi_result = run_variational_inference(
        xs, ys, num_iterations=500, learning_rate=0.01
    )

    print("Variational inference complete:")
    print(f"  Final ELBO: {vi_result['elbo_trace'][-1]:.4f}")
    print(
        f"  Frequency posterior: μ={vi_result['final_params']['freq_loc']:.3f}, σ={vi_result['final_params']['freq_scale']:.3f}"
    )
    print(
        f"  Offset posterior: α={vi_result['final_params']['offset_alpha']:.3f}, β={vi_result['final_params']['offset_beta']:.3f}"
    )

    # Sample from variational posterior for diagnostics
    vi_samples = sample_from_variational_posterior(xs, num_samples=5000)
    vi_freqs = [s["freq"] for s in vi_samples["samples"]]
    vi_offsets = [s["offset"] for s in vi_samples["samples"]]

    print(f"  VI Frequency range: [{min(vi_freqs):.3f}, {max(vi_freqs):.3f}]")
    print(f"  VI Offset range: [{min(vi_offsets):.3f}, {max(vi_offsets):.3f}]")

    print("\n=== Generating Diagnostic Plots ===")
    plot_inference_diagnostics(xs, ys, result, vi_result, vi_samples)
