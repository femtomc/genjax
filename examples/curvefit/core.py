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
import math

# NumPyro imports
try:
    import numpyro
    import numpyro.distributions as numpyro_dist
    from numpyro.handlers import replay, seed as numpyro_seed
    from numpyro.infer.util import log_density
    from numpyro.infer import MCMC, HMC

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

# Pyro imports
try:
    import torch
    import pyro
    import pyro.distributions as pyro_dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam

    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False

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


# NumPyro Implementation
if NUMPYRO_AVAILABLE:

    def numpyro_sinfn(x, freq, offset):
        """Sine function with frequency and offset parameters."""
        return jnp.sin(2.0 * jnp.pi * freq * x + offset)

    def numpyro_point_model(x, freq, offset, obs=None):
        """NumPyro model for a single data point with Gaussian noise."""
        y_det = numpyro_sinfn(x, freq, offset)
        y_observed = numpyro.sample("obs", numpyro_dist.Normal(y_det, 0.3), obs=obs)
        return y_observed

    def numpyro_npoint_model(xs, obs_dict=None):
        """NumPyro model for multiple data points with shared sine wave parameters."""
        freq = numpyro.sample("freq", numpyro_dist.Exponential(10.0))
        offset = numpyro.sample("offset", numpyro_dist.Uniform(0.0, 2.0 * jnp.pi))

        with numpyro.plate("data", len(xs)):
            obs_vals = None
            if obs_dict is not None and "obs" in obs_dict:
                obs_vals = obs_dict["obs"]
            y_det = numpyro_sinfn(xs, freq, offset)
            y_observed = numpyro.sample(
                "obs", numpyro_dist.Normal(y_det, 0.3), obs=obs_vals
            )
        return y_observed

    def numpyro_guide_npoint(xs, obs_dict=None):
        """Guide for importance sampling that samples from the prior."""
        numpyro.sample("freq", numpyro_dist.Exponential(10.0))
        numpyro.sample("offset", numpyro_dist.Uniform(0.0, 2.0 * jnp.pi))

    def numpyro_single_importance_sample(key, xs, obs_dict):
        """Single importance sampling step for NumPyro."""
        key1, key2 = jrand.split(key)

        seeded_guide = numpyro_seed(numpyro_guide_npoint, key1)
        guide_log_density, guide_trace = log_density(seeded_guide, (xs, None), {}, {})

        seeded_model = numpyro_seed(numpyro_npoint_model, key2)
        replay_model = replay(seeded_model, guide_trace)
        model_log_density, model_trace = log_density(
            replay_model, (xs, obs_dict), {}, {}
        )

        log_weight = model_log_density - guide_log_density

        sample = {
            "freq": model_trace["freq"]["value"],
            "offset": model_trace["offset"]["value"],
        }

        return sample, log_weight

    def numpyro_run_importance_sampling(key, xs, ys, num_samples=1000):
        """Run importance sampling inference using NumPyro."""
        obs_dict = {"obs": ys}
        keys = jrand.split(key, num_samples)

        vectorized_sample = jax.vmap(
            lambda k: numpyro_single_importance_sample(k, xs, obs_dict), in_axes=0
        )

        samples, log_weights = vectorized_sample(keys)
        return {
            "samples": samples,
            "log_weights": log_weights,
            "num_samples": num_samples,
        }

    def numpyro_run_hmc_inference(key, xs, ys, num_samples=1000, num_warmup=500):
        """Run Hamiltonian Monte Carlo inference using NumPyro."""
        obs_dict = {"obs": ys}

        def conditioned_model():
            return numpyro_npoint_model(xs, obs_dict)

        hmc_kernel = HMC(conditioned_model, step_size=0.01)
        mcmc = MCMC(hmc_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(key)

        samples = mcmc.get_samples()
        diagnostics = {
            "divergences": mcmc.get_extra_fields().get("diverging", jnp.array([])),
            "accept_probs": mcmc.get_extra_fields().get("accept_prob", jnp.array([])),
            "step_size": mcmc.get_extra_fields().get("step_size", None),
        }

        return {
            "samples": {"freq": samples["freq"], "offset": samples["offset"]},
            "diagnostics": diagnostics,
            "num_samples": num_samples,
            "num_warmup": num_warmup,
        }

    def numpyro_hmc_summary_statistics(hmc_result):
        """Compute summary statistics for HMC results."""
        freq_samples = hmc_result["samples"]["freq"]
        offset_samples = hmc_result["samples"]["offset"]

        def eff_sample_size(x):
            n = len(x)
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

    # JIT compiled NumPyro functions
    numpyro_run_importance_sampling_jit = jax.jit(
        numpyro_run_importance_sampling, static_argnums=(3,)
    )
    numpyro_run_hmc_inference_jit = jax.jit(
        numpyro_run_hmc_inference, static_argnums=(3, 4)
    )


# Pyro Implementation
if PYRO_AVAILABLE:

    def pyro_sinfn(x, freq, offset):
        """Sine function with frequency and offset parameters."""
        return torch.sin(2.0 * math.pi * freq * x + offset)

    def pyro_point_model(x, freq, offset, idx, obs=None):
        """Pyro model for a single data point with Gaussian noise."""
        y_det = pyro_sinfn(x, freq, offset)
        y_observed = pyro.sample(f"obs_{idx}", pyro_dist.Normal(y_det, 0.3), obs=obs)
        return y_observed

    def pyro_npoint_model(xs, obs_dict=None):
        """Pyro model for multiple data points with shared sine wave parameters."""
        freq = pyro.sample("freq", pyro_dist.Exponential(10.0))
        offset = pyro.sample("offset", pyro_dist.Uniform(0.0, 2.0 * math.pi))

        ys = []
        for i, x in enumerate(xs):
            with pyro.plate(f"data_{i}", 1):
                obs_val = None
                if obs_dict is not None and "obs" in obs_dict:
                    obs_val = obs_dict["obs"][i] if i < len(obs_dict["obs"]) else None
                y = pyro_point_model(x, freq, offset, i, obs=obs_val)
                ys.append(y)

        return torch.stack(ys)

    def pyro_guide_npoint(xs, obs_dict=None):
        """Guide for importance sampling that samples from the prior."""
        pyro.sample("freq", pyro_dist.Exponential(10.0))
        pyro.sample("offset", pyro_dist.Uniform(0.0, 2.0 * math.pi))

    def pyro_run_importance_sampling(xs, ys, num_samples=1000):
        """Run importance sampling inference using Pyro."""
        # Convert JAX arrays to numpy first, then to torch tensors
        if hasattr(xs, "__array__"):  # JAX array
            xs = np.array(xs)
        if hasattr(ys, "__array__"):  # JAX array
            ys = np.array(ys)
        if isinstance(xs, (list, np.ndarray)):
            xs = torch.tensor(xs, dtype=torch.float32)
        if isinstance(ys, (list, np.ndarray)):
            ys = torch.tensor(ys, dtype=torch.float32)

        pyro.clear_param_store()

        samples = []
        log_weights = []

        for _ in range(num_samples):
            guide_trace = pyro.poutine.trace(pyro_guide_npoint).get_trace(xs)
            guide_log_prob = guide_trace.log_prob_sum()

            conditions = {}
            for i, y in enumerate(ys):
                conditions[f"obs_{i}"] = y

            conditioned_model = pyro.poutine.condition(
                pyro_npoint_model, data=conditions
            )
            replayed_model = pyro.poutine.replay(conditioned_model, trace=guide_trace)
            model_trace = pyro.poutine.trace(replayed_model).get_trace(xs)

            model_log_prob = model_trace.log_prob_sum()
            log_weight = model_log_prob - guide_log_prob

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

    def pyro_variational_guide(xs, obs_dict=None):
        """Variational guide for SVI."""
        freq_loc = pyro.param("freq_loc", torch.tensor(-1.0))
        freq_scale = pyro.param(
            "freq_scale", torch.tensor(0.5), constraint=pyro_dist.constraints.positive
        )

        offset_alpha = pyro.param(
            "offset_alpha", torch.tensor(2.0), constraint=pyro_dist.constraints.positive
        )
        offset_beta = pyro.param(
            "offset_beta", torch.tensor(2.0), constraint=pyro_dist.constraints.positive
        )

        pyro.sample("freq", pyro_dist.LogNormal(freq_loc, freq_scale))

        beta_sample = pyro.sample(
            "beta_sample", pyro_dist.Beta(offset_alpha, offset_beta)
        )
        offset = beta_sample * (2.0 * math.pi)
        pyro.sample("offset", pyro_dist.Delta(offset))

    def pyro_run_variational_inference(xs, ys, num_iterations=1000, learning_rate=0.01):
        """Run stochastic variational inference using Pyro."""
        # Convert JAX arrays to numpy first, then to torch tensors
        if hasattr(xs, "__array__"):  # JAX array
            xs = np.array(xs)
        if hasattr(ys, "__array__"):  # JAX array
            ys = np.array(ys)
        if isinstance(xs, (list, np.ndarray)):
            xs = torch.tensor(xs, dtype=torch.float32)
        if isinstance(ys, (list, np.ndarray)):
            ys = torch.tensor(ys, dtype=torch.float32)

        pyro.clear_param_store()

        conditions = {}
        for i, y in enumerate(ys):
            conditions[f"obs_{i}"] = y
        conditioned_model = pyro.poutine.condition(pyro_npoint_model, data=conditions)

        optimizer = Adam({"lr": learning_rate})
        svi = SVI(
            conditioned_model, pyro_variational_guide, optimizer, loss=Trace_ELBO()
        )

        elbo_trace = []
        for step in range(num_iterations):
            loss = svi.step(xs)
            elbo_trace.append(-loss)

            if step % 100 == 0:
                print(f"  Step {step}, ELBO: {-loss:.3f}")

        final_params = {}
        for name, param in pyro.get_param_store().items():
            final_params[name] = param.detach().clone()

        return {
            "final_params": final_params,
            "elbo_trace": torch.tensor(elbo_trace),
            "num_iterations": num_iterations,
        }

    def pyro_sample_from_variational_posterior(xs, num_samples=1000):
        """Sample from the fitted variational posterior."""
        samples = []

        for _ in range(num_samples):
            trace = pyro.poutine.trace(pyro_variational_guide).get_trace(xs)

            sample = {
                "freq": trace.nodes["freq"]["value"].item(),
                "offset": trace.nodes["offset"]["value"].item(),
            }
            samples.append(sample)

        return {"samples": samples, "num_samples": num_samples}

    def pyro_log_marginal_likelihood(log_weights):
        """Estimate log marginal likelihood from importance weights."""
        return torch.logsumexp(log_weights, dim=0) - math.log(len(log_weights))

    def pyro_effective_sample_size(log_weights):
        """Compute effective sample size from log importance weights."""
        log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
        weights_normalized = torch.exp(log_weights_normalized)
        return 1.0 / torch.sum(weights_normalized**2)


# Example usage and testing
if __name__ == "__main__":
    import jax.random as jrand
    from data import generate_test_dataset, print_dataset_summary

    # Generate common test data
    data = generate_test_dataset(seed=42, n_points=20)
    xs, ys = data["xs"], data["ys"]
    print_dataset_summary(data, "Test Dataset")

    print("\n=== GenJAX: Importance Sampling (SMC) ===")
    # Run GenJAX inference
    key = jrand.key(42)
    samples, log_weights = infer_latents(key, ys, 5000)

    print("GenJAX inference complete:")
    print(f"  Number of samples: {len(log_weights)}")
    print(f"  Log marginal likelihood: {log_marginal_likelihood(log_weights):.4f}")
    print(f"  Effective sample size: {effective_sample_size(log_weights):.1f}")

    # Extract parameter statistics
    freqs = samples.get_choices()["curve"]["freq"]
    offsets = samples.get_choices()["curve"]["off"]

    print(f"  Frequency range: [{freqs.min():.3f}, {freqs.max():.3f}]")
    print(f"  Offset range: [{offsets.min():.3f}, {offsets.max():.3f}]")

    print("\n=== Generating GenJAX Diagnostic Plots ===")
    plot_inference_diagnostics(None, xs, ys, samples, log_weights)

    # NumPyro comparison
    if NUMPYRO_AVAILABLE:
        print("\n=== NumPyro: Importance Sampling ===")
        key1, key2 = jrand.split(key, 2)
        numpyro_result = numpyro_run_importance_sampling(key1, xs, ys, num_samples=5000)

        print("NumPyro inference complete:")
        print(f"  Number of samples: {numpyro_result['num_samples']}")
        print(
            f"  Log marginal likelihood: {log_marginal_likelihood(numpyro_result['log_weights']):.4f}"
        )
        print(
            f"  Effective sample size: {effective_sample_size(numpyro_result['log_weights']):.1f}"
        )

        print("\n=== NumPyro: Hamiltonian Monte Carlo ===")
        hmc_result = numpyro_run_hmc_inference(
            key2, xs, ys, num_samples=2000, num_warmup=1000
        )
        summary = numpyro_hmc_summary_statistics(hmc_result)

        print("NumPyro HMC inference complete:")
        print(
            f"  Frequency posterior: μ={summary['freq']['mean']:.3f}, σ={summary['freq']['std']:.3f}"
        )
        print(
            f"  Offset posterior: μ={summary['offset']['mean']:.3f}, σ={summary['offset']['std']:.3f}"
        )
        print(f"  Number of divergences: {summary['num_divergences']}")
        print(f"  Mean acceptance probability: {summary['mean_accept_prob']:.3f}")
    else:
        print("\n=== NumPyro: Not Available (install numpyro for comparison) ===")

    # Pyro comparison
    if PYRO_AVAILABLE:
        print("\n=== Pyro: Importance Sampling ===")
        pyro_result = pyro_run_importance_sampling(xs, ys, num_samples=5000)

        print("Pyro inference complete:")
        print(f"  Number of samples: {pyro_result['num_samples']}")
        print(
            f"  Log marginal likelihood: {pyro_log_marginal_likelihood(pyro_result['log_weights']):.4f}"
        )
        print(
            f"  Effective sample size: {pyro_effective_sample_size(pyro_result['log_weights']):.1f}"
        )

        print("\n=== Pyro: Variational Inference ===")
        vi_result = pyro_run_variational_inference(
            xs, ys, num_iterations=500, learning_rate=0.01
        )
        print(f"  Final ELBO: {vi_result['elbo_trace'][-1]:.4f}")
    else:
        print(
            "\n=== Pyro: Not Available (install torch and pyro-ppl for comparison) ==="
        )
