from genjax import gen, uniform, normal
from genjax.core import Const
from genjax.pjax import seed
import jax.numpy as jnp
import jax.random as jrand

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
from genjax import Pytree
import jax

# NumPyro imports
import numpyro
import numpyro.distributions as numpyro_dist
from numpyro.handlers import replay, seed as numpyro_seed
from numpyro.infer.util import log_density
from numpyro.infer import MCMC, HMC


pi = jnp.pi
tfd = tfp.distributions

exponential = tfp_distribution(tfd.Exponential)


@Pytree.dataclass
class Lambda(Pytree):
    f: Const[any]
    dynamic_vals: jnp.ndarray
    static_vals: Const[tuple] = Const(())

    def __call__(self, *x):
        return self.f.value(*x, *self.static_vals.value, self.dynamic_vals)


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
    return Lambda(Const(sinfn), jnp.array([freq, offset]))


@gen
def onepoint_curve(x):
    curve = sine() @ "curve"
    y = point(x, curve) @ "y"
    return curve, (x, y)


@gen
def npoint_curve(xs):
    """N-point curve model with xs as input."""
    curve = sine() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)


def infer_latents(xs, ys, n_samples: Const[int]):
    """
    Infer latent curve parameters using GenJAX SMC importance sampling.

    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of importance samples (wrapped in Const)
    """
    from genjax.inference import init

    constraints = {"ys": {"obs": ys}}

    # Use SMC init for importance sampling - seeding applied externally
    result = init(
        npoint_curve,  # target generative function
        (xs,),  # target args with xs as input
        n_samples,  # already wrapped in Const
        constraints,  # constraints
    )

    # Extract samples (traces) and weights for compatibility
    return result.traces, result.log_weights


def hmc_infer_latents(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.05),
    n_steps: Const[int] = Const(10),
):
    """
    Infer latent curve parameters using GenJAX HMC.

    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of MCMC samples (wrapped in Const)
        n_warmup: Number of warmup/burn-in samples (wrapped in Const)
        step_size: HMC step size (wrapped in Const)
        n_steps: Number of leapfrog steps (wrapped in Const)

    Returns:
        (samples, diagnostics): HMC samples and diagnostics
    """
    from genjax.inference import hmc, chain
    from genjax.core import sel

    constraints = {"ys": {"obs": ys}}
    # Generate initial trace - seeding applied externally
    initial_trace, _ = npoint_curve.generate(constraints, xs)

    # Define HMC kernel for continuous parameters
    def hmc_kernel(trace):
        # Select the entire curve (which contains freq and off parameters)
        selection = sel("curve")
        return hmc(trace, selection, step_size=step_size.value, n_steps=n_steps.value)

    # Create MCMC chain - seeding applied externally
    hmc_chain = chain(hmc_kernel)

    # Run HMC with burn-in
    total_steps = n_samples.value + n_warmup.value
    result = hmc_chain(initial_trace, n_steps=Const(total_steps), burn_in=n_warmup)

    return result.traces, {
        "acceptance_rate": result.acceptance_rate,
        "n_samples": result.n_steps.value,
        "n_chains": result.n_chains.value,
    }


def get_points_for_inference(n_points=20):
    """Generate test data for inference with xs as input."""
    # Create grid of input points
    xs = jnp.linspace(0, n_points / 4, n_points)
    # Simulate model to get observations
    trace = npoint_curve.simulate(xs)
    curve, (xs_ret, ys) = trace.get_retval()
    return xs, ys


def log_marginal_likelihood(log_weights):
    """Estimate log marginal likelihood from importance weights."""
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(len(log_weights))


def effective_sample_size(log_weights):
    """Compute effective sample size from log importance weights."""
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    return 1.0 / jnp.sum(weights_normalized**2)


# NumPyro Implementation


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
    model_log_density, model_trace = log_density(replay_model, (xs, obs_dict), {}, {})

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
    """Run HMC inference using NumPyro (non-JIT version, same parameters as GenJAX)."""
    obs_dict = {"obs": ys}

    def conditioned_model(xs, obs_dict):
        return numpyro_npoint_model(xs, obs_dict)

    # Use basic HMC with same parameters as GenJAX for fair comparison
    # Note: Only specify num_steps to avoid trajectory_length conflict
    hmc_kernel = HMC(conditioned_model, step_size=0.01, num_steps=20)
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=False,
        progress_bar=False,
    )  # No JIT for this version
    mcmc.run(key, xs, obs_dict)

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()

    # Basic HMC diagnostics (same as GenJAX HMC for comparison)
    diagnostics = {
        "divergences": extra_fields.get("diverging", jnp.array([])),
        "accept_probs": extra_fields.get("accept_prob", jnp.array([])),
        "step_size": extra_fields.get("step_size", None),
    }

    return {
        "samples": {"freq": samples["freq"], "offset": samples["offset"]},
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
    }


def numpyro_run_hmc_inference_jit_impl(key, xs, ys, num_samples=1000, num_warmup=500):
    """Run HMC inference using NumPyro with JIT compilation (same parameters as GenJAX)."""
    obs_dict = {"obs": ys}

    def conditioned_model(xs, obs_dict):
        return numpyro_npoint_model(xs, obs_dict)

    # Use basic HMC with same parameters as GenJAX for fair comparison
    # Note: Only specify num_steps to avoid trajectory_length conflict
    hmc_kernel = HMC(conditioned_model, step_size=0.01, num_steps=20)
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=True,
        progress_bar=False,
    )  # Enable JIT for this version
    mcmc.run(key, xs, obs_dict)

    samples = mcmc.get_samples()
    diagnostics = {
        "divergences": mcmc.get_extra_fields().get("diverging", jnp.array([])),
        "accept_probs": mcmc.get_extra_fields().get("accept_prob", jnp.array([])),
        "step_size": mcmc.get_extra_fields().get("step_size", None),
    }

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()

    # Basic HMC diagnostics (same as GenJAX HMC for comparison)
    diagnostics = {
        "divergences": extra_fields.get("diverging", jnp.array([])),
        "accept_probs": extra_fields.get("accept_prob", jnp.array([])),
        "step_size": extra_fields.get("step_size", None),
    }

    return {
        "samples": {"freq": samples["freq"], "offset": samples["offset"]},
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
    }


# JIT-compiled version for fair performance comparison
numpyro_run_hmc_inference_jit = jax.jit(
    numpyro_run_hmc_inference_jit_impl, static_argnums=(3, 4)
)  # num_samples, num_warmup


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


# JIT compiled functions for performance benchmarks

# GenJAX JIT-compiled functions - apply seed() before jit()
infer_latents_seeded = seed(infer_latents)
hmc_infer_latents_seeded = seed(hmc_infer_latents)

infer_latents_jit = jax.jit(
    infer_latents_seeded
)  # Use Const pattern instead of static_argnums
hmc_infer_latents_jit = jax.jit(
    hmc_infer_latents_seeded
)  # Use Const pattern instead of static_argnums

# NumPyro JIT-compiled functions
numpyro_run_importance_sampling_jit = jax.jit(
    numpyro_run_importance_sampling, static_argnums=(3,)
)
numpyro_run_hmc_inference_jit = jax.jit(
    numpyro_run_hmc_inference, static_argnums=(3, 4)
)


#
def run_comprehensive_benchmark(
    n_points=20, n_samples=1000, n_warmup=500, seed=42, timing_repeats=50
):
    """
        Run comprehensive benchmarking across all frameworks and methods.

    Tests both importance sampling and HMC across GenJAX and NumPyro.

    Args:
        n_points: Number of data points for inference
        n_samples: Number of samples per method
        n_warmup: Number of warmup samples for MCMC methods
        seed: Random seed for reproducibility
        timing_repeats: Number of timing repetitions

    Returns:
        Dictionary with results for each framework and method
    """
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import benchmark_with_warmup
    from data import generate_test_dataset

    # Generate standardized test data
    data = generate_test_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]

    results = {}

    print(f"\n=== Comprehensive Benchmark: {n_points} points, {n_samples} samples ===")

    # GenJAX Importance Sampling
    print("Running GenJAX Importance Sampling...")

    # Use the pre-seeded JIT-compiled inference function
    def genjax_is_task():
        return infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples))

    genjax_is_times, genjax_is_stats = benchmark_with_warmup(
        genjax_is_task, repeats=timing_repeats
    )

    # Get samples for posterior analysis
    genjax_is_samples, genjax_is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples)
    )

    results["genjax_importance"] = {
        "method": "Importance Sampling",
        "framework": "GenJAX",
        "samples": genjax_is_samples,
        "weights": genjax_is_weights,
        "times": genjax_is_times,
        "timing_stats": genjax_is_stats,
        "log_marginal": log_marginal_likelihood(genjax_is_weights),
        "ess": effective_sample_size(genjax_is_weights),
    }

    # GenJAX HMC
    print("Running GenJAX HMC...")

    # Use the pre-seeded JIT-compiled HMC function
    def genjax_hmc_task():
        return hmc_infer_latents_jit(
            jrand.key(seed), xs, ys, Const(n_samples), Const(n_warmup)
        )

    genjax_hmc_times, genjax_hmc_stats = benchmark_with_warmup(
        genjax_hmc_task, repeats=timing_repeats
    )

    # Get samples for posterior analysis
    genjax_hmc_samples, genjax_hmc_diagnostics = hmc_infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples), Const(n_warmup)
    )

    results["genjax_hmc"] = {
        "method": "HMC",
        "framework": "GenJAX",
        "samples": genjax_hmc_samples,
        "diagnostics": genjax_hmc_diagnostics,
        "times": genjax_hmc_times,
        "timing_stats": genjax_hmc_stats,
    }

    # NumPyro methods
    print("Running NumPyro Importance Sampling...")
    key = jrand.key(seed)

    def numpyro_is_task():
        return numpyro_run_importance_sampling_jit(key, xs, ys, n_samples)

    numpyro_is_times, numpyro_is_stats = benchmark_with_warmup(
        numpyro_is_task, repeats=timing_repeats
    )

    numpyro_is_result = numpyro_run_importance_sampling_jit(key, xs, ys, n_samples)

    results["numpyro_importance"] = {
        "method": "Importance Sampling",
        "framework": "NumPyro",
        "samples": numpyro_is_result["samples"],
        "weights": numpyro_is_result["log_weights"],
        "times": numpyro_is_times,
        "timing_stats": numpyro_is_stats,
        "log_marginal": log_marginal_likelihood(numpyro_is_result["log_weights"]),
        "ess": effective_sample_size(numpyro_is_result["log_weights"]),
    }

    print("Running NumPyro HMC...")

    def numpyro_hmc_task():
        return numpyro_run_hmc_inference_jit(key, xs, ys, n_samples, n_warmup)

    numpyro_hmc_times, numpyro_hmc_stats = benchmark_with_warmup(
        numpyro_hmc_task, repeats=timing_repeats
    )

    numpyro_hmc_result = numpyro_run_hmc_inference_jit(key, xs, ys, n_samples, n_warmup)

    results["numpyro_hmc"] = {
        "method": "HMC",
        "framework": "NumPyro",
        "samples": numpyro_hmc_result["samples"],
        "diagnostics": numpyro_hmc_result["diagnostics"],
        "times": numpyro_hmc_times,
        "timing_stats": numpyro_hmc_stats,
    }

    return results


def extract_posterior_samples(benchmark_results):
    """
    Extract standardized posterior samples from benchmark results.

    Args:
        benchmark_results: Results dictionary from run_comprehensive_benchmark

    Returns:
        Dictionary with standardized posterior samples for each method
    """
    posterior_samples = {}

    for method_name, result in benchmark_results.items():
        framework = result["framework"]
        method = result["method"]

        if framework == "GenJAX":
            if method == "Importance Sampling":
                # Extract and resample using importance weights
                traces = result["samples"]
                weights = result["weights"]

                # Resample according to importance weights
                n_resample = min(1000, len(weights))
                key = jrand.key(123)
                indices = jrand.categorical(key, weights, shape=(n_resample,))

                freq_samples = traces.get_choices()["curve"]["freq"][indices]
                offset_samples = traces.get_choices()["curve"]["off"][indices]

            elif method == "HMC":
                # Extract MCMC samples directly
                traces = result["samples"]
                freq_samples = traces.get_choices()["curve"]["freq"]
                offset_samples = traces.get_choices()["curve"]["off"]

        elif framework == "NumPyro":
            if method == "Importance Sampling":
                # Resample using importance weights
                samples = result["samples"]
                weights = result["weights"]

                n_resample = min(1000, len(weights))
                key = jrand.key(123)
                indices = jrand.categorical(key, weights, shape=(n_resample,))

                freq_samples = samples["freq"][indices]
                offset_samples = samples["offset"][indices]

            elif method == "HMC":
                # MCMC samples
                samples = result["samples"]
                freq_samples = samples["freq"]
                offset_samples = samples["offset"]

        posterior_samples[method_name] = {
            "framework": framework,
            "method": method,
            "freq": freq_samples,
            "offset": offset_samples,
        }

    return posterior_samples


# Example usage and testing
if __name__ == "__main__":
    import jax.random as jrand
    from data import generate_test_dataset, print_dataset_summary
    from genjax.core import Const
    from figs import plot_inference_diagnostics

    # Generate common test data
    data = generate_test_dataset(seed=42, n_points=20)
    xs, ys = data["xs"], data["ys"]
    print_dataset_summary(data, "Test Dataset")

    print("\n=== GenJAX: Importance Sampling (SMC) ===")
    # Run GenJAX inference
    key = jrand.key(42)
    samples, log_weights = infer_latents_jit(key, xs, ys, Const(5000))

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
    # NumPyro is available in curvefit environment
    print("\n=== NumPyro: Importance Sampling ===")
    key1, key2 = jrand.split(key, 2)
    numpyro_result = numpyro_run_importance_sampling_jit(key1, xs, ys, num_samples=5000)

    print("NumPyro inference complete:")
    print(f"  Number of samples: {numpyro_result['num_samples']}")
    print(
        f"  Log marginal likelihood: {log_marginal_likelihood(numpyro_result['log_weights']):.4f}"
    )
    print(
        f"  Effective sample size: {effective_sample_size(numpyro_result['log_weights']):.1f}"
    )

    print("\n=== NumPyro: Hamiltonian Monte Carlo ===")
    hmc_result = numpyro_run_hmc_inference_jit(
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
