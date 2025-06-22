from genjax import gen, normal, Cond, flip
from genjax.core import Const
from genjax.pjax import seed
import jax.numpy as jnp
import jax.random as jrand

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
import jax

# NumPyro imports
import numpyro
import numpyro.distributions as numpyro_dist
from numpyro.handlers import replay, seed as numpyro_seed

# from numpyro.contrib.funsor import log_density  # Moved to function for lazy loading
from numpyro.infer import HMC, MCMC


from genjax import Pytree

pi = jnp.pi
tfd = tfp.distributions

exponential = tfp_distribution(tfd.Exponential)


@Pytree.dataclass
class Lambda(Pytree):
    """Lambda wrapper for curve functions with JAX compatibility."""

    f: Const[any]
    dynamic_vals: jnp.ndarray
    static_vals: Const[tuple] = Const(())

    def __call__(self, *x):
        return self.f.value(*x, *self.static_vals.value, self.dynamic_vals)


### Model + inference code ###
@gen
def point(x, curve):
    y_det = curve(x)
    y_observed = normal(y_det, 0.2) @ "obs"
    return y_observed


@gen
def point_with_noise(x, curve, noise_std):
    """Point model with configurable noise level."""
    y_det = curve(x)
    y_observed = normal(y_det, noise_std) @ "obs"
    return y_observed


def polyfn(x, coeffs):
    """Degree 2 polynomial: y = a + b*x + c*x^2"""
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    return a + b * x + c * x**2


@gen
def polynomial():
    # Use normal distributions for polynomial coefficients
    # Uniform priors with std=1.0 for all coefficients
    a = normal(0.0, 1.0) @ "a"  # Constant term (std=1.0)
    b = normal(0.0, 1.0) @ "b"  # Linear coefficient (std=1.0)
    c = normal(0.0, 1.0) @ "c"  # Quadratic coefficient (std=1.0)
    return Lambda(Const(polyfn), jnp.array([a, b, c]))


@gen
def onepoint_curve(x):
    curve = polynomial() @ "curve"
    y = point(x, curve) @ "y"
    return curve, (x, y)


@gen
def npoint_curve(xs):
    """N-point curve model with xs as input."""
    curve = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)


@gen
def npoint_curve_easy(xs, noise_std=Const(0.2)):
    """N-point curve model with configurable noise for easier inference."""
    curve = polynomial() @ "curve"
    ys = (
        point_with_noise.vmap(in_axes=(0, None, None))(xs, curve, noise_std.value)
        @ "ys"
    )
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


def infer_latents_easy(
    xs, ys, n_samples: Const[int], noise_std: Const[float] = Const(0.2)
):
    """
    Infer latent curve parameters using easier model with more noise.

    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of importance samples (wrapped in Const)
        noise_std: Observation noise std (default: 0.15, 3x standard)
    """
    from genjax.inference import init

    constraints = {"ys": {"obs": ys}}

    # Use easier model with configurable noise
    result = init(
        npoint_curve_easy,  # easier model
        (xs, noise_std),  # pass noise level
        n_samples,
        constraints,
    )

    return result.traces, result.log_weights


def hmc_infer_latents_vectorized(
    key,
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.05),
    n_steps: Const[int] = Const(10),
    n_chains: Const[int] = Const(4),
):
    """
    Infer latent curve parameters using vectorized GenJAX HMC across multiple chains.

    Args:
        key: JAX random key
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of MCMC samples per chain (wrapped in Const)
        n_warmup: Number of warmup/burn-in samples (wrapped in Const)
        step_size: HMC step size (wrapped in Const)
        n_steps: Number of leapfrog steps (wrapped in Const)
        n_chains: Number of parallel chains (wrapped in Const)

    Returns:
        (samples, diagnostics): HMC samples and diagnostics
    """
    from genjax.inference import hmc, chain
    from genjax.core import sel

    constraints = {"ys": {"obs": ys}}
    
    # Define HMC kernel for continuous parameters
    def hmc_kernel(trace):
        # Select the entire curve (which contains polynomial parameters)
        selection = sel("curve")
        return hmc(trace, selection, step_size=step_size.value, n_steps=n_steps.value)

    # Create MCMC chain
    hmc_chain = chain(hmc_kernel)
    
    # Function to run single chain
    def run_single_chain(chain_key):
        # Split key for initialization and chain
        init_key, chain_key = jrand.split(chain_key)
        
        # Generate initial trace with this chain's key
        initial_trace, _ = seed(npoint_curve.generate)(init_key, constraints, xs)
        
        # Run HMC with burn-in
        total_steps = n_samples.value + n_warmup.value
        result = seed(hmc_chain)(chain_key, initial_trace, n_steps=Const(total_steps), burn_in=n_warmup)
        return result
    
    # Vectorize across chains
    vectorized_chains = jax.vmap(run_single_chain)
    
    # Split keys for chains
    chain_keys = jrand.split(key, n_chains.value)
    results = vectorized_chains(chain_keys)
    
    # Combine results from all chains
    return results.traces, {
        "acceptance_rate": jnp.mean(results.acceptance_rate),
        "n_samples": n_samples.value,
        "n_chains": n_chains.value,
        "n_total_samples": n_samples.value * n_chains.value,
    }


# Keep old single-chain version for compatibility
def hmc_infer_latents(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.05),
    n_steps: Const[int] = Const(10),
):
    """
    Single-chain HMC for backward compatibility.
    This function expects seed() to be applied externally.
    """
    from genjax.inference import hmc, chain
    from genjax.core import sel

    constraints = {"ys": {"obs": ys}}
    # Generate initial trace - seeding applied externally
    initial_trace, _ = npoint_curve.generate(constraints, xs)

    # Define HMC kernel for continuous parameters
    def hmc_kernel(trace):
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
        "n_chains": 1,
    }


def get_points_for_inference(n_points=10):
    """Generate test data for inference with xs as input."""
    # Create grid of input points in [0, 1]
    xs = jnp.linspace(0, 1, n_points)
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


def numpyro_polyfn(x, a, b, c):
    """Degree 2 polynomial function: y = a + b*x + c*x^2"""
    return a + b * x + c * x**2


def numpyro_npoint_model(xs, obs_dict=None):
    """NumPyro model for multiple data points with shared polynomial parameters."""
    # Match GenJAX model with normal distributions
    a = numpyro.sample("a", numpyro_dist.Normal(0.0, 1.0))  # Constant term (std=1.0)
    b = numpyro.sample(
        "b", numpyro_dist.Normal(0.0, 1.0)
    )  # Linear coefficient (std=1.0)
    c = numpyro.sample(
        "c", numpyro_dist.Normal(0.0, 1.0)
    )  # Quadratic coefficient (std=1.0)

    with numpyro.plate("data", len(xs)):
        obs_vals = None
        if obs_dict is not None and "obs" in obs_dict:
            obs_vals = obs_dict["obs"]
        y_det = numpyro_polyfn(xs, a, b, c)
        y_observed = numpyro.sample(
            "obs", numpyro_dist.Normal(y_det, 0.2), obs=obs_vals
        )
    return y_observed


def numpyro_guide_npoint(xs, obs_dict=None):
    """Guide for importance sampling that samples from the prior."""
    numpyro.sample("a", numpyro_dist.Normal(0.0, 1.0))  # Constant term (std=1.0)
    numpyro.sample("b", numpyro_dist.Normal(0.0, 1.0))  # Linear coefficient (std=1.0)
    numpyro.sample(
        "c", numpyro_dist.Normal(0.0, 1.0)
    )  # Quadratic coefficient (std=1.0)


def numpyro_single_importance_sample(key, xs, obs_dict):
    """Single importance sampling step for NumPyro."""
    # Lazy import to avoid funsor dependency for non-NumPyro modes
    from numpyro.contrib.funsor import log_density

    key1, key2 = jrand.split(key)

    seeded_guide = numpyro_seed(numpyro_guide_npoint, key1)
    guide_log_density, guide_trace = log_density(seeded_guide, (xs, None), {}, {})

    seeded_model = numpyro_seed(numpyro_npoint_model, key2)
    replay_model = replay(seeded_model, guide_trace)
    model_log_density, model_trace = log_density(replay_model, (xs, obs_dict), {}, {})

    log_weight = model_log_density - guide_log_density

    sample = {
        "a": model_trace["a"]["value"],
        "b": model_trace["b"]["value"],
        "c": model_trace["c"]["value"],
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


def numpyro_run_hmc_inference(key, xs, ys, num_samples=1000, num_warmup=500, num_chains=4):
    """Run vectorized HMC inference using NumPyro across multiple chains."""
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
        num_chains=num_chains,
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
        "samples": {"a": samples["a"], "b": samples["b"], "c": samples["c"]},
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "num_chains": num_chains,
        "num_total_samples": num_samples * num_chains,
    }


def numpyro_run_hmc_inference_jit_impl(
    key, xs, ys, num_samples=1000, num_warmup=500, step_size=0.01, num_steps=20, num_chains=4
):
    """Run vectorized HMC inference using NumPyro with JIT compilation across multiple chains."""
    obs_dict = {"obs": ys}

    def conditioned_model(xs, obs_dict):
        return numpyro_npoint_model(xs, obs_dict)

    # Use basic HMC with same parameters as GenJAX for fair comparison
    # Note: Only specify num_steps to avoid trajectory_length conflict
    hmc_kernel = HMC(conditioned_model, step_size=step_size, num_steps=num_steps)
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        jit_model_args=True,
        progress_bar=False,
    )  # Enable JIT for this version
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
        "samples": {"a": samples["a"], "b": samples["b"], "c": samples["c"]},
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "num_chains": num_chains,
        "num_total_samples": num_samples * num_chains,
    }


# JIT-compiled version for fair performance comparison
numpyro_run_hmc_inference_jit = jax.jit(
    numpyro_run_hmc_inference_jit_impl, static_argnums=(3, 4, 6, 7)
)  # num_samples, num_warmup, num_steps, num_chains (step_size is not static)


# Outlier models for GenJAX


@gen
def point_with_outliers(x, curve, outlier_rate=0.1, outlier_std=1.0):
    """Point model that can be either inlier or outlier.

    Args:
        x: Input point
        curve: Curve function
        outlier_rate: Probability of being an outlier (default 0.1)
        outlier_std: Standard deviation for outlier distribution (default 1.0)
    """
    y_det = curve(x)

    # Sample whether this point is an outlier
    is_outlier = flip(outlier_rate) @ "is_outlier"

    # Define inlier and outlier branches with SAME address (now supported!)
    @gen
    def inlier_branch():
        # Normal observation with small noise
        return normal(y_det, 0.2) @ "obs"

    @gen
    def outlier_branch():
        # Outlier: wide normal centered at y_det
        return normal(y_det, outlier_std) @ "obs"

    # Use Cond to select between branches
    cond_model = Cond(outlier_branch, inlier_branch)
    y_observed = cond_model(is_outlier) @ "y"

    return y_observed


@gen
def npoint_curve_with_outliers(xs, outlier_rate=Const(0.1), outlier_std=Const(1.0)):
    """N-point curve model with outlier detection.

    Each point can be classified as inlier or outlier.
    """
    curve = polynomial() @ "curve"

    # Vectorize the outlier model
    ys = (
        point_with_outliers.vmap(in_axes=(0, None, None, None))(
            xs, curve, outlier_rate.value, outlier_std.value
        )
        @ "ys"
    )

    return curve, (xs, ys)


def infer_latents_with_outliers(
    xs,
    ys,
    n_samples: Const[int],
    outlier_rate: Const[float] = Const(0.1),
    outlier_std: Const[float] = Const(1.0),
):
    """
    Infer latent curve parameters and outlier indicators using GenJAX SMC.

    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of importance samples (wrapped in Const)
        outlier_rate: Prior probability of outlier (wrapped in Const)
        outlier_std: Std dev for outlier distribution (wrapped in Const)
    """
    from genjax.inference import init

    # With the fixed Cond, we can now properly constrain observations!
    constraints = {"ys": {"y": {"obs": ys}}}

    # Use SMC init for importance sampling
    result = init(
        npoint_curve_with_outliers,  # outlier-aware model
        (xs, outlier_rate, outlier_std),  # model args
        n_samples,
        constraints,
    )

    return result.traces, result.log_weights


def hmc_infer_latents_with_outliers(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.05),
    n_steps: Const[int] = Const(10),
    outlier_rate: Const[float] = Const(0.1),
    outlier_std: Const[float] = Const(1.0),
):
    """
    Infer latent curve parameters using HMC, marginalizing over outlier indicators.

    Note: HMC only updates continuous parameters (curve coefficients).
    The discrete outlier indicators are handled through regeneration.
    """
    from genjax.inference import hmc, mh, chain
    from genjax.core import sel

    constraints = {"ys": {"y": {"obs": ys}}}

    # Generate initial trace
    initial_trace, _ = npoint_curve_with_outliers.generate(
        constraints, xs, outlier_rate, outlier_std
    )

    # Define mixed kernel: HMC for continuous, MH for discrete
    def mixed_kernel(trace):
        # First update curve parameters with HMC
        trace = hmc(
            trace, sel("curve"), step_size=step_size.value, n_steps=n_steps.value
        )

        # Then update outlier indicators with MH
        # Select all outlier indicators
        trace = mh(trace, sel("ys") & sel("is_outlier"))

        return trace

    # Create MCMC chain
    mcmc_chain = chain(mixed_kernel)

    # Run MCMC with burn-in
    total_steps = n_samples.value + n_warmup.value
    result = mcmc_chain(initial_trace, n_steps=Const(total_steps), burn_in=n_warmup)

    return result.traces, {
        "acceptance_rate": result.acceptance_rate,
        "n_samples": result.n_steps.value,
        "n_chains": result.n_chains.value,
    }


# JIT compiled functions for performance benchmarks

# GenJAX JIT-compiled functions - apply seed() before jit()
infer_latents_seeded = seed(infer_latents)
hmc_infer_latents_seeded = seed(hmc_infer_latents)
hmc_infer_latents_vectorized_jit = jax.jit(hmc_infer_latents_vectorized)
infer_latents_with_outliers_seeded = seed(infer_latents_with_outliers)
hmc_infer_latents_with_outliers_seeded = seed(hmc_infer_latents_with_outliers)

infer_latents_jit = jax.jit(
    infer_latents_seeded
)  # Use Const pattern instead of static_argnums
hmc_infer_latents_jit = jax.jit(
    hmc_infer_latents_seeded
)  # Use Const pattern instead of static_argnums
infer_latents_with_outliers_jit = jax.jit(infer_latents_with_outliers_seeded)
hmc_infer_latents_with_outliers_jit = jax.jit(hmc_infer_latents_with_outliers_seeded)

# NumPyro JIT-compiled functions
numpyro_run_importance_sampling_jit = jax.jit(
    numpyro_run_importance_sampling, static_argnums=(3,)
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

    # GenJAX Vectorized HMC
    print("Running GenJAX Vectorized HMC (4 chains)...")

    # Use the JIT-compiled vectorized HMC function
    def genjax_hmc_task():
        return hmc_infer_latents_vectorized_jit(
            jrand.key(seed), xs, ys, 
            Const(n_samples), Const(n_warmup),
            Const(0.05), Const(10), Const(4)  # 4 chains
        )

    genjax_hmc_times, genjax_hmc_stats = benchmark_with_warmup(
        genjax_hmc_task, repeats=timing_repeats
    )

    # Get samples for posterior analysis
    genjax_hmc_samples, genjax_hmc_diagnostics = hmc_infer_latents_vectorized_jit(
        jrand.key(seed), xs, ys, 
        Const(n_samples), Const(n_warmup),
        Const(0.05), Const(10), Const(4)  # 4 chains
    )

    results["genjax_hmc"] = {
        "method": f"HMC (K=4, L={n_samples})",
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

    print("Running NumPyro Vectorized HMC (4 chains)...")

    def numpyro_hmc_task():
        return numpyro_run_hmc_inference_jit(key, xs, ys, n_samples, n_warmup, num_chains=4)

    numpyro_hmc_times, numpyro_hmc_stats = benchmark_with_warmup(
        numpyro_hmc_task, repeats=timing_repeats
    )

    numpyro_hmc_result = numpyro_run_hmc_inference_jit(key, xs, ys, n_samples, n_warmup, num_chains=4)

    results["numpyro_hmc"] = {
        "method": f"HMC (K=4, L={n_samples})",
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

                a_samples = traces.get_choices()["curve"]["a"][indices]
                b_samples = traces.get_choices()["curve"]["b"][indices]
                c_samples = traces.get_choices()["curve"]["c"][indices]

            elif method in ["HMC", "Vectorized HMC"]:
                # Extract MCMC samples directly
                traces = result["samples"]
                # For vectorized HMC, samples are shape (n_chains, n_samples, ...)
                # We'll flatten them to (n_chains * n_samples, ...)
                a_samples = traces.get_choices()["curve"]["a"]
                b_samples = traces.get_choices()["curve"]["b"]
                c_samples = traces.get_choices()["curve"]["c"]
                
                # Flatten if vectorized (shape has more than 1 dimension)
                if a_samples.ndim > 1:
                    a_samples = a_samples.reshape(-1)
                    b_samples = b_samples.reshape(-1)
                    c_samples = c_samples.reshape(-1)

        elif framework == "NumPyro":
            if method == "Importance Sampling":
                # Resample using importance weights
                samples = result["samples"]
                weights = result["weights"]

                n_resample = min(1000, len(weights))
                key = jrand.key(123)
                indices = jrand.categorical(key, weights, shape=(n_resample,))

                a_samples = samples["a"][indices]
                b_samples = samples["b"][indices]
                c_samples = samples["c"][indices]

            elif method in ["HMC", "Vectorized HMC"]:
                # MCMC samples
                samples = result["samples"]
                a_samples = samples["a"]
                b_samples = samples["b"]
                c_samples = samples["c"]
                
                # NumPyro already flattens chains, so shape is (n_chains * n_samples,)

        posterior_samples[method_name] = {
            "framework": framework,
            "method": method,
            "a": a_samples,
            "b": b_samples,
            "c": c_samples,
        }

    return posterior_samples
