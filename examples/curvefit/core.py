from genjax import gen, normal, Cond, flip, uniform, beta
from genjax.core import Const, sel
from genjax.pjax import seed
import jax.numpy as jnp
import jax.random as jrand

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
import jax

# NumPyro imports (conditional)
try:
    import numpyro
    import numpyro.distributions as numpyro_dist
    from numpyro.handlers import replay, seed as numpyro_seed
    # from numpyro.contrib.funsor import log_density  # Moved to function for lazy loading
    from numpyro.infer import HMC, MCMC
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False
    # Create dummy objects to prevent NameError
    numpyro = None
    numpyro_dist = None
    replay = None
    numpyro_seed = None
    HMC = None
    MCMC = None


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
    y_observed = normal(y_det, 0.05) @ "obs"  # Reduced observation noise
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


def hmc_infer_latents_vectorized(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.05),
    n_steps: Const[int] = Const(10),
    n_chains: Const[int] = Const(4),
):
    """
    Vectorized HMC inference running multiple chains in parallel.
    
    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of MCMC samples per chain (wrapped in Const)
        n_warmup: Number of warmup/burn-in samples (wrapped in Const)
        step_size: HMC step size (wrapped in Const)
        n_steps: Number of leapfrog steps (wrapped in Const)
        n_chains: Number of parallel chains (wrapped in Const)
        
    Returns:
        tuple: (trace with shape (n_chains, n_samples), diagnostics dict)
    """
    # Create a function that runs a single chain with key as argument
    def run_single_chain(key, xs, ys):
        return seed(hmc_infer_latents)(key, xs, ys, n_samples, n_warmup, step_size, n_steps)
    
    # Generate keys for each chain
    keys = jrand.split(jrand.key(0), n_chains.value)
    
    # Run chains in parallel using vmap
    vectorized_hmc = jax.vmap(run_single_chain, in_axes=(0, None, None))
    traces, diagnostics = vectorized_hmc(keys, xs, ys)
    
    # Update diagnostics to reflect multiple chains
    diagnostics["n_chains"] = n_chains.value
    
    return traces, diagnostics


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
            "obs", numpyro_dist.Normal(y_det, 0.05), obs=obs_vals  # Reduced observation noise
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
        "samples": {"a": samples["a"], "b": samples["b"], "c": samples["c"]},
        "diagnostics": diagnostics,
        "num_samples": num_samples,
        "num_warmup": num_warmup,
    }


def numpyro_run_hmc_inference_jit_impl(
    key, xs, ys, num_samples=1000, num_warmup=500, step_size=0.01, num_steps=20
):
    """Run HMC inference using NumPyro with JIT compilation (same parameters as GenJAX)."""
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
    }


# JIT-compiled version for fair performance comparison
numpyro_run_hmc_inference_jit = jax.jit(
    numpyro_run_hmc_inference_jit_impl, static_argnums=(3, 4, 6)
)  # num_samples, num_warmup, num_steps (step_size is not static)


def numpyro_hmc_summary_statistics(hmc_result):
    """Extract summary statistics from NumPyro HMC results."""
    diagnostics = hmc_result.get("diagnostics", {})
    accept_probs = diagnostics.get("accept_probs", jnp.array([]))
    
    # Calculate acceptance rate - handle empty arrays
    if accept_probs.size > 0:
        accept_rate = jnp.mean(accept_probs)
    else:
        accept_rate = 0.0
    
    return {
        "accept_rate": float(accept_rate),  # Convert to Python float to avoid JAX tracer issues
        "num_samples": hmc_result.get("num_samples", 0),
        "num_warmup": hmc_result.get("num_warmup", 0),
    }


# Outlier models for GenJAX


# Define branch functions outside to avoid JAX local function comparison issues
@gen
def inlier_branch(mean, std):
    """Inlier branch: follows the polynomial regression with small noise."""
    # For inliers, we use the mean (y_det) but ignore std, using fixed 0.05
    return normal(mean, 0.05) @ "obs"  # Same address in both branches


@gen
def outlier_branch(mean, std):
    """Outlier branch: samples from a uniform distribution [-4, 4]."""
    # For outliers, we ignore both mean and std and use uniform[-4, 4]
    return uniform(-4.0, 4.0) @ "obs"  # Same address in both branches


@gen
def point_with_outliers(x, curve, outlier_rate=0.1, outlier_mean=0.0, outlier_std=5.0):
    """Point model that can be either inlier or outlier.

    This model uses the Cond combinator to naturally express a mixture model where:
    - Inliers: follow the polynomial regression with small observation noise
    - Outliers: come from a uniform distribution on [-4, 4] independent of the curve
    
    This models realistic scenarios where outliers are measurement errors,
    sensor failures, or corrupted data unrelated to the true underlying curve.

    Args:
        x: Input point
        curve: Curve function
        outlier_rate: Probability of being an outlier (default 0.1)
        outlier_mean: (ignored - kept for API compatibility)
        outlier_std: (ignored - kept for API compatibility)
    """
    y_det = curve(x)

    # Sample whether this point is an outlier
    is_outlier = flip(outlier_rate) @ "is_outlier"

    # Use Cond combinator with proper branches
    cond_model = Cond(outlier_branch, inlier_branch)
    
    # Call Cond with appropriate arguments for each branch
    # When is_outlier=True: outlier_branch() - ignores parameters, uses uniform[-4,4]
    # When is_outlier=False: inlier_branch(y_det) - uses polynomial value
    # Both branches need same signature, so we pass dummy values to outlier branch
    y_observed = cond_model(is_outlier, jnp.where(is_outlier, 0.0, y_det),
                           outlier_std) @ "y"
    
    return y_observed


@gen
def npoint_curve_with_outliers(xs, outlier_rate=Const(0.1), outlier_mean=Const(0.0), 
                               outlier_std=Const(5.0)):
    """N-point curve model with outlier detection.

    Each point can be classified as inlier or outlier.
    Inliers follow the polynomial curve, outliers are from an independent Gaussian.
    """
    curve = polynomial() @ "curve"

    # Vectorize the outlier model
    ys = (
        point_with_outliers.vmap(in_axes=(0, None, None, None, None))(
            xs, curve, outlier_rate.value, outlier_mean.value, outlier_std.value
        )
        @ "ys"
    )

    return curve, (xs, ys)


@gen
def npoint_curve_with_outliers_beta(xs, alpha=Const(1.0), beta_param=Const(10.0)):
    """N-point curve model with outlier detection using beta prior.

    Uses a beta prior for the outlier probability, allowing inference
    over the outlier rate itself.

    Args:
        xs: Input points
        alpha: Beta distribution alpha parameter (default 1.0)
        beta_param: Beta distribution beta parameter (default 10.0)
    """
    # Sample outlier rate from beta prior
    # Beta(1, 10) has mean 1/11 ≈ 0.09, skewed towards low outlier rates
    outlier_rate = beta(alpha.value, beta_param.value) @ "outlier_rate"
    
    # Sample polynomial coefficients
    curve = polynomial() @ "curve"

    # Vectorize the outlier model with sampled outlier rate
    ys = (
        point_with_outliers.vmap(in_axes=(0, None, None, None, None))(
            xs, curve, outlier_rate, 0.0, 5.0  # outlier_mean and std are ignored
        )
        @ "ys"
    )

    return curve, (xs, ys)


def infer_latents_with_outliers(
    xs,
    ys,
    n_samples: Const[int],
    outlier_rate: Const[float] = Const(0.1),
    outlier_mean: Const[float] = Const(0.0),
    outlier_std: Const[float] = Const(5.0),
):
    """
    Infer latent curve parameters and outlier indicators using GenJAX SMC.

    Args:
        xs: Input points where observations were made
        ys: Observed values at xs
        n_samples: Number of importance samples (wrapped in Const)
        outlier_rate: Prior probability of outlier (wrapped in Const)
        outlier_mean: Mean of outlier distribution (wrapped in Const)
        outlier_std: Std dev for outlier distribution (wrapped in Const)
    """
    from genjax.inference import init

    # Constraint on observations - address the Cond output properly
    constraints = {"ys": {"y": {"obs": ys}}}

    # Use SMC init for importance sampling
    result = init(
        npoint_curve_with_outliers,  # outlier-aware model
        (xs, outlier_rate, outlier_mean, outlier_std),  # model args
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
    outlier_mean: Const[float] = Const(0.0),
    outlier_std: Const[float] = Const(5.0),
):
    """
    Infer latent curve parameters using HMC on continuous parameters only.

    For this simplified version, we only update the curve parameters with HMC,
    keeping outlier indicators fixed at their initial values.
    """
    from genjax.inference import hmc, chain
    from genjax.core import sel

    # Constraint on observations - address the Cond output properly
    constraints = {"ys": {"y": {"obs": ys}}}

    # Generate initial trace
    initial_trace, _ = npoint_curve_with_outliers.generate(
        constraints, xs, outlier_rate, outlier_mean, outlier_std
    )

    # Define HMC kernel for continuous parameters only
    def hmc_kernel(trace):
        # Only update curve parameters with HMC
        return hmc(
            trace, sel("curve"), step_size=step_size.value, n_steps=n_steps.value
        )

    # Create MCMC chain
    mcmc_chain = chain(hmc_kernel)

    # Run MCMC with burn-in
    total_steps = n_samples.value + n_warmup.value
    result = mcmc_chain(initial_trace, n_steps=Const(total_steps), burn_in=n_warmup)

    return result.traces, {
        "acceptance_rate": result.acceptance_rate,
        "n_samples": result.n_steps.value,
        "n_chains": result.n_chains.value,
    }


def mixed_infer_latents_with_outliers_beta(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    mh_moves_per_step: Const[int] = Const(5),
    hmc_step_size: Const[float] = Const(0.01),
    hmc_n_steps: Const[int] = Const(10),
    alpha: Const[float] = Const(1.0),
    beta_param: Const[float] = Const(10.0),
):
    """
    Infer latent parameters using mixed MCMC:
    - MH for discrete outlier indicators
    - HMC for continuous parameters (curve coefficients and outlier rate)

    Args:
        xs: Input points
        ys: Observed values
        n_samples: Number of MCMC samples
        n_warmup: Number of warmup samples
        mh_moves_per_step: Number of MH moves for outlier indicators per MCMC step
        hmc_step_size: Step size for HMC
        hmc_n_steps: Number of leapfrog steps for HMC
        alpha: Beta distribution alpha parameter
        beta_param: Beta distribution beta parameter
    """
    from genjax.inference import hmc, mh, chain

    # Constraint on observations
    constraints = {"ys": {"y": {"obs": ys}}}

    # Generate initial trace with beta model
    initial_trace, _ = npoint_curve_with_outliers_beta.generate(
        constraints, xs, alpha, beta_param
    )

    # Define mixed kernel
    def mixed_kernel(trace):
        # First, MH moves on outlier indicators
        for i in range(mh_moves_per_step.value):
            trace = mh(trace, sel({"ys": sel("is_outlier")}))
        
        # Then, HMC on continuous parameters (curve and outlier_rate)
        trace = hmc(
            trace, 
            sel("curve") | sel("outlier_rate"), 
            step_size=hmc_step_size.value, 
            n_steps=hmc_n_steps.value
        )
        
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
        "rhat": result.rhat,
        "ess_bulk": result.ess_bulk,
        "ess_tail": result.ess_tail,
    }


# JIT compiled functions for performance benchmarks

# GenJAX JIT-compiled functions - apply seed() before jit()
infer_latents_seeded = seed(infer_latents)
hmc_infer_latents_seeded = seed(hmc_infer_latents)
hmc_infer_latents_vectorized_seeded = seed(hmc_infer_latents_vectorized)
infer_latents_with_outliers_seeded = seed(infer_latents_with_outliers)
hmc_infer_latents_with_outliers_seeded = seed(hmc_infer_latents_with_outliers)
mixed_infer_latents_with_outliers_beta_seeded = seed(mixed_infer_latents_with_outliers_beta)

infer_latents_jit = jax.jit(
    infer_latents_seeded
)  # Use Const pattern instead of static_argnums
hmc_infer_latents_jit = jax.jit(
    hmc_infer_latents_seeded
)  # Use Const pattern instead of static_argnums
hmc_infer_latents_vectorized_jit = jax.jit(
    hmc_infer_latents_vectorized_seeded
)  # Use Const pattern instead of static_argnums
infer_latents_with_outliers_jit = jax.jit(infer_latents_with_outliers_seeded)
hmc_infer_latents_with_outliers_jit = jax.jit(hmc_infer_latents_with_outliers_seeded)
mixed_infer_latents_with_outliers_beta_jit = jax.jit(mixed_infer_latents_with_outliers_beta_seeded)

# NumPyro JIT-compiled functions
numpyro_run_importance_sampling_jit = jax.jit(
    numpyro_run_importance_sampling, static_argnums=(3,)
)


#
def run_comprehensive_benchmark(
    n_points=20,
    n_samples=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=20,
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

                a_samples = traces.get_choices()["curve"]["a"][indices]
                b_samples = traces.get_choices()["curve"]["b"][indices]
                c_samples = traces.get_choices()["curve"]["c"][indices]

            elif method == "HMC":
                # Extract MCMC samples directly
                traces = result["samples"]
                a_samples = traces.get_choices()["curve"]["a"]
                b_samples = traces.get_choices()["curve"]["b"]
                c_samples = traces.get_choices()["curve"]["c"]

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

            elif method == "HMC":
                # MCMC samples
                samples = result["samples"]
                a_samples = samples["a"]
                b_samples = samples["b"]
                c_samples = samples["c"]

        posterior_samples[method_name] = {
            "framework": framework,
            "method": method,
            "a": a_samples,
            "b": b_samples,
            "c": c_samples,
        }

    return posterior_samples
