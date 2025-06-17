"""Core model definitions and timing utilities for fair coin case study."""

import jax
import jax.numpy as jnp
import jax.random as jrand
import time
from genjax import seed
from genjax import modular_vmap as vmap
from genjax import beta, flip, gen


def timing(fn, repeats=200, inner_repeats=200, number=100):
    """Benchmark function execution time with multiple runs.

    Args:
        fn: Function to benchmark
        repeats: Number of outer timing runs
        inner_repeats: Number of inner timing runs per outer run
        number: Legacy parameter for compatibility

    Returns:
        Tuple of (times_array, (mean_time, std_time))
    """
    times = []
    for i in range(repeats):
        possible = []
        for i in range(inner_repeats):
            start_time = time.perf_counter()
            fn()
            interval = time.perf_counter() - start_time
            possible.append(interval)
        times.append(jnp.array(possible).min())
    times = jnp.array(times)
    return times, (jnp.mean(times), jnp.std(times))


@gen
def beta_ber():
    """Beta-Bernoulli model for fair coin inference.

    Models coin fairness with Beta(10, 10) prior and Bernoulli likelihood.
    """
    # define the hyperparameters that control the Beta prior
    alpha0 = jnp.array(10.0)
    beta0 = jnp.array(10.0)
    # sample f from the Beta prior
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip(f) @ "obs"


def genjax_timing(
    num_obs=50,
    repeats=50,
    num_samples=1000,
):
    """Time GenJAX importance sampling implementation."""
    data = {"obs": jnp.ones(num_obs)}

    def importance_(data):
        _, w = beta_ber.generate(data)
        return w

    imp_jit = jax.jit(
        seed(
            vmap(
                importance_,
                axis_size=num_samples,
                in_axes=None,
            )
        ),
    )
    key = jrand.key(1)
    _ = imp_jit(key, data)
    _ = imp_jit(key, data)
    times, (time_mu, time_std) = timing(
        lambda: imp_jit(key, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def numpyro_timing(
    num_obs=50,
    repeats=200,
    num_samples=1000,
):
    """Time NumPyro importance sampling implementation."""
    import numpyro
    import numpyro.distributions as dist
    from numpyro.handlers import block, replay, seed
    from numpyro.infer.util import (
        log_density,
    )

    key = jax.random.PRNGKey(314159)
    data = jnp.ones(num_obs)

    def model(data):
        # define the hyperparameters that control the Beta prior
        alpha0 = 10.0
        beta0 = 10.0
        # sample f from the Beta prior
        f = numpyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        # loop over the observed data
        with numpyro.plate("data", size=len(data)):
            # observe datapoint i using the Bernoulli
            # likelihood Bernoulli(f)
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    guide = block(model, hide=["obs"])

    def importance(model, guide):
        def fn(key, *args, **kwargs):
            key, sub_key = jax.random.split(key)
            seeded_guide = seed(guide, sub_key)
            guide_log_density, guide_trace = log_density(
                seeded_guide,
                args,
                kwargs,
                {},
            )
            seeded_model = seed(model, key)
            replay_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                replay_model,
                args,
                kwargs,
                {},
            )
            return model_log_density - guide_log_density

        return fn

    vectorized_importance_weights = jax.jit(
        jax.vmap(importance(model, guide), in_axes=(0, None)),
    )

    sub_keys = jax.random.split(key, num_samples)

    # Run to warm up the JIT.
    _ = vectorized_importance_weights(sub_keys, data)
    _ = vectorized_importance_weights(sub_keys, data)

    times, (time_mu, time_std) = timing(
        lambda: vectorized_importance_weights(sub_keys, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def handcoded_timing(
    num_obs=50,
    repeats=50,
    num_samples=1000,
):
    """Time handcoded JAX importance sampling implementation."""
    data = jnp.ones(num_obs)

    def importance_(data):
        alpha0 = 10.0
        beta0 = 10.0
        f = beta.sample(alpha0, beta0)
        w = flip.logpdf(data, f)
        return jnp.sum(w)

    imp_jit = jax.jit(
        seed(
            vmap(
                importance_,
                axis_size=num_samples,
                in_axes=None,
            )
        ),
    )

    key = jrand.key(1)
    _ = imp_jit(key, data)
    _ = imp_jit(key, data)
    times, (time_mu, time_std) = timing(
        lambda: imp_jit(key, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def pyro_timing(
    num_obs=50,
    repeats=200,
    num_samples=1000,
):
    """Time Pyro importance sampling implementation."""
    import torch
    import pyro
    import pyro.distributions as dist

    # Set PyTorch to use CPU for fair comparison with JAX CPU
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    # Convert JAX data to PyTorch tensor
    data = torch.ones(num_obs, dtype=torch.float32)

    def model(data):
        # Define hyperparameters for Beta prior
        alpha0 = torch.tensor(10.0, dtype=torch.float32)
        beta0 = torch.tensor(10.0, dtype=torch.float32)

        # Sample fairness parameter from Beta prior
        f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))

        # Observe data using Bernoulli likelihood
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Bernoulli(f), obs=data)

    def guide(data):
        # Prior as guide (importance sampling)
        alpha0 = torch.tensor(10.0, dtype=torch.float32)
        beta0 = torch.tensor(10.0, dtype=torch.float32)
        pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))

    def single_importance_sample():
        """Single importance sampling step."""
        pyro.clear_param_store()

        # Sample from guide and compute weight
        guide_trace = pyro.poutine.trace(guide).get_trace(data)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)
        ).get_trace(data)

        # Compute importance weight (log space)
        weight = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
        return weight

    # Vectorized importance sampling
    def run_inference():
        weights = []
        for _ in range(num_samples):
            weight = single_importance_sample()
            weights.append(weight)
        return torch.tensor(weights)

    # Warm up
    try:
        _ = run_inference()
        _ = run_inference()
    except Exception as e:
        print(f"Pyro warmup failed: {e}")
        raise

    # Time the inference
    times, (time_mu, time_std) = timing(
        lambda: run_inference(),
        repeats=repeats,
        inner_repeats=5,  # Fewer inner repeats for Pyro as it's typically slower
    )

    return times, (time_mu, time_std)
