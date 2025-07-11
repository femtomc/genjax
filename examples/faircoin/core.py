"""Core model definitions and timing utilities for fair coin case study."""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import seed
from genjax import modular_vmap as vmap
from genjax import beta, flip, gen, Const, const
from genjax.timing import timing


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
    inner_repeats=20,
):
    """Time GenJAX importance sampling implementation."""
    data = {"obs": jnp.ones(num_obs)}  # Multiple observations to match other frameworks

    def importance_(data):
        _, w = beta_ber_multi.generate(data, const(num_obs))
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
        inner_repeats=inner_repeats,
        auto_sync=False,  # Using manual block_until_ready()
    )
    return times, (time_mu, time_std)


def numpyro_timing(
    num_obs=50,
    repeats=200,
    num_samples=1000,
    inner_repeats=20,
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
        inner_repeats=inner_repeats,
        auto_sync=False,  # Using manual block_until_ready()
    )
    return times, (time_mu, time_std)


def handcoded_timing(
    num_obs=50,
    repeats=50,
    num_samples=1000,
    inner_repeats=20,
):
    """Time handcoded JAX importance sampling implementation."""
    data = jnp.ones(num_obs)  # Multiple observations to match other frameworks

    def importance_(data):
        alpha0 = 10.0
        beta0 = 10.0
        f = beta.sample(alpha0, beta0)
        w = flip.logpdf(data, f)  # Returns scalar regardless of observation shape
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
        inner_repeats=inner_repeats,
        auto_sync=False,  # Using manual block_until_ready()
    )
    return times, (time_mu, time_std)


# Posterior sampling functions for comparison visualization


@gen
def beta_ber_multi(num_obs: Const[int]):
    """Beta-Bernoulli model for multiple coin flips.

    Models coin fairness with Beta(10, 10) prior and multiple Bernoulli observations.
    """
    # define the hyperparameters that control the Beta prior
    alpha0 = jnp.array(10.0)
    beta0 = jnp.array(10.0)
    # sample f from the Beta prior
    f = beta(alpha0, beta0) @ "latent_fairness"
    # observe multiple coin flips
    return flip.repeat(n=num_obs.value)(f) @ "obs"


def genjax_posterior_samples(num_obs=50, num_samples=1000):
    """Generate posterior samples using GenJAX importance sampling.

    Args:
        num_obs: Number of observations
        num_samples: Number of importance samples

    Returns:
        Tuple of (samples, weights) where samples are fairness parameter values
    """
    # Generate 80% heads (1s) and 20% tails (0s)
    num_heads = int(0.8 * num_obs)
    num_tails = num_obs - num_heads
    data = {"obs": jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])}
    key = jrand.key(123)  # Different seed for GenJAX

    def importance_sample(data):
        trace, weight = beta_ber_multi.generate(data, const(num_obs))
        fairness = trace.get_choices()["latent_fairness"]
        return fairness, weight

    # Vectorized importance sampling with proper seed handling
    seeded_importance = seed(importance_sample)

    keys = jrand.split(key, num_samples)

    samples, log_weights = jax.vmap(seeded_importance, in_axes=(0, None))(
        keys,
        data,
    )

    # Convert log weights to normalized weights
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    weights = weights / jnp.sum(weights)

    return samples, weights


def handcoded_posterior_samples(num_obs=50, num_samples=1000):
    """Generate posterior samples using handcoded JAX importance sampling.

    Args:
        num_obs: Number of observations
        num_samples: Number of importance samples

    Returns:
        Tuple of (samples, weights) where samples are fairness parameter values
    """
    # Generate 80% heads (1s) and 20% tails (0s)
    num_heads = int(0.8 * num_obs)
    num_tails = num_obs - num_heads
    data = jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])
    key = jrand.key(456)  # Different seed for handcoded JAX

    def importance_sample(data):
        alpha0 = 10.0
        beta0 = 10.0
        # Sample from prior (which is also our proposal)
        f = beta.sample(alpha0, beta0)
        # Compute likelihood of data given f
        log_weight = jnp.sum(flip.logpdf(data, f))
        return f, log_weight

    # Vectorized importance sampling with proper seed handling
    seeded_importance = seed(importance_sample)

    keys = jrand.split(key, num_samples)

    samples, log_weights = jax.vmap(seeded_importance, in_axes=(0, None))(
        keys,
        data,
    )

    # Convert log weights to normalized weights
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    weights = weights / jnp.sum(weights)

    return samples, weights


def numpyro_posterior_samples(num_obs=50, num_samples=1000):
    """Generate posterior samples using NumPyro importance sampling.

    Args:
        num_obs: Number of observations
        num_samples: Number of importance samples

    Returns:
        Tuple of (samples, weights) where samples are fairness parameter values
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.handlers import block, replay, seed
    from numpyro.infer.util import log_density

    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    # Generate 80% heads (1s) and 20% tails (0s)
    num_heads = int(0.8 * num_obs)
    num_tails = num_obs - num_heads
    data = jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])

    def model(data):
        alpha0 = 10.0
        beta0 = 10.0
        f = numpyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        with numpyro.plate("data", size=len(data)):
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    guide = block(model, hide=["obs"])

    def importance_sample(key, data):
        key, sub_key = jax.random.split(key)
        seeded_guide = seed(guide, sub_key)
        guide_log_density, guide_trace = log_density(seeded_guide, (data,), {}, {})

        seeded_model = seed(model, key)
        replay_model = replay(seeded_model, guide_trace)
        model_log_density, model_trace = log_density(replay_model, (data,), {}, {})

        weight = model_log_density - guide_log_density
        fairness = guide_trace["latent_fairness"]["value"]
        return fairness, weight

    # Vectorized importance sampling
    vmap_importance = jax.vmap(importance_sample, in_axes=(0, None))

    keys = jax.random.split(key, num_samples)
    samples, log_weights = vmap_importance(keys, data)

    # Convert log weights to normalized weights
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    weights = weights / jnp.sum(weights)

    return samples, weights


def exact_beta_posterior_stats(num_obs=50):
    """Compute exact Beta posterior statistics.

    Args:
        num_obs: Number of observations (80% heads, 20% tails)

    Returns:
        Tuple of (alpha_post, beta_post, mean, mode, std)
    """
    # Prior parameters
    alpha_prior = 10.0
    beta_prior = 10.0

    # Observed data: 80% heads (1s), 20% tails (0s)
    num_heads = int(0.8 * num_obs)
    num_tails = num_obs - num_heads

    # Posterior parameters for Beta distribution
    alpha_post = alpha_prior + num_heads
    beta_post = beta_prior + num_tails

    # Posterior statistics
    mean = alpha_post / (alpha_post + beta_post)
    mode = (
        (alpha_post - 1) / (alpha_post + beta_post - 2)
        if alpha_post > 1 and beta_post > 1
        else mean
    )
    variance = (alpha_post * beta_post) / (
        (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
    )
    std = jnp.sqrt(variance)

    return alpha_post, beta_post, mean, mode, std
