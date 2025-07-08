"""GMM importance sampling benchmarks for GenJAX, JAX, and PyTorch."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from typing import Dict, Any, Tuple

from genjax import gen, normal, categorical, seed, trace, ChoiceMap
from .timing_utils import benchmark_with_warmup


# Model parameters
DEFAULT_MEANS = jnp.array([-2.0, 0.0, 2.0])
DEFAULT_STDS = jnp.array([0.5, 0.5, 0.5])
DEFAULT_WEIGHTS = jnp.array([0.3, 0.4, 0.3])


# ============================================================================
# GenJAX Implementation
# ============================================================================

@gen
def gmm_model(n_data, means, stds, weights):
    """Full GMM generative model."""
    z_samples = []
    x_samples = []
    
    for i in range(n_data):
        # Sample component
        z = categorical(jnp.log(weights)) @ f"z_{i}"
        z_samples.append(z)
        
        # Sample observation
        x = normal(means[z], stds[z]) @ f"x_{i}"
        x_samples.append(x)
    
    return (z_samples, x_samples)


@gen
def gmm_proposal(observations, means, stds, weights):
    """Proposal that samples z given observed x."""
    z_samples = []
    
    for i in range(len(observations)):
        x = observations[i]
        
        # Compute posterior over z given x
        log_likelihoods = jax.vmap(
            lambda m, s: jstats.norm.logpdf(x, m, s)
        )(means, stds)
        
        log_probs = log_likelihoods + jnp.log(weights)
        probs = jax.nn.softmax(log_probs)
        
        # Sample from posterior
        z = categorical(jnp.log(probs)) @ f"z_{i}"
        z_samples.append(z)
    
    return z_samples


# Compile once
gmm_model_seeded = seed(gmm_model)
gmm_proposal_seeded = seed(gmm_proposal)


def genjax_gmm_importance_sampling(key, observations, means, stds, weights, n_particles):
    """Run importance sampling with GenJAX."""
    n_data = len(observations)
    
    # Create constraints
    constraints = ChoiceMap()
    for i in range(n_data):
        constraints[f"x_{i}"] = observations[i]
    
    # Run importance sampling
    keys = jax.random.split(key, n_particles)
    
    def single_particle(key):
        trace, weight = gmm_model_seeded.importance(
            key,
            (n_data, means, stds, weights),
            constraints,
            gmm_proposal_seeded,
            (observations, means, stds, weights)
        )
        # Extract z samples
        z_samples = jnp.array([trace[f"z_{i}"] for i in range(n_data)])
        return z_samples, weight
    
    # Run for all particles
    z_particles, weights = jax.vmap(single_particle)(keys)
    
    # Return particles and normalized weights
    log_weights = weights
    normalized_weights = jax.nn.softmax(log_weights)
    
    return z_particles, normalized_weights


# ============================================================================
# Handcoded JAX Implementation
# ============================================================================

def jax_gmm_importance_sampling(key, observations, means, stds, weights, n_particles):
    """Handcoded JAX importance sampling."""
    n_data = len(observations)
    n_components = len(means)
    
    # Split keys
    keys = jax.random.split(key, n_particles)
    
    def single_particle(key):
        # For each observation, compute posterior and sample
        z_samples = []
        log_weight = 0.0
        
        keys_per_obs = jax.random.split(key, n_data)
        
        for i in range(n_data):
            x = observations[i]
            
            # Compute proposal distribution q(z|x)
            log_likelihoods = jnp.array([
                jstats.norm.logpdf(x, means[k], stds[k])
                for k in range(n_components)
            ])
            log_q = log_likelihoods + jnp.log(weights)
            q_probs = jax.nn.softmax(log_q)
            
            # Sample from proposal
            z = jax.random.categorical(keys_per_obs[i], jnp.log(q_probs))
            z_samples.append(z)
            
            # Compute importance weight contribution
            # log p(z,x) - log q(z|x)
            log_p_z = jnp.log(weights[z])
            log_p_x_given_z = jstats.norm.logpdf(x, means[z], stds[z])
            log_p = log_p_z + log_p_x_given_z
            log_q_z = jnp.log(q_probs[z])
            
            log_weight += log_p - log_q_z
        
        return jnp.array(z_samples), log_weight
    
    # Run for all particles
    z_particles, log_weights = jax.vmap(single_particle)(keys)
    
    # Normalize weights
    normalized_weights = jax.nn.softmax(log_weights)
    
    return z_particles, normalized_weights


# ============================================================================
# Data Generation
# ============================================================================

def generate_gmm_data(key, n_data: int, means=None, stds=None, weights=None):
    """Generate synthetic data from a GMM."""
    if means is None:
        means = DEFAULT_MEANS
    if stds is None:
        stds = DEFAULT_STDS
    if weights is None:
        weights = DEFAULT_WEIGHTS
        
    # Sample component assignments
    key, subkey = jax.random.split(key)
    z_true = jax.random.categorical(subkey, jnp.log(weights), shape=(n_data,))
    
    # Sample observations from components
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(n_data,))
    observations = means[z_true] + stds[z_true] * noise
    
    return observations, z_true


# ============================================================================
# Timing Functions
# ============================================================================

def time_genjax_gmm_is(n_data: int, n_particles: int, repeats: int = 100) -> Dict[str, Any]:
    """Time GenJAX GMM importance sampling."""
    # Generate data
    key = jax.random.PRNGKey(42)
    observations, z_true = generate_gmm_data(key, n_data)
    
    # JIT compile
    jitted_is = jax.jit(genjax_gmm_importance_sampling, static_argnums=(5,))
    
    # Warmup
    for _ in range(5):
        _ = jitted_is(key, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS, n_particles)
    
    # Create timing task
    def task():
        key_copy = key
        key_copy, subkey = jax.random.split(key_copy)
        result = jitted_is(subkey, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS, n_particles)
        return result
    
    # Run benchmark
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, 
        warmup_runs=5,
        repeats=repeats,
        inner_repeats=10,  # Fewer inner repeats for IS
        auto_sync=True
    )
    
    return {
        'framework': 'genjax',
        'algorithm': 'importance_sampling',
        'n_data': n_data,
        'n_particles': n_particles,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times.tolist()
    }


def time_jax_gmm_is(n_data: int, n_particles: int, repeats: int = 100) -> Dict[str, Any]:
    """Time handcoded JAX GMM importance sampling."""
    # Generate data
    key = jax.random.PRNGKey(42)
    observations, z_true = generate_gmm_data(key, n_data)
    
    # JIT compile
    jitted_is = jax.jit(jax_gmm_importance_sampling, static_argnums=(5,))
    
    # Warmup
    for _ in range(5):
        _ = jitted_is(key, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS, n_particles)
    
    # Create timing task
    def task():
        key_copy = key
        key_copy, subkey = jax.random.split(key_copy)
        result = jitted_is(subkey, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS, n_particles)
        return result
    
    # Run benchmark
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, 
        warmup_runs=5,
        repeats=repeats,
        inner_repeats=10,
        auto_sync=True
    )
    
    return {
        'framework': 'handcoded_jax',
        'algorithm': 'importance_sampling',
        'n_data': n_data,
        'n_particles': n_particles,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times.tolist()
    }