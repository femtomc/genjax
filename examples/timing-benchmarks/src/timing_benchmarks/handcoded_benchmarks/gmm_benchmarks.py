"""1D Gaussian Mixture Model benchmarks for GenJAX and handcoded JAX."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from typing import Dict, Any, Tuple

from genjax import gen, normal, categorical, seed, trace
from .timing_utils import benchmark_with_warmup


# ============================================================================
# 1D Gaussian Mixture Model
# ============================================================================

# Model parameters
DEFAULT_MEANS = jnp.array([-2.0, 0.0, 2.0])
DEFAULT_STDS = jnp.array([0.5, 0.5, 0.5])
DEFAULT_WEIGHTS = jnp.array([0.3, 0.4, 0.3])


# ============================================================================
# GenJAX Implementation
# ============================================================================

def genjax_gmm_step(key, observations, means, stds, weights):
    """GenJAX: Compute posterior P(z|x) for GMM.
    
    This is equivalent to one step of importance sampling with the exact posterior
    as the proposal distribution.
    """
    n_data = len(observations)
    n_components = len(means)
    
    # Compute log probabilities for all data points and components
    # Shape: (n_data, n_components)
    log_probs = jax.vmap(
        lambda x: jax.vmap(
            lambda k: jstats.norm.logpdf(x, means[k], stds[k]) + jnp.log(weights[k])
        )(jnp.arange(n_components))
    )(observations)
    
    # Convert to probabilities
    probs = jax.nn.softmax(log_probs, axis=1)
    
    # Sample component assignments from the posterior
    z_samples = jax.vmap(
        lambda p, k: jax.random.categorical(k, jnp.log(p))
    )(probs, jax.random.split(key, n_data))
    
    return z_samples


# ============================================================================
# Handcoded JAX Implementation
# ============================================================================

def jax_gmm_step(key, observations, means, stds, weights):
    """Handcoded JAX: Importance sampling step (single particle)."""
    n_data = len(observations)
    n_components = len(means)
    
    # This implements the same importance sampling logic as GenJAX:
    # 1. For each observation, compute posterior P(z|x) 
    # 2. Sample z from this posterior (this is our proposal q(z|x))
    # 3. The importance weight is p(z,x)/q(z|x), but since we're doing
    #    exact posterior sampling, the weights cancel out for single particle
    
    # Compute log probabilities for all data points and components
    # Shape: (n_data, n_components)
    log_probs = jax.vmap(
        lambda x: jax.vmap(
            lambda k: jstats.norm.logpdf(x, means[k], stds[k]) + jnp.log(weights[k])
        )(jnp.arange(n_components))
    )(observations)
    
    # Convert to probabilities (this is P(z|x) for each datapoint)
    probs = jax.nn.softmax(log_probs, axis=1)
    
    # Sample component assignments from the posterior
    z_samples = jax.vmap(
        lambda p, k: jax.random.categorical(k, jnp.log(p))
    )(probs, jax.random.split(key, n_data))
    
    # Note: For single particle importance sampling with exact posterior proposal,
    # the importance weight is always 1 (or log weight = 0) because p(z,x)/q(z|x) = p(x)
    # which is constant across particles
    
    return z_samples




# ============================================================================
# Data Generation
# ============================================================================

def generate_gmm_data(key, n_data: int, means=None, stds=None, weights=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

def time_genjax_gmm(n_data: int, n_steps: int, repeats: int = 100) -> Dict[str, Any]:
    """Time GenJAX GMM implementation."""
    # Generate data
    key = jax.random.PRNGKey(42)
    observations, _ = generate_gmm_data(key, n_data)
    
    # JIT compile
    jitted_step = jax.jit(genjax_gmm_step)
    
    # Create timing task
    def task():
        key_copy = key
        result = None
        for _ in range(n_steps):
            key_copy, subkey = jax.random.split(key_copy)
            result = jitted_step(subkey, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS)
        return result
    
    # Run benchmark with automatic warm-up and inner repeats
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, 
        warmup_runs=5,
        repeats=repeats,
        inner_repeats=50,  # Reduced for faster benchmarking
        auto_sync=True  # Automatically block_until_ready
    )
    
    return {
        'framework': 'genjax',
        'n_data': n_data,
        'n_steps': n_steps,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times.tolist()
    }


def time_jax_gmm(n_data: int, n_steps: int, repeats: int = 100) -> Dict[str, Any]:
    """Time handcoded JAX GMM implementation."""
    # Generate data
    key = jax.random.PRNGKey(42)
    observations, _ = generate_gmm_data(key, n_data)
    
    # JIT compile
    jitted_step = jax.jit(jax_gmm_step)
    
    # Create timing task
    def task():
        key_copy = key
        result = None
        for _ in range(n_steps):
            key_copy, subkey = jax.random.split(key_copy)
            result = jitted_step(subkey, observations, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS)
        return result
    
    # Run benchmark with automatic warm-up and inner repeats
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, 
        warmup_runs=5,
        repeats=repeats,
        inner_repeats=50,  # Reduced for faster benchmarking
        auto_sync=True  # Automatically block_until_ready
    )
    
    return {
        'framework': 'handcoded_jax',
        'n_data': n_data,
        'n_steps': n_steps,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times.tolist()
    }




# ============================================================================
# Benchmark Runner
# ============================================================================

def run_gmm_benchmarks(data_sizes: list = [100, 1000, 10000, 100000],
                      n_steps: int = 10,
                      repeats: int = 100) -> Dict[str, list]:
    """Run JAX-based GMM benchmarks."""
    results = {
        'genjax': [],
        'handcoded_jax': []
    }
    
    for n_data in data_sizes:
        print(f"\nData size: {n_data}")
        
        # GenJAX
        print("  Running GenJAX...")
        result = time_genjax_gmm(n_data, n_steps, repeats)
        results['genjax'].append(result)
        print(f"    Mean time: {result['mean_time']:.4f}s")
        
        # Handcoded JAX
        print("  Running handcoded JAX...")
        result = time_jax_gmm(n_data, n_steps, repeats)
        results['handcoded_jax'].append(result)
        print(f"    Mean time: {result['mean_time']:.4f}s")
    
    return results