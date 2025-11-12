"""Handcoded JAX benchmark (using TFP distributions) for polynomial regression.

This module contains direct JAX implementations using TensorFlow Probability distributions,
serving as a baseline for performance comparison with proper probability distribution APIs.
"""

import time
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jrand
import tensorflow_probability.substrates.jax as tfp

from genjax.timing import benchmark_with_warmup
from ..data.generation import PolynomialDataset, polyfn

# TFP distributions
tfd = tfp.distributions


def handcoded_jax_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 50,
    key: Optional[jrand.PRNGKey] = None,
    inner_repeats: int = 10,
) -> Dict[str, Any]:
    """Handcoded JAX importance sampling with TFP distributions on polynomial regression.
    
    This uses TensorFlow Probability's JAX backend for proper probability distributions.
    
    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        key: Random key (optional)
        
    Returns:
        Dictionary with timing results and samples
    """
    if key is None:
        key = jrand.key(42)
    
    xs, ys = dataset.xs, dataset.ys
    
    def importance_sampling(key, xs, ys, n_particles):
        """Vectorized pure-JAX importance sampler."""
        key_a, key_b, key_c = jrand.split(key, 3)
        a_samples = jrand.normal(key_a, (n_particles,))
        b_samples = jrand.normal(key_b, (n_particles,))
        c_samples = jrand.normal(key_c, (n_particles,))

        xs_batched = xs[None, :]
        y_pred = (
            a_samples[:, None]
            + b_samples[:, None] * xs_batched
            + c_samples[:, None] * xs_batched**2
        )
        log_lik = jax.scipy.stats.norm.logpdf(ys, y_pred, 0.05).sum(axis=1)
        return {
            "a": a_samples,
            "b": b_samples,
            "c": c_samples,
            "log_weights": log_lik,
        }
    
    # JIT compile the inference function with static n_particles
    jitted_is = jax.jit(importance_sampling, static_argnums=(3,))
    
    # Define task for benchmarking  
    def task():
        result = jitted_is(key, xs, ys, n_particles)
        # Block only on log weights for fair comparison
        jax.block_until_ready(result['log_weights'])
        return result
    
    # Run benchmark with automatic warm-up - more inner repeats for accuracy
    times, (mean_time, std_time) = benchmark_with_warmup(
        task,
        warmup_runs=5,
        repeats=repeats,
        inner_repeats=inner_repeats,
        auto_sync=False,
    )
    
    # Get samples for validation
    samples = task()
    
    return {
        "framework": "handcoded_jax",
        "method": "importance_sampling",
        "n_particles": n_particles,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "samples": {
            "a": samples['a'],
            "b": samples['b'],
            "c": samples['c'],
        },
        "log_weights": samples['log_weights'],
    }




def handcoded_jax_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int = 1000,
    n_warmup: int = 50,
    repeats: int = 100,
    key: Optional[jax.Array] = None,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    inner_repeats: int = 10,
) -> Dict[str, Any]:
    """Handcoded HMC timing for polynomial regression - direct JAX implementation."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    xs, ys = dataset.xs, dataset.ys
    n_points = len(xs)
    
    # Log joint density
    def log_joint(params):
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs**2
        
        # Likelihood: Normal(y | y_pred, 0.05)
        log_lik = jax.scipy.stats.norm.logpdf(ys, y_pred, 0.05).sum()
        
        # Priors: Normal(0, 1) for all parameters
        log_prior = jax.scipy.stats.norm.logpdf(params, 0., 1.).sum()
        
        return log_lik + log_prior
    
    # HMC implementation
    def leapfrog(q, p, step_size, n_leapfrog):
        """Leapfrog integrator for HMC (legacy timing-benchmarks version)."""
        grad = jax.grad(log_joint)(q)
        p = p + 0.5 * step_size * grad
        for _ in range(n_leapfrog - 1):
            q = q + step_size * p
            grad = jax.grad(log_joint)(q)
            p = p + step_size * grad
        q = q + step_size * p
        grad = jax.grad(log_joint)(q)
        p = p + 0.5 * step_size * grad
        return q, p
    
    def hmc_step(state, key):
        """Single HMC step."""
        q, log_p = state
        key, subkey = jax.random.split(key)
        
        # Sample momentum
        p = jax.random.normal(subkey, shape=q.shape)
        initial_energy = -log_p + 0.5 * jnp.sum(p**2)
        
        # Leapfrog integration
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog)
        
        # Compute acceptance probability
        log_p_new = log_joint(q_new)
        new_energy = -log_p_new + 0.5 * jnp.sum(p_new**2)
        
        # Metropolis accept/reject
        key, subkey = jax.random.split(key)
        accept_prob = jnp.minimum(1., jnp.exp(initial_energy - new_energy))
        accept = jax.random.uniform(subkey) < accept_prob
        
        q = jax.lax.cond(accept, lambda _: q_new, lambda _: q, operand=None)
        log_p = jax.lax.cond(accept, lambda _: log_p_new, lambda _: log_p, operand=None)
        
        return (q, log_p), q
    
    def run_hmc(key):
        # Initialize
        key, subkey = jax.random.split(key)
        q_init = jax.random.normal(subkey, shape=(3,))
        log_p_init = log_joint(q_init)
        
        # Run chain
        total_steps = n_warmup + n_samples
        keys = jax.random.split(key, total_steps + 1)
        
        # Use scan for efficiency
        (q_final, log_p_final), samples = jax.lax.scan(
            hmc_step, (q_init, log_p_init), keys[1:]
        )
        
        # Return samples after warmup
        return samples[n_warmup:]
    
    # JIT compile
    jitted_hmc = jax.jit(run_hmc)
    
    # Timing function
    def task():
        samples = jitted_hmc(key)
        jax.block_until_ready(samples)
        return samples
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(
        task,
        warmup_runs=3,
        repeats=repeats,
        inner_repeats=inner_repeats,
        auto_sync=False,
    )
    
    # Get final samples for validation
    samples = task()
    
    return {
        "framework": "handcoded_jax",
        "method": "hmc",
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "step_size": step_size,
        "n_leapfrog": n_leapfrog,
        "samples": {
            "a": samples[:, 0],
            "b": samples[:, 1],
            "c": samples[:, 2],
        }
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    
    from ..data.generation import generate_polynomial_data
    
    parser = argparse.ArgumentParser(description="Run Handcoded JAX+TFP benchmarks")
    parser.add_argument("--n-particles", type=int, nargs="+", 
                        default=[100, 1000, 10000, 100000],
                        help="Number of particles for IS")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Number of timing repetitions")
    parser.add_argument("--inner-repeats", type=int, default=50,
                        help="Inner timing repeats for IS")
    parser.add_argument("--output-dir", type=str, default="data/handcoded_jax",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = {}
    
    print("Running Handcoded JAX+TFP Importance Sampling benchmarks...")
    is_results = {}
    for n_particles in args.n_particles:
        print(f"  N = {n_particles:,} particles...")
        result = handcoded_jax_polynomial_is_timing(
            dataset,
            n_particles,
            repeats=args.repeats,
            inner_repeats=args.inner_repeats,
        )
        is_results[f"n{n_particles}"] = result
        
        # Save individual result (without samples to avoid JAX array serialization)
        result_file = output_dir / f"is_n{n_particles}.json"
        result_to_save = {k: v for k, v in result.items() if k not in ['samples', 'log_weights']}
        result_to_save['times'] = [float(t) for t in result['times']]  # Convert to Python floats
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
    
    results["is"] = is_results
    
    # Save summary (with cleaned results)
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Clean up results for JSON serialization
    clean_results = {}
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            clean_results[method] = {}
            for key, result in method_results.items():
                if isinstance(result, dict):
                    clean_result = {
                        k: v for k, v in result.items() 
                        if k not in ['samples', 'log_weights']
                    }
                    # Convert times to Python floats
                    if 'times' in clean_result:
                        clean_result['times'] = [float(t) for t in clean_result['times']]
                    clean_results[method][key] = clean_result
                else:
                    clean_results[method] = result
        else:
            clean_results[method] = method_results
    
    with open(summary_file, "w") as f:
        json.dump({
            "framework": "handcoded_jax",
            "dataset": {
                "n_points": dataset.n_points,
                "noise_std": float(dataset.noise_std),
            },
            "config": vars(args),
            "results": clean_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
