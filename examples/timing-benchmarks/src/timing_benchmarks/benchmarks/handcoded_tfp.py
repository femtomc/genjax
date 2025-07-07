"""Handcoded JAX with TFP distributions benchmark implementations for polynomial regression.

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


def handcoded_tfp_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None
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
    
    # Direct importance sampling implementation - optimized pure JAX
    def importance_sampling(key, xs, ys, n_particles):
        """Optimized pure JAX implementation without TFP overhead."""
        # Split keys for each particle
        sample_keys = jrand.split(key, n_particles)
        
        # Sample parameters for each particle - pure JAX
        def sample_particle(k):
            k1, k2, k3 = jrand.split(k, 3)
            a = jrand.normal(k1, ())
            b = jrand.normal(k2, ())
            c = jrand.normal(k3, ())
            return a, b, c
        
        a_samples, b_samples, c_samples = jax.vmap(sample_particle)(sample_keys)
        
        # Compute log likelihood for each sample - pure JAX
        def log_likelihood(a, b, c):
            y_pred = a + b * xs + c * xs**2
            # Use JAX's scipy for log pdf computation
            return jax.scipy.stats.norm.logpdf(ys, y_pred, 0.05).sum()
        
        log_weights = jax.vmap(log_likelihood)(a_samples, b_samples, c_samples)
        
        return {
            'a': a_samples,
            'b': b_samples,
            'c': c_samples,
            'log_weights': log_weights
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
        inner_repeats=200,  # Increased for better accuracy on fast operations
        auto_sync=False
    )
    
    # Get samples for validation
    samples = task()
    
    return {
        "framework": "handcoded_tfp",
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
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of timing repetitions")
    parser.add_argument("--output-dir", type=str, default="data/handcoded_tfp",
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
        result = handcoded_tfp_polynomial_is_timing(
            dataset, n_particles, repeats=args.repeats
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
            "framework": "handcoded_tfp",
            "dataset": {
                "n_points": dataset.n_points,
                "noise_std": float(dataset.noise_std),
            },
            "config": vars(args),
            "results": clean_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")