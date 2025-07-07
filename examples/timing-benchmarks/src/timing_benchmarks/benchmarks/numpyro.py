"""NumPyro benchmark implementations for polynomial regression.

This module contains NumPyro-specific models and timing functions for
polynomial regression with importance sampling, using direct distribution sampling
for optimal performance.
"""

import time
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, Predictive
from numpyro.primitives import plate, param, sample
from numpyro import handlers

from genjax.timing import benchmark_with_warmup
from ..data.generation import PolynomialDataset, polyfn


# NumPyro model definition
def polynomial_model(xs, ys=None):
    """Polynomial regression model in NumPyro."""
    # Sample coefficients from prior
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    c = numpyro.sample("c", dist.Normal(0.0, 1.0))
    
    # Compute predictions
    y_pred = a + b * xs + c * xs**2
    
    # Observe data
    numpyro.sample("ys", dist.Normal(y_pred, 0.05), obs=ys)
    
    return y_pred


def numpyro_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None
) -> Dict[str, Any]:
    """Time NumPyro importance sampling on polynomial regression.
    
    This implementation uses direct sampling and likelihood computation for minimal overhead.
    
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
    
    # Importance sampling using NumPyro's trace handler (like GenJAX)
    def importance_sampling_traced(key, xs, ys, n_particles):
        """Importance sampling using NumPyro's trace handler."""
        keys = jrand.split(key, n_particles)
        
        def sample_and_weight(k):
            # Run model and capture trace
            trace_fn = handlers.trace(handlers.seed(polynomial_model, k))
            tr = trace_fn.get_trace(xs, ys)
            
            # Extract samples
            a = tr['a']['value']
            b = tr['b']['value']
            c = tr['c']['value']
            
            # Compute log weight (sum of log probs)
            log_weight = 0.0
            for site in tr.values():
                if site['type'] == 'sample':
                    log_weight += site['fn'].log_prob(site['value']).sum()
            
            return a, b, c, log_weight
        
        # Vectorize over particles
        a_samples, b_samples, c_samples, log_weights = jax.vmap(sample_and_weight)(keys)
        
        return {
            'a': a_samples,
            'b': b_samples,
            'c': c_samples,
            'log_weights': log_weights
        }
    
    # JIT compile the inference function with static n_particles
    jitted_is = jax.jit(importance_sampling_traced, static_argnums=(3,))
    
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
        inner_repeats=200,  # Increased for better accuracy
        auto_sync=False
    )
    
    # Get samples for validation
    samples = task()
    
    return {
        "framework": "numpyro",
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


def numpyro_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int,
    n_warmup: int = 500,
    repeats: int = 100,
    key: Optional[jax.Array] = None
) -> Dict[str, Any]:
    """Time NumPyro HMC on polynomial regression.
    
    Note: NumPyro HMC implementation would go here.
    For now, this returns a placeholder result.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        key: Random key (optional)
        
    Returns:
        Dictionary with timing results
    """
    # Placeholder implementation
    return {
        "framework": "numpyro",
        "method": "hmc",
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_points": dataset.n_points,
        "times": [],
        "mean_time": 0.0,
        "std_time": 0.0,
        "error": "HMC not implemented in NumPyro benchmarks"
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    
    from ..data.generation import generate_polynomial_data
    
    parser = argparse.ArgumentParser(description="Run NumPyro benchmarks")
    parser.add_argument("--n-particles", type=int, nargs="+", 
                        default=[100, 1000, 10000, 100000],
                        help="Number of particles for IS")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of timing repetitions")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path("data/numpyro")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = {}
    
    print("Running NumPyro Importance Sampling benchmarks...")
    is_results = {}
    for n_particles in args.n_particles:
        print(f"  N = {n_particles:,} particles...")
        result = numpyro_polynomial_is_timing(
            dataset, n_particles, repeats=args.repeats
        )
        is_results[f"n{n_particles}"] = result
        
        # Save individual result (without samples)
        result_file = output_dir / f"is_n{n_particles}.json"
        result_to_save = {k: v for k, v in result.items() if k not in ['samples', 'log_weights']}
        result_to_save['times'] = [float(t) for t in result['times']]
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
    
    results["is"] = is_results
    
    # Save summary
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
            "framework": "numpyro",
            "dataset": {
                "n_points": dataset.n_points,
                "noise_std": float(dataset.noise_std),
            },
            "config": vars(args),
            "results": clean_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")