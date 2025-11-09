"""Optimized GenJAX benchmark implementation for polynomial regression.

This implementation follows the same pattern as the faircoin example
to minimize overhead and achieve comparable performance.
"""

import time
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax.inference import init
from genjax import gen, normal, Cond, Const, sel
from genjax.core import Pytree
from genjax.pjax import seed
from genjax import modular_vmap as vmap

from genjax.timing import benchmark_with_warmup
from ..data.generation import PolynomialDataset, polyfn


### Optimized GenJAX Model ###

@gen
def polynomial_model(xs):
    """Polynomial model matching handcoded structure."""
    # Sample coefficients from prior
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    
    # Compute predictions
    y_pred = a + b * xs + c * xs**2
    
    # Observe with vectorized normal
    ys = normal.vmap(in_axes=(0, None))(y_pred, 0.05) @ "ys"
    
    return ys


### Direct Importance Sampling (matching faircoin pattern) ###

def genjax_polynomial_is_direct(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None,
) -> Dict[str, Any]:
    """Direct GenJAX importance sampling following faircoin pattern.
    
    This implementation minimizes overhead by:
    1. Using direct vmap without intermediate functions
    2. Avoiding unnecessary seed wrapping
    3. Not creating dummy arrays
    
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
    constraints = {"ys": ys}
    
    # Direct importance sampling function
    def importance_sample(constraints):
        trace, weight = polynomial_model.generate(constraints, xs)
        return trace, weight
    
    # JIT compile with vmap - matching faircoin pattern exactly
    imp_jit = jax.jit(
        seed(
            vmap(
                importance_sample,
                axis_size=n_particles,
                in_axes=None,  # constraints are not vmapped
            )
        )
    )
    
    # Task for benchmarking
    def task():
        traces, log_weights = imp_jit(key, constraints)
        jax.block_until_ready(log_weights)  # Only block on weights
        return traces, log_weights
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, warmup_runs=2, repeats=repeats, inner_repeats=50, auto_sync=False
    )
    
    # Get samples for validation
    traces, log_weights = task()
    
    # Extract samples
    samples_a = traces.get_choices()["a"]
    samples_b = traces.get_choices()["b"]
    samples_c = traces.get_choices()["c"]
    
    return {
        "framework": "genjax_optimized",
        "method": "importance_sampling",
        "n_particles": n_particles,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "samples": {
            "a": samples_a,
            "b": samples_b,
            "c": samples_c,
        },
        "log_weights": log_weights,
    }


### Alternative: Match handcoded structure exactly ###

def genjax_polynomial_is_minimal(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None,
) -> Dict[str, Any]:
    """Minimal GenJAX implementation that matches handcoded structure.
    
    This version only computes weights, not full traces.
    
    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        key: Random key (optional)
        
    Returns:
        Dictionary with timing results
    """
    if key is None:
        key = jrand.key(42)
    
    xs, ys = dataset.xs, dataset.ys
    constraints = {"ys": ys}
    
    # Only compute weights
    def importance_weight_only(constraints):
        _, weight = polynomial_model.generate(constraints, xs)
        return weight
    
    # JIT compile with vmap
    imp_jit = jax.jit(
        seed(
            vmap(
                importance_weight_only,
                axis_size=n_particles,
                in_axes=None,
            )
        )
    )
    
    # Task for benchmarking
    def task():
        log_weights = imp_jit(key, constraints)
        jax.block_until_ready(log_weights)
        return log_weights
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, warmup_runs=2, repeats=repeats, inner_repeats=50, auto_sync=False
    )
    
    # Get weights for validation
    log_weights = task()
    
    return {
        "framework": "genjax_minimal",
        "method": "importance_sampling",
        "n_particles": n_particles,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "log_weights": log_weights,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    
    from ..data.generation import generate_polynomial_data
    
    parser = argparse.ArgumentParser(description="Run optimized GenJAX benchmarks")
    parser.add_argument(
        "--n-particles",
        type=int,
        nargs="+",
        default=[100, 1000, 10000, 100000],
        help="Number of particles for IS",
    )
    parser.add_argument(
        "--n-points", type=int, default=50, help="Number of data points"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of timing repetitions"
    )
    parser.add_argument(
        "--minimal", action="store_true", help="Use minimal implementation (weights only)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Choose implementation
    if args.minimal:
        timing_fn = genjax_polynomial_is_minimal
        framework_name = "genjax_minimal"
    else:
        timing_fn = genjax_polynomial_is_direct
        framework_name = "genjax_optimized"
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path(f"data/{framework_name}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = {}
    
    print(f"Running {framework_name} Importance Sampling benchmarks...")
    is_results = {}
    for n_particles in args.n_particles:
        print(f"  N = {n_particles:,} particles...")
        result = timing_fn(dataset, n_particles, repeats=args.repeats)
        is_results[f"n{n_particles}"] = result
        
        # Save individual result
        result_file = output_dir / f"is_n{n_particles}.json"
        result_to_save = {
            k: v for k, v in result.items() if k not in ["samples", "log_weights"]
        }
        result_to_save["times"] = [float(t) for t in result["times"]]
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
    
    results["is"] = is_results
    
    # Save summary
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Clean results for JSON
    clean_results = {}
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            clean_results[method] = {}
            for key, result in method_results.items():
                if isinstance(result, dict):
                    clean_result = {
                        k: v for k, v in result.items() 
                        if k not in ["samples", "log_weights"]
                    }
                    if "times" in clean_result:
                        clean_result["times"] = [float(t) for t in clean_result["times"]]
                    clean_results[method][key] = clean_result
                else:
                    clean_results[method] = result
        else:
            clean_results[method] = method_results
    
    with open(summary_file, "w") as f:
        json.dump(
            {
                "framework": framework_name,
                "dataset": {
                    "n_points": dataset.n_points,
                    "noise_std": float(dataset.noise_std),
                },
                "config": vars(args),
                "results": clean_results,
            },
            f,
            indent=2,
        )
    
    print(f"\nResults saved to {output_dir}")