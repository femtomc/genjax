"""GenJAX benchmark implementation for polynomial regression.

This implementation uses a flattened model structure for optimal performance.
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
from timing_benchmarks.data.generation import PolynomialDataset, polyfn


### Optimized GenJAX Model ###


@gen
def polynomial_flat(xs):
    """Flattened polynomial model for reduced overhead."""
    # Sample coefficients
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"

    # Compute predictions directly
    y_pred = a + b * xs + c * xs**2

    # Observe all points at once with normal.vmap
    ys = normal.vmap(in_axes=(0, None))(y_pred, 0.05) @ "ys"

    return ys


### Optimized Inference Function ###


def make_genjax_infer_is(n_particles: int):
    """Create optimized GenJAX importance sampling inference function.
    
    This follows the faircoin pattern for minimal overhead.
    """
    def infer(xs, ys):
        constraints = {"ys": ys}
        
        def importance_(constraints):
            _, w = polynomial_flat.generate(constraints, xs)
            return w
        
        # Direct vmap without dummy arrays
        imp = vmap(importance_, axis_size=n_particles, in_axes=None)
        return imp(constraints)
    
    return infer


### Timing Functions ###


def genjax_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None,
    use_direct: bool = False,
) -> Dict[str, Any]:
    """Time optimized GenJAX importance sampling on polynomial regression.

    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        key: Random key (optional)
        use_direct: Whether to use direct implementation

    Returns:
        Dictionary with timing results and samples
    """
    if key is None:
        key = jrand.key(42)

    xs, ys = dataset.xs, dataset.ys

    # Create and JIT the inference function using faircoin pattern
    genjax_infer_is = make_genjax_infer_is(n_particles)
    infer_jit = jax.jit(seed(genjax_infer_is))

    def task():
        log_weights = infer_jit(key, xs, ys)
        jax.block_until_ready(log_weights)
        return log_weights

    # Run benchmark with automatic warm-up - more inner repeats for accuracy
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, warmup_runs=5, repeats=repeats, inner_repeats=200, auto_sync=False
    )

    # Get final weights
    log_weights = task()

    # For compatibility, also get samples - run full version once
    def get_samples():
        constraints = {"ys": ys}
        def sample_particle(_):
            trace, _ = polynomial_flat.generate(constraints, xs)
            return trace
        traces = vmap(sample_particle)(jnp.arange(n_particles))
        return traces
    
    jitted_samples = jax.jit(seed(get_samples))
    traces = jitted_samples(key)
    
    # Extract samples
    samples_a = traces.get_choices()["a"]
    samples_b = traces.get_choices()["b"]
    samples_c = traces.get_choices()["c"]

    return {
        "framework": "genjax" if not use_direct else "genjax_direct",
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


def genjax_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int,
    n_warmup: int = 50,
    repeats: int = 100,
    key: Optional[jax.Array] = None,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
) -> Dict[str, Any]:
    """Time GenJAX HMC on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        key: Random key (optional)
        step_size: HMC step size (default: 0.01)
        n_leapfrog: Number of leapfrog steps (default: 20)
        
    Returns:
        Dictionary with timing results
    """
    if key is None:
        key = jrand.key(42)
    
    xs, ys = dataset.xs, dataset.ys
    
    # Import HMC from GenJAX
    from genjax.inference import hmc, chain
    from genjax import sel, const, seed
    from genjax.core import Trace
    
    # Create HMC kernel
    def hmc_kernel(trace: Trace):
        # Select all continuous parameters
        selection = sel("a") | sel("b") | sel("c")
        return hmc(trace, selection, step_size, n_leapfrog)
    
    # Create MCMC chain with burn-in
    mcmc_algorithm = chain(hmc_kernel)
    
    # Initialize with importance sampling
    constraints = {"ys": ys}
    init_trace, _ = polynomial_flat.generate(constraints, xs)
    
    # Create inference function WITHOUT key parameter - seed will add it
    def run_hmc():
        # Total steps = warmup + samples
        total_steps = n_warmup + n_samples
        
        # Run MCMC with single chain (for timing comparison)
        result = mcmc_algorithm(
            init_trace,
            n_steps=const(total_steps),
            n_chains=const(1),  # Single chain for fair comparison
            burn_in=const(n_warmup),
            autocorrelation_resampling=const(1)  # No thinning
        )
        
        # Extract samples
        choices = result.traces.get_choices()
        return choices
    
    # seed(run_hmc) will create a function that takes (key, *original_args)
    # Since run_hmc has no args, seeded version will just take (key)
    jitted_hmc = jax.jit(seed(run_hmc))
    
    # Timing function
    def task():
        samples = jitted_hmc(key)
        jax.block_until_ready(samples["a"])
        return samples
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, warmup_runs=3, repeats=repeats, inner_repeats=1, auto_sync=False
    )
    
    # Get final samples for validation
    samples = task()
    
    return {
        "framework": "genjax",
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
            "a": samples["a"].squeeze(),  # Remove chain dimension
            "b": samples["b"].squeeze(),
            "c": samples["c"].squeeze(),
        }
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path

    from ..data.generation import generate_polynomial_data

    parser = argparse.ArgumentParser(description="Run GenJAX benchmarks")
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
        "--use-direct", action="store_true", help="Use direct implementation"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--method", choices=["is", "hmc", "all"], default="is",
        help="Which method to benchmark"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of HMC samples"
    )
    parser.add_argument(
        "--n-warmup", type=int, default=500, help="Number of HMC warmup samples"
    )
    parser.add_argument(
        "--step-size", type=float, default=0.01, help="HMC step size"
    )
    parser.add_argument(
        "--n-leapfrog", type=int, default=20, help="HMC leapfrog steps"
    )

    args = parser.parse_args()

    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)

    # Create output directory
    if args.output_dir is None:
        output_dir = Path("data/genjax_direct" if args.use_direct else "data/genjax")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = {}

    # Run importance sampling benchmarks
    if args.method in ["is", "all"]:
        print(
            f"Running {'Direct' if args.use_direct else ''} GenJAX Importance Sampling benchmarks..."
        )
        is_results = {}
        for n_particles in args.n_particles:
            print(f"  N = {n_particles:,} particles...")
            result = genjax_polynomial_is_timing(
                dataset, n_particles, repeats=args.repeats, use_direct=args.use_direct
            )
            is_results[f"n{n_particles}"] = result

            # Save individual result (without samples to avoid JAX array serialization)
            result_file = output_dir / f"is_n{n_particles}.json"
            result_to_save = {
                k: v for k, v in result.items() if k not in ["samples", "log_weights"]
            }
            result_to_save["times"] = [
                float(t) for t in result["times"]
            ]  # Convert to Python floats
            with open(result_file, "w") as f:
                json.dump(result_to_save, f, indent=2)

        results["is"] = is_results

    # Run HMC benchmarks
    if args.method in ["hmc", "all"]:
        print("Running GenJAX HMC benchmarks...")
        print(f"  N = {args.n_samples:,} samples (warmup: {args.n_warmup})...")
        print(f"  Step size = {args.step_size}, Leapfrog steps = {args.n_leapfrog}")
        
        hmc_result = genjax_polynomial_hmc_timing(
            dataset,
            n_samples=args.n_samples,
            n_warmup=args.n_warmup,
            repeats=args.repeats,
            step_size=args.step_size,
            n_leapfrog=args.n_leapfrog,
        )
        
        # Save HMC result
        result_file = output_dir / f"hmc_n{args.n_samples}.json"
        result_to_save = {
            k: v for k, v in hmc_result.items() if k != "samples"
        }
        result_to_save["times"] = [
            float(t) for t in hmc_result["times"]
        ]  # Convert to Python floats
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
            
        results["hmc"] = hmc_result

    # Save summary
    summary_file = (
        output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Clean up results for JSON serialization
    clean_results = {}
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            clean_results[method] = {}
            for key, result in method_results.items():
                if isinstance(result, dict):
                    clean_result = {
                        k: v
                        for k, v in result.items()
                        if k not in ["samples", "log_weights"]
                    }
                    # Convert times to Python floats
                    if "times" in clean_result:
                        clean_result["times"] = [
                            float(t) for t in clean_result["times"]
                        ]
                    clean_results[method][key] = clean_result
                else:
                    clean_results[method] = result
        else:
            clean_results[method] = method_results

    with open(summary_file, "w") as f:
        json.dump(
            {
                "framework": "genjax" if not args.use_direct else "genjax_direct",
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
