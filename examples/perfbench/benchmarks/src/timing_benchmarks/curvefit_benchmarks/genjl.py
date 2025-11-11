"""Gen.jl benchmark runner."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import os

from ..data.generation import generate_polynomial_data
from ..julia_interface import GenJLBenchmark


def genjl_polynomial_is_timing(dataset, n_particles, repeats=100, setup_julia=False):
    """Run Gen.jl polynomial regression importance sampling timing."""
    gen_jl = GenJLBenchmark()
    
    if setup_julia and gen_jl.julia_available:
        gen_jl.setup_julia_environment()
    
    # Run the benchmark
    result = gen_jl.run_polynomial_is(
        dataset.xs, dataset.ys, n_particles, repeats=repeats
    )
    
    # Add dataset info
    result['n_points'] = dataset.n_points
    
    return result


def genjl_polynomial_hmc_timing(dataset, n_samples, n_warmup=50, repeats=100, setup_julia=False, **kwargs):
    """Run Gen.jl polynomial regression HMC timing."""
    gen_jl = GenJLBenchmark()
    
    if setup_julia and gen_jl.julia_available:
        gen_jl.setup_julia_environment()
    
    # Extract HMC parameters if provided
    step_size = kwargs.get('step_size', 0.01)
    n_leapfrog = kwargs.get('n_leapfrog', 20)
    
    # Run the benchmark
    result = gen_jl.run_polynomial_hmc(
        dataset.xs, dataset.ys, n_samples, n_warmup=n_warmup, repeats=repeats,
        step_size=step_size, n_leapfrog=n_leapfrog
    )
    
    # Add dataset info
    result['n_points'] = dataset.n_points
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gen.jl benchmarks")
    parser.add_argument("--method", choices=["is", "hmc", "all"], default="is",
                        help="Inference method to benchmark")
    parser.add_argument("--n-particles", type=int, nargs="+", 
                        default=[100, 1000, 10000, 100000],
                        help="Number of particles for IS")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples for HMC")
    parser.add_argument("--n-warmup", type=int, default=500,
                        help="Number of warmup samples for HMC")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Number of timing repetitions")
    parser.add_argument("--output-dir", type=str, default="data/genjl",
                        help="Output directory for results")
    parser.add_argument("--setup-julia", action="store_true",
                        help="Run Julia setup before benchmarks")
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Julia if requested
    gen_jl = GenJLBenchmark()
    if args.setup_julia and gen_jl.julia_available:
        print("Setting up Julia environment...")
        gen_jl.setup_julia_environment()
    
    if not gen_jl.julia_available:
        print("Julia not available. Please install Julia to run Gen.jl benchmarks.")
        sys.exit(1)
    
    # Run benchmarks
    results = {}
    
    if args.method in ["is", "all"]:
        print("Running Gen.jl Importance Sampling benchmarks...")
        is_results = {}
        for n_particles in args.n_particles:
            print(f"  N = {n_particles:,} particles...")
            result = genjl_polynomial_is_timing(
                dataset, n_particles, repeats=args.repeats, setup_julia=False
            )
            is_results[f"n{n_particles}"] = result
            
            # Save individual result
            result_file = output_dir / f"is_n{n_particles}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        results["is"] = is_results
    
    if args.method in ["hmc", "all"]:
        print("Running Gen.jl HMC benchmarks...")
        result = genjl_polynomial_hmc_timing(
            dataset, args.n_samples, n_warmup=args.n_warmup, 
            repeats=args.repeats, setup_julia=False
        )
        results["hmc"] = result
        
        # Save HMC result
        result_file = output_dir / f"hmc_n{args.n_samples}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Save summary
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "framework": "gen.jl",
            "dataset": {
                "n_points": dataset.n_points,
                "noise_std": dataset.noise_std,
            },
            "config": vars(args),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
