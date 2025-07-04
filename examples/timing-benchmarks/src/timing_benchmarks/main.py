"""Main CLI for timing benchmarks.

This module provides the command-line interface for running timing benchmarks
across probabilistic programming frameworks.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import benchmark modules
from .data import get_benchmark_datasets, get_scaling_datasets, summarize_dataset
from .analysis import (
    run_polynomial_is_comparison,
    run_polynomial_hmc_comparison,
)
from .benchmarks import (
    genjax_polynomial_is_timing,
    genjax_polynomial_hmc_timing,
)
from .analysis.combine import (
    gen_polynomial_is_timing,
    gen_polynomial_hmc_timing,
)
from .export import save_benchmark_results, get_latest_experiment
from .visualization import create_all_figures


def run_test_mode(args):
    """Run quick test mode with minimal data."""
    print("Running test mode with minimal configuration...")
    
    # Get tiny dataset
    datasets = get_benchmark_datasets()
    dataset = datasets["tiny"]
    
    print(summarize_dataset(dataset))
    
    # Test GenJAX IS
    print("\nTesting GenJAX IS with 100 particles...")
    genjax_is_result = genjax_polynomial_is_timing(
        dataset, n_particles=100, repeats=args.repeats
    )
    print(f"Mean time: {genjax_is_result['mean_time']:.4f}s ± {genjax_is_result['std_time']:.4f}s")
    
    # Test GenJAX HMC
    print("\nTesting GenJAX HMC with 100 samples...")
    genjax_hmc_result = genjax_polynomial_hmc_timing(
        dataset, n_samples=100, n_warmup=50, repeats=args.repeats
    )
    print(f"Mean time: {genjax_hmc_result['mean_time']:.4f}s ± {genjax_hmc_result['std_time']:.4f}s")
    
    # Test Gen.jl if requested
    if "gen.jl" in args.frameworks:
        print("\nTesting Gen.jl IS with 100 particles...")
        try:
            gen_is_result = gen_polynomial_is_timing(
                dataset, n_particles=100, repeats=args.repeats, setup_julia=args.setup_julia
            )
            print(f"Mean time: {gen_is_result['mean_time']:.4f}s ± {gen_is_result['std_time']:.4f}s")
        except Exception as e:
            print(f"Gen.jl test failed: {e}")
            print("Make sure Julia is installed and Gen.jl environment is set up")
    
    print("\nTest mode completed!")


def run_polynomial_is(args):
    """Run polynomial regression importance sampling benchmarks."""
    print("Running Polynomial Regression IS Benchmarks")
    print("=" * 50)
    
    # Get dataset
    datasets = get_benchmark_datasets()
    dataset = datasets[args.data_size]
    
    print(summarize_dataset(dataset))
    print()
    
    # Parse particle counts
    if args.n_particles:
        n_particles_list = args.n_particles
    else:
        n_particles_list = [100, 1000, 10000]
    
    # Run comparison
    results = run_polynomial_is_comparison(
        dataset,
        n_particles_list=n_particles_list,
        repeats=args.repeats,
        frameworks=args.frameworks
    )
    
    # Save results if requested
    if args.export_data:
        config = {
            "benchmark": "polynomial_is",
            "data_size": args.data_size,
            "n_particles_list": n_particles_list,
            "repeats": args.repeats,
            "frameworks": args.frameworks
        }
        
        save_results = {
            "config": config,
            "is_comparison": results
        }
        
        exp_dir = save_benchmark_results(
            save_results,
            description=f"Polynomial IS benchmark with {args.data_size} dataset"
        )
        
        # Generate figures
        if args.plot:
            create_all_figures(exp_dir)
    
    print("\nBenchmark completed!")


def run_polynomial_hmc(args):
    """Run polynomial regression HMC benchmarks."""
    print("Running Polynomial Regression HMC Benchmarks")
    print("=" * 50)
    
    # Get dataset
    datasets = get_benchmark_datasets()
    dataset = datasets[args.data_size]
    
    print(summarize_dataset(dataset))
    print()
    
    # Run comparison
    results = run_polynomial_hmc_comparison(
        dataset,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
        frameworks=args.frameworks
    )
    
    # Save results if requested
    if args.export_data:
        config = {
            "benchmark": "polynomial_hmc",
            "data_size": args.data_size,
            "n_samples": args.n_samples,
            "n_warmup": args.n_warmup,
            "repeats": args.repeats,
            "frameworks": args.frameworks
        }
        
        save_results = {
            "config": config,
            "hmc_comparison": results
        }
        
        exp_dir = save_benchmark_results(
            save_results,
            description=f"Polynomial HMC benchmark with {args.data_size} dataset"
        )
        
        # Generate figures
        if args.plot:
            create_all_figures(exp_dir)
    
    print("\nBenchmark completed!")


def run_polynomial_all(args):
    """Run all polynomial regression benchmarks."""
    print("Running All Polynomial Regression Benchmarks")
    print("=" * 50)
    
    # Get dataset
    datasets = get_benchmark_datasets()
    dataset = datasets[args.data_size]
    
    print(summarize_dataset(dataset))
    print()
    
    # Run IS comparison
    print("\n--- Importance Sampling ---")
    n_particles_list = [100, 1000, 10000]
    is_results = run_polynomial_is_comparison(
        dataset,
        n_particles_list=n_particles_list,
        repeats=args.repeats,
        frameworks=args.frameworks
    )
    
    # Run HMC comparison
    print("\n--- Hamiltonian Monte Carlo ---")
    hmc_results = run_polynomial_hmc_comparison(
        dataset,
        n_samples=1000,
        n_warmup=500,
        repeats=args.repeats,
        frameworks=args.frameworks
    )
    
    # Save combined results
    if args.export_data:
        config = {
            "benchmark": "polynomial_all",
            "data_size": args.data_size,
            "n_particles_list": n_particles_list,
            "n_samples": 1000,
            "n_warmup": 500,
            "repeats": args.repeats,
            "frameworks": args.frameworks
        }
        
        save_results = {
            "config": config,
            "is_comparison": is_results,
            "hmc_comparison": hmc_results
        }
        
        exp_dir = save_benchmark_results(
            save_results,
            description=f"Complete polynomial benchmark suite with {args.data_size} dataset"
        )
        
        # Generate figures
        if args.plot:
            create_all_figures(exp_dir)
    
    print("\nAll benchmarks completed!")


def plot_from_data(args):
    """Generate plots from saved benchmark data."""
    # Determine experiment directory
    if args.data:
        exp_dir = args.data
    else:
        exp_dir = get_latest_experiment()
        if exp_dir is None:
            print("No experiment data found. Run benchmarks first.")
            return
    
    print(f"Loading data from: {exp_dir}")
    
    # Create figures
    output_dir = args.output_dir or "figs"
    create_all_figures(exp_dir, output_dir)
    
    print(f"Figures saved to: {output_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Timing benchmarks for probabilistic programming frameworks"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Benchmark commands")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Quick test mode")
    test_parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of timing repetitions (default: 5)"
    )
    test_parser.add_argument(
        "--frameworks", nargs="+", default=["genjax"],
        choices=["genjax", "gen.jl", "numpyro", "pyro"],
        help="Frameworks to test (default: genjax)"
    )
    test_parser.add_argument(
        "--setup-julia", action="store_true",
        help="Run Julia setup before benchmarks"
    )
    
    # Polynomial IS
    poly_is_parser = subparsers.add_parser(
        "polynomial-is", help="Polynomial regression importance sampling"
    )
    poly_is_parser.add_argument(
        "--n-particles", nargs="+", type=int,
        help="Particle counts to test (default: 100, 1000, 10000)"
    )
    poly_is_parser.add_argument(
        "--data-size", default="medium",
        choices=["tiny", "small", "medium", "large", "xlarge", "xxlarge"],
        help="Dataset size (default: medium)"
    )
    poly_is_parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of timing repetitions (default: 100)"
    )
    poly_is_parser.add_argument(
        "--frameworks", nargs="+", default=["genjax", "gen.jl"],
        choices=["genjax", "gen.jl", "numpyro", "pyro"],
        help="Frameworks to benchmark (default: genjax gen.jl)"
    )
    poly_is_parser.add_argument(
        "--export-data", action="store_true",
        help="Save benchmark results to data directory"
    )
    poly_is_parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots after benchmarking"
    )
    
    # Polynomial HMC
    poly_hmc_parser = subparsers.add_parser(
        "polynomial-hmc", help="Polynomial regression HMC"
    )
    poly_hmc_parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of HMC samples (default: 1000)"
    )
    poly_hmc_parser.add_argument(
        "--n-warmup", type=int, default=500,
        help="Number of warmup samples (default: 500)"
    )
    poly_hmc_parser.add_argument(
        "--data-size", default="medium",
        choices=["tiny", "small", "medium", "large", "xlarge", "xxlarge"],
        help="Dataset size (default: medium)"
    )
    poly_hmc_parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of timing repetitions (default: 100)"
    )
    poly_hmc_parser.add_argument(
        "--frameworks", nargs="+", default=["genjax", "gen.jl"],
        choices=["genjax", "gen.jl", "numpyro", "pyro", "stan"],
        help="Frameworks to benchmark (default: genjax gen.jl)"
    )
    poly_hmc_parser.add_argument(
        "--export-data", action="store_true",
        help="Save benchmark results to data directory"
    )
    poly_hmc_parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots after benchmarking"
    )
    
    # Polynomial all
    poly_all_parser = subparsers.add_parser(
        "polynomial-all", help="All polynomial regression benchmarks"
    )
    poly_all_parser.add_argument(
        "--data-size", default="medium",
        choices=["tiny", "small", "medium", "large", "xlarge", "xxlarge"],
        help="Dataset size (default: medium)"
    )
    poly_all_parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of timing repetitions (default: 100)"
    )
    poly_all_parser.add_argument(
        "--frameworks", nargs="+", default=["genjax", "gen.jl"],
        choices=["genjax", "gen.jl", "numpyro", "pyro", "stan"],
        help="Frameworks to benchmark (default: genjax gen.jl)"
    )
    poly_all_parser.add_argument(
        "--export-data", action="store_true",
        help="Save benchmark results to data directory"
    )
    poly_all_parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots after benchmarking"
    )
    
    # Plot from saved data
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from saved benchmark data"
    )
    plot_parser.add_argument(
        "--data", type=str,
        help="Path to experiment data directory (default: latest)"
    )
    plot_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: timing-benchmarks/figs)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "test":
        run_test_mode(args)
    elif args.command == "polynomial-is":
        run_polynomial_is(args)
    elif args.command == "polynomial-hmc":
        run_polynomial_hmc(args)
    elif args.command == "polynomial-all":
        run_polynomial_all(args)
    elif args.command == "plot":
        plot_from_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()