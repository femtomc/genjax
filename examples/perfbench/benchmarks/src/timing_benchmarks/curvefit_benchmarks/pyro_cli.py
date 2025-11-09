"""Pyro-specific CLI for running benchmarks in the Pyro environment.

This module provides entry points for Pyro benchmarks that need to be run
in a separate environment with PyTorch and Pyro installed.
"""

import argparse
import numpy as np
from ..data import get_benchmark_datasets
from ..export import save_benchmark_results
from .pyro import (
    pyro_polynomial_is_timing,
    pyro_polynomial_hmc_timing,
)


def main():
    """Main entry point for Pyro benchmarks."""
    parser = argparse.ArgumentParser(description="Run Pyro benchmarks")
    parser.add_argument("command", choices=["pyro-is", "pyro-hmc", "pyro-all"])
    parser.add_argument("--data-size", default="small", 
                       choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    # Get dataset
    datasets = get_benchmark_datasets()
    dataset = datasets[args.data_size]
    
    results = {}
    
    if args.command in ["pyro-is", "pyro-all"]:
        print("Running Pyro IS benchmarks...")
        is_results = {}
        for n_particles in [100, 1000, 10000]:
            key = f"n{n_particles}"
            is_results[key] = {
                "pyro": pyro_polynomial_is_timing(
                    dataset, n_particles, repeats=args.repeats, device=args.device
                )
            }
        results["is_comparison"] = is_results
    
    if args.command in ["pyro-hmc", "pyro-all"]:
        print("Running Pyro HMC benchmarks...")
        hmc_results = {
            "pyro": pyro_polynomial_hmc_timing(
                dataset, n_samples=1000, n_warmup=500, 
                repeats=args.repeats, device=args.device
            )
        }
        results["hmc_comparison"] = hmc_results
    
    if args.export and results:
        config = {
            "benchmark": f"pyro_{args.command}",
            "data_size": args.data_size,
            "repeats": args.repeats,
            "device": args.device,
            "frameworks": ["pyro"]
        }
        
        save_results = {"config": config, **results}
        exp_dir = save_benchmark_results(
            save_results,
            description=f"Pyro benchmarks on {args.device}"
        )
        print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()