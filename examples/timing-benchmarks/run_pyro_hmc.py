#!/usr/bin/env python
"""Run Pyro HMC benchmarks for polynomial regression."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'timing_benchmarks', 'curvefit-benchmarks'))
from src.timing_benchmarks.data.generation import generate_polynomial_data
import pyro as pyro_module


def main():
    parser = argparse.ArgumentParser(description="Run Pyro HMC benchmarks")
    parser.add_argument("--chain-lengths", type=int, nargs="+", 
                        default=[100, 500, 1000],
                        help="HMC chain lengths to benchmark")
    parser.add_argument("--n-warmup", type=int, default=500,
                        help="Number of warmup samples")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of timing repetitions")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="data/pyro",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks for each chain length
    print("Running Pyro HMC benchmarks...")
    for n_samples in args.chain_lengths:
        print(f"  Chain length = {n_samples:,} samples...")
        
        # Run benchmark
        result = pyro_polynomial_hmc_timing(
            dataset, 
            n_samples=n_samples,
            n_warmup=args.n_warmup,
            repeats=args.repeats, 
            device=args.device
        )
        
        # Save result (without samples to avoid serialization issues)
        result_file = output_dir / f"hmc_n{n_samples}.json"
        result_to_save = {k: v for k, v in result.items() if k not in ['samples']}
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
        
        print(f"    Mean time: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()