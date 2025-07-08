#!/usr/bin/env python
"""Run HMC benchmarks for PyTorch only (no JAX imports)."""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import os

# Configure PyTorch
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import data generation
from timing_benchmarks.data.generation import generate_polynomial_data

# Import PyTorch HMC  
import importlib.util
spec = importlib.util.spec_from_file_location(
    "handcoded_torch", 
    Path(__file__).parent / "src" / "timing_benchmarks" / "curvefit-benchmarks" / "handcoded_torch.py"
)
handcoded_torch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handcoded_torch)
handcoded_torch_polynomial_hmc_timing = handcoded_torch.handcoded_torch_polynomial_hmc_timing


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch HMC benchmarks")
    parser.add_argument("--chain-lengths", type=int, nargs="+", 
                        default=[100, 500, 1000, 5000],
                        help="Chain lengths to benchmark")
    parser.add_argument("--n-warmup", type=int, default=500,
                        help="Number of warmup samples")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of timing repetitions")
    parser.add_argument("--device", default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    
    args = parser.parse_args()
    
    # Generate dataset
    print(f"\nGenerating polynomial dataset with {args.n_points} points...")
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    output_dir = Path("data/handcoded_torch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"Running HMC benchmarks for handcoded_torch")
    print("="*60)
    
    print(f"\n{'='*60}")
    print(f"Device Configuration:")
    print(f"{'='*60}")
    print(f"Device setting: {args.device}")
    
    results = []
    
    for chain_length in args.chain_lengths:
        print(f"\nChain length: {chain_length}")
        print("-" * 40)
        
        try:
            # Run benchmark
            result = handcoded_torch_polynomial_hmc_timing(
                dataset,
                n_samples=chain_length,
                n_warmup=args.n_warmup,
                repeats=args.repeats,
                step_size=0.01,
                n_leapfrog=20,
                device=args.device
            )
            
            # Convert numpy arrays to lists for JSON serialization
            result_json = {
                "framework": result["framework"],
                "method": result["method"],
                "n_samples": result["n_samples"],
                "n_warmup": result["n_warmup"],
                "n_points": result["n_points"],
                "times": result["times"].tolist() if hasattr(result["times"], 'tolist') else result["times"],
                "mean_time": float(result["mean_time"]),
                "std_time": float(result["std_time"]),
                "step_size": result["step_size"],
                "n_leapfrog": result["n_leapfrog"],
            }
            
            # Save result
            result_file = output_dir / f"hmc_n{chain_length}.json"
            with open(result_file, "w") as f:
                json.dump(result_json, f, indent=2)
            
            print(f"✓ handcoded_torch HMC (n={chain_length}): {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
            print(f"  Saved to: {result_file}")
            
            results.append(result_json)
            
        except Exception as e:
            print(f"✗ handcoded_torch HMC failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "framework": "handcoded_torch",
            "method": "hmc",
            "dataset": {
                "n_points": args.n_points,
                "noise_std": 0.05,
            },
            "config": vars(args),
            "results": results
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("HMC benchmarking complete!")
    print("="*60)
    print("\nRun the following to generate comparison plots:")
    print("python combine_results.py --frameworks handcoded_torch")


if __name__ == "__main__":
    main()