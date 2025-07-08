#!/usr/bin/env python
"""Run Pyro HMC benchmarks for polynomial regression."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Set CUDA environment
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"  # Pyro uses PyTorch, not JAX

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
    
    # Import PyTorch and Pyro
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, HMC
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Generate dataset
    np.random.seed(42)
    xs = np.linspace(-1, 1, args.n_points)
    true_params = np.array([0.7, -0.3, 0.5])
    ys = true_params[0] + true_params[1] * xs + true_params[2] * xs**2 + 0.05 * np.random.randn(args.n_points)
    
    # Convert to PyTorch tensors
    xs_torch = torch.tensor(xs, dtype=torch.float32, device=device)
    ys_torch = torch.tensor(ys, dtype=torch.float32, device=device)
    
    # Define Pyro model
    def polynomial_model(xs, ys=None):
        # Priors
        a = pyro.sample("a", dist.Normal(0.0, 1.0))
        b = pyro.sample("b", dist.Normal(0.0, 1.0))
        c = pyro.sample("c", dist.Normal(0.0, 1.0))
        
        # Deterministic transformation
        y_det = a + b * xs + c * xs**2
        
        # Likelihood
        with pyro.plate("data", len(xs)):
            pyro.sample("y", dist.Normal(y_det, 0.05), obs=ys)
        
        return a, b, c
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks for each chain length
    print("Running Pyro HMC benchmarks...")
    for n_samples in args.chain_lengths:
        print(f"  Chain length = {n_samples:,} samples...")
        
        # Warm-up run
        hmc_kernel = HMC(polynomial_model, step_size=0.01, num_steps=20)
        mcmc = MCMC(hmc_kernel, num_samples=10, warmup_steps=10)
        mcmc.run(xs_torch, ys_torch)
        
        # Timing runs
        times = []
        for rep in range(args.repeats):
            # Create fresh kernel for each run
            hmc_kernel = HMC(polynomial_model, step_size=0.01, num_steps=20)
            mcmc = MCMC(hmc_kernel, num_samples=n_samples, warmup_steps=args.n_warmup)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            mcmc.run(xs_torch, ys_torch)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            if (rep + 1) % 10 == 0:
                print(f"    Completed {rep + 1}/{args.repeats} repetitions...")
        
        # Calculate timing statistics
        times = np.array(times)
        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        
        # Get final samples for validation
        hmc_kernel = HMC(polynomial_model, step_size=0.01, num_steps=20)
        mcmc = MCMC(hmc_kernel, num_samples=n_samples, warmup_steps=args.n_warmup)
        mcmc.run(xs_torch, ys_torch)
        samples = mcmc.get_samples()
        
        # Prepare result
        result = {
            "framework": "pyro",
            "method": "hmc",
            "n_samples": n_samples,
            "n_warmup": args.n_warmup,
            "n_points": args.n_points,
            "times": times.tolist(),
            "mean_time": mean_time,
            "std_time": std_time,
            "step_size": 0.01,
            "num_steps": 20,
            "device": args.device
        }
        
        # Save result
        result_file = output_dir / f"hmc_n{n_samples}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"    Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()