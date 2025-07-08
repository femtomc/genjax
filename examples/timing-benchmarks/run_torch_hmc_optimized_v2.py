#!/usr/bin/env python
"""Highly optimized PyTorch HMC implementation for polynomial regression."""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import without JAX
from timing_benchmarks.data.generation import generate_polynomial_data

def handcoded_torch_polynomial_hmc_timing(
    dataset,
    n_samples=1000,
    n_warmup=500,
    repeats=100,
    key=None,
    step_size=0.01,
    n_leapfrog=20,
    device="cuda",
):
    """Highly optimized handcoded PyTorch HMC timing for polynomial regression."""
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # Convert data to PyTorch tensors
    xs = torch.tensor(dataset.xs.__array__(), dtype=torch.float32, device=device)
    ys = torch.tensor(dataset.ys.__array__(), dtype=torch.float32, device=device)
    n_points = len(xs)
    
    # Precompute xs squared
    xs_squared = xs ** 2
    
    # Precompute constants
    inv_sigma_sq = 1.0 / (0.05 ** 2)
    log_normalizer_lik = -n_points * np.log(0.05 * np.sqrt(2 * np.pi))
    log_normalizer_prior = -3 * np.log(np.sqrt(2 * np.pi))
    
    # Optimized log joint computation without autograd
    def log_joint_manual(params):
        """Compute log joint density manually (no autograd needed)."""
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs_squared
        
        # Likelihood
        residuals = ys - y_pred
        log_lik = -0.5 * inv_sigma_sq * torch.sum(residuals ** 2) + log_normalizer_lik
        
        # Prior
        log_prior = -0.5 * torch.sum(params ** 2) + log_normalizer_prior
        
        return log_lik + log_prior
    
    # Manual gradient computation (polynomial regression has closed-form gradients)
    def grad_log_joint_manual(params):
        """Compute gradient of log joint manually."""
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs_squared
        residuals = ys - y_pred
        
        # Gradient of log likelihood
        common_factor = inv_sigma_sq * torch.sum(residuals)
        grad_a = common_factor
        grad_b = inv_sigma_sq * torch.sum(residuals * xs)
        grad_c = inv_sigma_sq * torch.sum(residuals * xs_squared)
        
        # Add gradient of log prior
        grad_a -= a
        grad_b -= b
        grad_c -= c
        
        return torch.stack([grad_a, grad_b, grad_c])
    
    # Ultra-fast leapfrog using manual gradients
    def leapfrog_fast(q, p):
        """Optimized leapfrog integrator."""
        # Initial half step for momentum
        grad = grad_log_joint_manual(q)
        p = p + 0.5 * step_size * grad
        
        # Full steps
        for _ in range(n_leapfrog - 1):
            q = q + step_size * p
            grad = grad_log_joint_manual(q)
            p = p + step_size * grad
        
        # Final position update and half step
        q = q + step_size * p
        grad = grad_log_joint_manual(q)
        p = p + 0.5 * step_size * grad
        
        return q, p
    
    def hmc_kernel_fast(q_init, total_steps):
        """Run HMC chain with optimized kernel."""
        q = q_init.clone()
        log_p = log_joint_manual(q)
        
        samples = torch.zeros((n_samples, 3), device=device)
        sample_idx = 0
        
        for step in range(total_steps):
            # Sample momentum
            p = torch.randn(3, device=device)
            initial_kinetic = 0.5 * torch.sum(p ** 2)
            initial_energy = -log_p + initial_kinetic
            
            # Leapfrog integration
            q_new, p_new = leapfrog_fast(q.clone(), p)
            
            # Compute new energy
            log_p_new = log_joint_manual(q_new)
            new_kinetic = 0.5 * torch.sum(p_new ** 2)
            new_energy = -log_p_new + new_kinetic
            
            # Metropolis accept/reject
            energy_diff = initial_energy - new_energy
            if torch.log(torch.rand(1, device=device)) < energy_diff:
                q = q_new
                log_p = log_p_new
            
            # Store sample after warmup
            if step >= n_warmup:
                samples[sample_idx] = q
                sample_idx += 1
        
        return samples
    
    # Initialize chain
    torch.manual_seed(42)
    q_init = torch.randn(3, device=device)
    
    # Warm-up run
    with torch.no_grad():
        _ = hmc_kernel_fast(q_init, n_warmup + n_samples)
    
    # Timing runs
    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            samples = hmc_kernel_fast(q_init, n_warmup + n_samples)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    
    # Get final samples for validation
    with torch.no_grad():
        samples = hmc_kernel_fast(q_init, n_warmup + n_samples)
    
    return {
        "framework": "handcoded_torch",
        "method": "hmc",
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "step_size": step_size,
        "n_leapfrog": n_leapfrog,
        "device": str(device),
        "samples": {
            "a": samples[:, 0].cpu().numpy(),
            "b": samples[:, 1].cpu().numpy(),
            "c": samples[:, 2].cpu().numpy(),
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run optimized PyTorch HMC benchmarks")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-warmup", type=int, default=500, help="Number of warmup samples")
    parser.add_argument("--repeats", type=int, default=20, help="Number of timing repetitions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    # Generate dataset
    print("Generating polynomial dataset...")
    dataset = generate_polynomial_data(n_points=50, seed=42)
    
    print(f"\nRunning highly optimized HMC with {args.n_samples} samples on {args.device}...")
    
    result = handcoded_torch_polynomial_hmc_timing(
        dataset,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
        step_size=0.01,
        n_leapfrog=20,
        device=args.device
    )
    
    # Save result
    output_dir = Path("data/handcoded_torch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_json = {
        "framework": result["framework"],
        "method": result["method"],
        "n_samples": result["n_samples"],
        "n_warmup": result["n_warmup"],
        "n_points": result["n_points"],
        "times": result["times"].tolist(),
        "mean_time": result["mean_time"],
        "std_time": result["std_time"],
        "step_size": result["step_size"],
        "n_leapfrog": result["n_leapfrog"],
        "device": result["device"],
    }
    
    result_file = output_dir / f"hmc_n{args.n_samples}.json"
    with open(result_file, "w") as f:
        json.dump(result_json, f, indent=2)
    
    print(f"✓ handcoded_torch HMC (n={args.n_samples}): {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
    print(f"  Saved to: {result_file}")
    
    # Compare with Pyro if available
    pyro_file = Path(f"data/pyro/hmc_n{args.n_samples}.json")
    if pyro_file.exists():
        with open(pyro_file) as f:
            pyro_result = json.load(f)
        pyro_time = pyro_result["mean_time"]
        speedup = pyro_time / result["mean_time"]
        print(f"  Speedup vs Pyro: {speedup:.2f}×")


if __name__ == "__main__":
    main()