#!/usr/bin/env python
"""Run PyTorch HMC benchmarks directly."""

import sys
import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import without JAX
from timing_benchmarks.data.generation import generate_polynomial_data

# Now import the PyTorch HMC function
import torch
import torch.distributions as dist

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
    """Handcoded PyTorch HMC timing for polynomial regression."""
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # Convert data to PyTorch tensors
    xs = torch.tensor(dataset.xs.__array__(), dtype=torch.float32, device=device)
    ys = torch.tensor(dataset.ys.__array__(), dtype=torch.float32, device=device)
    n_points = len(xs)
    
    # Log joint density
    def log_joint(params):
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs**2
        
        # Likelihood: Normal(y | y_pred, 0.05)
        log_lik = torch.distributions.Normal(y_pred, 0.05).log_prob(ys).sum()
        
        # Priors: Normal(0, 1) for all parameters
        log_prior = torch.distributions.Normal(0., 1.).log_prob(params).sum()
        
        return log_lik + log_prior
    
    # Compute gradient of log joint
    def grad_log_joint(params):
        params = params.detach().requires_grad_(True)
        log_p = log_joint(params)
        log_p.backward()
        return params.grad
    
    # HMC implementation
    def leapfrog(q, p, step_size, n_leapfrog):
        """Leapfrog integrator for HMC."""
        q = q.clone()
        p = p.clone()
        
        # Initial half step for momentum
        grad = grad_log_joint(q)
        p = p + 0.5 * step_size * grad
        
        # Full steps
        for _ in range(n_leapfrog - 1):
            q = q + step_size * p
            grad = grad_log_joint(q)
            p = p + step_size * grad
        
        # Final position update and half step for momentum
        q = q + step_size * p
        grad = grad_log_joint(q)
        p = p + 0.5 * step_size * grad
        
        return q, p
    
    def hmc_step(q, log_p):
        """Single HMC step."""
        # Sample momentum
        p = torch.randn_like(q)
        initial_energy = -log_p + 0.5 * (p**2).sum()
        
        # Leapfrog integration
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog)
        
        # Compute acceptance probability
        log_p_new = log_joint(q_new)
        new_energy = -log_p_new + 0.5 * (p_new**2).sum()
        
        # Metropolis accept/reject
        accept_prob = torch.minimum(torch.tensor(1.), torch.exp(initial_energy - new_energy))
        accept = torch.rand(1, device=device) < accept_prob
        
        if accept:
            return q_new, log_p_new
        else:
            return q, log_p
    
    def run_hmc():
        # Initialize
        q = torch.randn(3, device=device)
        log_p = log_joint(q)
        
        # Collect samples
        samples = []
        total_steps = n_warmup + n_samples
        
        # Run chain
        for i in range(total_steps):
            q, log_p = hmc_step(q, log_p)
            if i >= n_warmup:
                samples.append(q.clone())
        
        return torch.stack(samples)
    
    # Try to use torch.compile if available (PyTorch 2.0+)
    try:
        run_hmc_compiled = torch.compile(run_hmc)
        print("Using torch.compile for acceleration")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), falling back to eager mode")
        run_hmc_compiled = run_hmc
    
    # Warm-up run
    _ = run_hmc_compiled()
    
    # Timing function
    def task():
        samples = run_hmc_compiled()
        if device == "cuda":
            torch.cuda.synchronize()
        return samples
    
    # Timing runs
    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        samples = task()
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    
    # Get final samples for validation
    samples = run_hmc_compiled()
    
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
        "samples": {
            "a": samples[:, 0].cpu().numpy(),
            "b": samples[:, 1].cpu().numpy(),
            "c": samples[:, 2].cpu().numpy(),
        }
    }


def main():
    # Generate dataset
    print("Generating polynomial dataset...")
    dataset = generate_polynomial_data(n_points=50, seed=42)
    
    # Run benchmarks for 100 and 500
    chain_lengths = [100, 500]
    
    output_dir = Path("data/handcoded_torch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for n_samples in chain_lengths:
        print(f"\nRunning HMC with {n_samples} samples...")
        
        result = handcoded_torch_polynomial_hmc_timing(
            dataset,
            n_samples=n_samples,
            n_warmup=500,
            repeats=50,
            step_size=0.01,
            n_leapfrog=20,
            device="cuda"
        )
        
        # Save result (without samples for JSON)
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
        }
        
        result_file = output_dir / f"hmc_n{n_samples}.json"
        with open(result_file, "w") as f:
            json.dump(result_json, f, indent=2)
        
        print(f"✓ handcoded_torch HMC (n={n_samples}): {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        print(f"  Saved to: {result_file}")


if __name__ == "__main__":
    main()