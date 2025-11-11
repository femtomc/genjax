"""Handcoded PyTorch benchmark for polynomial regression."""

import argparse
import numpy as np
import torch
import torch.distributions as dist
from pathlib import Path
import json
from datetime import datetime
import time

from ..data.generation import generate_polynomial_data, PolynomialDataset
from .timing_utils import benchmark_with_warmup
from typing import Dict, Any, Optional


def importance_sampling(xs, ys, n_particles, device='cuda'):
    """Handcoded PyTorch importance sampling for polynomial regression."""
    # Move data to device
    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    ys = torch.tensor(ys, dtype=torch.float32, device=device)
    
    # Pre-compute x^2
    xs_squared = xs * xs
    
    # Sample parameters for all particles at once
    a_dist = dist.Normal(0.0, 1.0)
    b_dist = dist.Normal(0.0, 1.0)
    c_dist = dist.Normal(0.0, 1.0)
    
    a_samples = a_dist.sample((n_particles,)).to(device)
    b_samples = b_dist.sample((n_particles,)).to(device)
    c_samples = c_dist.sample((n_particles,)).to(device)
    
    # Compute predictions for all particles
    # Shape: (n_particles, n_data)
    y_pred = (a_samples.unsqueeze(1) + 
              b_samples.unsqueeze(1) * xs.unsqueeze(0) + 
              c_samples.unsqueeze(1) * xs_squared.unsqueeze(0))
    
    # Compute log likelihood
    obs_dist = dist.Normal(y_pred, 0.05)
    log_weights = obs_dist.log_prob(ys.unsqueeze(0)).sum(dim=1)
    
    return log_weights


def handcoded_torch_timing(dataset, n_particles, repeats=10, device='cuda'):
    """Time handcoded PyTorch importance sampling."""
    # Convert JAX arrays to numpy if needed
    xs = np.array(dataset.xs) if hasattr(dataset.xs, '__array__') else dataset.xs
    ys = np.array(dataset.ys) if hasattr(dataset.ys, '__array__') else dataset.ys

    def task():
        with torch.no_grad():
            result = importance_sampling(xs, ys, n_particles, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        return result

    # Use common timing utility with standardized parameters
    times, (mean_time, std_time) = benchmark_with_warmup(
        task,
        warmup_runs=5,
        repeats=10,
        inner_repeats=10,
        auto_sync=False,  # We handle synchronization manually in task()
    )

    return {
        'framework': 'handcoded_torch',
        'method': 'importance_sampling',
        'n_particles': n_particles,
        'n_points': dataset.n_points,
        'times': times.tolist(),  # Convert numpy array to list for JSON serialization
        'mean_time': mean_time,
        'std_time': std_time,
        'device': device
    }


def main():
    """Main entry point for handcoded PyTorch benchmarks."""
    parser = argparse.ArgumentParser(description="Run handcoded PyTorch benchmarks")
    parser.add_argument("--n-particles", type=int, nargs="+",
                        default=[100, 1000, 10000, 100000],
                        help="Number of particles for IS")
    parser.add_argument("--n-points", type=int, default=50,
                        help="Number of data points")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of timing repetitions (outer repeats)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path("data/handcoded_torch")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Handcoded PyTorch Importance Sampling benchmarks...")
    
    for n_particles in args.n_particles:
        print(f"  N = {n_particles:,} particles...")
        result = handcoded_torch_timing(dataset, n_particles, 
                                       repeats=args.repeats, device=args.device)
        
        # Save individual result
        result_file = output_dir / f"is_n{n_particles}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
    
    # Save summary
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "framework": "handcoded_torch",
            "n_points": dataset.n_points, 
            "noise_std": dataset.noise_std,
            "config": vars(args)
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")




def handcoded_torch_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int = 1000,
    n_warmup: int = 50,
    repeats: int = 100,
    key: Optional[Any] = None,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Handcoded PyTorch HMC timing for polynomial regression."""
    import torch
    
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
    except:
        # Fall back to regular function if compile not available
        run_hmc_compiled = run_hmc

    # Timing function
    def task():
        samples = run_hmc_compiled()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return samples

    # Use common timing utility with standardized parameters
    times, (mean_time, std_time) = benchmark_with_warmup(
        task,
        warmup_runs=3,
        repeats=10,
        inner_repeats=10,
        auto_sync=False,  # We handle synchronization manually in task()
    )

    # Get final samples for validation
    samples = task()
    
    return {
        "framework": "handcoded_torch",
        "method": "hmc",
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_points": dataset.n_points,
        "times": times.tolist(),  # Convert numpy array to list for JSON serialization
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


if __name__ == "__main__":
    main()
