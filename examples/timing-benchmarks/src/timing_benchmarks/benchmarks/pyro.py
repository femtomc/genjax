"""Pyro benchmark implementations for polynomial regression.

This module contains Pyro-specific models and timing functions for
polynomial regression with importance sampling and HMC.
"""

import time
from typing import Dict, Any
import numpy as np

from ..data.generation import PolynomialDataset


def pyro_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Time Pyro importance sampling on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        device: PyTorch device ("cpu" or "cuda")
        
    Returns:
        Dictionary with timing results
    """
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import Importance, EmpiricalMarginal
        
        # Set device
        device = torch.device(device)
        
        # Convert data to PyTorch tensors
        xs = torch.tensor(dataset.xs, dtype=torch.float32, device=device)
        ys = torch.tensor(dataset.ys, dtype=torch.float32, device=device)
        
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
        
        # Create importance sampling inference
        importance = Importance(polynomial_model, num_samples=n_particles)
        
        # Warm-up run
        _ = importance.run(xs, ys)
        
        # Timing runs
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            posterior = importance.run(xs, ys)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append(time.time() - start_time)
        
        # Get samples for validation
        posterior = importance.run(xs, ys)
        marginal_a = EmpiricalMarginal(posterior, "a")
        marginal_b = EmpiricalMarginal(posterior, "b")
        marginal_c = EmpiricalMarginal(posterior, "c")
        
        samples_a = marginal_a.sample((n_particles,)).cpu().numpy()
        samples_b = marginal_b.sample((n_particles,)).cpu().numpy()
        samples_c = marginal_c.sample((n_particles,)).cpu().numpy()
        
        # Get log weights
        log_weights = torch.tensor([
            posterior.log_prob(i).item() for i in range(n_particles)
        ])
        
        return {
            "framework": "pyro",
            "method": "is",
            "n_particles": n_particles,
            "n_points": dataset.n_points,
            "times": times,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "samples": {
                "a": samples_a,
                "b": samples_b,
                "c": samples_c,
            },
            "log_weights": log_weights.cpu().numpy(),
        }
        
    except ImportError:
        print("Pyro not installed. Skipping Pyro benchmarks.")
        return {
            "framework": "pyro",
            "method": "is",
            "n_particles": n_particles,
            "n_points": dataset.n_points,
            "times": [],
            "mean_time": np.nan,
            "std_time": np.nan,
            "error": "Pyro not installed"
        }


def pyro_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int,
    n_warmup: int = 500,
    repeats: int = 100,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Time Pyro HMC on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        device: PyTorch device ("cpu" or "cuda")
        
    Returns:
        Dictionary with timing results
    """
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import MCMC, HMC
        
        # Set device
        device = torch.device(device)
        
        # Convert data to PyTorch tensors
        xs = torch.tensor(dataset.xs, dtype=torch.float32, device=device)
        ys = torch.tensor(dataset.ys, dtype=torch.float32, device=device)
        
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
        
        # Create HMC kernel
        hmc_kernel = HMC(polynomial_model, step_size=0.01, num_steps=20)
        
        # Warm-up run
        mcmc = MCMC(hmc_kernel, num_samples=10, warmup_steps=10)
        mcmc.run(xs, ys)
        
        # Timing runs
        times = []
        for _ in range(repeats):
            mcmc = MCMC(hmc_kernel, num_samples=n_samples, warmup_steps=n_warmup)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            mcmc.run(xs, ys)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append(time.time() - start_time)
        
        # Get final samples for validation
        mcmc = MCMC(hmc_kernel, num_samples=n_samples, warmup_steps=n_warmup)
        mcmc.run(xs, ys)
        samples = mcmc.get_samples()
        
        return {
            "framework": "pyro",
            "method": "hmc",
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "n_points": dataset.n_points,
            "times": times,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "samples": {
                "a": samples["a"].cpu().numpy(),
                "b": samples["b"].cpu().numpy(),
                "c": samples["c"].cpu().numpy(),
            },
            "diagnostics": {
                "num_samples": n_samples,
                "warmup_steps": n_warmup,
            }
        }
        
    except ImportError:
        print("Pyro not installed. Skipping Pyro benchmarks.")
        return {
            "framework": "pyro",
            "method": "hmc",
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "n_points": dataset.n_points,
            "times": [],
            "mean_time": np.nan,
            "std_time": np.nan,
            "error": "Pyro not installed"
        }