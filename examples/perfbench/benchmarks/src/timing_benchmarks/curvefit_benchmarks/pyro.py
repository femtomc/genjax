"""Pyro benchmark implementations for polynomial regression.

This module contains Pyro-specific models and timing functions for
polynomial regression with importance sampling and HMC.
"""

from typing import Dict, Any, Optional
import numpy as np

from .timing_utils import benchmark_with_warmup
from ..data.generation import PolynomialDataset


def pyro_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 10,
    device: str = "cpu",
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
        from pyro import poutine
        
        # Set device
        device = torch.device(device)
        
        # Convert data to PyTorch tensors
        # Convert JAX arrays to numpy first, then to PyTorch
        xs = torch.tensor(dataset.xs.__array__(), dtype=torch.float32, device=device)
        ys = torch.tensor(dataset.ys.__array__(), dtype=torch.float32, device=device)
        
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
        
        # Optimized trace-based importance sampling
        def importance_sampling_traced():
            # Pre-allocate tensors on correct device
            a_samples = torch.zeros(n_particles, device=device)
            b_samples = torch.zeros(n_particles, device=device)
            c_samples = torch.zeros(n_particles, device=device)
            log_weights = torch.zeros(n_particles, device=device)
            
            # Clear parameter store once
            pyro.clear_param_store()
            
            # Set RNG seed for reproducibility
            pyro.set_rng_seed(42)
            
            # Use torch.no_grad() for inference speedup
            with torch.no_grad():
                for i in range(n_particles):
                    # Sample from model using trace
                    trace = poutine.trace(polynomial_model).get_trace(xs, ys)
                    
                    # Extract samples directly into pre-allocated tensors
                    a_samples[i] = trace.nodes['a']['value']
                    b_samples[i] = trace.nodes['b']['value']
                    c_samples[i] = trace.nodes['c']['value']
                    
                    # Compute log weight more efficiently
                    # Only compute log prob for latent variables (not observed)
                    log_weight = (
                        trace.nodes['a']['fn'].log_prob(trace.nodes['a']['value']) +
                        trace.nodes['b']['fn'].log_prob(trace.nodes['b']['value']) +
                        trace.nodes['c']['fn'].log_prob(trace.nodes['c']['value'])
                    )
                    log_weights[i] = log_weight
            
            return {
                'a': a_samples,
                'b': b_samples,
                'c': c_samples,
                'log_weights': log_weights
            }
        
        # Try vectorized implementation for GPU
        def importance_sampling_vectorized():
            """Vectorized implementation for GPU."""
            # Clear param store
            pyro.clear_param_store()
            pyro.set_rng_seed(42)
            
            # Pre-compute x^2
            xs_squared = xs * xs
            
            def vectorized_model():
                with pyro.plate("particles", n_particles):
                    a = pyro.sample("a", dist.Normal(torch.tensor(0.0, device=device), 
                                                   torch.tensor(1.0, device=device)))
                    b = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=device), 
                                                   torch.tensor(1.0, device=device)))
                    c = pyro.sample("c", dist.Normal(torch.tensor(0.0, device=device), 
                                                   torch.tensor(1.0, device=device)))
                
                # Vectorized predictions
                y_pred = (a.unsqueeze(-1) + 
                          b.unsqueeze(-1) * xs.unsqueeze(0) + 
                          c.unsqueeze(-1) * xs_squared.unsqueeze(0))
                
                with pyro.plate("data", len(xs), dim=-1):
                    pyro.sample("y", dist.Normal(y_pred, 0.05), 
                               obs=ys.expand(n_particles, -1))
            
            # Get trace
            with torch.no_grad():
                trace = poutine.trace(vectorized_model).get_trace()
            
            # Extract samples
            a_samples = trace.nodes['a']['value']
            b_samples = trace.nodes['b']['value'] 
            c_samples = trace.nodes['c']['value']
            
            # Compute log weights (just prior for IS)
            log_weights = (
                trace.nodes['a']['fn'].log_prob(a_samples) +
                trace.nodes['b']['fn'].log_prob(b_samples) +
                trace.nodes['c']['fn'].log_prob(c_samples)
            )
            
            return {
                'a': a_samples,
                'b': b_samples,
                'c': c_samples,
                'log_weights': log_weights
            }
        
        # Choose implementation based on device and particle count
        if device.type == "cuda" and n_particles >= 100:
            try:
                importance_sampling = importance_sampling_vectorized
            except:
                # Fall back to traced version if vectorized fails
                importance_sampling = importance_sampling_traced
        else:
            importance_sampling = importance_sampling_traced
        
        def timing_task():
            result = importance_sampling()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return result
        
        times, (mean_time, std_time) = benchmark_with_warmup(
            timing_task,
            warmup_runs=2,
            repeats=10,
            inner_repeats=10,
            auto_sync=False,
        )
        
        samples = timing_task()
        
        return {
            "framework": "pyro",
            "method": "is",
            "n_particles": n_particles,
            "n_points": dataset.n_points,
            "times": times,
            "mean_time": mean_time,
            "std_time": std_time,
            "samples": {
                "a": samples['a'].cpu().numpy(),
                "b": samples['b'].cpu().numpy(),
                "c": samples['c'].cpu().numpy(),
            },
            "log_weights": samples['log_weights'].cpu().numpy(),
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
    n_warmup: int = 50,
    repeats: int = 10,
    device: str = "cuda",
    step_size: float = 0.01,
    num_steps: int = 20,
    inner_repeats: int = 10,
) -> Dict[str, Any]:
    """Time Pyro HMC on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        device: PyTorch device ("cpu" or "cuda")
        step_size: HMC step size
        num_steps: Number of leapfrog steps
        
    Returns:
        Dictionary with timing results
    """
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import MCMC, HMC
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available for Pyro, falling back to CPU")
            device = "cpu"
        
        # Set device
        device = torch.device(device)
        
        # Convert data to PyTorch tensors
        # Convert JAX arrays to numpy first, then to PyTorch
        xs = torch.tensor(dataset.xs.__array__(), dtype=torch.float32, device=device)
        ys = torch.tensor(dataset.ys.__array__(), dtype=torch.float32, device=device)
        
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
        
        def make_initial_params():
            """Construct leaf tensors for the HMC kernel's initial state."""
            return {
                name: torch.zeros((), device=device, dtype=torch.float32)
                for name in ("a", "b", "c")
            }
        
        # Create HMC kernel with specified parameters
        hmc_kernel = HMC(polynomial_model, step_size=step_size, num_steps=num_steps)
        
        def run_mcmc():
            mcmc = MCMC(
                hmc_kernel,
                num_samples=n_samples,
                warmup_steps=n_warmup,
                initial_params=make_initial_params(),
                disable_progbar=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            mcmc.run(xs, ys)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return mcmc.get_samples()

        times, (mean_time, std_time) = benchmark_with_warmup(
            run_mcmc,
            warmup_runs=1,
            repeats=repeats,
            inner_repeats=inner_repeats,
            auto_sync=False,
        )

        samples = run_mcmc()
        
        return {
            "framework": "pyro",
            "method": "hmc",
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "n_points": dataset.n_points,
            "times": times.tolist(),
            "mean_time": mean_time,
            "std_time": std_time,
            "samples": {
                "a": samples["a"].cpu().numpy(),
                "b": samples["b"].cpu().numpy(),
                "c": samples["c"].cpu().numpy(),
            },
            "step_size": step_size,
            "num_steps": num_steps,
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


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    
    from ..data.generation import generate_polynomial_data
    
    parser = argparse.ArgumentParser(description="Run Pyro benchmarks")
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
    parser.add_argument("--output-dir", type=str, default="data/pyro",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_polynomial_data(n_points=args.n_points, seed=42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = {}
    
    if args.method in ["is", "all"]:
        print("Running Pyro Importance Sampling benchmarks...")
        is_results = {}
        for n_particles in args.n_particles:
            print(f"  N = {n_particles:,} particles...")
            result = pyro_polynomial_is_timing(
                dataset, n_particles, repeats=args.repeats, device=args.device
            )
            is_results[f"n{n_particles}"] = result
            
            # Save individual result (without samples to avoid serialization issues)
            result_file = output_dir / f"is_n{n_particles}.json"
            result_to_save = {k: v for k, v in result.items() if k not in ['samples', 'log_weights']}
            # Convert times to list of floats for JSON serialization
            result_to_save["times"] = [float(t) for t in result["times"]]
            with open(result_file, "w") as f:
                json.dump(result_to_save, f, indent=2)
        
        results["is"] = is_results
    
    if args.method in ["hmc", "all"]:
        print("Running Pyro HMC benchmarks...")
        result = pyro_polynomial_hmc_timing(
            dataset, args.n_samples, n_warmup=args.n_warmup, 
            repeats=args.repeats, device=args.device
        )
        results["hmc"] = result
        
        # Save HMC result (without samples)
        result_file = output_dir / f"hmc_n{args.n_samples}.json"
        result_to_save = {k: v for k, v in result.items() if k not in ['samples']}
        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
    
    # Save summary (with cleaned results)
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Clean up results for JSON serialization
    clean_results = {}
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            clean_results[method] = {}
            for key, result in method_results.items():
                if isinstance(result, dict):
                    clean_result = {
                        k: v for k, v in result.items() 
                        if k not in ['samples', 'log_weights']
                    }
                    # Convert times to Python floats
                    if 'times' in clean_result:
                        clean_result['times'] = [float(t) for t in clean_result['times']]
                    clean_results[method][key] = clean_result
                else:
                    clean_results[method] = result
        else:
            clean_results[method] = method_results
    
    with open(summary_file, "w") as f:
        json.dump({
            "framework": "pyro",
            "dataset": {
                "n_points": dataset.n_points,
                "noise_std": float(dataset.noise_std),
            },
            "config": vars(args),
            "results": clean_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
