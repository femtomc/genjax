"""1D Gaussian Mixture Model benchmarks for PyTorch."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.distributions as dist
from typing import Dict, Any

# Import shared data generation from gmm_benchmarks
from .gmm_benchmarks import generate_gmm_data, DEFAULT_MEANS, DEFAULT_STDS, DEFAULT_WEIGHTS


# ============================================================================
# Handcoded PyTorch Implementation  
# ============================================================================

def torch_gmm_step(observations_torch, means_torch, stds_torch, weights_torch, device='cuda'):
    """Handcoded PyTorch: Infer component assignments."""
    # Ensure everything is on the correct device
    if not isinstance(observations_torch, torch.Tensor):
        observations_torch = torch.tensor(observations_torch, dtype=torch.float32, device=device)
    else:
        observations_torch = observations_torch.to(device)
        
    if not isinstance(means_torch, torch.Tensor):
        means_torch = torch.tensor(means_torch, dtype=torch.float32, device=device)
        stds_torch = torch.tensor(stds_torch, dtype=torch.float32, device=device)
        weights_torch = torch.tensor(weights_torch, dtype=torch.float32, device=device)
    else:
        means_torch = means_torch.to(device)
        stds_torch = stds_torch.to(device)
        weights_torch = weights_torch.to(device)
    
    n_data = len(observations_torch)
    n_components = len(means_torch)
    
    # Compute log probabilities for all data points and components
    # Shape: (n_data, n_components)
    log_probs = torch.zeros(n_data, n_components, device=device)
    
    for k in range(n_components):
        # Create normal distribution for component k
        component_dist = dist.Normal(means_torch[k], stds_torch[k])
        # Compute log prob for all data points
        log_probs[:, k] = component_dist.log_prob(observations_torch) + torch.log(weights_torch[k])
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(log_probs, dim=1)
    
    # Sample component assignments
    categorical_dist = dist.Categorical(probs)
    z_samples = categorical_dist.sample()
    
    return z_samples


# ============================================================================
# Timing Functions
# ============================================================================

def time_torch_gmm(n_data: int, n_steps: int, repeats: int = 100, device: str = 'cuda') -> Dict[str, Any]:
    """Time handcoded PyTorch GMM implementation."""
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Generate data using JAX and convert to PyTorch
    key = jax.random.PRNGKey(42)
    observations_jax, _ = generate_gmm_data(key, n_data)
    observations = torch.tensor(np.array(observations_jax), dtype=torch.float32, device=device)
    
    # Convert parameters to torch
    means = torch.tensor(np.array(DEFAULT_MEANS), dtype=torch.float32, device=device)
    stds = torch.tensor(np.array(DEFAULT_STDS), dtype=torch.float32, device=device)
    weights = torch.tensor(np.array(DEFAULT_WEIGHTS), dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = torch_gmm_step(observations, means, stds, weights, device)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Time multiple steps
    times = []
    for _ in range(repeats):
        # Take minimum of inner runs to reduce noise
        inner_times = []
        for _ in range(200):  # Inner repeats for accuracy
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_steps):
                    result = torch_gmm_step(observations, means, stds, weights, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            inner_times.append(time.perf_counter() - start)
        
        times.append(min(inner_times))
    
    times = np.array(times)
    return {
        'framework': 'handcoded_torch',
        'n_data': n_data,
        'n_steps': n_steps,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'times': times.tolist(),
        'device': device
    }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_torch_gmm_benchmarks(data_sizes: list = [100, 1000, 10000, 100000],
                            n_steps: int = 10,
                            repeats: int = 100,
                            device: str = 'cuda') -> Dict[str, list]:
    """Run PyTorch GMM benchmarks."""
    results = {
        'handcoded_torch': []
    }
    
    for n_data in data_sizes:
        print(f"\nData size: {n_data}")
        
        # Handcoded PyTorch
        print("  Running handcoded PyTorch...")
        result = time_torch_gmm(n_data, n_steps, repeats, device)
        results['handcoded_torch'].append(result)
        print(f"    Mean time: {result['mean_time']:.4f}s")
    
    return results