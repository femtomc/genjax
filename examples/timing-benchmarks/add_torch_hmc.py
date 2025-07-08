#!/usr/bin/env python
"""Add handcoded HMC implementation to handcoded_torch.py"""

import sys
from pathlib import Path

# Read the current handcoded_torch.py
torch_path = Path("src/timing_benchmarks/curvefit-benchmarks/handcoded_torch.py")
content = torch_path.read_text()

# Define the HMC implementation to add
hmc_impl = '''

def handcoded_torch_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int = 1000,
    n_warmup: int = 500,
    repeats: int = 100,
    key: Optional[Any] = None,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Handcoded PyTorch HMC timing for polynomial regression."""
    import torch
    
    device = torch.device(device)
    
    # Convert data to PyTorch tensors
    xs = torch.tensor(dataset.xs.__array__(), dtype=torch.float32, device=device)
    ys = torch.tensor(dataset.ys.__array__(), dtype=torch.float32, device=device)
    n_points = len(xs)
    
    # Log joint density
    def log_joint(params):
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs**2
        
        # Likelihood: Normal(y | y_pred, 0.1)
        log_lik = torch.distributions.Normal(y_pred, 0.1).log_prob(ys).sum()
        
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
    
    # Warm-up run
    _ = run_hmc()
    
    # Timing function
    def task():
        samples = run_hmc()
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
    samples = run_hmc()
    
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
'''

# Find where to insert the HMC implementation (after the imports and before if __name__)
import_end = content.find('if __name__ == "__main__":')
if import_end == -1:
    print("Could not find main block")
    sys.exit(1)

# Insert the HMC implementation
new_content = content[:import_end] + hmc_impl + "\n\n" + content[import_end:]

# Write back
torch_path.write_text(new_content)
print(f"Added handcoded PyTorch HMC implementation to {torch_path}")