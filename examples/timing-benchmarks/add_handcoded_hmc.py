#!/usr/bin/env python
"""Add handcoded HMC implementation to handcoded_tfp.py"""

import sys
from pathlib import Path

# Read the current handcoded_tfp.py
tfp_path = Path("src/timing_benchmarks/curvefit-benchmarks/handcoded_tfp.py")
content = tfp_path.read_text()

# Define the HMC implementation to add
hmc_impl = '''

def handcoded_tfp_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int = 1000,
    n_warmup: int = 500,
    repeats: int = 100,
    key: Optional[jax.Array] = None,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
) -> Dict[str, Any]:
    """Handcoded HMC timing for polynomial regression - direct JAX implementation."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    xs, ys = dataset.xs, dataset.ys
    n_points = len(xs)
    
    # Log joint density
    def log_joint(params):
        a, b, c = params[0], params[1], params[2]
        y_pred = a + b * xs + c * xs**2
        
        # Likelihood: Normal(y | y_pred, 0.1)
        log_lik = jax.scipy.stats.norm.logpdf(ys, y_pred, 0.1).sum()
        
        # Priors: Normal(0, 1) for all parameters
        log_prior = jax.scipy.stats.norm.logpdf(params, 0., 1.).sum()
        
        return log_lik + log_prior
    
    # HMC implementation
    def leapfrog(q, p, step_size, n_leapfrog):
        """Leapfrog integrator for HMC."""
        # Initial half step for momentum
        grad = jax.grad(log_joint)(q)
        p = p + 0.5 * step_size * grad
        
        # Full steps
        for _ in range(n_leapfrog - 1):
            q = q + step_size * p
            grad = jax.grad(log_joint)(q)
            p = p + step_size * grad
        
        # Final position update and half step for momentum
        q = q + step_size * p
        grad = jax.grad(log_joint)(q)
        p = p + 0.5 * step_size * grad
        
        return q, p
    
    def hmc_step(state, key):
        """Single HMC step."""
        q, log_p = state
        key, subkey = jax.random.split(key)
        
        # Sample momentum
        p = jax.random.normal(subkey, shape=q.shape)
        initial_energy = -log_p + 0.5 * jnp.sum(p**2)
        
        # Leapfrog integration
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog)
        
        # Compute acceptance probability
        log_p_new = log_joint(q_new)
        new_energy = -log_p_new + 0.5 * jnp.sum(p_new**2)
        
        # Metropolis accept/reject
        key, subkey = jax.random.split(key)
        accept_prob = jnp.minimum(1., jnp.exp(initial_energy - new_energy))
        accept = jax.random.uniform(subkey) < accept_prob
        
        q = jax.lax.cond(accept, lambda: q_new, lambda: q, operand=None)
        log_p = jax.lax.cond(accept, lambda: log_p_new, lambda: log_p, operand=None)
        
        return (q, log_p), q
    
    def run_hmc(key):
        # Initialize
        key, subkey = jax.random.split(key)
        q_init = jax.random.normal(subkey, shape=(3,))
        log_p_init = log_joint(q_init)
        
        # Run chain
        total_steps = n_warmup + n_samples
        keys = jax.random.split(key, total_steps + 1)
        
        # Use scan for efficiency
        (q_final, log_p_final), samples = jax.lax.scan(
            hmc_step, (q_init, log_p_init), keys[1:]
        )
        
        # Return samples after warmup
        return samples[n_warmup:]
    
    # JIT compile
    jitted_hmc = jax.jit(run_hmc)
    
    # Timing function
    def task():
        samples = jitted_hmc(key)
        jax.block_until_ready(samples)
        return samples
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(
        task, warmup_runs=3, repeats=repeats, inner_repeats=1, auto_sync=False
    )
    
    # Get final samples for validation
    samples = task()
    
    return {
        "framework": "handcoded_tfp",
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
            "a": samples[:, 0],
            "b": samples[:, 1],
            "c": samples[:, 2],
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
tfp_path.write_text(new_content)
print(f"Added handcoded HMC implementation to {tfp_path}")