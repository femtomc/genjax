"""GenJAX benchmark implementations for polynomial regression.

This module contains GenJAX-specific models and timing functions for
polynomial regression with importance sampling and HMC.
"""

import time
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import gen, normal, Cond, Const, sel
from genjax.core import Pytree
from genjax.pjax import seed
from genjax.inference import init, hmc, chain

# Import shared utilities
import sys
import os
# Add path to genjax/examples directory to import shared utils
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(examples_dir)
from utils import benchmark_with_warmup

from ..data.generation import PolynomialDataset, polyfn


### GenJAX Models ###

@Pytree.dataclass
class Lambda(Pytree):
    """Lambda wrapper for curve functions with JAX compatibility."""
    f: Const[any]
    dynamic_vals: jnp.ndarray
    static_vals: Const[tuple] = Const(())

    def __call__(self, x):
        # Unpack the dynamic values (coefficients)
        return self.f.value(x, *self.dynamic_vals)


@gen
def polynomial():
    """Polynomial coefficient prior model (degree 2)."""
    a = normal(0.0, 1.0) @ "a"  # Constant term
    b = normal(0.0, 1.0) @ "b"  # Linear coefficient
    c = normal(0.0, 1.0) @ "c"  # Quadratic coefficient
    return Lambda(Const(polyfn), jnp.array([a, b, c]))


@gen
def point(x, curve):
    """Single data point model with Gaussian noise."""
    y_det = curve(x)
    y_observed = normal(y_det, 0.05) @ "obs"
    return y_observed


@gen
def npoint_curve(xs):
    """N-point curve model with xs as input."""
    curve = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)


### GenJAX Inference Functions ###

def genjax_infer_is(xs, ys, n_particles: Const[int]):
    """GenJAX importance sampling inference."""
    from genjax.inference import init
    
    constraints = {"ys": {"obs": ys}}
    
    # Use SMC init for importance sampling
    result = init(
        npoint_curve,
        (xs,),
        n_particles,
        constraints,
    )
    
    return result.traces, result.log_weights


def genjax_infer_hmc(
    xs,
    ys,
    n_samples: Const[int],
    n_warmup: Const[int] = Const(500),
    step_size: Const[float] = Const(0.01),
    n_steps: Const[int] = Const(20),
):
    """GenJAX HMC inference."""
    from genjax.inference import hmc, chain
    from genjax.core import sel
    
    constraints = {"ys": {"obs": ys}}
    initial_trace, _ = npoint_curve.generate(constraints, xs)
    
    # Define HMC kernel for continuous parameters
    def hmc_kernel(trace):
        selection = sel("curve")
        return hmc(trace, selection, step_size=step_size.value, n_steps=n_steps.value)
    
    # Create MCMC chain
    hmc_chain = chain(hmc_kernel)
    
    # Run HMC with burn-in
    total_steps = n_samples.value + n_warmup.value
    result = hmc_chain(initial_trace, n_steps=Const(total_steps), burn_in=n_warmup)
    
    return result.traces, {
        "acceptance_rate": result.acceptance_rate,
        "n_samples": result.n_steps.value,
        "n_chains": result.n_chains.value,
    }


### GenJAX Timing Functions ###

def genjax_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None
) -> Dict[str, Any]:
    """Time GenJAX importance sampling on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        key: Random key (optional)
        
    Returns:
        Dictionary with timing results and samples
    """
    if key is None:
        key = jrand.key(42)
    
    xs, ys = dataset.xs, dataset.ys
    
    # JIT compile the inference function
    infer_jit = jax.jit(seed(genjax_infer_is))
    
    # Define task for benchmarking
    def task():
        return infer_jit(key, xs, ys, Const(n_particles))
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(task, repeats=repeats)
    
    # Get samples for validation
    traces, log_weights = task()
    
    # Extract samples
    samples_a = traces.get_choices()["curve"]["a"]
    samples_b = traces.get_choices()["curve"]["b"]
    samples_c = traces.get_choices()["curve"]["c"]
    
    return {
        "framework": "genjax",
        "method": "importance_sampling",
        "n_particles": n_particles,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "samples": {
            "a": samples_a,
            "b": samples_b,
            "c": samples_c,
        },
        "log_weights": log_weights,
    }


def genjax_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int,
    n_warmup: int = 500,
    repeats: int = 100,
    key: Optional[jrand.PRNGKey] = None
) -> Dict[str, Any]:
    """Time GenJAX HMC on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        key: Random key (optional)
        
    Returns:
        Dictionary with timing results and samples
    """
    if key is None:
        key = jrand.key(42)
    
    xs, ys = dataset.xs, dataset.ys
    
    # JIT compile the inference function
    infer_jit = jax.jit(seed(genjax_infer_hmc))
    
    # Define task for benchmarking
    def task():
        return infer_jit(key, xs, ys, Const(n_samples), Const(n_warmup))
    
    # Run benchmark with automatic warm-up
    times, (mean_time, std_time) = benchmark_with_warmup(task, repeats=repeats)
    
    # Get samples for validation
    traces, diagnostics = task()
    
    # Extract samples
    samples_a = traces.get_choices()["curve"]["a"]
    samples_b = traces.get_choices()["curve"]["b"]
    samples_c = traces.get_choices()["curve"]["c"]
    
    return {
        "framework": "genjax",
        "method": "hmc",
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "samples": {
            "a": samples_a,
            "b": samples_b,
            "c": samples_c,
        },
        "diagnostics": diagnostics,
    }


def genjax_polynomial_is_timing_simple(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100
) -> Dict[str, Any]:
    """Simplified GenJAX IS timing - just measure JAX operations.
    
    This is a minimal benchmark that tests raw JAX performance without
    the full GenJAX inference machinery. Useful for debugging and
    establishing a performance baseline.
    """
    key = jrand.PRNGKey(42)
    
    xs = jnp.array(dataset.xs)
    ys = jnp.array(dataset.ys)
    
    # Define a simple vectorized operation similar to IS
    @jax.jit
    def simple_is(key, xs, ys, n_particles):
        keys = jrand.split(key, n_particles)
        
        # Sample parameters
        a_samples = jax.vmap(lambda k: jrand.normal(k, shape=()))(keys)
        b_samples = jax.vmap(lambda k: jrand.normal(jrand.split(k)[1], shape=()))(keys) 
        c_samples = jax.vmap(lambda k: jrand.normal(jrand.split(k, 3)[2], shape=()))(keys)
        
        # Compute predictions for each particle
        def compute_predictions(a, b, c):
            return a + b * xs + c * xs**2
        
        y_preds = jax.vmap(compute_predictions)(a_samples, b_samples, c_samples)
        
        # Compute log weights (simplified)
        log_weights = jax.vmap(
            lambda y_pred: -0.5 * jnp.sum((y_pred - ys)**2 / 0.05**2)
        )(y_preds)
        
        return a_samples, b_samples, c_samples, log_weights
    
    # Warm-up
    _ = simple_is(key, xs, ys, n_particles)
    
    # Timing
    times = []
    for _ in range(repeats):
        start = time.time()
        a, b, c, lw = simple_is(key, xs, ys, n_particles)
        a.block_until_ready()  # Ensure computation completes
        times.append(time.time() - start)
    
    return {
        "framework": "genjax",
        "method": "importance_sampling_simple",
        "n_particles": n_particles,
        "n_points": dataset.n_points,
        "times": times,
        "mean_time": jnp.mean(jnp.array(times)),
        "std_time": jnp.std(jnp.array(times)),
        "samples": {
            "a": a,
            "b": b,
            "c": c,
        },
        "log_weights": lw,
    }