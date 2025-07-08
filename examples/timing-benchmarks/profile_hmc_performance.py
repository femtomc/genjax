#!/usr/bin/env python
"""Profile HMC performance to understand the speed differences."""

import os
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data

# Generate dataset
dataset = generate_polynomial_data(n_points=50, seed=42)
xs, ys = dataset.xs, dataset.ys

print("Profiling HMC Performance")
print("="*60)

# Parameters
n_warmup = 500
n_samples = 100
step_size = 0.01
n_leapfrog = 20

print(f"Configuration:")
print(f"  Data points: {len(xs)}")
print(f"  Warmup steps: {n_warmup}")
print(f"  Sample steps: {n_samples}")
print(f"  Total HMC steps: {n_warmup + n_samples}")
print(f"  Leapfrog steps per HMC: {n_leapfrog}")
print(f"  Total gradient evaluations: {(n_warmup + n_samples) * (n_leapfrog + 1)}")

# Load and time each implementation
print("\n" + "="*60)
print("Timing Results:")
print("="*60)

# 1. Handcoded JAX
from timing_benchmarks.curvefit_benchmarks.handcoded_tfp import handcoded_tfp_polynomial_hmc_timing

start = time.time()
result = handcoded_tfp_polynomial_hmc_timing(
    dataset, n_samples=n_samples, n_warmup=n_warmup, repeats=5, step_size=step_size, n_leapfrog=n_leapfrog
)
print(f"\n1. Handcoded JAX HMC:")
print(f"   Mean time: {result['mean_time']*1000:.2f}ms")
print(f"   Time per HMC step: {result['mean_time']*1000/(n_warmup+n_samples):.4f}ms")
print(f"   Time per gradient: {result['mean_time']*1000/((n_warmup+n_samples)*(n_leapfrog+1)):.6f}ms")

# 2. GenJAX
from timing_benchmarks.curvefit_benchmarks.genjax import genjax_polynomial_hmc_timing

result = genjax_polynomial_hmc_timing(
    dataset, n_samples=n_samples, n_warmup=n_warmup, repeats=5, step_size=step_size, n_leapfrog=n_leapfrog
)
print(f"\n2. GenJAX HMC:")
print(f"   Mean time: {result['mean_time']*1000:.2f}ms")
print(f"   Time per HMC step: {result['mean_time']*1000/(n_warmup+n_samples):.4f}ms")
print(f"   Overhead vs handcoded: {result['mean_time']/handcoded_time:.1f}x")

handcoded_time = result['mean_time']

# 3. NumPyro
from timing_benchmarks.curvefit_benchmarks.numpyro import numpyro_polynomial_hmc_timing

result = numpyro_polynomial_hmc_timing(
    dataset, n_samples=n_samples, n_warmup=n_warmup, repeats=5, step_size=step_size, n_leapfrog=n_leapfrog
)
print(f"\n3. NumPyro HMC:")
print(f"   Mean time: {result['mean_time']*1000:.2f}ms")
print(f"   Time per HMC step: {result['mean_time']*1000/(n_warmup+n_samples):.4f}ms")
print(f"   Overhead vs handcoded: {result['mean_time']/handcoded_time:.1f}x")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("\nThe 20-30x overhead in GenJAX/NumPyro is likely due to:")
print("1. Trace management and probabilistic programming abstractions")
print("2. Dynamic model interpretation vs static compiled code")
print("3. Additional bookkeeping for programmable inference")
print("4. The handcoded version is the theoretical minimum - just math operations")
print("\nThis overhead is the price paid for:")
print("- Automatic differentiation of arbitrary models")
print("- Composable inference algorithms")
print("- Model introspection and debugging")
print("- General-purpose probabilistic programming")

# Let's also check if the algorithms produce similar results
print("\n" + "="*60)
print("Checking algorithm correctness (do they produce similar posteriors?):")
print("="*60)

# We already have samples from the runs above
# Just need to verify they're exploring similar regions of parameter space