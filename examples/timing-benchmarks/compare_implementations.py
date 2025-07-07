"""Compare different GenJAX implementations to identify overhead."""

import jax
import jax.numpy as jnp
import jax.random as jrand
from pathlib import Path
import sys

# Add the timing benchmarks to path
sys.path.append(str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data
from timing_benchmarks.benchmarks.genjax import genjax_polynomial_is_timing
from timing_benchmarks.benchmarks.genjax_optimized import (
    genjax_polynomial_is_direct,
    genjax_polynomial_is_minimal,
)
from timing_benchmarks.benchmarks.handcoded_tfp import handcoded_tfp_polynomial_is_timing

# Test configuration
N_PARTICLES = 10000
N_POINTS = 50
REPEATS = 100

# Generate dataset
dataset = generate_polynomial_data(n_points=N_POINTS, seed=42)

print("GenJAX Implementation Comparison")
print("=" * 60)
print(f"Configuration: {N_PARTICLES} particles, {N_POINTS} data points, {REPEATS} repeats")
print()

# 1. Handcoded TFP baseline
print("1. Handcoded JAX + TFP (baseline)")
print("-" * 40)
tfp_result = handcoded_tfp_polynomial_is_timing(dataset, N_PARTICLES, repeats=REPEATS)
tfp_time = tfp_result["mean_time"]
print(f"Time: {tfp_time*1000:.3f} ± {tfp_result['std_time']*1000:.3f} ms")

# 2. Original GenJAX implementation
print("\n2. Original GenJAX implementation")
print("-" * 40)
original_result = genjax_polynomial_is_timing(dataset, N_PARTICLES, repeats=REPEATS)
original_time = original_result["mean_time"]
print(f"Time: {original_time*1000:.3f} ± {original_result['std_time']*1000:.3f} ms")
print(f"Overhead vs TFP: {(original_time/tfp_time):.1f}x")

# 3. Optimized GenJAX (direct pattern)
print("\n3. Optimized GenJAX (faircoin pattern)")
print("-" * 40)
optimized_result = genjax_polynomial_is_direct(dataset, N_PARTICLES, repeats=REPEATS)
optimized_time = optimized_result["mean_time"]
print(f"Time: {optimized_time*1000:.3f} ± {optimized_result['std_time']*1000:.3f} ms")
print(f"Overhead vs TFP: {(optimized_time/tfp_time):.1f}x")
print(f"Improvement vs original: {(original_time/optimized_time - 1)*100:.1f}%")

# 4. Minimal GenJAX (weights only)
print("\n4. Minimal GenJAX (weights only)")
print("-" * 40)
minimal_result = genjax_polynomial_is_minimal(dataset, N_PARTICLES, repeats=REPEATS)
minimal_time = minimal_result["mean_time"]
print(f"Time: {minimal_time*1000:.3f} ± {minimal_result['std_time']*1000:.3f} ms")
print(f"Overhead vs TFP: {(minimal_time/tfp_time):.1f}x")
print(f"Improvement vs original: {(original_time/minimal_time - 1)*100:.1f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Implementation':<30} {'Time (ms)':<12} {'vs TFP':<10} {'vs Original':<12}")
print("-" * 64)
print(f"{'Handcoded TFP':<30} {tfp_time*1000:>9.3f} ms {'':<10} {'':<12}")
print(f"{'Original GenJAX':<30} {original_time*1000:>9.3f} ms {original_time/tfp_time:>8.1f}x {'':<12}")
print(f"{'Optimized GenJAX':<30} {optimized_time*1000:>9.3f} ms {optimized_time/tfp_time:>8.1f}x {(1-optimized_time/original_time)*100:>10.1f}%")
print(f"{'Minimal GenJAX':<30} {minimal_time*1000:>9.3f} ms {minimal_time/tfp_time:>8.1f}x {(1-minimal_time/original_time)*100:>10.1f}%")

# Check if results are correct
print("\n" + "=" * 60)
print("CORRECTNESS CHECK")
print("=" * 60)

# Compare log marginal likelihoods
from jax.scipy.special import logsumexp
tfp_lml = logsumexp(tfp_result["log_weights"]) - jnp.log(N_PARTICLES)
original_lml = logsumexp(original_result["log_weights"]) - jnp.log(N_PARTICLES)
optimized_lml = logsumexp(optimized_result["log_weights"]) - jnp.log(N_PARTICLES)
minimal_lml = logsumexp(minimal_result["log_weights"]) - jnp.log(N_PARTICLES)

print(f"Log marginal likelihood estimates:")
print(f"  Handcoded TFP:  {tfp_lml:.3f}")
print(f"  Original:       {original_lml:.3f} (diff: {abs(original_lml - tfp_lml):.3f})")
print(f"  Optimized:      {optimized_lml:.3f} (diff: {abs(optimized_lml - tfp_lml):.3f})")
print(f"  Minimal:        {minimal_lml:.3f} (diff: {abs(minimal_lml - tfp_lml):.3f})")

print("\nAll implementations produce similar results ✓" if max(
    abs(original_lml - tfp_lml),
    abs(optimized_lml - tfp_lml),
    abs(minimal_lml - tfp_lml)
) < 1.0 else "WARNING: Results differ significantly!")