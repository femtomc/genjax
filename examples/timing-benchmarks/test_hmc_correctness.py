#!/usr/bin/env python
"""Test HMC correctness across implementations."""

import os
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import jax.random as jrand
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data

# Generate test data
dataset = generate_polynomial_data(n_points=50, seed=42)
xs, ys = dataset.xs, dataset.ys

print("Testing HMC implementations...")
print("="*60)

# Test 1: Check if handcoded is computing correct number of gradient evaluations
print("\n1. Counting gradient evaluations per HMC step:")

# Count gradients in handcoded implementation
grad_count = 0

def count_grads(f):
    def wrapped(*args, **kwargs):
        global grad_count
        grad_count += 1
        return f(*args, **kwargs)
    return wrapped

# Simplified version to count gradients
def log_joint(params):
    a, b, c = params[0], params[1], params[2]
    y_pred = a + b * xs + c * xs**2
    log_lik = jax.scipy.stats.norm.logpdf(ys, y_pred, 0.05).sum()
    log_prior = jax.scipy.stats.norm.logpdf(params, 0., 1.).sum()
    return log_lik + log_prior

# Test leapfrog with gradient counting
def leapfrog_with_count(q, p, step_size, n_leapfrog):
    global grad_count
    grad_count = 0
    
    # Initial half step
    grad = jax.grad(log_joint)(q)
    grad_count += 1
    p = p + 0.5 * step_size * grad
    
    # Full steps
    for _ in range(n_leapfrog - 1):
        q = q + step_size * p
        grad = jax.grad(log_joint)(q)
        grad_count += 1
        p = p + step_size * grad
    
    # Final steps
    q = q + step_size * p
    grad = jax.grad(log_joint)(q)
    grad_count += 1
    p = p + 0.5 * step_size * grad
    
    return q, p, grad_count

# Test with initial values
q_test = jnp.array([0.1, 0.2, 0.3])
p_test = jnp.array([0.5, -0.3, 0.7])

q_new, p_new, n_grads = leapfrog_with_count(q_test, p_test, 0.01, 20)
print(f"Handcoded leapfrog: {n_grads} gradient evaluations for 20 leapfrog steps")
print(f"Expected: 21 (initial + 19 full + final)")

# Test 2: Check total HMC operations
print("\n2. Testing total operations for 100 HMC samples:")
print(f"- Warmup steps: 500")
print(f"- Sample steps: 100")
print(f"- Total HMC iterations: 600")
print(f"- Leapfrog steps per iteration: 20")
print(f"- Total leapfrog steps: 600 * 20 = 12,000")
print(f"- Total gradient evaluations: ~600 * 21 = 12,600")

# Test 3: Compare execution patterns
print("\n3. Checking JAX compilation behavior:")

# Create traced version
@jax.jit
def leapfrog_jitted(q, p, step_size, n_leapfrog):
    # This will unroll the loop at compile time
    grad = jax.grad(log_joint)(q)
    p = p + 0.5 * step_size * grad
    
    for _ in range(n_leapfrog - 1):
        q = q + step_size * p
        grad = jax.grad(log_joint)(q)
        p = p + step_size * grad
    
    q = q + step_size * p
    grad = jax.grad(log_joint)(q)
    p = p + 0.5 * step_size * grad
    
    return q, p

# Time a single leapfrog
import time

# Warm up
_ = leapfrog_jitted(q_test, p_test, 0.01, 20)
jax.block_until_ready(_[0])

# Time it
start = time.time()
for _ in range(1000):
    q_out, p_out = leapfrog_jitted(q_test, p_test, 0.01, 20)
    jax.block_until_ready(q_out)
elapsed = time.time() - start

print(f"1000 leapfrog integrations took: {elapsed:.3f}s")
print(f"Per leapfrog: {elapsed/1000*1000:.3f}ms")

# Test 4: Check if scan vs loop makes a difference
print("\n4. Testing scan vs loop implementation:")

def leapfrog_scan(q, p, step_size, n_leapfrog):
    """Leapfrog using scan instead of Python loop."""
    # Initial half step
    grad = jax.grad(log_joint)(q)
    p = p + 0.5 * step_size * grad
    
    # Define scan function
    def scan_fn(carry, _):
        q, p = carry
        q = q + step_size * p
        grad = jax.grad(log_joint)(q)
        p = p + step_size * grad
        return (q, p), None
    
    # Run scan for n_leapfrog-1 steps
    (q, p), _ = jax.lax.scan(scan_fn, (q, p), None, length=n_leapfrog-1)
    
    # Final steps
    q = q + step_size * p
    grad = jax.grad(log_joint)(q)
    p = p + 0.5 * step_size * grad
    
    return q, p

leapfrog_scan_jit = jax.jit(leapfrog_scan)

# Warm up
_ = leapfrog_scan_jit(q_test, p_test, 0.01, 20)
jax.block_until_ready(_[0])

# Time it
start = time.time()
for _ in range(1000):
    q_out, p_out = leapfrog_scan_jit(q_test, p_test, 0.01, 20)
    jax.block_until_ready(q_out)
elapsed_scan = time.time() - start

print(f"1000 scan-based leapfrogs took: {elapsed_scan:.3f}s")
print(f"Per leapfrog: {elapsed_scan/1000*1000:.3f}ms")
print(f"Speedup from scan: {elapsed/elapsed_scan:.2f}x")

print("\n" + "="*60)
print("CONCLUSION:")
print("The handcoded implementation may be faster because:")
print("1. Python loops get unrolled at JAX compile time (no overhead)")
print("2. Direct gradient calls may be more optimized than framework abstractions")
print("3. Minimal overhead from trace management or other framework features")