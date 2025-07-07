"""Diagnostic script to identify GenJAX overhead sources."""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import gen, normal, seed
from genjax import modular_vmap as vmap
from genjax.timing import benchmark_with_warmup
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

# Test parameters
N_PARTICLES = 10000
N_POINTS = 50
REPEATS = 100

# Generate test data
key = jrand.key(42)
xs = jnp.linspace(-2, 2, N_POINTS)
true_a, true_b, true_c = 0.5, -0.3, 0.2
ys = true_a + true_b * xs + true_c * xs**2 + 0.05 * jrand.normal(key, (N_POINTS,))


print("GenJAX Overhead Diagnostic")
print("=" * 60)
print(f"Configuration: {N_PARTICLES} particles, {N_POINTS} data points")
print()


# 1. Test raw JAX operations
print("1. Raw JAX operations (baseline)")
print("-" * 40)

def raw_jax_is():
    """Pure JAX implementation without any abstractions."""
    keys = jrand.split(key, N_PARTICLES)
    
    def sample_particle(k):
        k1, k2, k3 = jrand.split(k, 3)
        a = jrand.normal(k1) * 1.0 + 0.0  # N(0, 1)
        b = jrand.normal(k2) * 1.0 + 0.0
        c = jrand.normal(k3) * 1.0 + 0.0
        
        y_pred = a + b * xs + c * xs**2
        log_lik = -0.5 * jnp.sum((ys - y_pred)**2 / (0.05**2)) - N_POINTS * jnp.log(0.05 * jnp.sqrt(2 * jnp.pi))
        return log_lik
    
    return jax.vmap(sample_particle)(keys)

raw_jax_jit = jax.jit(raw_jax_is)
_ = raw_jax_jit()  # warmup
_, (raw_time, raw_std) = benchmark_with_warmup(
    lambda: raw_jax_jit().block_until_ready(),
    repeats=REPEATS, auto_sync=False
)
print(f"Raw JAX: {raw_time*1000:.3f} ± {raw_std*1000:.3f} ms")


# 2. Test with TFP distributions
print("\n2. JAX + TFP distributions")
print("-" * 40)

def tfp_is():
    """JAX with TFP distributions."""
    keys = jrand.split(key, N_PARTICLES)
    prior = tfd.Normal(loc=0.0, scale=1.0)
    
    def sample_particle(k):
        k1, k2, k3 = jrand.split(k, 3)
        a = prior.sample(seed=k1)
        b = prior.sample(seed=k2)
        c = prior.sample(seed=k3)
        
        y_pred = a + b * xs + c * xs**2
        likelihood = tfd.Normal(loc=y_pred, scale=0.05)
        return jnp.sum(likelihood.log_prob(ys))
    
    return jax.vmap(sample_particle)(keys)

tfp_jit = jax.jit(tfp_is)
_ = tfp_jit()  # warmup
_, (tfp_time, tfp_std) = benchmark_with_warmup(
    lambda: tfp_jit().block_until_ready(),
    repeats=REPEATS, auto_sync=False
)
print(f"JAX + TFP: {tfp_time*1000:.3f} ± {tfp_std*1000:.3f} ms")
print(f"Overhead vs raw: {(tfp_time/raw_time - 1)*100:.1f}%")


# 3. Test GenJAX distributions only (no gen functions)
print("\n3. GenJAX distributions only")
print("-" * 40)

def genjax_dist_only():
    """Using GenJAX distributions without gen functions."""
    keys = jrand.split(key, N_PARTICLES)
    
    def sample_particle(k):
        k1, k2, k3 = jrand.split(k, 3)
        # Use the seed function to handle key properly
        a = seed(lambda k: normal.sample(0.0, 1.0))(k1)
        b = seed(lambda k: normal.sample(0.0, 1.0))(k2)
        c = seed(lambda k: normal.sample(0.0, 1.0))(k3)
        
        y_pred = a + b * xs + c * xs**2
        # Sum of individual normal logpdfs
        log_lik = jnp.sum(normal.logpdf(ys, y_pred, 0.05))
        return log_lik
    
    return jax.vmap(sample_particle)(keys)

genjax_dist_jit = jax.jit(genjax_dist_only)
_ = genjax_dist_jit()  # warmup
_, (dist_time, dist_std) = benchmark_with_warmup(
    lambda: genjax_dist_jit().block_until_ready(),
    repeats=REPEATS, auto_sync=False
)
print(f"GenJAX dists: {dist_time*1000:.3f} ± {dist_std*1000:.3f} ms")
print(f"Overhead vs raw: {(dist_time/raw_time - 1)*100:.1f}%")


# 4. Test minimal gen function
print("\n4. Minimal gen function (no vmap in model)")
print("-" * 40)

@gen
def minimal_model(xs, ys):
    """Minimal model without internal vmap."""
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b" 
    c = normal(0.0, 1.0) @ "c"
    
    # Manually compute likelihood
    y_pred = a + b * xs + c * xs**2
    # Return total log likelihood as a traced value
    return jnp.sum(normal.logpdf(ys, y_pred, 0.05))

def minimal_gen_is():
    """IS with minimal gen function."""
    def importance_weight(_):
        _, weight = minimal_model.generate({}, xs, ys)
        return weight
    
    return seed(vmap(importance_weight, axis_size=N_PARTICLES))(key, None)

minimal_jit = jax.jit(minimal_gen_is)
_ = minimal_jit()  # warmup
_, (minimal_time, minimal_std) = benchmark_with_warmup(
    lambda: minimal_jit().block_until_ready(),
    repeats=REPEATS, auto_sync=False
)
print(f"Minimal gen: {minimal_time*1000:.3f} ± {minimal_std*1000:.3f} ms")
print(f"Overhead vs raw: {(minimal_time/raw_time - 1)*100:.1f}%")


# 5. Test full gen function with vmap
print("\n5. Full gen function (with normal.vmap)")
print("-" * 40)

@gen
def full_model(xs):
    """Full model with vectorized observations."""
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    
    y_pred = a + b * xs + c * xs**2
    ys = normal.vmap(in_axes=(0, None))(y_pred, 0.05) @ "ys"
    return ys

def full_gen_is():
    """IS with full gen function."""
    constraints = {"ys": ys}
    
    def importance_sample(constraints):
        _, weight = full_model.generate(constraints, xs)
        return weight
    
    return seed(vmap(importance_sample, axis_size=N_PARTICLES, in_axes=None))(key, constraints)

full_jit = jax.jit(full_gen_is)
_ = full_jit()  # warmup
_, (full_time, full_std) = benchmark_with_warmup(
    lambda: full_jit().block_until_ready(),
    repeats=REPEATS, auto_sync=False
)
print(f"Full gen: {full_time*1000:.3f} ± {full_std*1000:.3f} ms")
print(f"Overhead vs raw: {(full_time/raw_time - 1)*100:.1f}%")


# 6. Test with trace extraction
print("\n6. Full gen with trace extraction")
print("-" * 40)

def full_gen_traces():
    """IS returning full traces."""
    constraints = {"ys": ys}
    
    def importance_sample(constraints):
        trace, weight = full_model.generate(constraints, xs)
        return trace, weight
    
    return seed(vmap(importance_sample, axis_size=N_PARTICLES, in_axes=None))(key, constraints)

traces_jit = jax.jit(full_gen_traces)
_ = traces_jit()  # warmup
_, (traces_time, traces_std) = benchmark_with_warmup(
    lambda: traces_jit()[1].block_until_ready(),  # block on weights only
    repeats=REPEATS, auto_sync=False
)
print(f"With traces: {traces_time*1000:.3f} ± {traces_std*1000:.3f} ms")
print(f"Overhead vs raw: {(traces_time/raw_time - 1)*100:.1f}%")


# Summary
print("\n" + "=" * 60)
print("SUMMARY (all times in milliseconds)")
print("=" * 60)
print(f"Raw JAX:           {raw_time*1000:7.3f} ms (1.0x baseline)")
print(f"JAX + TFP:         {tfp_time*1000:7.3f} ms ({tfp_time/raw_time:3.1f}x)")
print(f"GenJAX dists only: {dist_time*1000:7.3f} ms ({dist_time/raw_time:3.1f}x)")
print(f"Minimal gen func:  {minimal_time*1000:7.3f} ms ({minimal_time/raw_time:3.1f}x)")
print(f"Full gen func:     {full_time*1000:7.3f} ms ({full_time/raw_time:3.1f}x)")
print(f"Gen with traces:   {traces_time*1000:7.3f} ms ({traces_time/raw_time:3.1f}x)")

print("\nOverhead breakdown:")
print(f"- TFP distributions: {(tfp_time/raw_time - 1)*100:5.1f}%")
print(f"- GenJAX distributions: {(dist_time/raw_time - 1)*100:5.1f}%") 
print(f"- Gen function machinery: {(minimal_time/dist_time - 1)*100:5.1f}%")
print(f"- Vectorized observations: {(full_time/minimal_time - 1)*100:5.1f}%")
print(f"- Trace extraction: {(traces_time/full_time - 1)*100:5.1f}%")