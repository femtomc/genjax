#!/usr/bin/env python3
"""Simple test of vectorization benefits on GPU."""

from examples.curvefit.core import (
    gibbs_infer_latents_with_outliers_jit,
    enumerative_gibbs_infer_latents_with_outliers_jit,
)
from examples.curvefit.data import polyfn
from genjax.core import Const
import jax.numpy as jnp
import jax.random as jrand
import jax
import time


def simple_benchmark(func, repeats=3):
    """Simple timing without complex benchmarking."""
    times = []
    for _ in range(repeats):
        start = time.time()
        result = func()
        jax.block_until_ready(result)  # Wait for GPU completion
        times.append(time.time() - start)
    return min(times)  # Best time


def main():
    print(f"JAX backend: {jax.devices()[0].platform}")
    print("\nSimple vectorization test with reduced samples...")

    # Generate fixed test data
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    key = jrand.key(42)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    for n_points in [10, 25, 50, 100]:
        print(f"\nTesting with {n_points} data points...")

        xs_obs = jnp.linspace(0.0, 1.0, n_points)
        y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
        noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
        is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < 0.25
        outlier_vals = jrand.uniform(
            outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
        )
        ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

        # Use smaller sample sizes for speed
        n_samples, n_warmup = 100, 50

        # Test regular Gibbs
        def run_regular():
            return gibbs_infer_latents_with_outliers_jit(
                jrand.key(42),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(0.25),
            )

        reg_time = simple_benchmark(run_regular)

        # Test enumerative Gibbs
        def run_enumerative():
            return enumerative_gibbs_infer_latents_with_outliers_jit(
                jrand.key(42),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(0.25),
            )

        enum_time = simple_benchmark(run_enumerative)

        if enum_time < reg_time:
            speedup = reg_time / enum_time
            faster = "Enumerative"
        else:
            speedup = enum_time / reg_time
            faster = "Regular"

        print(
            f"  Regular: {reg_time * 1000:.1f}ms, Enumerative: {enum_time * 1000:.1f}ms"
        )
        print(f"  {faster} is {speedup:.2f}x faster")


if __name__ == "__main__":
    main()
