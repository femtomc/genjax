#!/usr/bin/env python3
"""Test vectorization benefits of enumerative Gibbs on GPU."""

from examples.curvefit.core import (
    gibbs_infer_latents_with_outliers_jit,
    enumerative_gibbs_infer_latents_with_outliers_jit,
)
from examples.curvefit.data import polyfn
from examples.utils import benchmark_with_warmup
from genjax.core import Const
import jax.numpy as jnp
import jax.random as jrand
import jax


def main():
    print("Testing vectorization benefits with larger datasets on GPU...")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")

    # Test with much larger datasets
    for n_points in [50, 100, 200, 500, 1000]:
        print(f"\nTesting with {n_points} data points...")

        # Generate data
        true_a, true_b, true_c = -0.211, -0.395, 0.673
        key = jrand.key(42)
        x_key, noise_key, outlier_key = jrand.split(key, 3)

        xs_obs = jnp.linspace(0.0, 1.0, n_points)
        y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
        noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
        is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < 0.25
        outlier_vals = jrand.uniform(
            outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
        )
        ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

        # Benchmark regular Gibbs
        def run_regular():
            return gibbs_infer_latents_with_outliers_jit(
                jrand.key(42),
                xs_obs,
                ys_obs,
                n_samples=Const(500),
                n_warmup=Const(100),
                outlier_rate=Const(0.25),
            )

        _, (reg_time, _) = benchmark_with_warmup(run_regular, repeats=3)

        # Benchmark enumerative Gibbs
        def run_enumerative():
            return enumerative_gibbs_infer_latents_with_outliers_jit(
                jrand.key(42),
                xs_obs,
                ys_obs,
                n_samples=Const(500),
                n_warmup=Const(100),
                outlier_rate=Const(0.25),
            )

        _, (enum_time, _) = benchmark_with_warmup(run_enumerative, repeats=3)

        if enum_time < reg_time:
            speedup = reg_time / enum_time
            faster_method = "Enumerative"
        else:
            speedup = enum_time / reg_time
            faster_method = "Regular"

        print(
            f"  Regular: {reg_time * 1000:.1f}ms, Enumerative: {enum_time * 1000:.1f}ms"
        )
        print(f"  {faster_method} is {speedup:.2f}x faster")


if __name__ == "__main__":
    main()
