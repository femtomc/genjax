#!/usr/bin/env python3
"""
Quick test: What's the absolute minimum for typical outlier rates?
"""

import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.curvefit.core import enumerative_gibbs_infer_latents_with_outliers_jit
from examples.curvefit.data import polyfn
from genjax.core import Const
import time


def test_minimal_configs():
    """Test very aggressive minimal configurations."""
    print("🚀 Testing Aggressive Minimal Gibbs Configurations")
    print("=" * 50)

    # Generate easier test case (lower outlier rate)
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    key = jrand.key(42)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    n_points = 15
    outlier_rate = 0.25  # More typical outlier rate

    xs_obs = jnp.linspace(0.0, 1.0, n_points)
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )
    ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

    print(
        f"  Test data: {n_points} points, {jnp.sum(is_outlier_true)} outliers ({outlier_rate * 100:.0f}%)"
    )

    # Test very aggressive configurations
    configs = [
        (5, 10),  # 15 total - extremely minimal
        (10, 15),  # 25 total - very minimal
        (10, 25),  # 35 total - minimal
        (15, 35),  # 50 total - fast
        (25, 50),  # 75 total - reasonable
        (50, 100),  # 150 total - conservative
    ]

    print("\n  Testing ultra-minimal configurations:")

    best_minimal = None

    for n_warmup, n_samples in configs:
        total = n_warmup + n_samples

        # Run Gibbs
        start = time.time()
        result = enumerative_gibbs_infer_latents_with_outliers_jit(
            jrand.key(42 + total),
            xs_obs,
            ys_obs,
            n_samples=Const(n_samples),
            n_warmup=Const(n_warmup),
            outlier_rate=Const(outlier_rate),
        )
        jax.block_until_ready(result)
        runtime = (time.time() - start) * 1000

        # Evaluate
        curve_samples = result["curve_samples"]
        outlier_samples = result["outlier_samples"]

        a_mean = jnp.mean(curve_samples["a"])
        b_mean = jnp.mean(curve_samples["b"])
        c_mean = jnp.mean(curve_samples["c"])

        param_mse = (
            (a_mean - true_a) ** 2 + (b_mean - true_b) ** 2 + (c_mean - true_c) ** 2
        )

        outlier_probs = jnp.mean(outlier_samples, axis=0)
        predicted_outliers = outlier_probs > 0.5

        tp = jnp.sum(predicted_outliers & is_outlier_true)
        fp = jnp.sum(predicted_outliers & ~is_outlier_true)
        fn = jnp.sum(~predicted_outliers & is_outlier_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        status = "✅" if f1 >= 0.9 else "⚠️" if f1 >= 0.8 else "❌"

        print(
            f"    {total:3d} sweeps (w={n_warmup:2d}, s={n_samples:2d}): F1={f1:.3f}, MSE={param_mse:.3f}, {runtime:5.1f}ms {status}"
        )

        # Track first config with good performance
        if f1 >= 0.9 and best_minimal is None:
            best_minimal = {
                "total": total,
                "warmup": n_warmup,
                "samples": n_samples,
                "f1": f1,
                "mse": param_mse,
                "runtime": runtime,
            }

    if best_minimal:
        print("\n🎯 Minimum viable for 90% F1:")
        print(
            f"  • {best_minimal['total']} total sweeps (warmup={best_minimal['warmup']}, samples={best_minimal['samples']})"
        )
        print(
            f"  • Performance: F1={best_minimal['f1']:.3f}, MSE={best_minimal['mse']:.3f}"
        )
        print(f"  • Runtime: {best_minimal['runtime']:.1f}ms")
    else:
        print("\n⚠️ No configuration achieved 90% F1 - need more sweeps!")

    return best_minimal


def test_very_fast_gibbs():
    """Test if we can get away with ~50 total sweeps for good performance."""
    print("\n🏃 Ultra-Fast Gibbs Challenge: Can we do it in ~50 sweeps?")
    print("=" * 55)

    # Easier case: fewer points, lower outlier rate
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    key = jrand.key(123)  # Different seed
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    n_points = 10  # Smaller problem
    outlier_rate = 0.2  # Lower outlier rate

    xs_obs = jnp.linspace(0.0, 1.0, n_points)
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )
    ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

    print(
        f"  Easier test: {n_points} points, {jnp.sum(is_outlier_true)} outliers ({outlier_rate * 100:.0f}%)"
    )

    # Focus on ~50 sweep configurations
    fast_configs = [
        (10, 25),  # 35 total
        (15, 30),  # 45 total
        (20, 35),  # 55 total
        (25, 40),  # 65 total
    ]

    print("\n  Testing ultra-fast configurations:")

    for n_warmup, n_samples in fast_configs:
        total = n_warmup + n_samples

        # Run multiple trials for robustness
        f1_scores = []
        mse_scores = []

        for trial in range(3):
            result = enumerative_gibbs_infer_latents_with_outliers_jit(
                jrand.key(200 + trial + total),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(outlier_rate),
            )

            curve_samples = result["curve_samples"]
            outlier_samples = result["outlier_samples"]

            a_mean = jnp.mean(curve_samples["a"])
            b_mean = jnp.mean(curve_samples["b"])
            c_mean = jnp.mean(curve_samples["c"])

            param_mse = (
                (a_mean - true_a) ** 2 + (b_mean - true_b) ** 2 + (c_mean - true_c) ** 2
            )

            outlier_probs = jnp.mean(outlier_samples, axis=0)
            predicted_outliers = outlier_probs > 0.5

            tp = jnp.sum(predicted_outliers & is_outlier_true)
            fp = jnp.sum(predicted_outliers & ~is_outlier_true)
            fn = jnp.sum(~predicted_outliers & is_outlier_true)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            f1_scores.append(f1)
            mse_scores.append(param_mse)

        avg_f1 = jnp.mean(jnp.array(f1_scores))
        avg_mse = jnp.mean(jnp.array(mse_scores))
        std_f1 = jnp.std(jnp.array(f1_scores))

        status = "🚀" if avg_f1 >= 0.95 else "✅" if avg_f1 >= 0.9 else "⚠️"

        print(
            f"    {total:2d} sweeps: F1={avg_f1:.3f}±{std_f1:.3f}, MSE={avg_mse:.3f} {status}"
        )


if __name__ == "__main__":
    # Test 1: Find minimum for typical case
    best = test_minimal_configs()

    # Test 2: Challenge ultra-fast configs
    test_very_fast_gibbs()

    print("\n🏆 Summary:")
    if best:
        print(f"  • For typical cases (25% outliers): {best['total']} sweeps minimum")
        print(f"  • Runtime: {best['runtime']:.1f}ms for excellent performance")
    print("  • For easier cases (20% outliers): ~45-55 sweeps may suffice")
    print("  • Vectorized Gibbs is remarkably efficient!")
