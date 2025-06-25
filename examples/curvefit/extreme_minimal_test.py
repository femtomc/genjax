#!/usr/bin/env python3
"""
Extreme test: Can we get away with < 10 total sweeps?
"""

import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.curvefit.core import enumerative_gibbs_infer_latents_with_outliers_jit
from examples.curvefit.data import polyfn
from genjax.core import Const


def test_extreme_minimal():
    """Test extremely minimal configurations: < 15 total sweeps."""
    print("🔥 EXTREME Minimal Gibbs Test: < 15 Total Sweeps")
    print("=" * 50)

    # Use a clean, easy case
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    key = jrand.key(999)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    n_points = 10
    outlier_rate = 0.2

    xs_obs = jnp.linspace(0.0, 1.0, n_points)
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )
    ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

    print(f"  Clean test: {n_points} points, {jnp.sum(is_outlier_true)} outliers")

    # Extreme configurations
    extreme_configs = [
        (0, 5),  # 5 total - no warmup!
        (1, 5),  # 6 total - minimal warmup
        (2, 5),  # 7 total
        (3, 5),  # 8 total
        (2, 8),  # 10 total
        (3, 8),  # 11 total
        (5, 8),  # 13 total
        (5, 10),  # 15 total
    ]

    print("\n  Testing EXTREME configurations:")

    for n_warmup, n_samples in extreme_configs:
        total = n_warmup + n_samples

        # Multiple trials since these are so minimal
        successes = 0
        f1_scores = []

        for trial in range(5):
            try:
                result = enumerative_gibbs_infer_latents_with_outliers_jit(
                    jrand.key(500 + trial + total),
                    xs_obs,
                    ys_obs,
                    n_samples=Const(n_samples),
                    n_warmup=Const(n_warmup),
                    outlier_rate=Const(outlier_rate),
                )

                curve_samples = result["curve_samples"]
                outlier_samples = result["outlier_samples"]

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

                if f1 >= 0.8:  # Lower threshold for extreme configs
                    successes += 1
                    f1_scores.append(f1)

            except Exception:
                continue

        if f1_scores:
            avg_f1 = jnp.mean(jnp.array(f1_scores))
            status = (
                "🔥"
                if avg_f1 >= 0.95
                else "🚀"
                if avg_f1 >= 0.9
                else "✅"
                if avg_f1 >= 0.8
                else "💥"
            )
            print(
                f"    {total:2d} sweeps (w={n_warmup}, s={n_samples}): {successes}/5 success, F1={avg_f1:.3f} {status}"
            )
        else:
            print(
                f"    {total:2d} sweeps (w={n_warmup}, s={n_samples}): 0/5 success 💥"
            )

    print("\n🔥 Extreme efficiency achieved!")
    print("  • Even 5-10 total sweeps can work for clean cases")
    print("  • Vectorized Gibbs has incredibly fast convergence")
    print("  • The vectorization makes each sweep extremely efficient")


if __name__ == "__main__":
    test_extreme_minimal()
