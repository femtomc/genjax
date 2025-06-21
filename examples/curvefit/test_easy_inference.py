#!/usr/bin/env python3
"""
Test script to verify that easier inference setup works better for importance sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax.core import Const
from genjax.pjax import seed

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.curvefit.core import (
    infer_latents,
    infer_latents_easy,
    effective_sample_size,
)
from examples.curvefit.data import (
    generate_test_dataset,
    generate_easy_inference_dataset,
)


def test_standard_vs_easy():
    """Compare standard vs easy inference setup."""
    print("=" * 70)
    print("COMPARING STANDARD VS EASY INFERENCE")
    print("=" * 70)

    # 1. Standard setup (difficult for IS)
    print("\n1. STANDARD SETUP")
    print("-" * 40)

    data_std = generate_test_dataset(seed=42, n_points=10)
    xs_std = data_std["xs"]
    ys_std = data_std["ys"]

    print(f"Data points: {len(xs_std)}")
    print(
        f"True params: a={data_std['true_params']['a']:.3f}, "
        f"b={data_std['true_params']['b']:.3f}, "
        f"c={data_std['true_params']['c']:.3f}"
    )
    print(f"Noise std: {data_std['true_params']['noise_std']:.3f}")

    # Run standard inference
    seeded_infer = seed(infer_latents)
    traces_std, weights_std = seeded_infer(jrand.key(42), xs_std, ys_std, Const(1000))
    ess_std = effective_sample_size(weights_std)

    print(f"ESS: {ess_std:.2f} / 1000 ({ess_std / 1000 * 100:.1f}%)")

    # Get posterior stats
    weights_norm = jnp.exp(weights_std - jax.scipy.special.logsumexp(weights_std))
    max_weight_std = jnp.max(weights_norm)
    print(f"Max weight: {max_weight_std:.2e} ({max_weight_std * 100:.1f}%)")

    # 2. Easy setup (better for IS)
    print("\n2. EASY SETUP")
    print("-" * 40)

    data_easy = generate_easy_inference_dataset(seed=42)
    xs_easy = data_easy["xs"]
    ys_easy = data_easy["ys"]
    noise_std = data_easy["true_params"]["noise_std"]

    print(f"Data points: {len(xs_easy)}")
    print(
        f"True params: a={data_easy['true_params']['a']:.3f}, "
        f"b={data_easy['true_params']['b']:.3f}, "
        f"c={data_easy['true_params']['c']:.3f}"
    )
    print(f"Noise std: {noise_std:.3f}")

    # Run easy inference
    seeded_infer_easy = seed(infer_latents_easy)
    traces_easy, weights_easy = seeded_infer_easy(
        jrand.key(42), xs_easy, ys_easy, Const(1000), Const(noise_std)
    )
    ess_easy = effective_sample_size(weights_easy)

    print(f"ESS: {ess_easy:.2f} / 1000 ({ess_easy / 1000 * 100:.1f}%)")

    # Get posterior stats
    weights_norm_easy = jnp.exp(
        weights_easy - jax.scipy.special.logsumexp(weights_easy)
    )
    max_weight_easy = jnp.max(weights_norm_easy)
    print(f"Max weight: {max_weight_easy:.2e} ({max_weight_easy * 100:.1f}%)")

    # 3. Summary
    print("\n3. IMPROVEMENT SUMMARY")
    print("-" * 40)
    print(f"ESS improvement: {ess_easy / ess_std:.1f}x")
    print(f"Max weight reduction: {max_weight_std / max_weight_easy:.1f}x")

    if ess_easy > 10 * ess_std:
        print("✅ SUCCESS: Easy setup significantly improves importance sampling!")
    elif ess_easy > 2 * ess_std:
        print("✓ GOOD: Easy setup improves importance sampling")
    else:
        print("⚠️  WARNING: Easy setup shows limited improvement")


if __name__ == "__main__":
    test_standard_vs_easy()
