#!/usr/bin/env python3
"""
Debug script for investigating importance sampling issue in curvefit example.

This script runs GenJAX importance sampling and prints detailed statistics to
diagnose why the variance appears to be zero.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax.core import Const
from genjax.pjax import seed

# Import the curvefit example components
from curvefit.core import (
    infer_latents,
    effective_sample_size,
    log_marginal_likelihood,
)
from curvefit.data import generate_test_dataset


def analyze_importance_sampling(key, xs, ys, n_samples=1000):
    """Run importance sampling and analyze the results in detail."""
    print("\n=== Importance Sampling Debug Analysis ===")
    print(f"Number of samples: {n_samples}")
    print(f"Data points: {len(xs)}")

    # Run importance sampling
    seeded_infer = seed(infer_latents)
    traces, log_weights = seeded_infer(key, xs, ys, Const(n_samples))

    # Analyze log weights
    print("\n--- Log Weight Statistics ---")
    print(f"Min log weight: {jnp.min(log_weights):.6f}")
    print(f"Max log weight: {jnp.max(log_weights):.6f}")
    print(f"Mean log weight: {jnp.mean(log_weights):.6f}")
    print(f"Std log weight: {jnp.std(log_weights):.6f}")
    print(f"Range: {jnp.max(log_weights) - jnp.min(log_weights):.6f}")

    # Check for degenerate weights
    normalized_log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    normalized_weights = jnp.exp(normalized_log_weights)

    print("\n--- Normalized Weight Statistics ---")
    print(f"Min normalized weight: {jnp.min(normalized_weights):.2e}")
    print(f"Max normalized weight: {jnp.max(normalized_weights):.2e}")
    print(f"Sum of normalized weights: {jnp.sum(normalized_weights):.6f}")

    # Check weight concentration
    sorted_weights = jnp.sort(normalized_weights)[::-1]
    cumsum_weights = jnp.cumsum(sorted_weights)
    n_particles_90 = jnp.sum(cumsum_weights < 0.9) + 1
    n_particles_99 = jnp.sum(cumsum_weights < 0.99) + 1

    print("\n--- Weight Concentration ---")
    print(
        f"Top particle weight: {sorted_weights[0]:.2e} ({sorted_weights[0] * 100:.2f}%)"
    )
    print(
        f"Top 10 particles weight: {jnp.sum(sorted_weights[:10]):.2e} ({jnp.sum(sorted_weights[:10]) * 100:.2f}%)"
    )
    print(
        f"Particles containing 90% weight: {n_particles_90} ({n_particles_90 / n_samples * 100:.1f}%)"
    )
    print(
        f"Particles containing 99% weight: {n_particles_99} ({n_particles_99 / n_samples * 100:.1f}%)"
    )

    # Effective sample size
    ess = effective_sample_size(log_weights)
    print("\n--- Effective Sample Size ---")
    print(f"ESS: {ess:.2f}")
    print(f"ESS/N: {ess / n_samples:.2f}")

    # Analyze raw samples before resampling
    choices = traces.get_choices()
    a_samples = choices["curve"]["a"]
    b_samples = choices["curve"]["b"]
    c_samples = choices["curve"]["c"]

    print("\n--- Raw Sample Statistics (before resampling) ---")
    print(
        f"Parameter a: mean={jnp.mean(a_samples):.4f}, std={jnp.std(a_samples):.4f}, "
        f"min={jnp.min(a_samples):.4f}, max={jnp.max(a_samples):.4f}"
    )
    print(
        f"Parameter b: mean={jnp.mean(b_samples):.4f}, std={jnp.std(b_samples):.4f}, "
        f"min={jnp.min(b_samples):.4f}, max={jnp.max(b_samples):.4f}"
    )
    print(
        f"Parameter c: mean={jnp.mean(c_samples):.4f}, std={jnp.std(c_samples):.4f}, "
        f"min={jnp.min(c_samples):.4f}, max={jnp.max(c_samples):.4f}"
    )

    # Check resampling
    print("\n--- Resampling Analysis ---")
    n_resample = min(100, n_samples)
    resample_key = jrand.key(456)
    indices = jrand.categorical(resample_key, log_weights, shape=(n_resample,))

    # Count unique indices
    unique_indices = jnp.unique(indices)
    print(f"Resampling {n_resample} particles")
    print(f"Number of unique particles selected: {len(unique_indices)}")
    print(f"Most frequently selected particle: {jnp.argmax(jnp.bincount(indices))}")

    # Resampled statistics
    a_resampled = a_samples[indices]
    b_resampled = b_samples[indices]
    c_resampled = c_samples[indices]

    print("\n--- Resampled Sample Statistics ---")
    print(
        f"Parameter a: mean={jnp.mean(a_resampled):.4f}, std={jnp.std(a_resampled):.4f}"
    )
    print(
        f"Parameter b: mean={jnp.mean(b_resampled):.4f}, std={jnp.std(b_resampled):.4f}"
    )
    print(
        f"Parameter c: mean={jnp.mean(c_resampled):.4f}, std={jnp.std(c_resampled):.4f}"
    )

    # Check if all samples are identical
    print("\n--- Sample Diversity Check ---")
    print(f"Number of unique a values: {len(jnp.unique(a_samples))}")
    print(f"Number of unique b values: {len(jnp.unique(b_samples))}")
    print(f"Number of unique c values: {len(jnp.unique(c_samples))}")

    # Log marginal likelihood
    log_ml = log_marginal_likelihood(log_weights)
    print("\n--- Log Marginal Likelihood ---")
    print(f"Log ML estimate: {log_ml:.6f}")

    return traces, log_weights


def main():
    """Main debug function."""
    # Set random seed
    seed_val = 42
    key = jrand.key(seed_val)

    # Generate test data
    print("Generating test data...")
    data = generate_test_dataset(seed=seed_val, n_points=20)
    xs, ys = data["xs"], data["ys"]

    print(f"Data shape: xs={xs.shape}, ys={ys.shape}")
    true_params = data["true_params"]
    print(
        f"True parameters: a={true_params['a']:.4f}, b={true_params['b']:.4f}, c={true_params['c']:.4f}"
    )

    # Test with different sample sizes
    for n_samples in [100, 1000, 5000]:
        key, subkey = jrand.split(key)
        analyze_importance_sampling(subkey, xs, ys, n_samples)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
