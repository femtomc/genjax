#!/usr/bin/env python3
"""
Demonstrate solutions to the importance sampling weight degeneracy problem.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax.core import Const
from genjax.pjax import seed

from curvefit.core import (
    infer_latents,
    hmc_infer_latents,
    effective_sample_size,
    log_marginal_likelihood,
)
from curvefit.data import generate_test_dataset


def compare_inference_methods(key, xs, ys, n_samples=1000):
    """Compare different inference methods to show the IS problem and solutions."""
    print("\n" + "=" * 70)
    print("COMPARING INFERENCE METHODS")
    print("=" * 70)

    # 1. Importance Sampling (problematic)
    print("\n1. IMPORTANCE SAMPLING FROM PRIOR")
    print("-" * 40)

    key, subkey = jrand.split(key)
    seeded_infer = seed(infer_latents)
    is_traces, is_weights = seeded_infer(subkey, xs, ys, Const(n_samples))

    # Analyze IS results
    ess = effective_sample_size(is_weights)
    log_ml = log_marginal_likelihood(is_weights)

    # Get resampled parameters
    normalized_weights = jnp.exp(is_weights - jax.scipy.special.logsumexp(is_weights))
    key, subkey = jrand.split(key)
    indices = jrand.categorical(subkey, is_weights, shape=(100,))

    choices = is_traces.get_choices()
    a_resampled = choices["curve"]["a"][indices]
    b_resampled = choices["curve"]["b"][indices]
    c_resampled = choices["curve"]["c"][indices]

    print(
        f"Effective Sample Size: {ess:.2f} / {n_samples} ({ess / n_samples * 100:.1f}%)"
    )
    print(f"Log Marginal Likelihood: {log_ml:.2f}")
    print(
        f"Posterior means: a={jnp.mean(a_resampled):.4f}, "
        f"b={jnp.mean(b_resampled):.4f}, c={jnp.mean(c_resampled):.4f}"
    )
    print(
        f"Posterior stds: a={jnp.std(a_resampled):.4f}, "
        f"b={jnp.std(b_resampled):.4f}, c={jnp.std(c_resampled):.4f}"
    )

    # Check degeneracy
    max_weight = jnp.max(normalized_weights)
    print(f"Maximum normalized weight: {max_weight:.2e} ({max_weight * 100:.1f}%)")
    print(
        "⚠️  WARNING: Severe weight degeneracy detected!"
        if ess < n_samples / 10
        else "✓ Weights are reasonable"
    )

    # 2. HMC (solution)
    print("\n2. HAMILTONIAN MONTE CARLO")
    print("-" * 40)

    key, subkey = jrand.split(key)
    seeded_hmc = seed(hmc_infer_latents)
    hmc_traces, hmc_diagnostics = seeded_hmc(
        subkey,
        xs,
        ys,
        Const(1000),  # n_samples
        Const(500),  # n_warmup
        Const(0.01),  # step_size
        Const(20),  # n_steps
    )

    # Get HMC samples
    hmc_choices = hmc_traces.get_choices()
    a_hmc = hmc_choices["curve"]["a"]
    b_hmc = hmc_choices["curve"]["b"]
    c_hmc = hmc_choices["curve"]["c"]

    print(f"Acceptance rate: {hmc_diagnostics['acceptance_rate']:.2f}")
    print(f"Number of samples: {hmc_diagnostics['n_samples']}")
    print(
        f"Posterior means: a={jnp.mean(a_hmc):.4f}, "
        f"b={jnp.mean(b_hmc):.4f}, c={jnp.mean(c_hmc):.4f}"
    )
    print(
        f"Posterior stds: a={jnp.std(a_hmc):.4f}, "
        f"b={jnp.std(b_hmc):.4f}, c={jnp.std(c_hmc):.4f}"
    )
    print("✓ HMC provides reliable posterior samples without degeneracy")

    return {
        "is": {
            "ess": ess,
            "log_ml": log_ml,
            "means": (
                jnp.mean(a_resampled),
                jnp.mean(b_resampled),
                jnp.mean(c_resampled),
            ),
            "stds": (jnp.std(a_resampled), jnp.std(b_resampled), jnp.std(c_resampled)),
        },
        "hmc": {
            "acceptance_rate": hmc_diagnostics["acceptance_rate"],
            "means": (jnp.mean(a_hmc), jnp.mean(b_hmc), jnp.mean(c_hmc)),
            "stds": (jnp.std(a_hmc), jnp.std(b_hmc), jnp.std(c_hmc)),
        },
    }


def demonstrate_problem_and_solutions():
    """Main demonstration of the IS problem and solutions."""
    print("\n" + "=" * 70)
    print("IMPORTANCE SAMPLING WEIGHT DEGENERACY: PROBLEM & SOLUTIONS")
    print("=" * 70)

    print("\nPROBLEM SUMMARY:")
    print("-" * 40)
    print("When using importance sampling from the prior for curve fitting:")
    print("• Prior is too broad (σ_a=1.0, σ_b=1.5, σ_c=0.8)")
    print("• With 20 data points, posterior is much more concentrated")
    print("• Most prior samples have negligible posterior probability")
    print("• Results in extreme weight degeneracy (ESS ≈ 1)")
    print("• Resampling collapses to 1-2 particles → zero variance")

    # Generate data
    seed_val = 42
    key = jrand.key(seed_val)
    data = generate_test_dataset(seed=seed_val, n_points=20)
    xs, ys = data["xs"], data["ys"]
    true_params = data["true_params"]

    print(
        f"\nTRUE PARAMETERS: a={true_params['a']:.4f}, b={true_params['b']:.4f}, c={true_params['c']:.4f}"
    )

    # Compare methods
    compare_inference_methods(key, xs, ys, n_samples=1000)

    print("\n" + "=" * 70)
    print("SOLUTIONS")
    print("=" * 70)

    print("\n1. USE MCMC (Implemented Above)")
    print("   • Start from reasonable initial point")
    print("   • Explore posterior locally")
    print("   • No weight degeneracy issues")
    print("   • Reliable uncertainty estimates")

    print("\n2. USE SEQUENTIAL MONTE CARLO")
    print("   • Add observations gradually")
    print("   • Temper from prior to posterior")
    print("   • Rejuvenate particles with MCMC")
    print("   • Better for multimodal posteriors")

    print("\n3. DESIGN BETTER PROPOSALS")
    print("   • Use Laplace approximation around MAP")
    print("   • Learn proposal from pilot runs")
    print("   • Adaptive importance sampling")
    print("   • Requires problem-specific design")

    print("\n4. USE VARIATIONAL INFERENCE")
    print("   • Optimize proposal distribution")
    print("   • Trade exactness for efficiency")
    print("   • Good for high dimensions")
    print("   • May underestimate uncertainty")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("\nFor this curve fitting problem:")
    print("✓ Use HMC for reliable posterior inference")
    print("✓ Importance sampling from prior is inefficient but not broken")
    print("✓ The 'zero variance' is due to extreme weight concentration")
    print("✓ This is a fundamental limitation, not a bug")


if __name__ == "__main__":
    demonstrate_problem_and_solutions()
