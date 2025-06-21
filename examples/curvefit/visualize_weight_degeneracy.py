#!/usr/bin/env python3
"""
Visualize weight degeneracy in importance sampling for the curvefit example.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
from genjax.core import Const
from genjax.pjax import seed

from curvefit.core import infer_latents
from curvefit.data import generate_test_dataset


def visualize_weight_degeneracy(n_samples=1000, seed_val=42):
    """Create visualizations showing weight degeneracy."""
    # Generate test data
    key = jrand.key(seed_val)
    data = generate_test_dataset(seed=seed_val, n_points=20)
    xs, ys = data["xs"], data["ys"]
    true_params = data["true_params"]

    # Run importance sampling
    seeded_infer = seed(infer_latents)
    traces, log_weights = seeded_infer(key, xs, ys, Const(n_samples))

    # Normalize weights
    normalized_log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    normalized_weights = jnp.exp(normalized_log_weights)

    # Sort weights
    sorted_indices = jnp.argsort(normalized_weights)[::-1]
    sorted_weights = normalized_weights[sorted_indices]
    cumulative_weights = jnp.cumsum(sorted_weights)

    # Extract samples
    choices = traces.get_choices()
    a_samples = choices["curve"]["a"]
    b_samples = choices["curve"]["b"]
    c_samples = choices["curve"]["c"]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Importance Sampling Weight Degeneracy Analysis (N={n_samples})", fontsize=16
    )

    # Plot 1: Weight distribution (log scale)
    ax = axes[0, 0]
    ax.semilogy(range(min(100, n_samples)), sorted_weights[:100])
    ax.set_xlabel("Particle Index (sorted)")
    ax.set_ylabel("Normalized Weight")
    ax.set_title("Weight Distribution (top 100)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative weight
    ax = axes[0, 1]
    n_show = min(50, n_samples)
    ax.plot(range(n_show), cumulative_weights[:n_show], "b-", linewidth=2)
    ax.axhline(y=0.9, color="r", linestyle="--", label="90% weight")
    ax.axhline(y=0.99, color="g", linestyle="--", label="99% weight")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Cumulative Weight")
    ax.set_title("Cumulative Weight Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: ESS over particles
    ax = axes[0, 2]
    ess_values = []
    for i in range(10, n_samples, max(1, n_samples // 100)):
        sub_weights = log_weights[:i]
        sub_normalized = sub_weights - jax.scipy.special.logsumexp(sub_weights)
        sub_norm_weights = jnp.exp(sub_normalized)
        ess = 1.0 / jnp.sum(sub_norm_weights**2)
        ess_values.append(ess)

    particle_counts = list(range(10, n_samples, max(1, n_samples // 100)))
    ax.plot(particle_counts, ess_values, "b-")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Effective Sample Size")
    ax.set_title("ESS vs Number of Particles")
    ax.grid(True, alpha=0.3)

    # Plot 4-6: Parameter samples colored by weight
    param_names = ["a", "b", "c"]
    param_samples = [a_samples, b_samples, c_samples]
    param_true = [true_params["a"], true_params["b"], true_params["c"]]

    # Use log weights for coloring to better show variation
    color_weights = normalized_log_weights
    vmin, vmax = jnp.percentile(color_weights, jnp.array([1, 99]))

    for i, (name, samples, true_val) in enumerate(
        zip(param_names, param_samples, param_true)
    ):
        ax = axes[1, i]
        scatter = ax.scatter(
            range(n_samples),
            samples,
            c=color_weights,
            cmap="viridis",
            s=10,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(
            y=true_val, color="r", linestyle="--", label=f"True {name}={true_val:.3f}"
        )
        ax.set_xlabel("Particle Index")
        ax.set_ylabel(f"Parameter {name}")
        ax.set_title(f"Parameter {name} Samples (colored by log weight)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, :], location="right", pad=0.01)
    cbar.set_label("Log Normalized Weight")

    plt.tight_layout()

    # Save figure
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/weight_degeneracy_n{n_samples}.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved visualization to {output_dir}/weight_degeneracy_n{n_samples}.png")

    # Print summary statistics
    print(f"\n=== Weight Degeneracy Summary (N={n_samples}) ===")
    print(
        f"Top particle weight: {sorted_weights[0]:.2e} ({sorted_weights[0] * 100:.2f}%)"
    )
    print(f"Particles for 90% weight: {jnp.sum(cumulative_weights < 0.9) + 1}")
    print(f"Particles for 99% weight: {jnp.sum(cumulative_weights < 0.99) + 1}")
    print(f"Effective Sample Size: {1.0 / jnp.sum(normalized_weights**2):.2f}")
    print(f"Min log weight: {jnp.min(log_weights):.2f}")
    print(f"Max log weight: {jnp.max(log_weights):.2f}")
    print(f"Log weight range: {jnp.max(log_weights) - jnp.min(log_weights):.2f}")


def main():
    """Create visualizations for different sample sizes."""
    for n_samples in [100, 1000, 5000]:
        print(f"\nGenerating visualization for {n_samples} samples...")
        visualize_weight_degeneracy(n_samples=n_samples)


if __name__ == "__main__":
    main()
