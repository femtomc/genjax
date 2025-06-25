#!/usr/bin/env python3
"""
Study: What's the minimum number of Gibbs sweeps for good results?

Tests different combinations of warmup and sampling to find the efficiency frontier.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax

from examples.viz import (
    setup_publication_fonts,
    FIGURE_SIZES,
    get_method_color,
    apply_grid_style,
    apply_standard_ticks,
)
from examples.curvefit.core import (
    enumerative_gibbs_infer_latents_with_outliers_jit,
    infer_latents_with_outliers_jit,
)
from examples.curvefit.data import polyfn
from genjax.core import Const
import time


def _copy_to_main_figs(filename):
    """Copy figure to main figs directory."""
    import shutil
    import os

    src = f"examples/curvefit/figs/{filename}"
    dst = f"../../figs/{filename}"

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    try:
        shutil.copy2(src, dst)
        print(f"  ✓ Copied to main figs: {filename}")
    except Exception as e:
        print(f"  ⚠ Failed to copy {filename}: {e}")


def generate_challenging_outlier_data(n_points=20, outlier_rate=0.4, seed=42):
    """Generate challenging data with many outliers to stress-test convergence."""
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    key = jrand.key(seed)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    xs_obs = jnp.linspace(0.0, 1.0, n_points)
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )
    ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

    return {
        "xs": xs_obs,
        "ys": ys_obs,
        "is_outlier_true": is_outlier_true,
        "true_params": {"a": true_a, "b": true_b, "c": true_c},
        "n_true_outliers": jnp.sum(is_outlier_true),
    }


def evaluate_gibbs_performance(result, data):
    """Evaluate Gibbs performance on detection and parameter estimation."""
    # Extract results
    curve_samples = result["curve_samples"]
    outlier_samples = result["outlier_samples"]

    # Parameter estimation
    a_mean = jnp.mean(curve_samples["a"])
    b_mean = jnp.mean(curve_samples["b"])
    c_mean = jnp.mean(curve_samples["c"])

    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    param_mse = (a_mean - true_a) ** 2 + (b_mean - true_b) ** 2 + (c_mean - true_c) ** 2

    # Outlier detection
    outlier_probs = jnp.mean(outlier_samples, axis=0)
    predicted_outliers = outlier_probs > 0.5
    is_outlier_true = data["is_outlier_true"]

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

    return {
        "param_mse": param_mse,
        "detection_f1": f1,
        "precision": precision,
        "recall": recall,
    }


def study_gibbs_efficiency_frontier(seed=42):
    """
    Find the efficiency frontier: minimum sweeps for good performance.

    Tests different (warmup, sampling) combinations systematically.
    """
    print("\n=== Gibbs Efficiency Frontier Study ===")

    # Generate challenging test data
    data = generate_challenging_outlier_data(n_points=20, outlier_rate=0.4, seed=seed)
    print(
        f"  Generated challenging data: {data['n_true_outliers']}/{len(data['xs'])} outliers (40%)"
    )

    # Test different configurations
    configs = [
        # (n_warmup, n_samples, label)
        (10, 25, "Minimal (35 total)"),
        (20, 50, "Very Fast (70 total)"),
        (50, 100, "Fast (150 total)"),
        (100, 200, "Standard (300 total)"),
        (200, 300, "Conservative (500 total)"),
        (500, 500, "Thorough (1000 total)"),
    ]

    results = []

    print("\n  Testing configurations:")
    for n_warmup, n_samples, label in configs:
        print(f"    {label}: warmup={n_warmup}, samples={n_samples}")

        # Run Gibbs with this configuration
        start_time = time.time()
        gibbs_result = enumerative_gibbs_infer_latents_with_outliers_jit(
            jrand.key(seed + n_warmup + n_samples),
            data["xs"],
            data["ys"],
            n_samples=Const(n_samples),
            n_warmup=Const(n_warmup),
            outlier_rate=Const(0.4),
        )
        jax.block_until_ready(gibbs_result)
        runtime = time.time() - start_time

        # Evaluate performance
        perf = evaluate_gibbs_performance(gibbs_result, data)

        results.append(
            {
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "total_sweeps": n_warmup + n_samples,
                "label": label,
                "runtime": runtime * 1000,  # Convert to ms
                **perf,
            }
        )

        print(
            f"      F1: {perf['detection_f1']:.3f}, MSE: {perf['param_mse']:.3f}, Time: {runtime * 1000:.1f}ms"
        )

    return results


def save_efficiency_frontier_figure(results, seed=42):
    """Create figure showing the efficiency frontier."""
    print("\n  Creating efficiency frontier figure...")

    setup_publication_fonts()
    fig, ((ax_f1, ax_mse), (ax_time, ax_pareto)) = plt.subplots(
        2, 2, figsize=FIGURE_SIZES["framework_comparison"]
    )

    # Extract data
    total_sweeps = [r["total_sweeps"] for r in results]
    f1_scores = [r["detection_f1"] for r in results]
    mse_values = [r["param_mse"] for r in results]
    runtimes = [r["runtime"] for r in results]
    labels = [r["label"] for r in results]

    # Colors from fast (red) to slow (green)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results)))

    # Panel 1: F1 vs Total Sweeps
    ax_f1.plot(
        total_sweeps,
        f1_scores,
        "o-",
        linewidth=2,
        markersize=8,
        color=get_method_color("genjax_hmc"),
    )

    # Highlight the "good enough" threshold
    ax_f1.axhline(
        0.95, color="red", linestyle="--", alpha=0.7, label="95% F1 threshold"
    )

    # Find first config that hits 95% F1
    good_configs = [r for r in results if r["detection_f1"] >= 0.95]
    if good_configs:
        best_config = min(good_configs, key=lambda x: x["total_sweeps"])
        ax_f1.scatter(
            [best_config["total_sweeps"]],
            [best_config["detection_f1"]],
            s=200,
            color="red",
            marker="*",
            zorder=10,
            label="Minimum for 95% F1",
        )
        print(
            f"  Minimum config for 95% F1: {best_config['label']} ({best_config['total_sweeps']} sweeps)"
        )

    ax_f1.set_xlabel("Total Gibbs Sweeps", fontweight="bold")
    ax_f1.set_ylabel("Outlier Detection F1", fontweight="bold")
    ax_f1.legend()
    apply_grid_style(ax_f1)
    apply_standard_ticks(ax_f1)

    # Panel 2: MSE vs Total Sweeps
    ax_mse.plot(
        total_sweeps,
        mse_values,
        "s-",
        linewidth=2,
        markersize=8,
        color=get_method_color("genjax_is"),
    )

    # Show diminishing returns
    ax_mse.set_xlabel("Total Gibbs Sweeps", fontweight="bold")
    ax_mse.set_ylabel("Parameter MSE", fontweight="bold")
    apply_grid_style(ax_mse)
    apply_standard_ticks(ax_mse)

    # Panel 3: Runtime vs Total Sweeps
    ax_time.plot(
        total_sweeps,
        runtimes,
        "^-",
        linewidth=2,
        markersize=8,
        color=get_method_color("data_points"),
    )

    ax_time.set_xlabel("Total Gibbs Sweeps", fontweight="bold")
    ax_time.set_ylabel("Runtime (ms)", fontweight="bold")
    apply_grid_style(ax_time)
    apply_standard_ticks(ax_time)

    # Panel 4: Efficiency frontier (F1 vs Runtime)
    for i, (f1, runtime, label) in enumerate(zip(f1_scores, runtimes, labels)):
        ax_pareto.scatter(
            [runtime],
            [f1],
            s=150,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )

        # Label the efficient configurations
        if f1 > 0.9 or runtime < 100:  # Label high-performance or fast configs
            ax_pareto.annotate(
                f"{total_sweeps[i]}",
                (runtime, f1),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

    # Connect points to show the frontier
    sorted_by_runtime = sorted(zip(runtimes, f1_scores), key=lambda x: x[0])
    runtime_sorted, f1_sorted = zip(*sorted_by_runtime)
    ax_pareto.plot(runtime_sorted, f1_sorted, "-", color="gray", alpha=0.5, linewidth=1)

    ax_pareto.set_xlabel("Runtime (ms)", fontweight="bold")
    ax_pareto.set_ylabel("Detection F1", fontweight="bold")
    ax_pareto.set_title("Efficiency Frontier", fontweight="bold")
    apply_grid_style(ax_pareto)
    apply_standard_ticks(ax_pareto)

    plt.tight_layout()

    filename = "examples/curvefit/figs/gibbs_efficiency_frontier.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"  ✓ Saved: {filename}")
    _copy_to_main_figs("gibbs_efficiency_frontier.pdf")

    return good_configs[0] if good_configs else results[-1]


def compare_minimal_gibbs_vs_is(minimal_config, data, seed=42):
    """Compare minimal Gibbs configuration against IS baseline."""
    print("\n  Comparing minimal Gibbs vs IS baseline...")

    n_warmup = minimal_config["n_warmup"]
    n_samples = minimal_config["n_samples"]

    # Run minimal Gibbs
    gibbs_result = enumerative_gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 100),
        data["xs"],
        data["ys"],
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(0.4),
    )

    # Run IS with similar computational budget
    total_gibbs_sweeps = n_warmup + n_samples
    is_particles = (
        total_gibbs_sweeps * 2
    )  # IS should get more particles for similar compute

    is_result = infer_latents_with_outliers_jit(
        jrand.key(seed + 200),
        data["xs"],
        data["ys"],
        n_samples=Const(is_particles),
        outlier_rate=Const(0.4),
    )

    # Evaluate both
    gibbs_perf = evaluate_gibbs_performance(gibbs_result, data)

    # IS performance
    samples, log_weights = is_result
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights = weights / jnp.sum(weights)

    a_samples = samples.get_choices()["curve"]["a"]
    b_samples = samples.get_choices()["curve"]["b"]
    c_samples = samples.get_choices()["curve"]["c"]
    outlier_samples = samples.get_choices()["ys"]["is_outlier"]

    a_mean = jnp.average(a_samples, weights=weights)
    b_mean = jnp.average(b_samples, weights=weights)
    c_mean = jnp.average(c_samples, weights=weights)

    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    is_param_mse = (
        (a_mean - true_a) ** 2 + (b_mean - true_b) ** 2 + (c_mean - true_c) ** 2
    )

    outlier_probs = jnp.average(outlier_samples, weights=weights, axis=0)
    predicted_outliers = outlier_probs > 0.5
    is_outlier_true = data["is_outlier_true"]

    tp = jnp.sum(predicted_outliers & is_outlier_true)
    fp = jnp.sum(predicted_outliers & ~is_outlier_true)
    fn = jnp.sum(~predicted_outliers & is_outlier_true)

    is_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    is_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    is_f1 = (
        2 * is_precision * is_recall / (is_precision + is_recall)
        if (is_precision + is_recall) > 0
        else 0.0
    )

    is_perf = {
        "param_mse": is_param_mse,
        "detection_f1": is_f1,
        "precision": is_precision,
        "recall": is_recall,
    }

    print(
        f"    Minimal Gibbs ({total_gibbs_sweeps} sweeps): F1={gibbs_perf['detection_f1']:.3f}, MSE={gibbs_perf['param_mse']:.3f}"
    )
    print(
        f"    IS ({is_particles} particles): F1={is_perf['detection_f1']:.3f}, MSE={is_perf['param_mse']:.3f}"
    )

    return gibbs_perf, is_perf


def run_complete_efficiency_study(seed=42):
    """Run the complete Gibbs efficiency study."""
    print("🔬 Gibbs Efficiency Study: Finding the Minimum Viable Configuration")
    print("=" * 70)

    # Step 1: Find efficiency frontier
    results = study_gibbs_efficiency_frontier(seed=seed)

    # Step 2: Create visualization
    minimal_config = save_efficiency_frontier_figure(results, seed=seed)

    # Step 3: Generate test data for comparison
    data = generate_challenging_outlier_data(seed=seed)

    # Step 4: Compare minimal Gibbs vs IS
    gibbs_perf, is_perf = compare_minimal_gibbs_vs_is(minimal_config, data, seed=seed)

    print("\n✅ Efficiency Study Complete!")
    print("\n🎯 Key Findings:")
    print(f"  • Minimum viable config: {minimal_config['label']}")
    print(f"  • Total sweeps needed: {minimal_config['total_sweeps']}")
    print(
        f"  • Performance: F1={minimal_config['detection_f1']:.3f}, MSE={minimal_config['param_mse']:.3f}"
    )
    print(f"  • Runtime: {minimal_config['runtime']:.1f}ms")
    print("\n📊 Efficiency Comparison:")
    print(
        f"  • Minimal Gibbs: F1={gibbs_perf['detection_f1']:.3f}, MSE={gibbs_perf['param_mse']:.3f}"
    )
    print(
        f"  • IS baseline:   F1={is_perf['detection_f1']:.3f}, MSE={is_perf['param_mse']:.3f}"
    )

    return {
        "minimal_config": minimal_config,
        "gibbs_performance": gibbs_perf,
        "is_performance": is_perf,
        "all_results": results,
    }


if __name__ == "__main__":
    run_complete_efficiency_study()
