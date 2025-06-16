"""Visualization functions for fair coin case study timing comparisons."""

import matplotlib.pyplot as plt
import seaborn as sns
from examples.faircoin.core import (
    genjax_timing,
    numpyro_timing,
    handcoded_timing,
    pyro_timing,
)


def timing_comparison_fig(
    num_obs=50,
    repeats=200,
    num_samples=1000,
    include_pyro=False,
):
    """Generate horizontal bar plot comparing framework performance.

    Args:
        num_obs: Number of observations in the model
        repeats: Number of timing repetitions
        num_samples: Number of importance samples
        include_pyro: Whether to include Pyro in comparison
    """
    sns.set_style("white")

    print("Running GenJAX timing...")
    gj_times, (gj_mu, gj_std) = genjax_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )

    print("Running NumPyro timing...")
    np_times, (np_mu, np_std) = numpyro_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )

    print("Running Handcoded timing...")
    hc_times, (hc_mu, hc_std) = handcoded_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )

    pyro_results = None
    if include_pyro:
        print("Running Pyro timing...")
        try:
            pyro_times, (pyro_mu, pyro_std) = pyro_timing(
                repeats=repeats,
                num_obs=num_obs,
                num_samples=num_samples,
            )
            pyro_results = (pyro_times, pyro_mu, pyro_std)
        except Exception as e:
            print(f"Pyro timing failed: {e}")
            include_pyro = False

    if pyro_results:
        print(
            f"GenJAX: {gj_mu:.6f}s, Handcoded: {hc_mu:.6f}s, NumPyro: {np_mu:.6f}s, Pyro: {pyro_results[1]:.6f}s"
        )
    else:
        print(f"GenJAX: {gj_mu:.6f}s, Handcoded: {hc_mu:.6f}s, NumPyro: {np_mu:.6f}s")

    # Calculate relative performance compared to handcoded (baseline)
    frameworks = ["Handcoded", "GenJAX", "NumPyro"]
    times = [hc_mu, gj_mu, np_mu]
    colors = ["gold", "deepskyblue", "coral"]

    if include_pyro and pyro_results:
        frameworks.append("Pyro")
        times.append(pyro_results[1])
        colors.append("mediumpurple")

    # Calculate percentage relative to handcoded baseline
    relative_times = [(t / hc_mu) * 100 for t in times]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=240)

    y_pos = range(len(frameworks))
    bars = ax.barh(
        y_pos, relative_times, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frameworks)
    ax.set_xlabel("Relative Performance (% of Handcoded JAX time)")
    ax.set_title(
        "Probabilistic Programming Framework Performance\n(Beta-Bernoulli Importance Sampling)"
    )

    # Add a vertical line at 100% (handcoded baseline)
    ax.axvline(x=100, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(102, len(frameworks) - 0.5, "Handcoded\nBaseline", fontsize=9, alpha=0.7)

    # Add percentage labels on bars
    for i, (bar, rel_time, abs_time) in enumerate(zip(bars, relative_times, times)):
        width = bar.get_width()
        label = f"{rel_time:.1f}% ({abs_time * 1000:.2f}ms)"
        ax.text(
            width + max(relative_times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
        )

    # Set x-axis limits with some padding
    ax.set_xlim(0, max(relative_times) * 1.2)

    # Invert y-axis so handcoded is at the top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save with appropriate filename
    filename = (
        "examples/faircoin/figs/comparison_with_pyro.pdf"
        if (include_pyro and pyro_results)
        else "examples/faircoin/figs/comparison.pdf"
    )
    plt.savefig(filename)
    print(f"Saved comparison plot to {filename}")
