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

    # Create horizontal bar plot with larger fonts for research paper
    plt.rcParams.update({"font.size": 20})  # Set base font size
    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)  # Reduced height for thinner bars

    y_pos = range(len(frameworks))
    bars = ax.barh(
        y_pos, relative_times, color=colors, alpha=0.8, edgecolor="black", linewidth=0.8
    )

    # Customize the plot with larger fonts
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frameworks, fontsize=22)
    ax.set_xlabel("Relative Performance (% of Handcoded JAX time)", fontsize=22)

    # Add clarification in top right corner
    ax.text(
        0.98,
        0.98,
        "Smaller bar is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Customize tick labels
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=22)

    # Add a vertical line at 100% (handcoded baseline) - truncated to bar height
    ax.plot(
        [100, 100],
        [-0.5, len(frameworks) - 0.1],
        color="black",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
    )
    ax.text(
        100,
        len(frameworks),
        "Handcoded Baseline",
        fontsize=20,
        alpha=0.8,
        ha="center",
        va="top",
    )

    # Add percentage labels on bars with larger font
    for i, (bar, rel_time, abs_time) in enumerate(zip(bars, relative_times, times)):
        width = bar.get_width()
        label = f"{rel_time:.1f}% ({abs_time * 1000:.2f}ms)"
        ax.text(
            width + max(relative_times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=20,
            weight="bold",
        )

    # Set x-axis limits with some padding
    ax.set_xlim(0, max(relative_times) * 1.2)

    # Add padding below bars for baseline label and increase tick spacing
    ax.set_ylim(len(frameworks) + 0.8, -0.5)  # Extra space below for label
    ax.tick_params(axis="x", pad=15)  # Add separation between ticks and bars

    # Remove axis frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save with parametrized filename and research paper quality settings
    pyro_suffix = "_with_pyro" if (include_pyro and pyro_results) else ""
    filename = f"examples/faircoin/figs/comparison_obs{num_obs}_samples{num_samples}_repeats{repeats}{pyro_suffix}.pdf"

    plt.savefig(filename, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved comparison plot to {filename}")
