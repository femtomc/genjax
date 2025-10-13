"""Visualization functions for fair coin case study timing comparisons."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from examples.faircoin.core import (
    genjax_timing,
    numpyro_timing,
    handcoded_timing,
    genjax_posterior_samples,
    numpyro_posterior_samples,
    handcoded_posterior_samples,
    exact_beta_posterior_stats,
)


def timing_comparison_fig(
    num_obs=50,
    repeats=200,
    num_samples=1000,
):
    """Generate horizontal bar plot comparing framework performance.

    Args:
        num_obs: Number of observations in the model
        repeats: Number of timing repetitions
        num_samples: Number of importance samples
    """
    sns.set_style("white")

    print("Running Ours timing...")
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

    print(f"Ours: {gj_mu:.6f}s, Handcoded: {hc_mu:.6f}s, NumPyro: {np_mu:.6f}s")

    # Calculate relative performance compared to handcoded (baseline)
    frameworks = ["Handcoded", "Ours", "NumPyro"]
    times = [hc_mu, gj_mu, np_mu]
    colors = ["gold", "deepskyblue", "coral"]

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

    # Removed 'Smaller bar is better' text

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
    filename = f"figs/faircoin_timing_performance_comparison_obs{num_obs}_samples{num_samples}_repeats{repeats}.pdf"

    plt.savefig(filename, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved comparison plot to {filename}")


def posterior_comparison_fig(
    num_obs=50,
    num_samples=5000,
    num_bins=50,
):
    """Generate grid of histograms comparing posterior samples across frameworks.

    Args:
        num_obs: Number of observations in the model
        num_samples: Number of posterior samples to generate
        num_bins: Number of histogram bins
    """
    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 18})  # Increased font size

    # Collect posterior samples from all methods
    print("Generating Ours posterior samples...")
    genjax_samples, genjax_weights = genjax_posterior_samples(num_obs, num_samples)

    print("Generating handcoded posterior samples...")
    handcoded_samples, handcoded_weights = handcoded_posterior_samples(
        num_obs, num_samples
    )

    print("Generating NumPyro posterior samples...")
    numpyro_samples, numpyro_weights = numpyro_posterior_samples(num_obs, num_samples)

    # Get exact posterior statistics
    alpha_post, beta_post, true_mean, true_mode, true_std = exact_beta_posterior_stats(
        num_obs
    )

    # Generate exact posterior PDF for comparison
    x_range = np.linspace(0.0, 1.0, 1000)
    exact_pdf = stats.beta.pdf(x_range, alpha_post, beta_post)

    # Set up subplot grid - squeezed vertical aspect ratio
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)  # Further reduced height

    axes = axes.flatten()

    # Colors and labels for each method
    methods = [
        ("Ours", genjax_samples, genjax_weights, "deepskyblue"),
        ("Handcoded JAX", handcoded_samples, handcoded_weights, "gold"),
        ("NumPyro", numpyro_samples, numpyro_weights, "coral"),
    ]

    # Create histograms for each method
    for i, (method_name, samples, weights, color) in enumerate(methods):
        ax = axes[i]

        # Create weighted histogram
        counts, bins, patches = ax.hist(
            samples,
            bins=num_bins,
            weights=weights,
            density=True,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{method_name}",
        )

        # Plot exact posterior PDF
        ax.plot(
            x_range, exact_pdf, "k-", linewidth=3, alpha=0.8, label="Exact Posterior"
        )

        # Add vertical line at true posterior mean
        ax.axvline(
            true_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True Mean: {true_mean:.3f}",
        )

        # Customize subplot
        ax.set_title(
            f"{method_name}", fontsize=20, fontweight="bold"
        )  # Increased title font
        ax.set_xlabel("Fairness Parameter", fontsize=18)  # Increased label font
        # ax.set_ylabel('Posterior Density', fontsize=18)  # Removed y-axis label
        ax.legend(fontsize=14, loc="upper left")  # Increased legend font
        ax.grid(False)  # Remove gridlines

        # Set consistent x-axis limits and custom ticks
        ax.set_xlim(0.0, 1.0)  # Full range from 0 to 1
        ax.set_xticks([0, 0.5, 1.0])  # Custom x-axis ticks at 0, 0.5, 1

        # Remove y-axis tick marks
        ax.set_yticks([])

        # Calculate sample statistics (for numerical comparison, but don't display on plot)
        sample_mean = np.average(samples, weights=weights)
        sample_std = np.sqrt(np.average((samples - sample_mean) ** 2, weights=weights))

    # Remove overall title - commented out
    # fig.suptitle(
    #     f'Beta-Bernoulli Posterior Comparison\n'
    #     f'({num_obs} observations, {num_samples:,} samples each)',
    #     fontsize=18, fontweight='bold', y=0.95
    # )

    # Remove information box about the exact posterior
    # exact_info = (f'Exact Posterior: Beta({alpha_post:.0f}, {beta_post:.0f})\n'
    #              f'True Mean: {true_mean:.3f}, True Std: {true_std:.3f}')
    #
    # fig.text(0.02, 0.02, exact_info, fontsize=14,  # Increased info font
    #         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    # plt.subplots_adjust(top=0.90, bottom=0.12)  # No longer need room for title

    # Save with parametrized filename
    filename = f"figs/faircoin_posterior_accuracy_comparison_obs{num_obs}_samples{num_samples}.pdf"

    plt.savefig(filename, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved posterior comparison plot to {filename}")

    # Print numerical comparison
    print("\nPosterior Statistics Comparison:")
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Mean Error':<12} {'Std Error':<10}")
    print("-" * 60)
    print(f"{'Exact':<15} {true_mean:<8.3f} {true_std:<8.3f} {'—':<12} {'—':<10}")

    for method_name, samples, weights, _ in methods:
        sample_mean = np.average(samples, weights=weights)
        sample_std = np.sqrt(np.average((samples - sample_mean) ** 2, weights=weights))
        mean_error = abs(sample_mean - true_mean)
        std_error = abs(sample_std - true_std)
        print(
            f"{method_name:<15} {sample_mean:<8.3f} {sample_std:<8.3f} {mean_error:<12.3f} {std_error:<10.3f}"
        )


def combined_comparison_fig(
    num_obs=50,
    num_samples=3000,  # Reduced from 10000
    timing_repeats=50,  # Reduced from 100
    timing_samples=500,  # Reduced from 1000
    num_bins=50,
    inner_repeats=20,  # Inner timing repeats
):
    """Generate combined figure with posterior comparison (left) and timing comparison (right).

    Args:
        num_obs: Number of observations in the model
        num_samples: Number of posterior samples to generate
        timing_repeats: Number of timing repetitions
        timing_samples: Number of importance samples for timing
        num_bins: Number of histogram bins
        inner_repeats: Number of inner timing repeats per outer repeat
    """
    # Set up plot style
    sns.set_style("white")
    plt.rcParams.update({"font.size": 18})

    # Create figure with 3x2 layout - stretched horizontally
    fig = plt.figure(figsize=(18, 8), dpi=300)

    # Create a single gridspec for the entire figure
    gs = fig.add_gridspec(
        2, 3, left=0.08, right=0.95, top=0.92, bottom=0.15, hspace=0.4, wspace=0.3
    )

    ### TOP ROW: POSTERIOR COMPARISON ###

    # Collect posterior samples from all methods (skip Pyro for 3x2 layout)
    print("Generating Ours posterior samples...")
    genjax_samples, genjax_weights = genjax_posterior_samples(num_obs, num_samples)

    print("Generating handcoded posterior samples...")
    handcoded_samples, handcoded_weights = handcoded_posterior_samples(
        num_obs, num_samples
    )

    print("Generating NumPyro posterior samples...")
    numpyro_samples, numpyro_weights = numpyro_posterior_samples(num_obs, num_samples)

    # Get exact posterior statistics
    alpha_post, beta_post, true_mean, true_mode, true_std = exact_beta_posterior_stats(
        num_obs
    )

    # Generate exact posterior PDF for comparison
    x_range = np.linspace(0.0, 1.0, 1000)
    exact_pdf = stats.beta.pdf(x_range, alpha_post, beta_post)

    # Colors and labels for each method (3 methods only)
    methods = [
        ("Ours", genjax_samples, genjax_weights, "deepskyblue"),
        ("Handcoded JAX", handcoded_samples, handcoded_weights, "gold"),
        ("NumPyro", numpyro_samples, numpyro_weights, "coral"),
    ]

    # Create posterior histograms in top row (row 0, cols 0, 1, 2)
    axes_top = []
    for i, (method_name, samples, weights, color) in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])  # Top row, column i
        axes_top.append(ax)

        # Create weighted histogram
        counts, bins, patches = ax.hist(
            samples,
            bins=num_bins,
            weights=weights,
            density=True,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{method_name}",
        )

        # Plot exact posterior PDF
        ax.plot(
            x_range, exact_pdf, "k-", linewidth=3, alpha=0.8, label="Exact Posterior"
        )

        # Add vertical line at true posterior mean
        ax.axvline(
            true_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True Mean: {true_mean:.3f}",
        )

        # Customize subplot
        ax.set_title(f"{method_name}", fontsize=20, fontweight="bold")
        ax.set_xlabel("Fairness Parameter", fontsize=18)
        ax.legend(fontsize=14, loc="upper left")
        ax.grid(False)

        # Set consistent x-axis limits and custom ticks
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0, 0.5, 1.0])

        # Remove y-axis tick marks
        ax.set_yticks([])

        # Calculate sample statistics (for numerical comparison)
        sample_mean = np.average(samples, weights=weights)
        sample_std = np.sqrt(np.average((samples - sample_mean) ** 2, weights=weights))

    # Get y-limits from all axes and set them to be the same
    y_min = min(ax.get_ylim()[0] for ax in axes_top)
    y_max = max(ax.get_ylim()[1] for ax in axes_top)
    for ax in axes_top:
        ax.set_ylim(y_min, y_max)

    ### BOTTOM ROW: TIMING COMPARISON ###

    # Span the entire bottom row for timing comparison
    ax_timing = fig.add_subplot(gs[1, :])

    # Run timing comparisons
    print("Running Ours timing...")
    gj_times, (gj_mu, gj_std) = genjax_timing(
        repeats=timing_repeats,
        num_obs=num_obs,
        num_samples=timing_samples,
        inner_repeats=inner_repeats,
    )

    print("Running NumPyro timing...")
    np_times, (np_mu, np_std) = numpyro_timing(
        repeats=timing_repeats,
        num_obs=num_obs,
        num_samples=timing_samples,
        inner_repeats=inner_repeats,
    )

    print("Running Handcoded timing...")
    hc_times, (hc_mu, hc_std) = handcoded_timing(
        repeats=timing_repeats,
        num_obs=num_obs,
        num_samples=timing_samples,
        inner_repeats=inner_repeats,
    )

    # Calculate relative performance compared to handcoded (baseline) - 3 frameworks only
    frameworks = ["Handcoded", "Ours", "NumPyro"]
    times = [hc_mu, gj_mu, np_mu]
    colors = ["gold", "deepskyblue", "coral"]

    # Calculate percentage relative to handcoded baseline
    relative_times = [(t / hc_mu) * 100 for t in times]

    # Calculate relative standard deviations for error bars
    relative_stds = [
        (gj_std / hc_mu) * 100,
        (hc_std / hc_mu) * 100,
        (np_std / hc_mu) * 100,
    ]

    # Create horizontal bar plot with error bars
    y_pos = range(len(frameworks))
    bars = ax_timing.barh(
        y_pos,
        relative_times,
        xerr=relative_stds,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    # Customize the timing plot - remove y-axis labels for better centering
    ax_timing.set_yticks([])  # Remove y-axis ticks
    ax_timing.set_xlabel("Relative Performance (% of Handcoded JAX time)", fontsize=22)

    # Removed 'Smaller bar is better' text

    # Customize tick labels
    ax_timing.tick_params(axis="x", labelsize=20)

    # Add a vertical line at 100% (handcoded baseline)
    ax_timing.plot(
        [100, 100],
        [-0.5, len(frameworks) - 0.1],
        color="black",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
    )
    ax_timing.text(
        100,
        len(frameworks),
        "Handcoded Baseline",
        fontsize=20,
        alpha=0.8,
        ha="center",
        va="top",
    )

    # Add framework names and percentage labels on bars
    timing_stds = [gj_std, hc_std, np_std]  # Standard deviations in seconds
    for i, (bar, rel_time, abs_time, rel_std, abs_std, framework) in enumerate(
        zip(bars, relative_times, times, relative_stds, timing_stds, frameworks)
    ):
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2

        # Add framework name on the left side of the bar
        ax_timing.text(
            max(relative_times) * 0.02,  # Small offset from left edge
            y_center,
            framework,
            ha="left",
            va="center",
            fontsize=20,
            weight="bold",
            color="black",  # All text in black
        )

        # Add percentage and timing on the right side of the whiskers (beyond error bars)
        whisker_end = width + rel_std  # End of the error bar
        perf_label = f"{rel_time:.1f}% ({abs_time * 1000:.3f} ± {abs_std * 1000:.3f}ms)"
        ax_timing.text(
            whisker_end + max(relative_times) * 0.01,
            y_center,
            perf_label,
            ha="left",
            va="center",
            fontsize=20,
            weight="bold",
        )

    # Set x-axis limits with extra padding for labels beyond whiskers
    max_whisker_end = max([rt + rs for rt, rs in zip(relative_times, relative_stds)])
    ax_timing.set_xlim(0, max_whisker_end * 1.3)

    # Add padding below bars for baseline label
    ax_timing.set_ylim(len(frameworks) + 0.8, -0.5)
    ax_timing.tick_params(axis="x", pad=15)

    # Remove axis frame except bottom (x-axis)
    ax_timing.spines["top"].set_visible(False)
    ax_timing.spines["right"].set_visible(False)
    ax_timing.spines["bottom"].set_visible(True)  # Keep x-axis line
    ax_timing.spines["left"].set_visible(False)

    # Print timing results (3 frameworks only)
    print(f"Ours: {gj_mu:.6f}s, Handcoded: {hc_mu:.6f}s, NumPyro: {np_mu:.6f}s")

    # Save with parametrized filename (3x2 layout without Pyro)
    filename = f"figs/faircoin_combined_posterior_and_timing_obs{num_obs}_samples{num_samples}.pdf"

    plt.savefig(filename, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved combined comparison plot to {filename}")

    # Print numerical comparison
    print("\nPosterior Statistics Comparison:")
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Mean Error':<12} {'Std Error':<10}")
    print("-" * 60)
    print(f"{'Exact':<15} {true_mean:<8.3f} {true_std:<8.3f} {'—':<12} {'—':<10}")

    for method_name, samples, weights, _ in methods:
        sample_mean = np.average(samples, weights=weights)
        sample_std = np.sqrt(np.average((samples - sample_mean) ** 2, weights=weights))
        mean_error = abs(sample_mean - true_mean)
        std_error = abs(sample_std - true_std)
        print(
            f"{method_name:<15} {sample_mean:<8.3f} {sample_std:<8.3f} {mean_error:<12.3f} {std_error:<10.3f}"
        )


def save_all_figures(num_obs=50, num_samples=5000):
    """Generate and save all faircoin figures.

    Args:
        num_obs: Number of observations
        num_samples: Number of samples for posterior comparison
    """
    print("Generating timing comparison figure...")
    timing_comparison_fig(
        num_obs=num_obs,
        repeats=100,  # Fewer repeats for faster generation
        num_samples=1000,  # Standard timing samples
    )

    print("\nGenerating posterior comparison figure...")
    posterior_comparison_fig(
        num_obs=num_obs,
        num_samples=num_samples,
    )

    print("\nGenerating combined comparison figure...")
    combined_comparison_fig(
        num_obs=num_obs,
        num_samples=num_samples,
    )

    print("\nAll faircoin figures generated successfully!")
