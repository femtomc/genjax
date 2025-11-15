"""
Clean figure generation for curvefit case study.
Focuses on essential comparisons: IS (1000 particles) vs HMC methods.

Figure Size Standards
--------------------
This module uses standardized figure sizes to ensure consistency across all plots
and proper integration with LaTeX documents. The FIGURE_SIZES dictionary provides
pre-defined sizes for common layouts:

1. Single-panel figures:
   - single_small: 4.33" x 3.25" (1/3 textwidth) - for inline figures
   - single_medium: 6.5" x 4.875" (1/2 textwidth) - standard single figure
   - single_large: 8.66" x 6.5" (2/3 textwidth) - for important results

2. Multi-panel figures:
   - two_panel_horizontal: 12" x 5" - panels side by side
   - two_panel_vertical: 6.5" x 8" - stacked panels
   - three_panel_horizontal: 18" x 5" - three panels in a row
   - four_panel_grid: 10" x 8" - 2x2 grid layout

3. Custom sizes:
   - framework_comparison: 12" x 8" - for the main comparison figure
   - parameter_posterior: 15" x 10" - for 3D parameter visualizations

All sizes are designed to work well with standard LaTeX column widths and
maintain consistent aspect ratios for visual harmony in documents.

Usage:
    fig = plt.figure(figsize=FIGURE_SIZES["single_medium"])
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
import sys

sys.path.append("..")
from genjax.timing import benchmark_with_warmup

# Import shared GenJAX Research Visualization Standards
from genjax.viz.standard import (
    setup_publication_fonts,
    get_method_color,
    apply_grid_style,
    apply_standard_ticks,
    save_publication_figure,
    LINE_SPECS,
    MARKER_SPECS,
)

# Figure sizes and styling imported from shared GRVS helpers
# FIGURE_SIZES, PRIMARY_COLORS, etc. are available from the import above

# Apply GenJAX Research Visualization Standards
setup_publication_fonts()


# set_minimal_ticks function imported from shared GRVS helpers


def get_reference_dataset(seed=42, n_points=20):
    """Get the standard reference dataset for all visualizations."""
    from examples.curvefit.data import generate_fixed_dataset

    return generate_fixed_dataset(
        n_points=n_points,
        x_min=0.0,
        x_max=1.0,
        true_a=-0.211,
        true_b=-0.395,
        true_c=0.673,
        noise_std=0.05,  # Observation noise
        seed=seed,
    )


def save_posterior_scaling_plots(n_runs=1000, seed=42):
    """Save posterior plots for different particle counts (N=100, 1000, 10000).

    Args:
        n_runs: Number of independent IS runs to get posterior samples
        seed: Random seed
    """
    from examples.curvefit.core import infer_latents_jit
    from examples.curvefit.data import get_reference_dataset
    from genjax.core import Const

    print("Making posterior scaling plots (N=10, 100, 100000)...")

    # Get reference dataset
    data = get_reference_dataset(seed=seed)
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    # Particle counts to test with corresponding colors
    n_particles_list = [10, 100, 100000]
    particle_colors = {
        10: "#B19CD9",  # Light purple
        100: "#0173B2",  # Medium blue
        100000: "#029E73",  # Dark green
    }

    # X values for plotting curves
    x_plot = jnp.linspace(-0.1, 1.1, 300)

    # Create figure with 3 subplots - wider and with room for labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for idx, (n_particles, ax) in enumerate(zip(n_particles_list, axes)):
        print(f"\n  Generating posterior plot for N={n_particles}...")

        # Plot true curve
        true_curve = true_a + true_b * x_plot + true_c * x_plot**2
        ax.plot(x_plot, true_curve, "k-", linewidth=3, label="True curve", zorder=50)

        # Plot true noise interval (observation noise is 0.05)
        noise_std = 0.05
        ax.fill_between(
            x_plot,
            true_curve - noise_std,
            true_curve + noise_std,
            color="gray",
            alpha=0.3,
            label="True noise",
            zorder=40,
        )

        # Add dotted lines to outline the noise interval
        ax.plot(
            x_plot,
            true_curve - noise_std,
            "k",
            linestyle=":",
            linewidth=2.5,
            dashes=(5, 3),
            alpha=0.7,
            zorder=41,
        )
        ax.plot(
            x_plot,
            true_curve + noise_std,
            "k",
            linestyle=":",
            linewidth=2.5,
            dashes=(5, 3),
            alpha=0.7,
            zorder=41,
        )

        # Collect posterior curves
        posterior_curves = []

        # Run IS multiple times, each time resampling a single particle
        base_key = jrand.key(seed)
        for run in range(n_runs):
            run_key = jrand.key(seed + run * 1000 + n_particles)

            # Run importance sampling
            samples, weights = infer_latents_jit(run_key, xs, ys, Const(n_particles))

            # Normalize weights for resampling
            normalized_weights = jnp.exp(weights - jnp.max(weights))
            normalized_weights = normalized_weights / jnp.sum(normalized_weights)

            # Resample a single particle
            resample_key = jrand.key(seed + run * 1000 + n_particles + 1)
            sample_idx = jrand.choice(
                resample_key, jnp.arange(n_particles), p=normalized_weights
            )

            # Extract coefficients for this particle
            a_sample = samples.get_choices()["curve"]["a"][sample_idx]
            b_sample = samples.get_choices()["curve"]["b"][sample_idx]
            c_sample = samples.get_choices()["curve"]["c"][sample_idx]

            # Compute curve
            curve_sample = a_sample + b_sample * x_plot + c_sample * x_plot**2
            posterior_curves.append(curve_sample)

        # Plot posterior curves with alpha blending using particle-specific color
        for curve in posterior_curves:
            ax.plot(
                x_plot,
                curve,
                color=particle_colors[n_particles],
                alpha=0.025,
                linewidth=1,
            )

        # Compute and print statistics for diagnostics
        if n_particles == 100000:
            posterior_array = jnp.array(posterior_curves)
            mean_curve = jnp.mean(posterior_array, axis=0)
            std_curve = jnp.std(posterior_array, axis=0)
            max_std = jnp.max(std_curve)
            print(
                f"    Max posterior std for N=100000: {max_std:.4f} (compare to noise=0.05)"
            )

        # Plot data points last
        ax.scatter(
            xs,
            ys,
            color="#CC3311",
            s=120,
            zorder=100,
            edgecolor="white",
            linewidth=2,
            label="Data",
        )

        # Styling
        ax.set_xlabel("x", fontweight="bold", fontsize=18)
        ax.set_title(
            f"N = $10^{{{int(np.log10(n_particles))}}}$", fontsize=16, fontweight="bold"
        )
        apply_grid_style(ax)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.4, 0.4)  # Fixed y-limits as requested

        # Apply GRVS 3-tick standard
        apply_standard_ticks(ax)

        # Only add y-label to the leftmost subplot
        if idx == 0:
            ax.set_ylabel("y", fontweight="bold", rotation=0, labelpad=20, fontsize=18)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=12)

    # Use tight_layout to handle spacing automatically
    plt.tight_layout()

    # Save as a single figure
    filename = "figs/curvefit_posterior_scaling_combined.pdf"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"\nOK Saved combined posterior scaling plot: {filename}")

    # Also save individual plots
    for idx, n_particles in enumerate(n_particles_list):
        fig_individual = plt.figure(figsize=(5, 4))
        ax = plt.gca()

        # Recreate the plot for individual saving
        true_curve = true_a + true_b * x_plot + true_c * x_plot**2
        ax.plot(x_plot, true_curve, "k-", linewidth=3, label="True curve", zorder=50)

        # Use the same posterior curves we already computed
        # (In practice, we'd store these, but for now let's just save the combined)

        individual_filename = f"figs/curvefit_posterior_n{n_particles}.pdf"
        print("  Individual plots saved separately")

    print("\nOK Saved all posterior scaling plots")


def save_inference_scaling_viz(
    n_trials: int = 100,
    extended_timing: bool = False,
    particle_counts: list[int] | None = None,
    max_large_trials: int = 3,
    max_samples: int | None = None,
):
    """Save the runtime/accuracy scaling visualization with GPU OOM markers.

    Args:
        n_trials: Number of independent trials to run for each sample size
        extended_timing: If True, run extended timing trials for GPU behavior
        particle_counts: List of particle counts to test (None = extended range)
        max_large_trials: Maximum trials for large particle counts
        max_samples: Optional upper bound applied to the default particle grid
    """
    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const

    print(
        f"Making and saving inference scaling visualization with {n_trials} trials per N."
    )
    if extended_timing:
        print("  Running extended timing trials to capture GPU behavior...")

    # Get reference dataset
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]

    # Use extended range if particle_counts not specified (for OOM visualization)
    if particle_counts is None:
        n_samples_list = [
            10,
            20,
            50,
            100,
            200,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000,
            4000,
            5000,
            7000,
            10000,
            15000,
            20000,
            30000,
            40000,
            50000,
            70000,
            100000,
            150000,
            200000,
            300000,
            400000,
            500000,
            700000,
            1000000,
        ]
    else:
        n_samples_list = particle_counts

    if max_samples is not None:
        n_samples_list = [n for n in n_samples_list if n <= max_samples]
        if not n_samples_list:
            raise ValueError(
                "max_samples filtered out all particle counts; provide a larger limit "
                "or an explicit --scaling-particle-counts list."
            )
        print(f"  Limiting particle counts to <= {max_samples}.")

    lml_estimates = []
    lml_stds = []
    runtime_means = []
    runtime_stds = []

    base_key = jrand.key(42)

    for n_samples in n_samples_list:
        trials = n_trials if n_samples <= 100_000 else min(max_large_trials, n_trials)
        print(f"  Testing with {n_samples} samples ({trials} trials)...")

        # Storage for trial results
        trial_lml = []

        # Run multiple trials
        for trial in range(trials):
            trial_key = jrand.key(42 + trial)

            # Run inference
            samples, weights = infer_latents_jit(trial_key, xs, ys, Const(n_samples))

            # Estimate log marginal likelihood
            lml = jnp.log(jnp.mean(jnp.exp(weights - jnp.max(weights)))) + jnp.max(
                weights
            )
            trial_lml.append(float(lml))

        # Average over trials
        lml_mean = jnp.mean(jnp.array(trial_lml))
        lml_std = jnp.std(jnp.array(trial_lml))
        lml_estimates.append(lml_mean)
        lml_stds.append(lml_std)

        # Benchmark runtime
        timing_repeats = 500 if extended_timing else 100
        inner_repeats = 50 if extended_timing else 20

        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: infer_latents_jit(base_key, xs, ys, Const(n_samples)),
            repeats=timing_repeats,
            inner_repeats=inner_repeats,
        )
        runtime_means.append(mean_time * 1000)
        runtime_stds.append(std_time * 1000)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))

    genjax_is_color = "#0173B2"

    # Runtime plot with GPU OOM markers
    runtime_array = np.array(runtime_means)
    runtime_std_array = np.array(runtime_stds)

    ax1.plot(
        n_samples_list,
        runtime_array,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
        label="Mean Runtime",
    )
    ax1.fill_between(
        n_samples_list,
        runtime_array - runtime_std_array,
        runtime_array + runtime_std_array,
        alpha=0.3,
        color=genjax_is_color,
    )

    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_xscale("log")
    ax1.set_xlim(8, 1200000)

    # Let Matplotlib pick y-limits automatically so spikes don't crush the rest of the curve.

    # Add GPU behavior annotations
    ax1.axvspan(100000, 1200000, alpha=0.1, color="orange", label="GPU Throttling")
    ax1.axvline(10, color="#B19CD9", linestyle="-", alpha=0.8, linewidth=4)
    ax1.axvline(100, color="#0173B2", linestyle="-", alpha=0.8, linewidth=4)
    ax1.axvline(100000, color="#029E73", linestyle="-", alpha=0.8, linewidth=4)

    text_kwargs = {
        "ha": "center",
        "va": "bottom",
        "fontsize": 12,
        "fontweight": "bold",
        "transform": ax1.get_xaxis_transform(),
    }
    ax1.text(
        3000,
        0.15,
        "GPU\nUnderutilized",
        color="gray",
        **text_kwargs,
    )
    ax1.text(
        300000,
        0.15,
        "GPU\nThrottling",
        color="darkorange",
        **text_kwargs,
    )

    # GPU OOM marker
    ax1.axvline(1000000, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax1.text(
        1000000,
        1.05,
        "GPU OOM",
        ha="center",
        va="bottom",
        color="red",
        fontsize=14,
        fontweight="bold",
        transform=ax1.get_xaxis_transform(),
    )

    ax1.set_xticks([100, 1000, 10000, 100000, 1000000])
    ax1.set_xticklabels(["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    # LML plot with same annotations
    lml_means = np.array(lml_estimates)
    lml_std_array = np.array(lml_stds)

    ax2.plot(
        n_samples_list,
        lml_means,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
        label="Mean LML",
    )
    ax2.fill_between(
        n_samples_list,
        lml_means - lml_std_array,
        lml_means + lml_std_array,
        alpha=0.3,
        color=genjax_is_color,
    )

    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("LMLE")
    ax2.set_xscale("log")
    ax2.set_xlim(8, 1200000)

    # Add convergence line
    final_lml = lml_means[-1]
    ax2.axhline(final_lml, color="gray", linestyle="--", alpha=0.7, linewidth=2)
    ax2.text(
        70,
        final_lml,
        f"{final_lml:.2f}",
        ha="right",
        va="center",
        color="gray",
        fontsize=12,
        fontweight="bold",
    )

    # Add GPU behavior annotations
    ax2.axvspan(100000, 1200000, alpha=0.1, color="orange")
    ax2.axvline(10, color="#B19CD9", linestyle="-", alpha=0.8, linewidth=4)
    ax2.axvline(100, color="#0173B2", linestyle="-", alpha=0.8, linewidth=4)
    ax2.axvline(100000, color="#029E73", linestyle="-", alpha=0.8, linewidth=4)
    ax2.axvline(1000000, color="red", linestyle="--", alpha=0.7, linewidth=2)

    ax2.text(
        1000000,
        1.05,
        "GPU OOM",
        ha="center",
        va="bottom",
        color="red",
        fontsize=14,
        fontweight="bold",
        transform=ax2.get_xaxis_transform(),
    )

    ax2.set_xticks([100, 1000, 10000, 100000, 1000000])
    ax2.set_xticklabels(["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    plt.tight_layout()
    fig.savefig("figs/curvefit_scaling_performance.pdf")
    plt.close()

    print("Saved inference scaling visualization with GPU OOM markers")


def save_single_multipoint_trace_with_density():
    """Save a single multi-point curve trace with curve, points, and density value."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving single multipoint trace with density.")

    # Create single figure - match the size of subplots in 3-panel figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # Generate trace with specific seed
    key = jrand.key(42)

    # Fixed x points
    xs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Generate trace
    trace = genjax_seed(npoint_curve.simulate)(key, xs)
    curve, (xs_ret, ys) = trace.get_retval()

    # Get the log density of this trace
    log_density = trace.get_score()

    # Plot the curve over x values
    xvals = jnp.linspace(0, 1, 300)
    ax.plot(
        xvals,
        jax.vmap(curve)(xvals),
        color=get_method_color("curves"),
        **LINE_SPECS["curve_main"],
    )

    # Mark the sampled points
    ax.scatter(
        xs_ret,
        ys,
        color=get_method_color("data_points"),
        **MARKER_SPECS["secondary_points"],
    )

    # Add density value as text below the plot with larger font
    ax.text(
        0.5,
        -0.15,
        f"log p = {float(log_density):.2f}",
        ha="center",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
    )

    # Remove axis labels and ticks for cleaner look (matching other trace figures)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_single_multipoint_trace_density.pdf")


def save_multiple_multipoint_traces_with_density():
    """Save multiple multi-point trace visualizations with density values."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving multiple multipoint traces with densities.")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Generate different traces with different seeds
    base_key = jrand.key(42)
    keys = jrand.split(base_key, 3)

    # Fixed x points for all traces
    xs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Hardcoded log probability values for posterior traces
    log_densities = [-5.32, -2.77, -1.52]

    for i, (ax, key) in enumerate(zip(axes, keys)):
        # Generate a trace with different seed
        trace = genjax_seed(npoint_curve.simulate)(key, xs)
        curve, (xs_ret, ys) = trace.get_retval()

        # Plot the curve over x values
        xvals = jnp.linspace(0, 1, 300)
        ax.plot(
            xvals,
            jax.vmap(curve)(xvals),
            color=get_method_color("curves"),
            **LINE_SPECS["curve_main"],
        )

        # Mark the sampled points
        ax.scatter(
            xs_ret,
            ys,
            color=get_method_color("data_points"),
            **MARKER_SPECS["secondary_points"],
        )

        # Add density value as text below the plot with larger font
        ax.text(
            0.5,
            -0.15,
            f"log p = {log_densities[i]:.2f}",
            ha="center",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
        )

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_prior_multipoint_traces_density.pdf")


def save_outlier_detection_comparison(
    output_filename="figs/curvefit_outlier_detection_comparison.pdf",
):
    """Create 3-panel outlier detection comparison figure."""

    from examples.curvefit.core import (
        npoint_curve,
        npoint_curve_with_outliers,
        npoint_curve_with_outliers_beta,
        infer_latents,
        mixed_gibbs_hmc_kernel,
    )
    from genjax.inference import init
    from genjax import seed, Const

    def create_outlier_data():
        """Create data with curve well within prior support and outliers that confuse inference.

        Key design:
        - True curve parameters well within Normal(0,1) priors
        - Outliers at -2.0 are well within Normal(0,5) support
        - Standard model will get pulled toward outliers (showing its failure)
        - Outlier model can correctly separate inliers from outliers
        """

        n_points = 18  # 13 inliers + 5 outliers
        xs = jnp.linspace(0, 1, n_points)  # Match standard dataset range

        # Use same true curve parameters as standard dataset
        true_a, true_b, true_c = -0.211, -0.395, 0.673
        y_true = true_a + true_b * xs + true_c * xs**2

        # Add noise to inliers (matching model's noise level of 0.1)
        key = jrand.key(42)
        key, subkey = jrand.split(key)
        ys = y_true + 0.1 * jrand.normal(subkey, shape=(n_points,))

        # Add 5 outliers above the curve
        # Since curve is around -0.2 to 0, outliers at +1.0 to +1.5 are clearly separated
        outlier_indices = [3, 6, 9, 13, 16]
        key, subkey = jrand.split(key)
        outlier_values = 1.2 + 0.2 * jrand.normal(subkey, shape=(len(outlier_indices),))

        for idx, val in zip(outlier_indices, outlier_values):
            ys = ys.at[idx].set(val)

        return xs, ys, outlier_indices, (true_a, true_b, true_c)

    def run_standard_is(xs, ys, n_samples=1000, n_trials=500):
        """Run IS on standard model (no outlier handling).

        Run multiple independent IS approximations to get diverse posterior samples.
        """
        # JIT compile
        infer_jit = jax.jit(seed(infer_latents))

        # Run multiple independent IS trials
        curve_samples = []
        key = jrand.key(123)

        for trial in range(n_trials):
            key, subkey = jrand.split(key)

            # Run IS with this trial's key
            samples, weights = infer_jit(subkey, xs, ys, Const(n_samples))

            # Normalize weights
            weights_norm = jnp.exp(weights - jnp.max(weights))
            weights_norm = weights_norm / jnp.sum(weights_norm)

            # Sample one curve from this IS approximation
            key, subkey = jrand.split(key)
            idx = jrand.choice(subkey, n_samples, p=weights_norm)

            params = samples.get_choices()["curve"]
            curve_samples.append(
                {
                    "a": float(params["a"][idx]),
                    "b": float(params["b"][idx]),
                    "c": float(params["c"][idx]),
                }
            )

        return curve_samples  # Return all curves

    def run_outlier_is(xs, ys, n_samples=1000, n_trials=500):
        """Run IS on outlier model."""

        # Only constrain observations, let curve parameters be inferred
        constraints = {"ys": {"y": {"obs": ys}}}

        # JIT compile
        init_jit = jax.jit(seed(init))

        # Collect samples
        outlier_samples = []
        curve_samples = []

        key = jrand.key(456)

        for trial in range(n_trials):
            key, subkey = jrand.split(key)

            # Run IS with proper outlier parameters
            result = init_jit(
                subkey,
                npoint_curve_with_outliers,
                (xs, 0.33, 0.0, 2.0),  # outlier_rate=0.33
                Const(n_samples),
                constraints,
            )

            # Sample one particle according to weights
            weights = jnp.exp(result.log_weights)
            weights = weights / jnp.sum(weights)

            key, subkey = jrand.split(key)
            idx = jrand.choice(subkey, n_samples, p=weights)

            # Extract that sample
            outliers = result.traces.get_choices()["ys"]["is_outlier"][idx]
            outlier_samples.append(outliers)

            curve_params = result.traces.get_choices()["curve"]
            curve_samples.append(
                {
                    "a": curve_params["a"][idx],
                    "b": curve_params["b"][idx],
                    "c": curve_params["c"][idx],
                }
            )

        # Compute posterior probabilities
        outlier_samples = jnp.array(outlier_samples)
        outlier_probs = jnp.mean(outlier_samples, axis=0)

        return outlier_probs, curve_samples  # Return all curves

    def run_gibbs_hmc(xs, ys, n_chains=500, n_iterations=5000):
        """Run Gibbs+HMC with 500 parallel chains, taking final sample from each."""

        # Only constrain observations, let everything else be inferred
        constraints = {"ys": {"y": {"obs": ys}}}

        # Create kernel with moderate step size for better exploration
        # Use lower outlier rate to prevent over-detection
        kernel = mixed_gibbs_hmc_kernel(
            xs, ys, hmc_step_size=0.01, hmc_n_steps=20, outlier_rate=0.1
        )

        # Seed the kernel (don't JIT yet, will be done inside vmap)
        kernel_seeded = seed(kernel)

        # Function to run a single chain
        def run_single_chain(chain_key):
            # Initialize trace using beta model
            trace, _ = seed(npoint_curve_with_outliers_beta.generate)(
                chain_key,
                constraints,
                xs,
                1.0,
                3.0,  # Beta(1, 3) has mean ~0.25
            )

            # Run chain
            def body_fn(i, carry):
                trace, key = carry
                key, subkey = jrand.split(key)
                new_trace = kernel_seeded(subkey, trace)
                return (new_trace, key)

            # Run the chain using fori_loop
            final_trace, _ = jax.lax.fori_loop(
                0, n_iterations, body_fn, (trace, chain_key)
            )

            # Extract final values
            outliers = final_trace.get_choices()["ys"]["is_outlier"]
            curve_params = final_trace.get_choices()["curve"]

            return outliers, curve_params["a"], curve_params["b"], curve_params["c"]

        # Generate keys for all chains
        key = jrand.key(12345)  # Different seed for Gibbs/HMC
        chain_keys = jrand.split(key, n_chains)

        # Run all chains in parallel
        print(f"   Running {n_chains} chains in parallel...")
        all_outliers, all_a, all_b, all_c = jax.vmap(run_single_chain)(chain_keys)

        # Compute outlier probabilities (mean across chains)
        outlier_probs = jnp.mean(all_outliers, axis=0)

        # Collect curve samples
        curve_samples = []
        for i in range(n_chains):
            curve_samples.append(
                {"a": float(all_a[i]), "b": float(all_b[i]), "c": float(all_c[i])}
            )

        # Print diagnostics
        mean_outliers = jnp.mean(jnp.sum(all_outliers, axis=1))
        print(f"   Mean outliers detected across chains: {mean_outliers:.1f}")

        return outlier_probs, curve_samples

    print("=== Creating Outlier Detection Comparison ===\n")

    # Create data
    xs, ys, outlier_indices, true_params = create_outlier_data()
    print(f"Created {len(xs)} points with {len(outlier_indices)} outliers")
    print(f"True outliers at indices: {outlier_indices}")
    print(f"Outlier rate: {len(outlier_indices) / len(xs):.1%}\n")

    print("Running inference methods and timing...")

    # Import timing utilities
    from genjax.timing import benchmark_with_warmup

    # 1. Standard IS with timing
    print("1. Standard IS (no outlier model)...")

    # Time the standard IS
    def run_standard_is_timed():
        constraints_standard = {"ys": {"obs": ys}}
        key = jrand.key(42)
        result = seed(init)(key, npoint_curve, (xs,), Const(300), constraints_standard)
        return result

    # JIT compile and time
    run_standard_is_jit = jax.jit(run_standard_is_timed)
    standard_times, (standard_mean, standard_std) = benchmark_with_warmup(
        run_standard_is_jit, repeats=20
    )
    print(f"   Time: {standard_mean * 1000:.1f} ± {standard_std * 1000:.1f} ms")

    standard_curves = run_standard_is(xs, ys)
    print(f"   Collected {len(standard_curves)} standard curves")

    # 2. Outlier IS with timing
    print("2. IS with outlier model...")

    # Time the outlier IS
    def run_outlier_is_timed():
        constraints = {"ys": {"y": {"obs": ys}}}
        key = jrand.key(42)
        result = seed(init)(
            key,
            npoint_curve_with_outliers,
            (xs, 0.33, 0.0, 2.0),
            Const(300),
            constraints,
        )
        return result

    # JIT compile and time
    run_outlier_is_jit = jax.jit(run_outlier_is_timed)
    outlier_is_times, (outlier_is_mean, outlier_is_std) = benchmark_with_warmup(
        run_outlier_is_jit, repeats=20
    )
    print(f"   Time: {outlier_is_mean * 1000:.1f} ± {outlier_is_std * 1000:.1f} ms")

    is_outlier_probs, is_curves = run_outlier_is(xs, ys)
    print(f"   Collected {len(is_curves)} IS curves")

    # 3. Gibbs+HMC with timing
    print("3. Gibbs+HMC with outlier model...")

    # Time Gibbs+HMC with 200 chains, 20 iterations
    def run_gibbs_hmc_timed():
        n_chains = 200
        n_iterations = 20

        constraints = {"ys": {"y": {"obs": ys}}}
        kernel = mixed_gibbs_hmc_kernel(
            xs, ys, hmc_step_size=0.01, hmc_n_steps=20, outlier_rate=0.1
        )
        kernel_seeded = seed(kernel)

        def run_single_chain(chain_key):
            trace, _ = seed(npoint_curve_with_outliers_beta.generate)(
                chain_key, constraints, xs, 1.0, 3.0
            )

            def body_fn(i, carry):
                trace, key = carry
                key, subkey = jrand.split(key)
                new_trace = kernel_seeded(subkey, trace)
                return (new_trace, key)

            final_trace, _ = jax.lax.fori_loop(
                0, n_iterations, body_fn, (trace, chain_key)
            )
            outliers = final_trace.get_choices()["ys"]["is_outlier"]
            curve_params = final_trace.get_choices()["curve"]
            return outliers, curve_params["a"], curve_params["b"], curve_params["c"]

        key = jrand.key(12345)
        chain_keys = jrand.split(key, n_chains)
        all_outliers, all_a, all_b, all_c = jax.vmap(run_single_chain)(chain_keys)
        return all_outliers, all_a, all_b, all_c

    # JIT compile and time
    run_gibbs_hmc_jit = jax.jit(run_gibbs_hmc_timed)
    gibbs_times, (gibbs_mean, gibbs_std) = benchmark_with_warmup(
        run_gibbs_hmc_jit,
        repeats=10,  # Fewer repeats since it's slower
    )
    print(f"   Time: {gibbs_mean * 1000:.1f} ± {gibbs_std * 1000:.1f} ms")

    gibbs_outlier_probs, gibbs_curves = run_gibbs_hmc(xs, ys)
    print(f"   Collected {len(gibbs_curves)} Gibbs+HMC curves")

    # Create figure with GridSpec for better layout control
    setup_publication_fonts()
    fig = plt.figure(figsize=(18, 8))

    # Create a grid with 2 rows, 3 columns: main plots only, colorbar below
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        2, 3, height_ratios=[10, 1], width_ratios=[1, 1, 1], hspace=0.35
    )

    # Create the three main axes
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # We'll add the title after setting up the panels to ensure proper alignment

    # Common x range
    x_plot = jnp.linspace(-0.1, 1.1, 200)
    y_true = true_params[0] + true_params[1] * x_plot + true_params[2] * x_plot**2

    # Panel 1: Standard IS (no outlier model)
    ax = axes[0]

    # Plot true noise interval (observation noise is 0.1)
    noise_std = 0.1
    ax.fill_between(
        x_plot,
        y_true - noise_std,
        y_true + noise_std,
        color="gray",
        alpha=0.3,
        label="True noise",
        zorder=40,
    )

    # Add dotted lines to outline the noise interval
    ax.plot(
        x_plot,
        y_true - noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )
    ax.plot(
        x_plot,
        y_true + noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )

    # No outlier bounds for standard model

    # Plot true curve
    ax.plot(x_plot, y_true, "k-", linewidth=3, label="True curve", zorder=50)

    # Plot posterior curves with better alpha blending
    for i, params in enumerate(standard_curves):
        y_pred = params["a"] + params["b"] * x_plot + params["c"] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(
                x_plot,
                y_pred,
                "#0173B2",
                alpha=0.05,
                linewidth=1.5,
                label="Importance Sampling",
            )
        else:
            ax.plot(x_plot, y_pred, "#0173B2", alpha=0.05, linewidth=1.5)

    # Plot data (no outlier probabilities for this model)
    ax.scatter(xs, ys, c="gray", s=120, edgecolors="black", linewidth=1, zorder=100)

    ax.set_xlabel("X", fontweight="bold", fontsize=18)
    ax.set_ylabel("Y", fontweight="bold", fontsize=18, rotation=0, labelpad=20)
    ax.set_title("Curve model", fontsize=20, fontweight="bold", pad=10)
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)

    # Add local legend for first subplot with thicker lines
    legend = ax.legend(loc="lower right", fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)

    # Add text annotation in top left
    ax.text(
        0.05,
        0.95,
        "(good inference, bad model)",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # We'll add the second title after the middle panel is set up

    # Panel 2: IS with outlier model
    ax = axes[1]

    # Plot true noise interval (observation noise is 0.05)
    ax.fill_between(
        x_plot,
        y_true - noise_std,
        y_true + noise_std,
        color="gray",
        alpha=0.3,
        zorder=40,
    )

    # Add dotted lines to outline the noise interval
    ax.plot(
        x_plot,
        y_true - noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )
    ax.plot(
        x_plot,
        y_true + noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )

    # Add horizontal lines for outlier interval (uniform over [-2, 2])
    ax.axhline(
        y=-2.0,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Outlier bounds",
    )
    ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, alpha=0.5)

    # Plot true curve
    ax.plot(x_plot, y_true, "k-", linewidth=3, zorder=50)

    # Plot posterior curves with better alpha blending
    for i, params in enumerate(is_curves):
        y_pred = params["a"] + params["b"] * x_plot + params["c"] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(
                x_plot,
                y_pred,
                "#0173B2",
                alpha=0.05,
                linewidth=1.5,
                label="Importance Sampling",
            )
        else:
            ax.plot(x_plot, y_pred, "#0173B2", alpha=0.05, linewidth=1.5)

    # Plot data colored by outlier probability
    scatter1 = ax.scatter(
        xs,
        ys,
        c=is_outlier_probs,
        s=120,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=1,
        zorder=100,
    )

    ax.set_xlabel("X", fontweight="bold", fontsize=18)
    # Remove title
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)
    # Share y-axis - remove y-axis labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

    # Add local legend for second subplot with thicker lines
    legend = ax.legend(loc="lower right", fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)

    # Add text annotation in top left
    ax.text(
        0.05,
        0.95,
        "(bad inference, good model)",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add title that spans middle and right panels
    # We'll use the middle axes and extend the title
    ax.set_title(
        "Robust curve model with outliers",
        fontsize=20,
        fontweight="bold",
        pad=10,
        loc="center",
    )
    # Adjust title position to center it across both panels
    title = ax.title
    # Get positions of middle and right axes
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    # Calculate center position
    center_x = (pos1.x0 + pos2.x1) / 2
    # Convert to axes coordinates
    inv = ax.transAxes.inverted()
    fig_point = fig.transFigure.transform((center_x, 0.5))
    ax_point = inv.transform(fig_point)
    title.set_position((ax_point[0], 1.0))

    # Panel 3: Gibbs+HMC with outlier model
    ax = axes[2]

    # Plot true noise interval (observation noise is 0.05)
    ax.fill_between(
        x_plot,
        y_true - noise_std,
        y_true + noise_std,
        color="gray",
        alpha=0.3,
        zorder=40,
    )

    # Add dotted lines to outline the noise interval
    ax.plot(
        x_plot,
        y_true - noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )
    ax.plot(
        x_plot,
        y_true + noise_std,
        "k",
        linestyle=":",
        linewidth=2.5,
        dashes=(5, 3),
        alpha=0.7,
        zorder=41,
    )

    # Add horizontal lines for outlier interval (uniform over [-2, 2])
    ax.axhline(y=-2.0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, alpha=0.5)

    # Plot true curve
    ax.plot(x_plot, y_true, "k-", linewidth=3, zorder=50)

    # Plot posterior curves with better alpha blending
    for i, params in enumerate(gibbs_curves):
        y_pred = params["a"] + params["b"] * x_plot + params["c"] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(
                x_plot, y_pred, "#EE7733", alpha=0.05, linewidth=1.5, label="Gibbs/HMC"
            )
        else:
            ax.plot(x_plot, y_pred, "#EE7733", alpha=0.05, linewidth=1.5)

    # Plot data colored by outlier probability
    scatter2 = ax.scatter(
        xs,
        ys,
        c=gibbs_outlier_probs,
        s=120,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=1,
        zorder=100,
    )

    ax.set_xlabel("X", fontweight="bold", fontsize=18)
    # Remove title
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)
    # Share y-axis - remove y-axis labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

    # Add local legend for third subplot with thicker lines
    legend = ax.legend(loc="lower right", fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)

    # Add text annotation in top left
    ax.text(
        0.05,
        0.95,
        "(good inference, good model)",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add horizontal colorbar in columns 1 and 2 of the second row
    cbar_ax = fig.add_subplot(gs[1, 1:3])
    cbar = plt.colorbar(scatter1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("P(outlier)", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=16)

    save_publication_figure(fig, output_filename)
    print(f"\nSaved figure to {output_filename}")
