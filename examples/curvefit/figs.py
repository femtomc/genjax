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
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.utils import benchmark_with_warmup

# Standardized figure sizes for consistent layout
# All sizes are in inches and follow LaTeX document conventions
FIGURE_SIZES = {
    # Single-panel figures (aspect ratio ~4:3 for readability)
    "single_small": (4.33, 3.25),  # 1/3 of textwidth, for inline figures
    "single_medium": (6.5, 4.875),  # 1/2 of textwidth, standard single figure
    "single_large": (8.66, 6.5),  # 2/3 of textwidth, for important results
    # Two-panel figures (horizontal layout)
    "two_panel_horizontal": (12, 5),  # Full textwidth, panels side by side
    "two_panel_square": (12, 6),  # Full textwidth, square panels
    # Two-panel figures (vertical layout)
    "two_panel_vertical": (6.5, 8),  # Half textwidth, stacked panels
    # Three-panel figures
    "three_panel_horizontal": (18, 5),  # Extended width for 3 panels
    "three_panel_grid": (15, 10),  # 3x2 grid layout
    # Four-panel figures
    "four_panel_grid": (10, 8),  # 2x2 grid, common for comparisons
    # Custom sizes for specific visualizations
    "framework_comparison": (12, 8),  # Two stacked panels with legend
    "parameter_posterior": (15, 10),  # 3x2 grid with 3D plots
}

# Publication-quality plot settings with improved readability
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    }
)


def set_minimal_ticks(ax, x_ticks=3, y_ticks=3):
    """Set minimal number of ticks on both axes."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, prune="both"))


def get_reference_dataset(seed=42, n_points=10, use_easy=False):
    """
    Generate a dataset for use across visualizations.

    This function generates the exact same "Reference Dataset" as shown in
    the four_multipoint_traces visualization (first subplot).

    Args:
        seed: Random seed for reproducibility
        n_points: Number of data points
        use_easy: If True, use easier dataset for importance sampling

    Returns:
        dict with xs, ys, true_params, trace
    """
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    # Match the exact setup from save_four_multipoint_trace_vizs
    xs = jnp.linspace(0, 1, 10)  # Always use 10 points like the reference

    # Use seeded simulation for reproducibility
    seeded_simulate = genjax_seed(npoint_curve.simulate)

    # Generate the same key sequence as in four_multipoint_traces
    key = jrand.key(seed)
    keys = jrand.split(key, 4)

    # Use the first key (index 0) which is the "Reference Dataset"
    trace = seeded_simulate(keys[0], xs)

    # Extract the data
    curve, (xs_ret, ys) = trace.get_retval()
    choices = trace.get_choices()

    # Extract true parameters
    true_a = float(choices["curve"]["a"])
    true_b = float(choices["curve"]["b"])
    true_c = float(choices["curve"]["c"])

    return {
        "xs": xs,
        "ys": ys,
        "true_params": {
            "a": true_a,
            "b": true_b,
            "c": true_c,
            "noise_std": 0.2,  # Current model noise level
        },
        "trace": trace,
    }


def save_framework_comparison_figure(
    n_points=10,
    n_samples_is=5000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=10,
    figsize=None,
):
    """
    Create a clean framework comparison figure.

    Compares:
    - GenJAX IS with 5000 particles
    - GenJAX HMC
    - NumPyro HMC

    HMC Parameters (identical for both frameworks):
    - step_size: 0.01
    - num_steps: 20 (leapfrog steps)
    - n_warmup: as specified (burn-in samples)
    - n_samples: as specified (post-warmup samples)

    Creates a 2-panel figure:
    - Top: Posterior curves comparison
    - Bottom: Timing comparison
    """
    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
        numpyro_run_importance_sampling_jit,
        numpyro_run_hmc_inference_jit,
    )
    from genjax.core import Const
    import jax

    print("=== Framework Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is} (fixed)")
    print(f"HMC samples: {n_samples_hmc}")
    print(f"HMC warmup: {n_warmup} (critical for convergence)")

    # Use the exact reference dataset from four_multipoint_traces
    data = get_reference_dataset(seed=seed)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]
    noise_std = data["true_params"]["noise_std"]

    print(
        f"Reference Dataset - True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}"
    )
    print(f"Observation noise std: {noise_std:.3f}")
    print(f"Number of data points: {len(xs)}")

    results = {}

    # 1. GenJAX Importance Sampling
    print(f"\n1. GenJAX IS ({n_samples_is} particles)...")

    def is_task():
        return infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples_is))

    is_times, is_timing_stats = benchmark_with_warmup(
        is_task, warmup_runs=2, repeats=timing_repeats, inner_repeats=10
    )
    is_samples, is_weights = is_task()

    # Extract and resample
    is_a = is_samples.get_choices()["curve"]["a"]
    is_b = is_samples.get_choices()["curve"]["b"]
    is_c = is_samples.get_choices()["curve"]["c"]

    # Importance resample for visualization - increase for better coverage
    n_resample = 500
    is_indices = jrand.categorical(jrand.key(123), is_weights, shape=(n_resample,))
    is_a_plot = is_a[is_indices]
    is_b_plot = is_b[is_indices]
    is_c_plot = is_c[is_indices]

    # Calculate weighted mean for IS (important!)
    normalized_weights = jnp.exp(is_weights - jax.scipy.special.logsumexp(is_weights))
    is_a_mean = jnp.sum(is_a * normalized_weights)
    is_b_mean = jnp.sum(is_b * normalized_weights)
    is_c_mean = jnp.sum(is_c * normalized_weights)

    results["genjax_is"] = {
        "a": is_a_plot,
        "b": is_b_plot,
        "c": is_c_plot,
        "timing": is_timing_stats,
        "method": f"GenJAX IS (N={n_samples_is})",
        "weighted_mean_a": is_a_mean,
        "weighted_mean_b": is_b_mean,
        "weighted_mean_c": is_c_mean,
    }

    print(
        f"  Time: {is_timing_stats[0] * 1000:.1f} ± {is_timing_stats[1] * 1000:.1f} ms"
    )
    print(
        f"  IS a mean (resampled): {is_a_plot.mean():.3f}, std: {is_a_plot.std():.3f}"
    )
    print(
        f"  IS b mean (resampled): {is_b_plot.mean():.3f}, std: {is_b_plot.std():.3f}"
    )
    print(
        f"  IS c mean (resampled): {is_c_plot.mean():.3f}, std: {is_c_plot.std():.3f}"
    )
    print(
        f"  IS weighted mean: a={is_a_mean:.3f}, b={is_b_mean:.3f}, c={is_c_mean:.3f}"
    )

    # 2. GenJAX HMC
    print("\n2. GenJAX HMC...")

    def hmc_task():
        return hmc_infer_latents_jit(
            jrand.key(seed),
            xs,
            ys,  # Use same seed as other methods
            Const(n_samples_hmc),
            Const(n_warmup),
            Const(0.001),
            Const(50),  # step_size=0.001, n_steps=50 for better acceptance
        )

    hmc_times, hmc_timing_stats = benchmark_with_warmup(
        hmc_task,
        warmup_runs=2,
        repeats=max(5, timing_repeats // 10),  # Reduced for HMC since it's slower
        inner_repeats=2,  # Very few inner repeats for HMC
    )
    hmc_samples, hmc_diagnostics = hmc_task()

    # Extract samples
    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]

    # Thin for plotting - increase to 500 samples
    thin_factor = max(1, len(hmc_a) // 500)
    hmc_a_plot = hmc_a[::thin_factor][:500]
    hmc_b_plot = hmc_b[::thin_factor][:500]
    hmc_c_plot = hmc_c[::thin_factor][:500]

    results["genjax_hmc"] = {
        "a": hmc_a_plot,
        "b": hmc_b_plot,
        "c": hmc_c_plot,
        "timing": hmc_timing_stats,
        "method": "GenJAX HMC",
        "accept_rate": float(hmc_diagnostics["acceptance_rate"]),
    }

    print(
        f"  Time: {hmc_timing_stats[0] * 1000:.1f} ± {hmc_timing_stats[1] * 1000:.1f} ms"
    )
    print(f"  Accept rate: {hmc_diagnostics['acceptance_rate']:.3f}")
    print(f"  HMC a mean: {hmc_a_plot.mean():.3f}, std: {hmc_a_plot.std():.3f}")
    print(f"  HMC b mean: {hmc_b_plot.mean():.3f}, std: {hmc_b_plot.std():.3f}")
    print(f"  HMC c mean: {hmc_c_plot.mean():.3f}, std: {hmc_c_plot.std():.3f}")
    print(f"  GenJAX HMC: {len(hmc_a)} total samples")

    # 3. NumPyro IS
    print(f"\n3. NumPyro IS ({n_samples_is} particles)...")

    def numpyro_is_task():
        return numpyro_run_importance_sampling_jit(
            jrand.key(seed), xs, ys, n_samples_is
        )

    numpyro_is_times, numpyro_is_timing_stats = benchmark_with_warmup(
        numpyro_is_task, warmup_runs=2, repeats=timing_repeats, inner_repeats=10
    )
    numpyro_is_result = numpyro_is_task()

    # Extract samples
    numpyro_is_samples = numpyro_is_result["samples"]
    numpyro_is_weights = numpyro_is_result["log_weights"]

    # Resample for visualization
    numpyro_is_indices = jrand.categorical(
        jrand.key(124), numpyro_is_weights, shape=(n_resample,)
    )
    numpyro_is_a_plot = numpyro_is_samples["a"][numpyro_is_indices]
    numpyro_is_b_plot = numpyro_is_samples["b"][numpyro_is_indices]
    numpyro_is_c_plot = numpyro_is_samples["c"][numpyro_is_indices]

    # Calculate weighted mean for NumPyro IS
    normalized_weights_np = jnp.exp(
        numpyro_is_weights - jax.scipy.special.logsumexp(numpyro_is_weights)
    )
    numpyro_is_a_mean = jnp.sum(numpyro_is_samples["a"] * normalized_weights_np)
    numpyro_is_b_mean = jnp.sum(numpyro_is_samples["b"] * normalized_weights_np)
    numpyro_is_c_mean = jnp.sum(numpyro_is_samples["c"] * normalized_weights_np)

    results["numpyro_is"] = {
        "a": numpyro_is_a_plot,
        "b": numpyro_is_b_plot,
        "c": numpyro_is_c_plot,
        "timing": numpyro_is_timing_stats,
        "method": f"NumPyro IS (N={n_samples_is})",
        "weighted_mean_a": numpyro_is_a_mean,
        "weighted_mean_b": numpyro_is_b_mean,
        "weighted_mean_c": numpyro_is_c_mean,
    }

    print(
        f"  Time: {numpyro_is_timing_stats[0] * 1000:.1f} ± {numpyro_is_timing_stats[1] * 1000:.1f} ms"
    )
    print(
        f"  NumPyro IS a mean (resampled): {numpyro_is_a_plot.mean():.3f}, std: {numpyro_is_a_plot.std():.3f}"
    )
    print(
        f"  NumPyro IS b mean (resampled): {numpyro_is_b_plot.mean():.3f}, std: {numpyro_is_b_plot.std():.3f}"
    )
    print(
        f"  NumPyro IS c mean (resampled): {numpyro_is_c_plot.mean():.3f}, std: {numpyro_is_c_plot.std():.3f}"
    )
    print(
        f"  NumPyro IS weighted mean: a={numpyro_is_a_mean:.3f}, b={numpyro_is_b_mean:.3f}, c={numpyro_is_c_mean:.3f}"
    )

    # 4. NumPyro HMC
    print("\n4. NumPyro HMC...")

    def numpyro_task():
        # Use same key as GenJAX HMC for fair comparison
        return numpyro_run_hmc_inference_jit(
            jrand.key(seed),
            xs,
            ys,
            n_samples_hmc,
            n_warmup,
            0.001,
            50,  # Same step_size and n_steps as GenJAX
        )

    numpyro_times, numpyro_timing_stats = benchmark_with_warmup(
        numpyro_task,
        warmup_runs=2,
        repeats=max(5, timing_repeats // 10),  # Reduced for HMC since it's slower
        inner_repeats=2,  # Very few inner repeats for HMC
    )
    numpyro_result = numpyro_task()

    # Extract samples
    numpyro_a = numpyro_result["samples"]["a"]
    numpyro_b = numpyro_result["samples"]["b"]
    numpyro_c = numpyro_result["samples"]["c"]

    # Thin for plotting - increase to 500 samples
    numpyro_a_plot = numpyro_a[::thin_factor][:500]
    numpyro_b_plot = numpyro_b[::thin_factor][:500]
    numpyro_c_plot = numpyro_c[::thin_factor][:500]

    results["numpyro_hmc"] = {
        "a": numpyro_a_plot,
        "b": numpyro_b_plot,
        "c": numpyro_c_plot,
        "timing": numpyro_timing_stats,
        "method": "NumPyro HMC",
    }

    print(
        f"  Time: {numpyro_timing_stats[0] * 1000:.1f} ± {numpyro_timing_stats[1] * 1000:.1f} ms"
    )
    print(f"  NumPyro a mean: {numpyro_a.mean():.3f}, std: {numpyro_a.std():.3f}")
    print(f"  NumPyro b mean: {numpyro_b.mean():.3f}, std: {numpyro_b.std():.3f}")
    print(f"  NumPyro c mean: {numpyro_c.mean():.3f}, std: {numpyro_c.std():.3f}")
    print(f"  NumPyro HMC: {len(numpyro_a)} total samples")

    # Create figure
    if figsize is None:
        figsize = FIGURE_SIZES["framework_comparison"]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.5)

    # Colors following visualization guide - distinct, colorblind-friendly palette
    colors = {
        "genjax_is": "#0173B2",  # Blue
        "genjax_hmc": "#DE8F05",  # Orange
        "numpyro_is": "#CC3311",  # Red
        "numpyro_hmc": "#029E73",  # Green
    }

    # Top panel: Posterior curves
    ax1 = fig.add_subplot(gs[0])

    x_fine = np.linspace(0, 1, 300)
    true_curve = true_a + true_b * x_fine + true_c * x_fine**2

    # Plot true curve and data with improved visibility
    ax1.plot(x_fine, true_curve, "k-", linewidth=2.5, label="True curve", zorder=10)
    ax1.scatter(
        xs,
        ys,
        c="#CC3311",
        s=80,
        alpha=1.0,
        label="Observed data",
        zorder=15,
        edgecolor="#882255",
        linewidth=1.5,
    )

    # Plot posterior samples and mean curves
    legend_handles = []
    for method_key, result in results.items():
        a_samples = result["a"]
        b_samples = result["b"]
        c_samples = result["c"]
        color = colors[method_key]

        # Skip plotting individual posterior curves - only plot mean curves

        # Calculate and plot mean curve
        # For IS, use the weighted mean instead of simple mean
        if method_key == "genjax_is" and "weighted_mean_a" in result:
            mean_a = float(result["weighted_mean_a"])
            mean_b = float(result["weighted_mean_b"])
            mean_c = float(result["weighted_mean_c"])
        else:
            mean_a = np.mean(a_samples)
            mean_b = np.mean(b_samples)
            mean_c = np.mean(c_samples)

        mean_curve = mean_a + mean_b * x_fine + mean_c * x_fine**2

        # Debug print for all mean curves
        print(
            f"  {result['method']} mean curve: a={mean_a:.3f}, b={mean_b:.3f}, c={mean_c:.3f}"
        )
        print(
            f"  {result['method']} mean curve range: [{np.min(mean_curve):.3f}, {np.max(mean_curve):.3f}]"
        )

        # Use consistent styling for all mean curves
        linewidth = 3.0
        zorder = 25  # High z-order for all mean curves

        ax1.plot(
            x_fine,
            mean_curve,
            color=color,
            linewidth=linewidth,
            alpha=1.0,
            linestyle="-",
            zorder=zorder,
        )  # Solid line, high z-order

        # Add a single legend entry for this method with higher alpha
        legend_line = ax1.plot(
            [], [], color=color, alpha=0.8, linewidth=3, label=result["method"]
        )[0]
        legend_handles.append(legend_line)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Posterior Comparison", fontweight="normal")
    set_minimal_ticks(ax1, x_ticks=4, y_ticks=3)

    # Create legend with only true curve and data
    from matplotlib.lines import Line2D

    # Legend elements - just true curve and data
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2.5, label="True curve"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#CC3311",
            markersize=10,
            label="Observed data",
            linestyle="",
        ),
    ]

    ax1.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=12)

    # Bottom panel: Timing comparison
    ax2 = fig.add_subplot(gs[1])

    methods = []
    times = []
    errors = []
    method_colors = []

    for method_key, result in results.items():
        methods.append(result["method"])
        times.append(result["timing"][0] * 1000)
        errors.append(result["timing"][1] * 1000)
        method_colors.append(colors[method_key])

    y_pos = np.arange(len(methods))
    bars = ax2.barh(
        y_pos,
        times,
        xerr=errors,
        color=method_colors,
        alpha=0.9,
        capsize=6,
        linewidth=1,
        error_kw={"linewidth": 1.5, "capthick": 1.5},
    )

    ax2.set_yticks([])  # Remove y-axis ticks
    ax2.set_yticklabels([])  # Remove y-axis labels
    ax2.set_xlabel("Time (ms)")
    ax2.set_title("Inference Time Comparison", fontweight="normal")
    # Minimize x-axis ticks
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    # Remove left spine for cleaner look
    ax2.spines["left"].set_visible(False)

    # Add time labels
    for i, (bar, time, error) in enumerate(zip(bars, times, errors)):
        ax2.text(
            bar.get_width() + error + max(times) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{time:.1f}±{error:.1f}",
            ha="left",
            va="center",
            fontsize=16,
        )

    # Create horizontal legend at the bottom for methods only
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,  # Three methods in one row
        frameon=False,
        columnspacing=2.0,
        fontsize=16,
    )

    # Save figure with extra space for legend
    filename = f"examples/curvefit/figs/framework_comparison_n{n_points}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Saved framework comparison: {filename}")

    return results


# Basic visualizations from original figs.py
def visualize_onepoint_trace(trace, ylim=(-1.5, 1.5)):
    """Visualize a single point trace."""
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)
    fig = plt.figure(figsize=FIGURE_SIZES["single_small"])
    plt.plot(xvals, jax.vmap(curve)(xvals), color="#333333", linewidth=2)
    plt.scatter(
        pt[0],
        pt[1],
        color="#CC3311",
        s=80,
        zorder=10,
        edgecolor="#882255",
        linewidth=1.5,
    )
    plt.ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    set_minimal_ticks(plt.gca(), x_ticks=3, y_ticks=3)
    plt.tight_layout(pad=0.5)
    return fig


def save_onepoint_trace_viz():
    """Save onepoint trace visualization."""
    from examples.curvefit.core import onepoint_curve

    print("Making and saving onepoint trace visualization.")
    trace = onepoint_curve.simulate(0.0)
    fig = visualize_onepoint_trace(trace)
    fig.savefig("examples/curvefit/figs/010_onepoint_trace.pdf")
    plt.close()


def visualize_multipoint_trace(trace, figsize=None, yrange=None, ax=None):
    """Visualize a multipoint trace."""
    curve, (xs, ys) = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)

    if ax is None:
        if figsize is None:
            figsize = FIGURE_SIZES["single_medium"]
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.plot(xvals, jax.vmap(curve)(xvals), color="#333333", linewidth=2)
    ax.scatter(
        xs, ys, color="#CC3311", s=60, zorder=10, edgecolor="#882255", linewidth=1.2
    )
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    set_minimal_ticks(ax, x_ticks=3, y_ticks=3)
    if fig is not None:
        fig.tight_layout(pad=0.5)
    return fig


def save_multipoint_trace_viz():
    """Save multipoint trace visualization."""
    from examples.curvefit.core import npoint_curve

    print("Making and saving multipoint trace visualization.")
    xs = jnp.linspace(0, 1, 10)
    trace = npoint_curve.simulate(xs)
    fig = visualize_multipoint_trace(trace)
    fig.savefig("examples/curvefit/figs/020_multipoint_trace.pdf")
    plt.close()


def save_four_multipoint_trace_vizs():
    """Save four different multipoint trace visualizations with density evaluations."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving four multipoint trace visualizations.")

    xs = jnp.linspace(0, 1, 10)

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["four_panel_grid"])
    axes = axes.flatten()

    # Use seeded simulation for reproducibility
    seeded_simulate = genjax_seed(npoint_curve.simulate)

    # Initialize key and split for multiple samples
    key = jrand.key(42)
    keys = jrand.split(key, 4)  # Split into 4 keys for 4 subplots

    for i, (ax, subkey) in enumerate(zip(axes, keys)):
        # Generate trace with properly split key
        # First trace (i=0) uses first split from seed=42, which is our reference dataset
        trace = seeded_simulate(subkey, xs)

        # Get the log density of the trace (score is negative log probability)
        log_density = -float(trace.get_score())

        visualize_multipoint_trace(trace, ax=ax)

        # Highlight the first sample as the reference dataset
        if i == 0:
            ax.set_title(
                f"Sample {i + 1} (Reference Dataset)",
                fontsize=16,
                pad=8,
                color="#CC3311",
                fontweight="bold",
            )
        else:
            ax.set_title(f"Sample {i + 1}", fontsize=16, pad=8)

        # Remove tick marks and labels but keep the box
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Keep all spines to create a box
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        # Make the box more prominent (extra thick for reference dataset)
        for spine in ax.spines.values():
            if i == 0:
                spine.set_linewidth(3.0)
                spine.set_color("#CC3311")
            else:
                spine.set_linewidth(1.5)
                spine.set_color("black")

        # Add log density text in lower left corner
        ax.text(
            0.02,
            0.02,
            f"log p = {log_density:.2f}",
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor="#666666",
                linewidth=0.5,
            ),
        )

    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/030_four_multipoint_traces.pdf")
    plt.close()


def save_inference_viz(n_curves_to_plot=200, seed=42):
    """Save inference visualization using the reference dataset.

    Args:
        n_curves_to_plot: Number of posterior curves to visualize
        seed: Random seed for reproducibility (default: 42)
    """
    from examples.curvefit.core import infer_latents
    from genjax.core import Const
    from genjax.pjax import seed as genjax_seed

    print("Making and saving inference visualization.")

    # Use the reference dataset for consistency
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    # Adjust xvals range based on actual data range
    xvals = jnp.linspace(0, 1, 300)

    # Apply seeding to inference function
    seeded_infer_latents = genjax_seed(infer_latents)
    samples, weights = seeded_infer_latents(jrand.key(seed), xs, ys, Const(int(10_000)))

    # Get top weighted samples
    order = jnp.argsort(weights, descending=True)
    samples, weights = jax.tree.map(
        lambda x: x[order[:n_curves_to_plot]], (samples, weights)
    )

    # Extract curve functions from samples
    # The model returns (curve, (xs, ys)), so we want the first element
    curves = []
    for i in range(n_curves_to_plot):
        # Get the i-th trace
        trace_i = jax.tree.map(lambda x: x[i], samples)
        # Extract the curve from the return value
        curve, _ = trace_i.get_retval()
        curves.append(curve)

    # Calculate alpha values for visualization
    # Use normalized weights for alpha, with minimum visibility
    weights_normalized = jax.nn.softmax(weights)
    alphas = jnp.maximum(
        0.05, jnp.sqrt(weights_normalized)
    )  # Reduced minimum alpha for denser look

    fig = plt.figure(figsize=FIGURE_SIZES["single_medium"])

    # Plot true curve
    true_curve = true_a + true_b * xvals + true_c * xvals**2
    plt.plot(xvals, true_curve, "#333333", linewidth=2.5, label="True curve", zorder=50)

    # Plot posterior samples
    for i, curve in enumerate(curves):
        plt.plot(
            xvals, curve(xvals), color="#0173B2", alpha=float(alphas[i]), linewidth=0.8
        )

    # Plot data points
    plt.scatter(
        xs,
        ys,
        color="#CC3311",
        s=80,
        zorder=100,
        edgecolor="#882255",
        linewidth=1.5,
        label="Data",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Posterior Inference", fontsize=22, pad=15)
    set_minimal_ticks(plt.gca(), x_ticks=4, y_ticks=3)
    plt.tight_layout(pad=0.5)
    fig.savefig("examples/curvefit/figs/050_inference_viz.pdf")
    plt.close()


def save_genjax_posterior_comparison(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=2,
    seed=42,
    timing_repeats=10,
):
    """Save GenJAX-only posterior comparison (IS vs HMC)."""
    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
    )
    from genjax.core import Const

    print("=== GenJAX Posterior Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is}")
    print(f"HMC samples: {n_samples_hmc}")

    # Use the reference dataset for consistency
    data = get_reference_dataset(n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    # Run IS inference
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )

    # Run HMC inference
    hmc_samples, hmc_diagnostics = hmc_infer_latents_jit(
        jrand.key(seed),
        xs,
        ys,
        Const(n_samples_hmc),
        Const(n_warmup),
        Const(0.001),
        Const(50),  # Smaller step size for better acceptance
    )

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel_horizontal"])

    x_fine = np.linspace(0, 1, 300)
    true_curve = true_a + true_b * x_fine + true_c * x_fine**2

    # IS posterior
    ax = axes[0]
    ax.plot(x_fine, true_curve, "#333333", linewidth=2.5, label="True curve")
    ax.scatter(
        xs,
        ys,
        c="#CC3311",
        s=80,
        label="Data",
        zorder=10,
        edgecolor="#882255",
        linewidth=1.5,
    )

    # Resample IS
    n_resample = 200
    is_indices = jrand.categorical(jrand.key(123), is_weights, shape=(n_resample,))
    is_a = is_samples.get_choices()["curve"]["a"][is_indices]
    is_b = is_samples.get_choices()["curve"]["b"][is_indices]
    is_c = is_samples.get_choices()["curve"]["c"][is_indices]

    for i in range(min(200, n_resample)):
        curve = is_a[i] + is_b[i] * x_fine + is_c[i] * x_fine**2
        ax.plot(x_fine, curve, color="#0173B2", alpha=0.03, linewidth=0.6)

    ax.set_title(f"GenJAX IS ({n_samples_is} particles)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

    # HMC posterior
    ax = axes[1]
    ax.plot(x_fine, true_curve, "#333333", linewidth=2.5, label="True curve")
    ax.scatter(
        xs,
        ys,
        c="#CC3311",
        s=80,
        label="Data",
        zorder=10,
        edgecolor="#882255",
        linewidth=1.5,
    )

    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]

    # Thin HMC samples
    thin_factor = max(1, len(hmc_a) // 200)
    hmc_a_thin = hmc_a[::thin_factor][:200]
    hmc_b_thin = hmc_b[::thin_factor][:200]
    hmc_c_thin = hmc_c[::thin_factor][:200]

    for i in range(len(hmc_a_thin)):
        curve = hmc_a_thin[i] + hmc_b_thin[i] * x_fine + hmc_c_thin[i] * x_fine**2
        ax.plot(x_fine, curve, color="#DE8F05", alpha=0.03, linewidth=0.6)

    ax.set_title(f"GenJAX HMC ({n_samples_hmc} samples)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/060_genjax_posterior_comparison.pdf")
    plt.close()

    print("✓ Saved GenJAX posterior comparison")


def save_inference_scaling_viz():
    """Save inference scaling visualization."""
    from examples.curvefit.core import infer_latents_jit
    from examples.curvefit.data import generate_test_dataset
    from genjax.core import Const
    from examples.utils import benchmark_with_warmup

    print("Making and saving inference scaling visualization.")

    # Test different numbers of samples
    n_samples_list = [100, 500, 1000, 2000, 5000, 10000]

    # Generate test data
    data = generate_test_dataset(seed=42, n_points=10)
    xs, ys = data["xs"], data["ys"]

    times_mean = []
    times_std = []
    lml_estimates = []
    ess_values = []

    for n_samples in n_samples_list:
        print(f"  Testing with {n_samples} samples...")

        # Time inference
        def task():
            return infer_latents_jit(jrand.key(42), xs, ys, Const(n_samples))

        _, (mean_time, std_time) = benchmark_with_warmup(
            task, warmup_runs=2, repeats=10, inner_repeats=10
        )
        times_mean.append(mean_time * 1000)  # Convert to ms
        times_std.append(std_time * 1000)

        # Compute log marginal likelihood and ESS
        samples, log_weights = task()

        # Compute LML - the weights are already log weights
        lml = jax.scipy.special.logsumexp(log_weights) - jnp.log(n_samples)
        lml_estimates.append(float(lml))

        # Compute effective sample size
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights_normalized = jnp.exp(log_weights_normalized)
        ess = 1.0 / jnp.sum(weights_normalized**2)
        ess_values.append(float(ess))

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=FIGURE_SIZES["three_panel_horizontal"]
    )

    # Define consistent color for GenJAX IS
    genjax_is_color = "#0173B2"

    # Timing plot
    ax1.errorbar(
        n_samples_list,
        times_mean,
        yerr=times_std,
        marker="o",
        capsize=8,
        linewidth=3,
        markersize=10,
        color=genjax_is_color,
        elinewidth=2,
    )
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Inference Time Scaling", fontweight="normal")
    ax1.set_xscale("log")
    ax1.set_xlim(80, 12000)
    # Use only 3 ticks on log scale
    ax1.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=3))
    set_minimal_ticks(ax1, x_ticks=3, y_ticks=3)

    # LML plot
    ax2.plot(
        n_samples_list,
        lml_estimates,
        marker="o",
        linewidth=3,
        markersize=10,
        color=genjax_is_color,
    )
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Log Marginal Likelihood")
    ax2.set_title("LML Estimates", fontweight="normal")
    ax2.set_xscale("log")
    ax2.set_xlim(80, 12000)
    # Use only 3 ticks on log scale
    ax2.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=3))
    set_minimal_ticks(ax2, x_ticks=3, y_ticks=3)

    # ESS plot
    ax3.plot(
        n_samples_list,
        ess_values,
        marker="o",
        linewidth=3,
        markersize=10,
        color=genjax_is_color,
    )
    ax3.set_xlabel("Number of Samples")
    ax3.set_ylabel("Effective Sample Size")
    ax3.set_title("ESS Scaling", fontweight="normal")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(80, 12000)
    # Use only 3 ticks on log scales
    ax3.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=3))
    ax3.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=3))

    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/040_inference_scaling.pdf")
    plt.close()

    print("✓ Saved inference scaling visualization")


def save_log_density_viz():
    """Save log density visualization using the reference dataset."""
    from examples.curvefit.core import npoint_curve

    print("Making and saving log density visualization.")

    # Use the reference dataset for consistency
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    # Define grid for polynomial parameters - match the normal prior ranges
    n_grid = 50
    a_range = jnp.linspace(-3.0, 3.0, n_grid)  # 3 std devs for a ~ Normal(0, 1.0)
    b_range = jnp.linspace(-4.5, 4.5, n_grid)  # 3 std devs for b ~ Normal(0, 1.5)

    # Compute log densities on grid
    log_densities = jnp.zeros((n_grid, n_grid))

    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            # Create trace with these parameters (fixing c to true value for 2D viz)
            constraints = {"curve": {"a": a, "b": b, "c": true_c}, "ys": {"obs": ys}}
            trace, log_weight = npoint_curve.generate(constraints, xs)
            log_densities = log_densities.at[i, j].set(log_weight)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])

    # Plot density as heatmap
    im = ax.imshow(
        log_densities.T,
        origin="lower",
        aspect="auto",
        extent=[a_range.min(), a_range.max(), b_range.min(), b_range.max()],
        cmap="viridis",
    )

    # Add true parameters from reference dataset
    ax.scatter(
        true_a,
        true_b,
        c="#CC3311",
        s=150,
        marker="*",
        edgecolor="white",
        linewidth=2,
        label="True params",
    )

    ax.set_xlabel("a (constant term)")
    ax.set_ylabel("b (linear coefficient)")
    ax.set_title("Log Joint Density (c fixed)", fontweight="normal")
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log Density", rotation=270, labelpad=20)

    set_minimal_ticks(ax, x_ticks=4, y_ticks=4)
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/070_log_density.pdf")
    plt.close()

    print("✓ Saved log density visualization")


def save_multiple_curves_single_point_viz():
    """Save visualization of multiple (curve + single point) samples.

    This demonstrates nested vectorization where we sample multiple
    independent curves, each with a single observation point.
    """
    from examples.curvefit.core import onepoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving multiple curves with single point visualization.")

    # Fixed x position for all samples
    x_position = 0.5

    # Generate multiple independent samples
    n_samples = 6
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["four_panel_grid"])
    axes = axes.flatten()

    # Use seeded simulation
    seeded_simulate = genjax_seed(onepoint_curve.simulate)

    # Generate keys
    key = jrand.key(42)
    keys = jrand.split(key, n_samples)

    # Common x values for plotting curves
    xvals = jnp.linspace(0, 1, 300)

    for i, (ax, subkey) in enumerate(zip(axes, keys)):
        # Generate independent curve + point sample
        trace = seeded_simulate(subkey, x_position)
        curve, (x, y) = trace.get_retval()

        # Plot the curve
        ax.plot(
            xvals, jax.vmap(curve)(xvals), color="#0173B2", linewidth=2.5, alpha=0.9
        )

        # Plot the observation point
        ax.scatter(
            x, y, color="#CC3311", s=120, zorder=10, edgecolor="#882255", linewidth=2
        )

        # Add vertical line to show x position
        ax.axvline(x=x_position, color="gray", linestyle="--", alpha=0.3, linewidth=1)

        # Set consistent y limits
        ax.set_ylim(-2, 2)
        ax.set_xlim(0, 1)

        # Minimal labeling
        if i >= 3:  # Bottom row
            ax.set_xlabel("x")
        if i % 3 == 0:  # Left column
            ax.set_ylabel("y")

        # Add sample number
        ax.text(
            0.05,
            0.95,
            f"Sample {i + 1}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Minimal ticks
        set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

    plt.suptitle("Multiple Independent (Curve + Point) Samples", fontsize=18, y=0.98)
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/025_multiple_curves_single_point.pdf")
    plt.close()

    print("✓ Saved multiple curves single point visualization")


def save_single_curve_multiple_points_viz():
    """Save visualization of single curve + multiple points.

    This demonstrates vectorization where we have one curve
    with multiple observation points.
    """
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving single curve with multiple points visualization.")

    # Generate different numbers of points
    point_counts = [1, 3, 5, 10, 20, 50]

    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["four_panel_grid"])
    axes = axes.flatten()

    # Use seeded simulation
    seeded_simulate = genjax_seed(npoint_curve.simulate)

    # Use same key for consistent curve across subplots
    key = jrand.key(42)

    # Common x values for plotting curves
    xvals = jnp.linspace(0, 1, 300)

    for i, (ax, n_points) in enumerate(zip(axes, point_counts)):
        # Generate observation points
        xs = jnp.linspace(0.1, 0.9, n_points)

        # Generate trace
        trace = seeded_simulate(key, xs)
        curve, (xs_ret, ys) = trace.get_retval()

        # Plot the curve
        ax.plot(
            xvals,
            jax.vmap(curve)(xvals),
            color="#0173B2",
            linewidth=2.5,
            alpha=0.9,
            label="Same curve",
        )

        # Plot the observation points
        ax.scatter(
            xs_ret,
            ys,
            color="#CC3311",
            s=max(20, 120 / np.sqrt(n_points)),  # Smaller points as count increases
            zorder=10,
            edgecolor="#882255",
            linewidth=1.5,
            label=f"{n_points} points",
        )

        # Set consistent y limits
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 1)

        # Minimal labeling
        if i >= 3:  # Bottom row
            ax.set_xlabel("x")
        if i % 3 == 0:  # Left column
            ax.set_ylabel("y")

        # Add point count as title
        ax.set_title(f"N = {n_points}", fontsize=14)

        # Minimal ticks
        set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

    plt.suptitle("Single Curve with Varying Numbers of Points", fontsize=18, y=0.98)
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/026_single_curve_multiple_points.pdf")
    plt.close()

    print("✓ Saved single curve multiple points visualization")


def create_code_visualization_figure(
    code_text,
    visualization_func,
    vis_args=(),
    vis_kwargs={},
    main_title="",
    code_title="Code",
    vis_title="Result",
    figsize=(14, 6),
    filename="code_vis_figure.pdf",
    highlight_patterns=None,
    highlight_lines=None,
    code_fontsize=10,
):
    """
    Create a publication-quality two-pane figure with code and visualization.

    Left pane: Code with syntax highlighting
    Right pane: Visualization result

    Args:
        code_text: Python code to display
        visualization_func: Function that creates the visualization (should return fig)
        vis_args: Arguments for visualization function
        vis_kwargs: Keyword arguments for visualization function
        main_title: Overall figure title
        code_title: Title for code pane
        vis_title: Title for visualization pane
        figsize: Overall figure size
        filename: Output filename
        highlight_patterns: List of patterns to highlight in code
        highlight_lines: List of line numbers to highlight
        code_fontsize: Font size for code
    """

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Add main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=16, weight="bold", y=0.98)

    # Left panel: Code
    ax_code = fig.add_subplot(gs[0])
    ax_code.axis("off")

    # Code styling
    bg_color = "#f8f8f8"
    border_color = "#cccccc"

    # Draw code background
    code_bg = plt.Rectangle(
        (0.05, 0.05),
        0.9,
        0.85,
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5,
        transform=ax_code.transAxes,
    )
    ax_code.add_patch(code_bg)

    # Add code title
    ax_code.text(
        0.5,
        0.94,
        code_title,
        fontsize=12,
        weight="bold",
        ha="center",
        transform=ax_code.transAxes,
    )

    # Render code with basic syntax highlighting
    render_code_with_highlighting(
        ax_code,
        code_text,
        highlight_patterns=highlight_patterns,
        highlight_lines=highlight_lines,
        fontsize=code_fontsize,
    )

    # Right panel: Visualization
    ax_vis = fig.add_subplot(gs[1])

    # Generate the visualization
    # Save current figure to restore later
    current_fig = plt.gcf()

    # Call visualization function
    vis_fig = visualization_func(*vis_args, **vis_kwargs)

    # If the visualization function returned a figure, extract its contents
    if vis_fig is not None and hasattr(vis_fig, "axes"):
        # Get the axes from the visualization figure
        for vis_ax in vis_fig.axes:
            # Copy the contents to our subplot
            # This is a bit hacky but works for most cases
            for artist in vis_ax.get_children():
                try:
                    artist_copy = artist
                    artist_copy.remove()
                    ax_vis.add_artist(artist_copy)
                except Exception:
                    pass

            # Copy axis properties
            ax_vis.set_xlim(vis_ax.get_xlim())
            ax_vis.set_ylim(vis_ax.get_ylim())
            ax_vis.set_xlabel(vis_ax.get_xlabel())
            ax_vis.set_ylabel(vis_ax.get_ylabel())

        # Close the temporary figure
        plt.close(vis_fig)

    # Add visualization title
    ax_vis.text(
        0.5,
        0.94,
        vis_title,
        fontsize=12,
        weight="bold",
        ha="center",
        transform=ax_vis.transAxes,
    )

    # Restore current figure
    plt.figure(current_fig.number)

    # Save the combined figure
    plt.tight_layout()
    if main_title:
        plt.subplots_adjust(top=0.93)

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Created code+visualization figure: {filename}")
    return filename


def render_code_with_highlighting(
    ax, code_text, highlight_patterns=None, highlight_lines=None, fontsize=10
):
    """Helper function to render code with basic highlighting in an axes."""

    # Font
    mono_font = font_manager.FontProperties(family="monospace", size=fontsize)

    # Colors for different code elements
    colors = {
        "keyword": "#0000ff",
        "decorator": "#cc0000",
        "string": "#008000",
        "comment": "#808080",
        "function": "#008b8b",
        "normal": "#000000",
    }

    highlight_color = "#fff59d"

    # Split code into lines
    lines = code_text.strip().split("\n")

    # Calculate positions
    y_start = 0.85
    line_height = 0.75 / max(len(lines), 20)
    x_start = 0.08

    # Python keywords not used in current implementation
    # keywords = {"def", "return", "import", "from", "as", "for", "in", "if", "else"}

    for i, line in enumerate(lines):
        y_pos = y_start - i * line_height

        # Highlight line background if requested
        if highlight_lines and (i + 1) in highlight_lines:
            highlight_rect = plt.Rectangle(
                (x_start - 0.02, y_pos - line_height * 0.5),
                0.85,
                line_height * 0.9,
                facecolor=highlight_color,
                edgecolor="none",
                transform=ax.transAxes,
                alpha=0.6,
            )
            ax.add_patch(highlight_rect)

        # Simple syntax coloring
        color = colors["normal"]
        if line.strip().startswith("#"):
            color = colors["comment"]
        elif line.strip().startswith("@"):
            color = colors["decorator"]
        elif "def " in line:
            color = colors["keyword"]
        elif any(
            f'"{s}"' in line or f"'{s}'" in line
            for s in ["a", "b", "c", "obs", "curve", "y"]
        ):
            # Crude string detection
            pass

        # Check for highlight patterns
        text_to_render = line
        for pattern in highlight_patterns or []:
            if pattern in line:
                # Add inline highlighting (simplified)
                parts = line.split(pattern)
                x_offset = 0
                for j, part in enumerate(parts):
                    if j > 0:
                        # Draw highlighted pattern
                        ax.text(
                            x_start + x_offset * 0.007,
                            y_pos,
                            pattern,
                            fontproperties=mono_font,
                            color=color,
                            transform=ax.transAxes,
                            bbox=dict(
                                boxstyle="round,pad=0.1",
                                facecolor=highlight_color,
                                edgecolor="none",
                                alpha=0.7,
                            ),
                        )
                        x_offset += len(pattern)

                    # Draw normal part
                    if part:
                        ax.text(
                            x_start + x_offset * 0.007,
                            y_pos,
                            part,
                            fontproperties=mono_font,
                            color=color,
                            transform=ax.transAxes,
                        )
                        x_offset += len(part)

                text_to_render = None
                break

        # Render the whole line if no pattern matching
        if text_to_render:
            ax.text(
                x_start,
                y_pos,
                text_to_render,
                fontproperties=mono_font,
                color=color,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

        # Add line numbers
        ax.text(
            0.03,
            y_pos,
            f"{i + 1:2d}",
            fontproperties=font_manager.FontProperties(
                family="monospace", size=fontsize - 2
            ),
            color="#999999",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )


def save_programming_with_generative_functions_figure():
    """
    Create figure showing programming with generative functions syntax with code and visualization.

    Shows how users can specify probability distributions and sample from them.
    """
    from examples.curvefit.core import onepoint_curve

    # Code to display - split between left and right panels
    full_code = """@gen
def polynomial():
    # Sample polynomial coefficients
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    # Return function that evaluates polynomial
    return lambda x: a + b * x + c * x**2

@gen
def onepoint_curve(x):
    # Sample a curve
    curve = polynomial() @ "curve"
    # Sample observation with noise
    y = normal(curve(x), 0.2) @ "y"
    return curve, (x, y)

# Sample from the model
trace = onepoint_curve.simulate(0.5)
curve, point = trace.get_retval()"""

    # Split code - first 15 lines for left panel
    code_lines_all = full_code.strip().split("\n")
    code_text = "\n".join(code_lines_all[:-3])  # All but last 3 lines

    # Create figure with adjusted layout
    fig = plt.figure(figsize=(9.5, 4))  # Slightly wider for better spacing

    # Main title - updated
    fig.suptitle(
        "Probabilistic programming with generative functions",
        fontsize=14,
        weight="bold",
        y=0.96,
    )

    # Create custom layout for nestled visualization
    # Left panel: Code (full height)
    ax_code = fig.add_axes([0.05, 0.05, 0.65, 0.85])  # [left, bottom, width, height]
    ax_code.axis("off")

    # Add subtitle for left column (no box)
    ax_code.text(
        0.335,
        0.92,
        "Distributions as programs",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#444444",
        transform=ax_code.transAxes,
    )

    # Render code with Pygments Tango style
    render_code_with_pygments(
        ax_code, code_text, style_name="tango", highlight_lines=[1, 4, 13], y_start=0.88
    )

    # Add annotation text for highlighted lines
    # Get the actual number of lines being rendered
    rendered_lines = code_text.strip().split("\n")
    num_rendered_lines = len(rendered_lines)

    # Calculate line height based on actual rendering (matching render_code_simple)
    y_start_render = 0.88  # Lowered to make room for subtitle
    line_height_render = 0.83 / num_rendered_lines  # Adjusted for subtitle space

    # Annotations for highlighted lines with exact line alignment
    annotations = [
        (1, "Domain-specific syntax"),
        (4, "Addresses denote random variables"),
        (13, "Hierarchical models"),
    ]

    for line_num, text in annotations:
        # Calculate y position to match the line rendering exactly
        # Center vertically to match the centered line numbers and code
        y_pos = (
            y_start_render
            - (line_num - 1) * line_height_render
            - line_height_render * 0.5
        )

        # Add annotation text to the right, vertically centered
        ax_code.text(
            0.64,
            y_pos,
            text,  # Adjusted to 0.64 for better spacing
            ha="left",
            va="center",  # Center vertically to match code and line numbers
            fontsize=8,
            style="italic",
            color="#555555",
            transform=ax_code.transAxes,
        )

    # Right panel: Code fragment with last 3 lines
    # Use the same axes bounds as the left panel for perfect alignment
    ax_code_right = fig.add_axes(
        [0.69, 0.05, 0.26, 0.85]
    )  # Adjusted to 0.69 for better spacing
    ax_code_right.axis("off")

    # Add subtitle for right column (no box)
    ax_code_right.text(
        0.5,
        0.92,
        "Interfaces for probabilistic operations",
        ha="center",
        fontsize=10,
        weight="bold",
        color="#444444",
        transform=ax_code_right.transAxes,
    )

    # Extract last 2 lines for right panel (excluding line 18)
    last_three_lines = code_lines_all[-3:-1]  # Only lines 16-17, excluding line 18

    # Get proper font for code rendering
    font_options = [
        "Berkeley Mono",
        "Berkeley Mono Variable",
        "Fira Code",
        "Source Code Pro",
        "DejaVu Sans Mono",
        "monospace",
    ]
    selected_font = "monospace"
    for font_name in font_options:
        try:
            fp = font_manager.FontProperties(family=font_name)
            if font_manager.findfont(fp):
                selected_font = font_name
                break
        except Exception:
            pass

    mono_font = font_manager.FontProperties(family=selected_font, size=9, weight=500)

    # Tango-inspired colors (same as render_code_simple)
    colors = {
        "comment": "#8f5902",  # Brown
        "decorator": "#5c35cc",  # Purple
        "keyword": "#204a87",  # Dark blue
        "string": "#4e9a06",  # Green
        "number": "#0000cf",  # Blue
        "function": "#000000",  # Black
        "operator": "#ce5c00",  # Orange
        "default": "#000000",  # Black
    }

    keywords = {
        "def",
        "return",
        "lambda",
        "import",
        "from",
        "class",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "in",
        "and",
        "or",
        "not",
        "is",
        "with",
        "as",
    }

    # Manually render the lines to ensure perfect alignment with left panel
    # Use the exact same y_start and line_height as the left panel
    for i, line in enumerate(last_three_lines):
        line_num = i + 16  # Starting from line 16
        y_pos = 0.88 - i * line_height_render  # Use same y_start as left panel

        # Line number (centered vertically like in render_code_simple)
        ax_code_right.text(
            0.04,
            y_pos - line_height_render * 0.5,
            f"{line_num:2d}",
            fontproperties=font_manager.FontProperties(
                family=selected_font, size=7, weight=500
            ),
            color="#666666",
            ha="right",
            va="center",
            transform=ax_code_right.transAxes,
        )

        # Determine line color based on content
        stripped = line.strip()
        color = colors["default"]

        if stripped.startswith("#"):
            color = colors["comment"]
        elif stripped.startswith("@"):
            color = colors["decorator"]
        elif any(kw in line.split() for kw in keywords):
            color = colors["keyword"]
        elif '"' in line or "'" in line:
            color = colors["default"]

        # Code text (centered vertically like in render_code_simple)
        ax_code_right.text(
            0.08,
            y_pos - line_height_render * 0.5,
            line,
            fontproperties=mono_font,
            color=color,
            ha="left",
            va="center",
            transform=ax_code_right.transAxes,
        )

    # Add trace tree visualization below the code fragment
    # Position below the 2 lines of code
    trace_height = 0.25
    trace_top = (
        y_start_render - 2 * line_height_render - 0.1
    )  # Below line 17 with spacing
    trace_bottom = (
        0.05 + 0.85 * trace_top - trace_height
    )  # Convert to figure coordinates
    ax_trace = fig.add_axes([0.69, trace_bottom, 0.26, trace_height])
    ax_trace.axis("off")

    # Add subtitle for trace visualization
    ax_trace.text(
        0.5,
        1.05,
        "Probabilistic program traces",
        ha="center",
        fontsize=9,
        weight="bold",
        color="#444444",
        transform=ax_trace.transAxes,
    )

    # Simplified tree structure
    # Center positions for cleaner layout
    center_x = 0.5

    # Root at top center (invisible connection point)
    root_x, root_y = center_x, 0.90  # Moved up from 0.85

    # First level
    curve_x, curve_y = center_x - 0.2, 0.60  # Moved up from 0.55
    y_x, y_y = center_x + 0.2, 0.60  # Moved up from 0.55

    # Second level - coefficients below curve
    a_x, a_y = center_x - 0.35, 0.35  # Moved up from 0.30
    b_x, b_y = center_x - 0.2, 0.35  # Moved up from 0.30
    c_x, c_y = center_x - 0.05, 0.35  # Moved up from 0.30

    # Draw edges - create a T-junction from root (using axes coordinates)
    # Vertical line from root
    ax_trace.plot(
        [root_x, root_x],
        [root_y, root_y - 0.15],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )

    # Horizontal line at junction - stop exactly at the branch points
    junction_y = root_y - 0.15
    ax_trace.plot(
        [curve_x, y_x],
        [junction_y, junction_y],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )

    # Vertical lines down to nodes - ensure they reach the nodes
    ax_trace.plot(
        [curve_x, curve_x],
        [junction_y, curve_y + 0.08],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    ax_trace.plot(
        [y_x, y_x],
        [junction_y, y_y + 0.08],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )

    # Curve to coefficients
    ax_trace.plot(
        [curve_x, a_x],
        [curve_y, a_y],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    ax_trace.plot(
        [curve_x, b_x],
        [curve_y, b_y],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    ax_trace.plot(
        [curve_x, c_x],
        [curve_y, c_y],
        "k-",
        linewidth=1,
        alpha=0.6,
        zorder=1,
        transform=ax_trace.transAxes,
    )

    # Draw nodes with addresses
    node_props = dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="#666666",
        linewidth=1,
        zorder=10,
    )

    # Root (no visual representation - just a connection point)

    # Address nodes
    ax_trace.text(
        curve_x,
        curve_y,
        '"curve"',
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
        bbox=node_props,
        transform=ax_trace.transAxes,
        zorder=10,
    )

    ax_trace.text(
        y_x,
        y_y,
        '"y"',
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
        bbox=node_props,
        transform=ax_trace.transAxes,
        zorder=10,
    )

    # Coefficient nodes as simple addresses under "curve"
    ax_trace.text(
        a_x,
        a_y,
        '"a"',
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
        bbox=node_props,
        transform=ax_trace.transAxes,
        zorder=10,
    )

    ax_trace.text(
        b_x,
        b_y,
        '"b"',
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
        bbox=node_props,
        transform=ax_trace.transAxes,
        zorder=10,
    )

    ax_trace.text(
        c_x,
        c_y,
        '"c"',
        ha="center",
        va="center",
        fontsize=7,
        fontfamily="monospace",
        bbox=node_props,
        transform=ax_trace.transAxes,
        zorder=10,
    )

    # Add f32[] annotations to all leaf nodes
    # Positions for f32[] labels - straight down with more spacing
    f32_offset_y = -0.30  # Pushed even lower

    # Vertical branch lines from leaf nodes to f32[] annotations
    # From "a"
    ax_trace.plot(
        [a_x, a_x],
        [a_y - 0.05, a_y + f32_offset_y + 0.06],
        "k-",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    # From "b"
    ax_trace.plot(
        [b_x, b_x],
        [b_y - 0.05, b_y + f32_offset_y + 0.06],
        "k-",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    # From "c"
    ax_trace.plot(
        [c_x, c_x],
        [c_y - 0.05, c_y + f32_offset_y + 0.06],
        "k-",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
        transform=ax_trace.transAxes,
    )
    # From "y"
    ax_trace.plot(
        [y_x, y_x],
        [y_y - 0.05, y_y + f32_offset_y + 0.06],
        "k-",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
        transform=ax_trace.transAxes,
    )

    # f32[] text annotations
    f32_props = dict(
        fontsize=6, fontfamily="monospace", color="#666666", style="italic"
    )

    ax_trace.text(
        a_x,
        a_y + f32_offset_y,
        "f32[]",
        ha="center",
        va="center",
        transform=ax_trace.transAxes,
        **f32_props,
    )
    ax_trace.text(
        b_x,
        b_y + f32_offset_y,
        "f32[]",
        ha="center",
        va="center",
        transform=ax_trace.transAxes,
        **f32_props,
    )
    ax_trace.text(
        c_x,
        c_y + f32_offset_y,
        "f32[]",
        ha="center",
        va="center",
        transform=ax_trace.transAxes,
        **f32_props,
    )
    ax_trace.text(
        y_x,
        y_y + f32_offset_y,
        "f32[]",
        ha="center",
        va="center",
        transform=ax_trace.transAxes,
        **f32_props,
    )

    # Visualization (curve plot) below the trace tree
    vis_height = 0.17  # Reduced to 85% of 0.20
    vis_bottom = trace_bottom - vis_height - 0.08  # Increased spacing to move down
    ax_vis = fig.add_axes(
        [0.69, vis_bottom, 0.26, vis_height]
    )  # Adjusted to 0.69 to match code panel

    # Add subtitle for trace visualization
    ax_vis.text(
        0.5,
        0.95,
        "Trace visual",
        ha="center",
        fontsize=9,
        weight="bold",
        color="#444444",
        transform=ax_vis.transAxes,
    )

    # Generate and plot trace
    trace = onepoint_curve.simulate(0.5)
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)

    # Plot curve
    yvals = jax.vmap(curve)(xvals)
    ax_vis.plot(xvals, yvals, color="#0173B2", linewidth=2.5, alpha=0.9)

    # Add text label for the curve - positioned higher and to the left
    text_x = 0.2  # Closer to the y-axis
    # Find the y value at this x position
    text_idx = int(len(xvals) * 0.2)
    text_y = float(yvals[text_idx])

    # Add the text without rotation, positioned higher above the curve
    ax_vis.text(
        text_x,
        text_y + 0.5,
        'trace["curve"]',
        fontsize=8,
        fontfamily="monospace",
        color="#0173B2",
        ha="center",
        va="bottom",
    )

    # Plot point - smaller size for smaller figure
    ax_vis.scatter(
        pt[0],
        pt[1],
        color="#CC3311",
        s=60,
        zorder=10,
        edgecolor="#882255",
        linewidth=1.0,
    )

    # Styling - tighter limits
    ax_vis.set_ylim(-1.5, 1.5)
    ax_vis.set_xlim(0, 1)
    ax_vis.set_ylabel("y", fontsize=8)
    ax_vis.grid(True, alpha=0.3, linestyle="--")
    # No title for small nestled plot

    # Simpler annotation - smaller for smaller figure
    ax_vis.annotate(
        '(x, trace["y"])',
        xy=(pt[0], pt[1]),
        xytext=(pt[0] + 0.15, pt[1] + 0.4),
        fontsize=7.5,
        fontfamily="monospace",
        color="#CC3311",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#CC3311", lw=1.0),
    )

    # Clean up ticks
    set_minimal_ticks(ax_vis, x_ticks=3, y_ticks=3)
    ax_vis.tick_params(labelsize=7)

    # Save figure (no tight_layout with custom axes positioning)
    plt.savefig(
        "examples/curvefit/figs/001_programming_with_generative_functions.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(
        "✓ Created programming with generative functions figure: figs/001_programming_with_generative_functions.pdf"
    )


def render_code_with_pygments(
    ax, code_text, style_name="tango", highlight_lines=None, y_start=None
):
    """Render code with Pygments syntax highlighting using simple approach."""
    # Use simple rendering for better alignment
    render_code_simple(
        ax, code_text, highlight_lines=highlight_lines, y_start_override=y_start
    )


def render_code_simple(
    ax, code_text, highlight_lines=None, start_line_num=1, y_start_override=None
):
    """Simple code rendering with Tango-like syntax highlighting."""
    # Font - Berkeley Mono is our preferred font
    font_options = [
        "Berkeley Mono",
        "Berkeley Mono Variable",
        "Fira Code",
        "Source Code Pro",
        "DejaVu Sans Mono",
        "monospace",
    ]

    # Find first available font
    selected_font = "monospace"  # fallback
    for font_name in font_options:
        try:
            fp = font_manager.FontProperties(family=font_name)
            if font_manager.findfont(fp):
                selected_font = font_name
                break
        except Exception:
            pass

    # Use appropriate size for smaller figure with weight 500 (medium)
    mono_font = font_manager.FontProperties(family=selected_font, size=11, weight=500)

    # Split code into lines
    lines = code_text.strip().split("\n")

    # Calculate positions - adjusted for no background/title
    y_start = (
        y_start_override if y_start_override is not None else 0.95
    )  # Start higher since no title
    # Use a fixed line height for better spacing
    line_height = 0.045  # Tighter line height for more compact display
    x_start = 0.08

    # Set of lines to highlight
    highlight_lines = set(highlight_lines) if highlight_lines else set()

    # Tango-inspired colors
    colors = {
        "comment": "#8f5902",  # Brown
        "decorator": "#5c35cc",  # Purple
        "keyword": "#204a87",  # Dark blue
        "string": "#4e9a06",  # Green
        "number": "#0000cf",  # Blue
        "function": "#000000",  # Black
        "operator": "#ce5c00",  # Orange
        "default": "#000000",  # Black
    }

    # Keywords to highlight
    keywords = {
        "def",
        "return",
        "lambda",
        "import",
        "from",
        "class",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "in",
        "and",
        "or",
        "not",
        "is",
        "with",
        "as",
    }

    # Render each line
    for i, line in enumerate(lines):
        y_pos = y_start - i * line_height
        line_num = i + start_line_num

        # Add highlight background if this line should be highlighted
        if line_num in highlight_lines:
            # Draw highlight rectangle centered on the line
            highlight_rect = mpatches.Rectangle(
                (0.02, y_pos - line_height * 0.95),
                0.61,
                line_height * 0.9,  # Adjusted to 0.61 for better coverage
                facecolor="#e6f2ff",  # Light blue highlight
                edgecolor="none",
                alpha=0.8,  # Visible highlighting
                transform=ax.transAxes,
                zorder=-1,  # Ensure it's behind text
            )
            ax.add_patch(highlight_rect)

        # Line number
        ax.text(
            0.04,
            y_pos - line_height * 0.5,
            f"{line_num:2d}",
            fontproperties=font_manager.FontProperties(
                family=selected_font, size=9, weight=500
            ),
            color="#666666",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

        # Determine line color based on content
        stripped = line.strip()
        color = colors["default"]

        if stripped.startswith("#"):
            color = colors["comment"]
        elif stripped.startswith("@"):
            color = colors["decorator"]
        elif any(kw in line.split() for kw in keywords):
            # For lines with keywords, still render as one piece but with keyword color
            color = colors["keyword"]
        elif '"' in line or "'" in line:
            # Lines containing strings - render in default but we could enhance this
            color = colors["default"]

        # Render entire line as one text element for perfect alignment
        ax.text(
            x_start,
            y_pos - line_height * 0.5,
            line,
            fontproperties=mono_font,
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


def visualize_onepoint_trace_for_figure(ax=None):
    """Modified version of visualize_onepoint_trace that works with subplots."""
    from examples.curvefit.core import onepoint_curve

    # Generate a trace
    trace = onepoint_curve.simulate(0.5)
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])
    else:
        fig = ax.figure

    ax.plot(xvals, jax.vmap(curve)(xvals), color="#0173B2", linewidth=3, alpha=0.9)
    ax.scatter(
        pt[0],
        pt[1],
        color="#CC3311",
        s=120,
        zorder=10,
        edgecolor="#882255",
        linewidth=2,
    )
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

    # Add annotation
    ax.annotate(
        f"Sampled point\n({pt[0]:.1f}, {pt[1]:.2f})",
        xy=(pt[0], pt[1]),
        xytext=(0.7, -1.5),
        fontsize=11,
        ha="center",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.3", color="#CC3311", lw=2
        ),
    )

    return fig if ax is None else None


def save_parameter_posterior_methods_comparison(seed=42):
    """Save parameter posterior visualization comparing all methods.

    Args:
        seed: Random seed for reproducibility (default: 42)
    """
    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
        numpyro_run_hmc_inference_jit,
    )
    from genjax.core import Const

    print("Making and saving parameter posterior methods comparison.")

    # Use the reference dataset for consistency
    data = get_reference_dataset()
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    # true_c not used in 2D visualization

    # Define consistent colors - using visualization guide palette
    colors = {
        "genjax_is": "#0173B2",  # Blue
        "genjax_hmc": "#DE8F05",  # Orange
        "numpyro_hmc": "#029E73",  # Green
    }

    # 1. GenJAX IS
    is_samples, is_weights = infer_latents_jit(jrand.key(seed), xs, ys, Const(5000))
    is_a = is_samples.get_choices()["curve"]["a"]
    is_b = is_samples.get_choices()["curve"]["b"]
    # is_c not needed for 2D visualization
    # Resample according to weights
    n_resample = 1000
    is_indices = jrand.categorical(jrand.key(123), is_weights, shape=(n_resample,))
    is_a_resampled = is_a[is_indices]
    is_b_resampled = is_b[is_indices]
    # is_c_resampled not needed for 2D visualization

    # 2. GenJAX HMC
    hmc_samples, _ = hmc_infer_latents_jit(
        jrand.key(seed),
        xs,
        ys,
        Const(1000),
        Const(500),
        Const(0.001),
        Const(50),  # Smaller step size for better acceptance
    )
    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    # hmc_c not needed for 2D visualization

    # 3. NumPyro HMC - use same key as GenJAX methods for consistency
    numpyro_result = numpyro_run_hmc_inference_jit(
        jrand.key(seed), xs, ys, 1000, 500, 0.001, 50
    )  # Same HMC params as GenJAX
    numpyro_a = numpyro_result["samples"]["a"]
    numpyro_b = numpyro_result["samples"]["b"]
    numpyro_c = numpyro_result["samples"]["c"]

    print(
        f"  NumPyro samples shape: a={numpyro_a.shape}, b={numpyro_b.shape}, c={numpyro_c.shape}"
    )
    print(f"  NumPyro a range: [{numpyro_a.min():.3f}, {numpyro_a.max():.3f}]")
    print(f"  NumPyro b range: [{numpyro_b.min():.3f}, {numpyro_b.max():.3f}]")
    print(f"  NumPyro c range: [{numpyro_c.min():.3f}, {numpyro_c.max():.3f}]")

    # Create figure with 3 columns (one for each method)
    fig = plt.figure(figsize=FIGURE_SIZES["parameter_posterior"])

    # Create grid spec for mixed 2D/3D layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3)

    methods_data = [
        ("GenJAX IS (5000)", is_a_resampled, is_b_resampled, colors["genjax_is"]),
        ("GenJAX HMC", hmc_a, hmc_b, colors["genjax_hmc"]),
        ("NumPyro HMC", numpyro_a, numpyro_b, colors["numpyro_hmc"]),
    ]

    # Calculate consistent axis limits across all methods
    all_a = np.concatenate([is_a_resampled, hmc_a, numpyro_a])
    all_b = np.concatenate([is_b_resampled, hmc_b, numpyro_b])

    # Include ground truth in limits calculation
    a_min = min(all_a.min(), true_a)
    a_max = max(all_a.max(), true_a)
    b_min = min(all_b.min(), true_b)
    b_max = max(all_b.max(), true_b)

    # Add some padding to show full density
    a_range = a_max - a_min
    b_range = b_max - b_min

    a_lim = (a_min - 0.15 * a_range, a_max + 0.15 * a_range)
    b_lim = (b_min - 0.15 * b_range, b_max + 0.15 * b_range)

    print(f"  Consistent a limits: [{a_lim[0]:.3f}, {a_lim[1]:.3f}]")
    print(f"  Consistent b limits: [{b_lim[0]:.3f}, {b_lim[1]:.3f}]")

    # Import 3D plotting

    # Plot 2D histograms (top row)
    legend_elements = []
    for i, (method_name, a_samples, b_samples, color) in enumerate(methods_data):
        ax = fig.add_subplot(gs[0, i])

        # Create 2D histogram with method color and consistent binning
        ax.hist2d(
            a_samples,
            b_samples,
            bins=30,
            range=[a_lim, b_lim],  # Use consistent bins across methods
            cmap=plt.cm.Blues
            if i == 0
            else plt.cm.Oranges
            if i == 1
            else plt.cm.Greens,
        )

        # Add true parameters
        ax.scatter(
            true_a,
            true_b,
            c="#CC3311",
            s=150,
            marker="*",
            edgecolor="white",
            linewidth=2,
            zorder=10,
        )

        ax.set_xlabel("a (constant)")
        ax.set_ylabel("b (linear)" if i == 0 else "")  # Only label leftmost
        set_minimal_ticks(ax, x_ticks=3, y_ticks=3)

        # Set consistent axis limits across all methods
        ax.set_xlim(a_lim)
        ax.set_ylim(b_lim)

        # Create legend element
        from matplotlib.patches import Rectangle

        legend_elements.append(
            Rectangle((0, 0), 1, 1, fc=color, alpha=0.8, label=method_name)
        )

    # Add ground truth to legend
    from matplotlib.lines import Line2D

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#CC3311",
            markersize=15,
            label="Ground truth",
        )
    )

    # Plot 3D density surfaces (bottom row)
    # First pass: calculate max density for consistent z-axis
    max_density = 0
    n_bins = 30
    all_H_smooth = []

    for i, (method_name, a_samples, b_samples, color) in enumerate(methods_data):
        # Estimate 2D density
        H, _, _ = np.histogram2d(
            a_samples, b_samples, bins=n_bins, range=[a_lim, b_lim], density=True
        )
        H = H.T
        from scipy.ndimage import gaussian_filter

        H_smooth = gaussian_filter(H, sigma=1.0)
        all_H_smooth.append(H_smooth)
        max_density = max(max_density, H_smooth.max())

    # Second pass: create the plots
    for i, (method_name, a_samples, b_samples, color) in enumerate(methods_data):
        # Create 3D subplot
        ax = fig.add_subplot(gs[1, i], projection="3d")

        # Use pre-calculated H_smooth
        H_smooth = all_H_smooth[i]

        # Recompute histogram to get edges for mesh
        H, a_edges, b_edges = np.histogram2d(
            a_samples, b_samples, bins=n_bins, range=[a_lim, b_lim], density=True
        )

        # Get bin centers for mesh
        a_centers = (a_edges[:-1] + a_edges[1:]) / 2
        b_centers = (b_edges[:-1] + b_edges[1:]) / 2
        a_mesh, b_mesh = np.meshgrid(a_centers, b_centers)

        # Create surface plot with reduced opacity for better visibility of ground truth
        ax.plot_surface(
            a_mesh,
            b_mesh,
            H_smooth,
            cmap=plt.cm.Blues
            if i == 0
            else plt.cm.Oranges
            if i == 1
            else plt.cm.Greens,
            alpha=0.7,  # Reduced opacity
            antialiased=True,
            linewidth=0,
            rcount=50,
            ccount=50,
            shade=True,
        )

        # Find the density at the true parameter location
        a_idx = np.argmin(np.abs(a_centers - true_a))
        b_idx = np.argmin(np.abs(b_centers - true_b))
        true_density = H_smooth[b_idx, a_idx]

        # Add vertical line for true parameters with enhanced visibility
        # Draw from bottom to top of z-axis
        z_max = max_density * 1.1  # Use the max across all methods

        # Plot vertical line in segments to ensure visibility
        z_points = np.linspace(0, z_max, 50)
        a_points = np.full_like(z_points, true_a)
        b_points = np.full_like(z_points, true_b)

        # Use plot3D for better rendering
        ax.plot3D(
            a_points,
            b_points,
            z_points,
            color="red",
            linewidth=4,
            alpha=1.0,
            zorder=1000,
        )

        # Add large marker at base
        ax.scatter(
            [true_a],
            [true_b],
            [0],
            color="red",
            s=300,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=1001,
        )

        # Add marker at actual density height
        ax.scatter(
            [true_a],
            [true_b],
            [true_density],
            color="red",
            s=200,
            marker="o",
            edgecolor="black",
            linewidth=3,
            zorder=1002,
        )

        # Add cross-hairs on base plane with thicker lines
        ax.plot(
            [a_lim[0], a_lim[1]],
            [true_b, true_b],
            [0, 0],
            "k-",
            alpha=0.7,
            linewidth=2,
            zorder=500,
        )
        ax.plot(
            [true_a, true_a],
            [b_lim[0], b_lim[1]],
            [0, 0],
            "k-",
            alpha=0.7,
            linewidth=2,
            zorder=500,
        )

        # Remove text label - ground truth will be in legend instead

        # Set labels
        ax.set_xlabel("a (constant)", labelpad=10)
        ax.set_ylabel("b (linear)", labelpad=10)
        ax.set_zlabel("Density" if i == 0 else "", labelpad=10)

        # Set view angle for better visualization
        ax.view_init(elev=30, azim=-45)

        # Set consistent axis limits
        ax.set_xlim(a_lim)
        ax.set_ylim(b_lim)

        # Reduce number of ticks - match 2D plots
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

        # Set smaller tick label size for 3D
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="z", labelsize=12)

    # Set consistent z-limits after calculating max density
    for i in range(3):
        ax = fig.axes[3 + i]  # 3D axes are after the 2D ones
        ax.set_zlim(0, max_density * 1.1)

    # Remove overall title to make room for legend

    # Add legend at the bottom with box
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=True,
        fontsize=18,
        edgecolor="black",
        fancybox=False,
    )

    # Adjust layout manually for 3D plots
    plt.subplots_adjust(
        top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.3, wspace=0.2
    )
    fig.savefig(
        "examples/curvefit/figs/085_parameter_posterior_methods_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close()

    print("✓ Saved parameter posterior methods comparison")


def save_vectorization_patterns_figure():
    """Generate the vectorization patterns figure in the same style as programming with generative functions.

    Layout: (2, 3) grid
    - Row 1: Multiple curves, single point (vmap entire model)
    - Row 2: Single curve, multiple points (vmap observation model)
    - Column 1: Code in monospace font
    - Column 2: Trace data plots showing the pattern
    - Column 3: Tree trace diagrams showing the structure
    """

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 10))

    # Define the grid - use equal sizes for all subplots
    subplot_size = 0.4  # Square subplots
    gap = 0.03

    # Row heights and positions
    row_height = 0.4  # Square subplots
    row1_bottom = 0.53
    row2_bottom = 0.08

    # Create axes for each panel - all same size
    # Row 1 (Pattern 1: Multiple curves)
    ax_code1 = fig.add_axes([0.02, row1_bottom, subplot_size, row_height])
    ax_plot1 = fig.add_axes(
        [0.02 + subplot_size + gap, row1_bottom, subplot_size, row_height]
    )
    ax_tree1 = fig.add_axes(
        [0.02 + 2 * (subplot_size + gap), row1_bottom, subplot_size, row_height]
    )

    # Row 2 (Pattern 2: Single curve, multiple points)
    ax_code2 = fig.add_axes([0.02, row2_bottom, subplot_size, row_height])
    ax_plot2 = fig.add_axes(
        [0.02 + subplot_size + gap, row2_bottom, subplot_size, row_height]
    )
    ax_tree2 = fig.add_axes(
        [0.02 + 2 * (subplot_size + gap), row2_bottom, subplot_size, row_height]
    )

    # Pattern 1: Multiple curves, single point
    # Code panel
    code1 = """# Pattern 1: Multiple curves, each with one point
# Natural for: parameter uncertainty, model comparison

# Just use vmap on the entire model!
traces = onepoint_curve.vmap()(
    jnp.full(4, 0.5)  # Same x for all
)

# Result: 4 independent polynomial curves,
# each with their own observation"""

    # Remove axes elements for code panels
    for ax in [ax_code1, ax_code2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Render code for pattern 1 with pygments/simple rendering
    # Start very close to the top of the axes
    render_code_simple(ax_code1, code1, y_start_override=0.98)

    # Plot panel for pattern 1 - show multiple curves in 2x2 grid
    # Remove the main axes elements
    ax_plot1.set_xlim(0, 1)
    ax_plot1.set_ylim(0, 1)
    ax_plot1.axis("off")

    # Generate and plot multiple curves
    import jax.random as jrand
    from examples.curvefit.core import onepoint_curve
    from genjax import seed

    key = jrand.key(42)
    x_plot = jnp.linspace(-0.5, 1.5, 100)

    # Create 4 curves for the 2x2 grid
    # First, let's define the absolute positions based on the row layout
    # Row 1 spans from row1_bottom to (row1_bottom + row_height)
    # We need to fit the 2x2 grid within this space

    # Define subplot dimensions as fractions of the plot area
    subplot_width = 0.12  # Absolute width of each subplot
    subplot_height = 0.15  # Absolute height of each subplot
    h_gap = 0.02  # Horizontal gap between subplots
    v_gap = 0.02  # Vertical gap between subplots

    # Calculate total grid dimensions
    total_grid_width = 2 * subplot_width + h_gap
    total_grid_height = 2 * subplot_height + v_gap

    # Center the grid within column 2's space
    col2_center = 0.02 + subplot_size + gap + subplot_size / 2
    grid_left = col2_center - total_grid_width / 2

    # Center vertically within row 1
    row1_center = row1_bottom + row_height / 2
    grid_bottom = row1_center - total_grid_height / 2

    for idx in range(4):
        row = idx // 2
        col = idx % 2

        # Calculate absolute position for each subplot
        left = grid_left + col * (subplot_width + h_gap)
        bottom = grid_bottom + (1 - row) * (subplot_height + v_gap)

        # Create sub-axes with absolute positioning
        sub_ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
        sub_ax.set_xlim(-0.5, 1.5)
        sub_ax.set_ylim(-2.5, 2.5)

        # Generate trace
        subkey = jrand.fold_in(key, idx)
        trace = seed(onepoint_curve.simulate)(subkey, 0.5)

        # Extract parameters
        curve_params = trace.get_choices()["curve"]
        a = curve_params["a"]
        b = curve_params["b"]
        c = curve_params["c"]
        y_plot = a + b * x_plot + c * x_plot**2

        # Plot curve
        sub_ax.plot(x_plot, y_plot, "b-", alpha=0.8, linewidth=1.5)

        # Plot observation point
        y_obs = trace.get_choices()["y"]["obs"]
        sub_ax.scatter(
            0.5, y_obs, color="red", s=30, zorder=5, edgecolor="black", linewidth=0.5
        )

        # Add grid
        sub_ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
        sub_ax.axvline(x=0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)

        # Remove tick labels for cleaner look
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])

        # Add border
        for spine in sub_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

    # Add title above the 2x2 grid
    # Position it above the grid
    title_y = grid_bottom + total_grid_height + 0.02
    fig.text(
        col2_center,
        title_y,
        "4 Independent Curves at x=0.5",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    # Tree panel for pattern 1
    ax_tree1.set_xlim(0, 1)
    ax_tree1.set_ylim(0, 1)
    ax_tree1.axis("off")

    # Draw tree structure
    ax_tree1.text(
        0.5,
        0.9,
        "traces[0:4]",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="black"),
    )

    # Tree for 4 traces
    branch_y = 0.65
    # Show all 4 traces
    trace_positions = [0.2, 0.35, 0.65, 0.8]
    trace_labels = ["[0]", "[1]", "[2]", "[3]"]

    # Draw main vertical line from root
    ax_tree1.plot([0.5, 0.5], [0.85, 0.75], "k-", linewidth=1, alpha=0.6)
    # Draw horizontal line
    ax_tree1.plot([0.15, 0.85], [0.75, 0.75], "k-", linewidth=1, alpha=0.6)

    for i, (x_pos, label) in enumerate(zip(trace_positions, trace_labels)):
        if label == "...":
            # Just show ellipsis, no branches
            ax_tree1.text(
                x_pos,
                branch_y,
                "...",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
        else:
            # Branch down
            ax_tree1.plot(
                [x_pos, x_pos], [0.75, branch_y], "k-", linewidth=1, alpha=0.6
            )

            # Trace box
            ax_tree1.text(
                x_pos,
                branch_y - 0.05,
                label,
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"
                ),
            )

            # Sub-branches
            ax_tree1.plot(
                [x_pos, x_pos],
                [branch_y - 0.1, branch_y - 0.2],
                "k-",
                linewidth=0.8,
                alpha=0.6,
            )
            ax_tree1.plot(
                [x_pos - 0.03, x_pos + 0.03],
                [branch_y - 0.2, branch_y - 0.2],
                "k-",
                linewidth=0.8,
                alpha=0.6,
            )

            # Leaves
            ax_tree1.text(
                x_pos - 0.03,
                branch_y - 0.25,
                "curve",
                ha="center",
                va="top",
                fontsize=8,
            )
            ax_tree1.text(
                x_pos + 0.03, branch_y - 0.25, "y", ha="center", va="top", fontsize=8
            )

    ax_tree1.set_title("Vectorized traces", fontsize=14)

    # Pattern 2: Single curve, multiple points
    code2 = """# Pattern 2: One curve, multiple points
# Natural for: regression, time series

@gen
def npoint_curve(xs):
    curve = polynomial() @ "curve"
    # Vectorize just the observations
    ys = point.vmap(xs, curve) @ "ys"
    return curve, (xs, ys)

# Sample one curve with N observations
trace = npoint_curve.simulate(xs)"""

    # Render code for pattern 2 with pygments/simple rendering
    # Start near the top of the axes
    render_code_simple(ax_code2, code2, y_start_override=0.95)

    # Plot panel for pattern 2 - single curve, multiple points
    # Match the width and horizontal position of the 2x2 grid above
    total_grid_width = 2 * 0.12 + 0.02  # Same dimensions as 2x2 grid
    col2_center = 0.02 + subplot_size + gap + subplot_size / 2  # Same center as above

    # Position the single plot
    new_width = total_grid_width
    new_left = col2_center - new_width / 2
    new_height = 0.32  # Most of the row height
    new_bottom = row2_bottom + (row_height - new_height) / 2  # Center in row 2

    # Create a new axes with the adjusted position
    ax_plot2_new = fig.add_axes([new_left, new_bottom, new_width, new_height])
    ax_plot2.remove()  # Remove the old axes
    ax_plot2 = ax_plot2_new  # Use the new one

    ax_plot2.set_xlim(-0.5, 3.5)
    ax_plot2.set_ylim(-2, 3)
    ax_plot2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax_plot2.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Generate single curve with multiple points
    from examples.curvefit.core import npoint_curve

    xs_multi = jnp.linspace(0, 3, 10)
    trace2 = npoint_curve.simulate(xs_multi)

    curve_ret, (xs_ret, ys_ret) = trace2.get_retval()

    # Plot the curve
    x_plot = jnp.linspace(-0.5, 3.5, 100)
    y_plot = curve_ret(x_plot)
    ax_plot2.plot(x_plot, y_plot, "b-", alpha=0.8, linewidth=2, label="True curve")

    # Plot observations
    ax_plot2.scatter(
        xs_ret,
        ys_ret,
        color="red",
        s=60,
        zorder=5,
        edgecolor="black",
        linewidth=1,
        label="Observations",
    )

    ax_plot2.set_xlabel("x")
    ax_plot2.set_ylabel("y")
    ax_plot2.set_title("Single curve, 10 points", fontsize=14)
    ax_plot2.legend(fontsize=10)
    set_minimal_ticks(ax_plot2)

    # Tree panel for pattern 2
    ax_tree2.set_xlim(0, 1)
    ax_tree2.set_ylim(0, 1)
    ax_tree2.axis("off")

    # Draw tree structure
    ax_tree2.text(
        0.5,
        0.9,
        "trace",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="black"),
    )

    # Branch to curve and ys
    ax_tree2.plot([0.5, 0.5], [0.85, 0.7], "k-", linewidth=1, alpha=0.6)
    ax_tree2.plot([0.3, 0.7], [0.7, 0.7], "k-", linewidth=1, alpha=0.6)

    # Curve branch
    ax_tree2.plot([0.3, 0.3], [0.7, 0.6], "k-", linewidth=1, alpha=0.6)
    ax_tree2.text(
        0.3,
        0.55,
        "curve",
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"),
    )

    # Sub-branches for curve
    ax_tree2.plot([0.3, 0.3], [0.5, 0.4], "k-", linewidth=0.8, alpha=0.6)
    ax_tree2.plot([0.2, 0.4], [0.4, 0.4], "k-", linewidth=0.8, alpha=0.6)
    ax_tree2.text(0.2, 0.35, "a", ha="center", va="top", fontsize=9)
    ax_tree2.text(0.3, 0.35, "b", ha="center", va="top", fontsize=9)
    ax_tree2.text(0.4, 0.35, "c", ha="center", va="top", fontsize=9)

    # ys branch
    ax_tree2.plot([0.7, 0.7], [0.7, 0.6], "k-", linewidth=1, alpha=0.6)
    ax_tree2.text(
        0.7,
        0.55,
        "ys[0:10]",
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"),
    )

    # Show vectorized structure
    ax_tree2.plot([0.7, 0.7], [0.5, 0.4], "k-", linewidth=0.8, alpha=0.6)
    ax_tree2.plot([0.6, 0.8], [0.4, 0.4], "k-", linewidth=0.8, alpha=0.6)

    for i, x in enumerate([0.62, 0.7, 0.78]):
        ax_tree2.plot([x, x], [0.4, 0.35], "k-", linewidth=0.8, alpha=0.6)
        if i == 1:
            ax_tree2.text(x, 0.32, "...", ha="center", va="top", fontsize=9)
        else:
            ax_tree2.text(x, 0.32, f"obs[{i * 9}]", ha="center", va="top", fontsize=8)

    ax_tree2.set_title("Single trace structure", fontsize=14)

    # Add overall title
    fig.suptitle(
        "Two Natural Vectorization Patterns", fontsize=20, fontweight="bold", y=0.98
    )

    # Add pattern labels
    fig.text(
        0.02,
        0.95,
        "Pattern 1: Multiple Independent Models",
        fontsize=16,
        fontweight="bold",
        color="#333333",
    )
    fig.text(
        0.02,
        0.48,
        "Pattern 2: Single Model, Multiple Observations",
        fontsize=16,
        fontweight="bold",
        color="#333333",
    )

    # Save figure
    # Don't use tight_layout with absolute positioning
    fig.savefig(
        "examples/curvefit/figs/002_vectorization_patterns.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("✓ Saved vectorization patterns figure")
