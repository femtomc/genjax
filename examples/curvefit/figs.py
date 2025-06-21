"""
Clean figure generation for curvefit case study.
Focuses on essential comparisons: IS (1000 particles) vs HMC methods.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.utils import benchmark_with_warmup

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

    Args:
        seed: Random seed for reproducibility
        n_points: Number of data points
        use_easy: If True, use easier dataset for importance sampling

    Returns:
        dict with xs, ys, true_params, trace
    """
    from examples.curvefit.data import (
        generate_easy_inference_dataset,
        generate_test_dataset,
    )

    if use_easy:
        # Use easier dataset that works better with importance sampling
        return generate_easy_inference_dataset(seed=seed, n_points=5)
    else:
        # Use standard dataset generation
        return generate_test_dataset(seed=seed, n_points=n_points)


def save_framework_comparison_figure(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=2,
    seed=42,
    timing_repeats=10,
    figsize=(12, 8),
):
    """
    Create a clean framework comparison figure.

    Compares:
    - GenJAX IS with 1000 particles
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
        infer_latents_easy,
        hmc_infer_latents_jit,
        numpyro_run_hmc_inference_jit,
    )
    from genjax.core import Const
    from genjax.pjax import seed as genjax_seed
    import jax

    print("=== Framework Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is} (fixed)")
    print(f"HMC samples: {n_samples_hmc}")
    print(f"HMC warmup: {n_warmup} (critical for convergence)")

    # Use easier dataset for better importance sampling performance
    data = get_reference_dataset(n_points=5, use_easy=True)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]
    noise_std = data["true_params"]["noise_std"]

    print(f"True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    print(f"Using easier inference with noise_std={noise_std:.3f}")

    results = {}

    # 1. GenJAX Importance Sampling (1000 particles) - using easier model
    print("\n1. GenJAX IS (1000 particles) with easier model...")

    # Create seeded and jitted version of easy inference
    seeded_infer_easy = genjax_seed(infer_latents_easy)
    infer_latents_easy_jit = jax.jit(seeded_infer_easy)

    def is_task():
        return infer_latents_easy_jit(
            jrand.key(seed), xs, ys, Const(n_samples_is), Const(noise_std)
        )

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

    results["genjax_is"] = {
        "a": is_a_plot,
        "b": is_b_plot,
        "c": is_c_plot,
        "timing": is_timing_stats,
        "method": "GenJAX IS (1000)",
    }

    print(
        f"  Time: {is_timing_stats[0] * 1000:.1f} ± {is_timing_stats[1] * 1000:.1f} ms"
    )
    print(f"  IS a mean: {is_a_plot.mean():.3f}, std: {is_a_plot.std():.3f}")
    print(f"  IS b mean: {is_b_plot.mean():.3f}, std: {is_b_plot.std():.3f}")
    print(f"  IS c mean: {is_c_plot.mean():.3f}, std: {is_c_plot.std():.3f}")

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
    print(
        f"  GenJAX HMC: {len(hmc_a)} total samples, thinned by {thin_factor} for plotting"
    )

    # 3. NumPyro HMC
    print("\n3. NumPyro HMC...")

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
    print(
        f"  NumPyro HMC: {len(numpyro_a)} total samples, thinned by {thin_factor} for plotting"
    )

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.5)

    # Colors following visualization guide - distinct, colorblind-friendly palette
    colors = {
        "genjax_is": "#0173B2",  # Blue
        "genjax_hmc": "#DE8F05",  # Orange
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

        # Plot subset of curves - increase to 500 for better visualization
        n_plot = min(500, len(a_samples))
        print(f"  Plotting {n_plot} curves for {result['method']}")

        for i in range(n_plot):
            curve = a_samples[i] + b_samples[i] * x_fine + c_samples[i] * x_fine**2
            ax1.plot(x_fine, curve, color=color, alpha=0.01, linewidth=0.6)

        # Calculate and plot mean curve
        mean_a = np.mean(a_samples)
        mean_b = np.mean(b_samples)
        mean_c = np.mean(c_samples)
        mean_curve = mean_a + mean_b * x_fine + mean_c * x_fine**2
        ax1.plot(
            x_fine,
            mean_curve,
            color=color,
            linewidth=2.5,
            alpha=1.0,
            linestyle="-",
            zorder=20,
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

    # Simple legend for true curve and data only
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

    ax1.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=14)

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
    fig = plt.figure(figsize=(4, 3))
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


def visualize_multipoint_trace(trace, figsize=(6, 4), yrange=None, ax=None):
    """Visualize a multipoint trace."""
    curve, (xs, ys) = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)

    if ax is None:
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Use seeded simulation for reproducibility
    seeded_simulate = genjax_seed(npoint_curve.simulate)

    for i, ax in enumerate(axes):
        # Generate trace with seed based on index
        # First trace (i=0) uses seed=42, which is our reference dataset
        trace = seeded_simulate(jrand.key(42 + i), xs)

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

    fig = plt.figure(figsize=(6, 4))

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

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
    fig, ax = plt.subplots(figsize=(6, 5))

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
    fig = plt.figure(figsize=(15, 10))

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
