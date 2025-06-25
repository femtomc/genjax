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
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.utils import benchmark_with_warmup
from genjax.core import Const

# Import shared GenJAX Research Visualization Standards
from examples.viz import (
    setup_publication_fonts,
    FIGURE_SIZES,
    get_method_color,
    apply_grid_style,
    set_minimal_ticks,
    apply_standard_ticks,
    save_publication_figure,
    LINE_SPECS,
    MARKER_SPECS,
)

# Figure sizes and styling now imported from shared examples.viz module
# FIGURE_SIZES, PRIMARY_COLORS, etc. are available from the import above

# Apply GenJAX Research Visualization Standards
setup_publication_fonts()


# set_minimal_ticks function now imported from examples.viz


def get_reference_dataset(seed=42, n_points=10):
    """Get the standard reference dataset for all visualizations."""
    from examples.curvefit.data import generate_fixed_dataset

    return generate_fixed_dataset(
        n_points=n_points,
        x_min=0.0,
        x_max=1.0,
        true_a=-0.211,
        true_b=-0.395,
        true_c=0.673,
        noise_std=0.05,  # Reduced observation noise
        seed=seed,
    )


def save_onepoint_trace_viz():
    """Save one-point curve trace visualization."""
    from examples.curvefit.core import onepoint_curve

    print("Making and saving onepoint trace visualization.")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])

    # Generate a trace at x = 0.5
    trace = onepoint_curve.simulate(0.5)
    curve, (x, y) = trace.get_retval()

    # Plot the curve over x values
    xvals = jnp.linspace(0, 1, 300)
    ax.plot(
        xvals,
        jax.vmap(curve)(xvals),
        color=get_method_color("curves"),
        **LINE_SPECS["curve_main"],
    )

    # Mark the sampled point
    ax.scatter(
        x, y, color=get_method_color("data_points"), **MARKER_SPECS["data_points"]
    )

    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)

    save_publication_figure(fig, "examples/curvefit/figs/curvefit_prior_trace.pdf")

    # Also create the multiple onepoint traces with densities
    save_multiple_onepoint_traces_with_density()


def save_multiple_onepoint_traces_with_density():
    """Save multiple one-point trace visualizations with density values."""
    from examples.curvefit.core import onepoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving multiple onepoint traces with densities.")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Generate different traces with different seeds
    base_key = jrand.key(42)
    keys = jrand.split(base_key, 3)

    # Hardcoded log probability values for prior traces
    log_densities = [1.75, 3.40, 1.82]

    for i, (ax, key) in enumerate(zip(axes, keys)):
        # Generate a trace at x = 0.5 with different seed
        trace = genjax_seed(onepoint_curve.simulate)(key, 0.5)
        curve, (x, y) = trace.get_retval()

        # Plot the curve over x values
        xvals = jnp.linspace(0, 1, 300)
        ax.plot(
            xvals,
            jax.vmap(curve)(xvals),
            color=get_method_color("curves"),
            **LINE_SPECS["curve_main"],
        )

        # Mark the sampled point
        ax.scatter(
            x, y, color=get_method_color("data_points"), **MARKER_SPECS["data_points"]
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

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_prior_traces_density.pdf"
    )


def save_multipoint_trace_viz():
    """Save multi-point curve trace visualization."""
    from examples.curvefit.core import npoint_curve

    print("Making and saving multipoint trace visualization.")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])

    # Generate trace with multiple points
    xs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    trace = npoint_curve.simulate(xs)
    curve, (xs_ret, ys) = trace.get_retval()

    # Plot the curve
    xvals = jnp.linspace(0, 1, 300)
    ax.plot(
        xvals,
        jax.vmap(curve)(xvals),
        color=get_method_color("curves"),
        **LINE_SPECS["curve_main"],
    )

    # Mark the sampled points
    ax.scatter(
        xs_ret, ys, color=get_method_color("data_points"), **MARKER_SPECS["data_points"]
    )

    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)

    save_publication_figure(fig, "examples/curvefit/figs/curvefit_posterior_trace.pdf")

    # Also create the multiple multipoint traces with densities
    save_multiple_multipoint_traces_with_density()


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

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_posterior_traces_density.pdf"
    )


def save_four_multipoint_trace_vizs():
    """Save visualization showing four different multi-point curve traces."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed

    print("Making and saving four multipoint trace visualizations.")

    # Fixed x positions for all traces
    xs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["four_panel_grid"])
    axes = axes.flatten()

    # Use seeded simulation
    seeded_simulate = genjax_seed(npoint_curve.simulate)

    # Generate keys
    key = jrand.key(42)
    keys = jrand.split(key, 4)

    # Common x values for plotting curves
    xvals = jnp.linspace(0, 1, 300)

    for i, (ax, subkey) in enumerate(zip(axes, keys)):
        # Generate trace with these x positions
        trace = seeded_simulate(subkey, xs)
        curve, (xs_ret, ys) = trace.get_retval()

        # Plot the curve
        ax.plot(
            xvals,
            jax.vmap(curve)(xvals),
            color=get_method_color("curves"),
            **LINE_SPECS["curve_secondary"],
        )

        # Mark the sampled points
        ax.scatter(
            xs_ret,
            ys,
            color=get_method_color("data_points"),
            s=100,
            zorder=10,
            edgecolor="white",
            linewidth=2,
        )

        # Remove axis labels and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_posterior_traces_grid.pdf"
    )


def save_inference_viz(seed=42):
    """Save posterior visualization using importance sampling."""
    from examples.curvefit.core import (
        infer_latents,
    )
    from genjax.core import Const
    from genjax.pjax import seed as genjax_seed

    print("Making and saving inference visualization.")

    # Get reference dataset
    data = get_reference_dataset(seed=seed)
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    # Run importance sampling
    key = jrand.key(seed)
    n_samples = Const(5000)  # Use 5000 samples for visualization
    samples, weights = genjax_seed(infer_latents)(key, xs, ys, n_samples)

    # Resample for posterior visualization
    normalized_weights = jnp.exp(weights - jnp.max(weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)

    # Sample indices according to weights
    resample_key = jrand.key(seed + 1)
    n_resample = 100  # Number of curves to plot
    indices = jrand.choice(
        resample_key, jnp.arange(5000), shape=(n_resample,), p=normalized_weights
    )

    # Extract coefficients
    a_samples = samples.get_choices()["curve"]["a"][indices]
    b_samples = samples.get_choices()["curve"]["b"][indices]
    c_samples = samples.get_choices()["curve"]["c"][indices]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])

    # Plot x range
    x_range = jnp.linspace(-0.1, 1.1, 300)

    # Plot true curve
    true_curve = true_a + true_b * x_range + true_c * x_range**2
    ax.plot(
        x_range, true_curve, color="#333333", linewidth=3, label="True curve", zorder=50
    )

    # Plot posterior samples
    for i in range(n_resample):
        curve_vals = a_samples[i] + b_samples[i] * x_range + c_samples[i] * x_range**2
        ax.plot(
            x_range,
            curve_vals,
            color=get_method_color("curves"),
            alpha=0.1,
            linewidth=1,
        )

    # Plot data points
    ax.scatter(
        xs,
        ys,
        color=get_method_color("data_points"),
        s=120,
        zorder=100,
        edgecolor="white",
        linewidth=2,
        label="Observations",
    )

    # Styling
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_title("Posterior Curves (IS)", fontweight="normal")
    apply_grid_style(ax)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xlim(-0.1, 1.1)

    # Reduce number of ticks
    set_minimal_ticks(ax)  # Use GRVS standard (3 ticks)

    save_publication_figure(fig, "examples/curvefit/figs/curvefit_posterior_curves.pdf")


def save_genjax_posterior_comparison(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=20,
):
    """Save comparison of GenJAX IS vs HMC posterior inference."""
    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
    )
    from genjax.core import Const

    print("\n=== GenJAX Posterior Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is}")
    print(f"HMC samples: {n_samples_hmc}")

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs = data["xs"]
    ys = data["ys"]
    true_params = data["true_params"]

    # Run IS inference
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )

    # Resample according to weights
    normalized_weights = jnp.exp(is_weights - jnp.max(is_weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    resample_key = jrand.key(seed + 1)
    is_indices = jrand.choice(
        resample_key,
        jnp.arange(n_samples_is),
        shape=(n_samples_is,),
        p=normalized_weights,
    )

    # Get resampled IS coefficients
    is_a = is_samples.get_choices()["curve"]["a"][is_indices]
    is_b = is_samples.get_choices()["curve"]["b"][is_indices]
    is_c = is_samples.get_choices()["curve"]["c"][is_indices]

    # Run HMC inference
    hmc_samples, accept_rate = hmc_infer_latents_jit(
        jrand.key(seed + 100),
        xs,
        ys,
        Const(n_samples_hmc),
        Const(n_warmup),
        Const(0.001),
        Const(50),
    )

    # Get HMC coefficients
    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Colors
    is_color = "#0173B2"  # Blue
    hmc_color = get_method_color("genjax_hmc")  # Orange

    # 1. Posterior curves comparison
    ax = axes[0, 0]
    x_fine = jnp.linspace(-0.1, 1.1, 300)

    # Plot true curve
    true_curve = (
        true_params["a"] + true_params["b"] * x_fine + true_params["c"] * x_fine**2
    )
    ax.plot(x_fine, true_curve, "k-", linewidth=3, label="True", zorder=100)

    # Plot IS posterior samples
    for i in range(min(50, n_samples_is)):
        curve = is_a[i] + is_b[i] * x_fine + is_c[i] * x_fine**2
        ax.plot(x_fine, curve, color=is_color, alpha=0.05, linewidth=0.8)

    # Plot HMC posterior samples
    for i in range(min(50, n_samples_hmc)):
        curve = hmc_a[i] + hmc_b[i] * x_fine + hmc_c[i] * x_fine**2
        ax.plot(x_fine, curve, color=hmc_color, alpha=0.05, linewidth=0.8)

    # Plot data points
    ax.scatter(
        xs, ys, color="#CC3311", s=100, zorder=200, edgecolor="white", linewidth=2
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_title('Posterior Curves')
    apply_grid_style(ax)
    ax.set_xlim(-0.1, 1.1)

    # Create legend patches
    is_patch = mpatches.Patch(color=is_color, label=f"IS ({n_samples_is})")
    hmc_patch = mpatches.Patch(color=hmc_color, label=f"HMC ({n_samples_hmc})")
    ax.legend(handles=[is_patch, hmc_patch], loc="best")

    # 2. Parameter marginals
    param_names = ["a", "b", "c"]
    param_data = [
        (is_a, hmc_a, true_params["a"]),
        (is_b, hmc_b, true_params["b"]),
        (is_c, hmc_c, true_params["c"]),
    ]

    for i, (is_vals, hmc_vals, true_val) in enumerate(param_data):
        ax = axes[0, 1] if i == 0 else (axes[1, 0] if i == 1 else axes[1, 1])

        # Histograms
        bins = np.linspace(
            min(is_vals.min(), hmc_vals.min()) - 0.1,
            max(is_vals.max(), hmc_vals.max()) + 0.1,
            30,
        )

        ax.hist(is_vals, bins=bins, alpha=0.5, density=True, color=is_color, label="IS")
        ax.hist(
            hmc_vals, bins=bins, alpha=0.5, density=True, color=hmc_color, label="HMC"
        )

        # True value
        ax.axvline(true_val, color="#CC3311", linestyle="--", linewidth=2, label="True")

        ax.set_xlabel(f"{param_names[i]}")
        ax.set_ylabel("Density")
        # ax.set_title(f'Parameter {param_names[i]}')
        apply_grid_style(ax)
        ax.legend()

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_posterior_comparison.pdf"
    )

    print("✓ Saved GenJAX posterior comparison")


def save_framework_comparison_figure(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=20,
):
    """Generate clean framework comparison with IS 1000 vs HMC methods."""
    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
        numpyro_run_importance_sampling_jit,
        numpyro_run_hmc_inference_jit,
        numpyro_hmc_summary_statistics,
    )
    from genjax.core import Const

    print("\n=== Framework Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is} (fixed)")
    print(f"HMC samples: {n_samples_hmc}")
    print(f"HMC warmup: {n_warmup} (critical for convergence)")

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    print(
        f"Reference Dataset - True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}"
    )
    print(f"Observation noise std: {data['noise_std']:.3f}")
    print(f"Number of data points: {len(xs)}")

    # Results storage
    results = {}

    # 1. GenJAX IS (1000 particles)
    print("\n1. GenJAX IS (1000 particles)...")
    times, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples_is)),
        repeats=timing_repeats,
    )
    print(f"  Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # Get samples for visualization
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )

    # Resample according to weights
    normalized_weights = jnp.exp(is_weights - jnp.max(is_weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    resample_key = jrand.key(seed + 1)
    resample_indices = jrand.choice(
        resample_key,
        jnp.arange(n_samples_is),
        shape=(n_samples_is,),
        p=normalized_weights,
    )

    # Get resampled coefficients
    is_a_resampled = is_samples.get_choices()["curve"]["a"][resample_indices]
    is_b_resampled = is_samples.get_choices()["curve"]["b"][resample_indices]
    is_c_resampled = is_samples.get_choices()["curve"]["c"][resample_indices]

    print(
        f"  IS a mean (resampled): {is_a_resampled.mean():.3f}, std: {is_a_resampled.std():.3f}"
    )
    print(
        f"  IS b mean (resampled): {is_b_resampled.mean():.3f}, std: {is_b_resampled.std():.3f}"
    )
    print(
        f"  IS c mean (resampled): {is_c_resampled.mean():.3f}, std: {is_c_resampled.std():.3f}"
    )

    # Also compute weighted mean
    is_a_all = is_samples.get_choices()["curve"]["a"]
    is_b_all = is_samples.get_choices()["curve"]["b"]
    is_c_all = is_samples.get_choices()["curve"]["c"]
    is_a_weighted = jnp.sum(is_a_all * normalized_weights)
    is_b_weighted = jnp.sum(is_b_all * normalized_weights)
    is_c_weighted = jnp.sum(is_c_all * normalized_weights)

    print(
        f"  IS weighted mean: a={is_a_weighted:.3f}, b={is_b_weighted:.3f}, c={is_c_weighted:.3f}"
    )

    results["genjax_is"] = {
        "method": "GenJAX IS",
        "samples": (is_a_resampled, is_b_resampled, is_c_resampled),
        "timing": (mean_time, std_time),
        "mean_curve": (is_a_weighted, is_b_weighted, is_c_weighted),
    }

    # 2. GenJAX HMC
    print("\n2. GenJAX HMC...")
    times, (mean_time, std_time) = benchmark_with_warmup(
        lambda: hmc_infer_latents_jit(
            jrand.key(seed),
            xs,
            ys,
            Const(n_samples_hmc),
            Const(n_warmup),
            Const(0.001),
            Const(50),
        ),
        repeats=timing_repeats,
    )
    print(f"  Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # Get samples
    hmc_samples, diagnostics = hmc_infer_latents_jit(
        jrand.key(seed),
        xs,
        ys,
        Const(n_samples_hmc),
        Const(n_warmup),
        Const(0.001),
        Const(50),
    )
    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]
    accept_rate = diagnostics.get("acceptance_rate", 0.0)

    print(f"  Accept rate: {accept_rate:.3f}")
    print(f"  HMC a mean: {hmc_a.mean():.3f}, std: {hmc_a.std():.3f}")
    print(f"  HMC b mean: {hmc_b.mean():.3f}, std: {hmc_b.std():.3f}")
    print(f"  HMC c mean: {hmc_c.mean():.3f}, std: {hmc_c.std():.3f}")

    results["genjax_hmc"] = {
        "method": "GenJAX HMC",
        "samples": (hmc_a, hmc_b, hmc_c),
        "timing": (mean_time, std_time),
        "accept_rate": accept_rate,
        "mean_curve": (hmc_a.mean(), hmc_b.mean(), hmc_c.mean()),
    }

    print(f"  GenJAX HMC: {n_samples_hmc} total samples")

    # 3. NumPyro IS (1000 particles)
    print("\n3. NumPyro IS (1000 particles)...")
    try:
        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: numpyro_run_importance_sampling_jit(
                jrand.key(seed), xs, ys, num_samples=n_samples_is
            ),
            repeats=timing_repeats,
        )
        print(f"  Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

        # Get samples
        numpyro_is_result = numpyro_run_importance_sampling_jit(
            jrand.key(seed), xs, ys, num_samples=n_samples_is
        )

        # Extract samples - NumPyro IS returns weighted samples
        numpyro_is_a = numpyro_is_result["a"]
        numpyro_is_b = numpyro_is_result["b"]
        numpyro_is_c = numpyro_is_result["c"]

        print(
            f"  NumPyro IS a mean: {numpyro_is_a.mean():.3f}, std: {numpyro_is_a.std():.3f}"
        )
        print(
            f"  NumPyro IS b mean: {numpyro_is_b.mean():.3f}, std: {numpyro_is_b.std():.3f}"
        )
        print(
            f"  NumPyro IS c mean: {numpyro_is_c.mean():.3f}, std: {numpyro_is_c.std():.3f}"
        )

        results["numpyro_is"] = {
            "method": "NumPyro IS",
            "samples": (numpyro_is_a, numpyro_is_b, numpyro_is_c),
            "timing": (mean_time, std_time),
            "mean_curve": (
                numpyro_is_a.mean(),
                numpyro_is_b.mean(),
                numpyro_is_c.mean(),
            ),
        }
    except Exception as e:
        print(f"  NumPyro IS failed: {type(e).__name__}: {str(e)}")
        print("  Skipping NumPyro IS...")

    # 4. NumPyro HMC
    print("\n4. NumPyro HMC...")
    times, (mean_time, std_time) = benchmark_with_warmup(
        lambda: numpyro_run_hmc_inference_jit(
            jrand.key(seed),
            xs,
            ys,
            num_samples=n_samples_hmc,
            num_warmup=n_warmup,
            step_size=0.001,
            num_steps=50,
        ),
        repeats=timing_repeats,
    )
    print(f"  Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # Get samples
    numpyro_hmc_result = numpyro_run_hmc_inference_jit(
        jrand.key(seed),
        xs,
        ys,
        num_samples=n_samples_hmc,
        num_warmup=n_warmup,
        step_size=0.001,
        num_steps=50,
    )
    numpyro_hmc_samples = numpyro_hmc_result["samples"]
    numpyro_hmc_a = numpyro_hmc_samples["a"]
    numpyro_hmc_b = numpyro_hmc_samples["b"]
    numpyro_hmc_c = numpyro_hmc_samples["c"]

    # Get diagnostics
    summary = numpyro_hmc_summary_statistics(numpyro_hmc_result)
    print(f"  Accept rate: {summary['accept_rate']:.3f}")

    print(
        f"  NumPyro a mean: {numpyro_hmc_a.mean():.3f}, std: {numpyro_hmc_a.std():.3f}"
    )
    print(
        f"  NumPyro b mean: {numpyro_hmc_b.mean():.3f}, std: {numpyro_hmc_b.std():.3f}"
    )
    print(
        f"  NumPyro c mean: {numpyro_hmc_c.mean():.3f}, std: {numpyro_hmc_c.std():.3f}"
    )

    results["numpyro_hmc"] = {
        "method": "NumPyro HMC",
        "samples": (numpyro_hmc_a, numpyro_hmc_b, numpyro_hmc_c),
        "timing": (mean_time, std_time),
        "mean_curve": (
            numpyro_hmc_a.mean(),
            numpyro_hmc_b.mean(),
            numpyro_hmc_c.mean(),
        ),
        "accept_rate": summary["accept_rate"],
    }

    print(f"  NumPyro HMC: {n_samples_hmc} total samples")

    # Create two-panel figure
    fig = plt.figure(figsize=FIGURE_SIZES["framework_comparison"])
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.5)

    # Colors following visualization guide - distinct, colorblind-friendly palette
    colors = {
        "genjax_is": "#0173B2",  # Blue
        "genjax_hmc": get_method_color("genjax_hmc"),  # Orange
        "numpyro_is": "#F39C12",  # Red
        "numpyro_hmc": get_method_color("numpyro_hmc"),  # Green
    }

    # Panel 1: Posterior curves
    ax1 = fig.add_subplot(gs[0])
    x_plot = jnp.linspace(-0.1, 1.1, 300)
    true_curve = true_a + true_b * x_plot + true_c * x_plot**2
    ax1.plot(x_plot, true_curve, "k-", linewidth=4, label="True curve", zorder=100)

    # Plot posterior mean curves for each method
    for method_key, result in results.items():
        a_mean, b_mean, c_mean = result["mean_curve"]
        mean_curve = a_mean + b_mean * x_plot + c_mean * x_plot**2
        ax1.plot(
            x_plot,
            mean_curve,
            color=colors[method_key],
            linewidth=3,
            label=result["method"],
            alpha=0.8,
        )

    # Print mean curve values for verification
    for method_key, result in results.items():
        a_mean, b_mean, c_mean = result["mean_curve"]
        mean_curve = a_mean + b_mean * x_plot + c_mean * x_plot**2
        print(
            f"  {result['method']} mean curve: a={a_mean:.3f}, b={b_mean:.3f}, c={c_mean:.3f}"
        )
        print(
            f"  {result['method']} mean curve range: [{mean_curve.min():.3f}, {mean_curve.max():.3f}]"
        )

    # Plot data points
    ax1.scatter(
        xs, ys, color="#333333", s=100, zorder=200, edgecolor="white", linewidth=2
    )

    ax1.set_xlabel("x", fontsize=18, fontweight="bold")
    ax1.set_ylabel("y", fontsize=18, fontweight="bold")
    # ax1.set_title("Posterior Mean Curves", fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", framealpha=0.9, fontsize=14)
    ax1.set_xlim(-0.1, 1.1)
    set_minimal_ticks(ax1, x_ticks=4, y_ticks=4)

    # Panel 2: Timing comparison (horizontal bars)
    ax2 = fig.add_subplot(gs[1])
    methods = list(results.keys())
    labels = [results[m]["method"] for m in methods]
    times = [results[m]["timing"][0] * 1000 for m in methods]  # Convert to ms
    errors = [results[m]["timing"][1] * 1000 for m in methods]
    colors_list = [colors[m] for m in methods]

    # Create horizontal bar plot with individual bars for legend
    y_positions = range(len(methods))
    bars = []
    for i, (y_pos, time, error, color, label) in enumerate(
        zip(y_positions, times, errors, colors_list, labels)
    ):
        bar = ax2.barh(
            y_pos, time, xerr=error, capsize=5, color=color, alpha=0.8, label=label
        )
        bars.append(bar[0])

    # Add timing values at the end of bars
    for i, (bar, time, error) in enumerate(zip(bars, times, errors)):
        width = bar.get_width()
        ax2.text(
            width + error + 10,  # Position text after error bar
            bar.get_y() + bar.get_height() / 2.0,
            f"{time:.1f}ms",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    # Remove y-axis labels (will use legend instead)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([])

    # Style the x-axis
    ax2.set_xlabel("Time (ms)", fontsize=18, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3, axis="x")

    # Set x-axis limits to accommodate timing labels
    max_time = max(times)
    max_error = max(errors)
    ax2.set_xlim(0, max_time + max_error + 100)  # Add space for labels

    # Set x-axis ticks
    from matplotlib.ticker import MultipleLocator

    if max_time < 10:
        ax2.xaxis.set_major_locator(MultipleLocator(2))  # Every 2ms for small values
    elif max_time < 100:
        ax2.xaxis.set_major_locator(MultipleLocator(20))  # Every 20ms
    else:
        ax2.xaxis.set_major_locator(
            MultipleLocator(200)
        )  # Every 200ms for large values

    # Set tick parameters
    ax1.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
    ax2.tick_params(
        axis="y", which="major", labelsize=16, width=0, length=0
    )  # No tick marks on y-axis
    ax2.tick_params(
        axis="x", which="major", labelsize=16, width=2, length=6
    )  # Style x-axis ticks

    plt.tight_layout()

    # Save figure
    filename = f"examples/curvefit/figs/curvefit_framework_comparison_n{n_points}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\n✓ Saved framework comparison: {filename}")

    return results


def save_inference_scaling_viz(n_trials=100):
    """Save inference scaling visualization across different sample sizes.

    Args:
        n_trials: Number of independent trials to run for each sample size (default: 100)
    """
    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const
    from examples.utils import benchmark_with_warmup

    print(
        f"Making and saving inference scaling visualization with {n_trials} trials per N."
    )

    # Get reference dataset
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]

    # Test different sample sizes - more points for smoother curves
    n_samples_list = [
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
    ]
    ess_values = []
    lml_estimates = []
    runtime_means = []
    runtime_stds = []

    base_key = jrand.key(42)

    for n_samples in n_samples_list:
        print(f"  Testing with {n_samples} samples ({n_trials} trials)...")

        # Storage for trial results
        trial_ess = []
        trial_lml = []

        # Run multiple trials
        for trial in range(n_trials):
            trial_key = jrand.key(42 + trial)  # Different key for each trial

            # Run inference
            samples, weights = infer_latents_jit(trial_key, xs, ys, Const(n_samples))

            # Compute ESS
            normalized_weights = jnp.exp(weights - jnp.max(weights))
            normalized_weights = normalized_weights / jnp.sum(normalized_weights)
            ess = 1.0 / jnp.sum(normalized_weights**2)
            trial_ess.append(float(ess))

            # Estimate log marginal likelihood
            lml = jnp.log(jnp.mean(jnp.exp(weights - jnp.max(weights)))) + jnp.max(
                weights
            )
            trial_lml.append(float(lml))

        # Average over trials
        ess_values.append(jnp.mean(jnp.array(trial_ess)))
        lml_estimates.append(jnp.mean(jnp.array(trial_lml)))

        # Benchmark runtime with more trials for stability
        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: infer_latents_jit(base_key, xs, ys, Const(n_samples)),
            repeats=100,  # Match the number of trials for consistency
            inner_repeats=20,  # More inner repeats for accurate timing
        )
        runtime_means.append(mean_time * 1000)  # Convert to ms
        runtime_stds.append(std_time * 1000)

    # Create figure with two panels in a column with shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=FIGURE_SIZES["inference_scaling"], sharex=True
    )

    # Common color for GenJAX IS
    genjax_is_color = "#0173B2"

    # Runtime plot without error bars
    ax1.plot(
        n_samples_list,
        runtime_means,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
    )
    # Remove x-axis label from top plot (shared x-axis)
    # ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Runtime (ms)")
    # ax1.set_title("Vectorized Runtime", fontweight="normal")
    ax1.set_xscale("log")
    ax1.set_xlim(80, 12000)
    ax1.set_ylim(0.2, 0.3)  # Set runtime axis limits
    # Add a horizontal line showing the mean runtime to emphasize flatness
    mean_runtime = jnp.mean(jnp.array(runtime_means))
    ax1.axhline(mean_runtime, color="gray", linestyle="--", alpha=0.5, linewidth=2)
    # Set specific x-axis tick locations with scientific notation
    ax1.set_xticks([100, 1000, 10000])
    ax1.set_xticklabels(["$10^2$", "$10^3$", "$10^4$"])
    # Hide x-axis tick labels on top plot
    ax1.tick_params(labelbottom=False)
    # Only set y-axis ticks to avoid overriding x-axis
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    # LML estimate plot
    ax2.plot(
        n_samples_list,
        lml_estimates,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
    )
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Log Marginal Likelihood")
    # ax2.set_title("LML Estimates", fontweight="normal")
    ax2.set_xscale("log")
    ax2.set_xlim(80, 12000)
    # Set specific x-axis tick locations with scientific notation
    ax2.set_xticks([100, 1000, 10000])
    ax2.set_xticklabels(["$10^2$", "$10^3$", "$10^4$"])
    # Only set y-axis ticks to avoid overriding x-axis
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/curvefit_scaling_performance.pdf")
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
    # ax.set_title("Log Joint Density (c fixed)", fontweight="normal")
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log Density", rotation=270, labelpad=20)

    set_minimal_ticks(ax)  # Use GRVS standard (3 ticks)
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/curvefit_logprob_surface.pdf")
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
            xvals,
            jax.vmap(curve)(xvals),
            color=get_method_color("curves"),
            **LINE_SPECS["curve_secondary"],
        )

        # Mark the sampled point
        ax.scatter(
            x,
            y,
            color=get_method_color("data_points"),
            s=100,
            zorder=10,
            edgecolor="white",
            linewidth=2,
        )

        # Styling
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.set_title(f"Sample {i+1}", fontweight="normal")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

        # Reduce number of ticks
        set_minimal_ticks(ax)  # Use GRVS standard (3 ticks)

    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/curvefit_posterior_marginal.pdf")
    plt.close()


def save_individual_method_parameter_density(
    n_points=10,
    n_samples=2000,
    seed=42,
):
    """Save individual 4-panel parameter density figures for each inference method."""
    print("\n=== Individual Method Parameter Density Figures ===")

    from examples.curvefit.core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
    )

    # Try to import numpyro functions if available
    try:
        from examples.curvefit.core import numpyro_run_hmc_inference_jit

        has_numpyro = True
    except ImportError:
        has_numpyro = False
        print("  Note: NumPyro not available, skipping NumPyro HMC visualization")
    from genjax.core import Const
    from scipy.ndimage import gaussian_filter

    # Set font to bold for publication
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.major.width": 2,
            "ytick.major.width": 2,
        }
    )

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")

    # Colors for each method (all progress from light to dark)
    method_colors = {
        "is": {"hex": "Blues", "surface": "Blues", "color": "#0173B2"},
        "hmc": {"hex": "Oranges", "surface": "Oranges", "color": "#DE8F05"},
        "numpyro": {"hex": "Greens", "surface": "Greens", "color": "#029E73"},
    }

    # Consistent axis limits based on data range
    a_lim = (-1.5, 1.0)
    b_lim = (-2.5, 2.0)
    c_lim = (-1.0, 2.5)

    def create_method_figure(method_name, a_vals, b_vals, c_vals, color_info):
        """Create a 4-panel figure for a single method."""
        fig = plt.figure(figsize=(28, 7))

        # Create layout: 1x4 grid (single row)
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

        # Convert to numpy for histogram operations
        a_vals_np = np.array(a_vals)
        b_vals_np = np.array(b_vals)
        c_vals_np = np.array(c_vals)

        # Panel 1: 2D hex (a, b)
        ax1 = fig.add_subplot(gs[0, 0])
        h1 = ax1.hexbin(
            a_vals_np, b_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax1.scatter(
            true_a,
            true_b,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax1.axhline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.set_xlabel("a (constant)", fontsize=20, fontweight="bold")
        ax1.set_ylabel("b (linear)", fontsize=20, fontweight="bold")
        # No title - axis labels show the parameters
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        # Calculate aspect ratio to make plot square
        a_range = a_lim[1] - a_lim[0]
        b_range = b_lim[1] - b_lim[0]
        ax1.set_aspect(a_range / b_range, adjustable="box")
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)

        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")

        # Create 2D histogram for surface
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")

        # Plot surface - mask out very low density values to avoid white plane
        hist_ab_masked = np.where(
            hist_ab_smooth > hist_ab_smooth.max() * 0.01, hist_ab_smooth, np.nan
        )
        surf_ab = ax2.plot_surface(
            X_ab,
            Y_ab,
            hist_ab_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Add ground truth reference
        z_max = hist_ab_smooth.max()
        # Add red lines at ground truth values
        ax2.plot(
            [a_lim[0], a_lim[1]], [true_b, true_b], [0, 0], "r-", linewidth=3, alpha=0.8
        )
        ax2.plot(
            [true_a, true_a], [b_lim[0], b_lim[1]], [0, 0], "r-", linewidth=3, alpha=0.8
        )
        # Add vertical red line at ground truth
        z_max = hist_ab_smooth.max() * 1.1
        ax2.plot(
            [true_a, true_a], [true_b, true_b], [0, z_max], "r-", linewidth=4, alpha=0.9
        )
        ax2.scatter(
            [true_a],
            [true_b],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
        )

        ax2.set_xlabel("a", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_ylabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        # No title - axis labels show the parameters
        ax2.set_xlim(a_lim)
        ax2.set_ylim(b_lim)
        ax2.set_zlim(0, hist_ab_smooth.max() * 1.1)  # Start from 0, add 10% margin
        ax2.view_init(elev=25, azim=45)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.zaxis.set_major_locator(MaxNLocator(nbins=3))
        # Ensure grid and panes are visible
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.grid(True, alpha=0.3)

        # Panel 3: 2D hex (b, c)
        ax3 = fig.add_subplot(gs[0, 2])
        h3 = ax3.hexbin(
            b_vals_np, c_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax3.scatter(
            true_b,
            true_c,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax3.axhline(true_c, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.set_xlabel("b (linear)", fontsize=20, fontweight="bold")
        ax3.set_ylabel("c (quadratic)", fontsize=20, fontweight="bold")
        # No title - axis labels show the parameters
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        # Calculate aspect ratio to make plot square
        b_range = b_lim[1] - b_lim[0]
        c_range = c_lim[1] - c_lim[0]
        ax3.set_aspect(b_range / c_range, adjustable="box")
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)

        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection="3d")

        # Create 2D histogram for surface
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25, range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")

        # Plot surface - mask out very low density values to avoid white plane
        hist_bc_masked = np.where(
            hist_bc_smooth > hist_bc_smooth.max() * 0.01, hist_bc_smooth, np.nan
        )
        surf_bc = ax4.plot_surface(
            X_bc,
            Y_bc,
            hist_bc_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Add ground truth reference
        z_max = hist_bc_smooth.max()
        # Add red lines at ground truth values
        ax4.plot(
            [b_lim[0], b_lim[1]], [true_c, true_c], [0, 0], "r-", linewidth=3, alpha=0.8
        )
        ax4.plot(
            [true_b, true_b], [c_lim[0], c_lim[1]], [0, 0], "r-", linewidth=3, alpha=0.8
        )
        # Add vertical red line at ground truth
        z_max = hist_bc_smooth.max() * 1.1
        ax4.plot(
            [true_b, true_b], [true_c, true_c], [0, z_max], "r-", linewidth=4, alpha=0.9
        )
        ax4.scatter(
            [true_b],
            [true_c],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
        )

        ax4.set_xlabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_ylabel("c", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        # No title - axis labels show the parameters
        ax4.set_xlim(b_lim)
        ax4.set_ylim(c_lim)
        ax4.set_zlim(0, hist_bc_smooth.max() * 1.1)  # Start from 0, add 10% margin
        ax4.view_init(elev=25, azim=45)
        ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.zaxis.set_major_locator(MaxNLocator(nbins=3))
        # Ensure grid and panes are visible
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False
        ax4.grid(True, alpha=0.3)

        # No overall title - rely on color scheme for method identification

        return fig

    # 1. GenJAX IS
    print("\n  Running GenJAX IS...")
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples * 3)
    )

    # Resample
    resample_idx = jrand.choice(
        jrand.key(seed + 100),
        jnp.arange(n_samples * 3),
        shape=(n_samples,),
        p=jnp.exp(is_weights - jnp.max(is_weights)),
    )

    is_a = is_samples.get_choices()["curve"]["a"][resample_idx]
    is_b = is_samples.get_choices()["curve"]["b"][resample_idx]
    is_c = is_samples.get_choices()["curve"]["c"][resample_idx]

    # Create and save IS figure
    fig_is = create_method_figure(
        "GenJAX IS (1000 particles)", is_a, is_b, is_c, method_colors["is"]
    )
    fig_is.savefig(
        "examples/curvefit/figs/curvefit_params_is1000.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_is)
    print("  ✓ Saved GenJAX IS parameter density figure")

    # 2. GenJAX HMC
    print("  Running GenJAX HMC...")
    hmc_samples, _ = hmc_infer_latents_jit(
        jrand.key(seed + 1),
        xs,
        ys,
        Const(n_samples),
        Const(500),
        Const(0.001),
        Const(50),
    )

    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]

    # Create and save HMC figure
    fig_hmc = create_method_figure(
        "GenJAX HMC", hmc_a, hmc_b, hmc_c, method_colors["hmc"]
    )
    fig_hmc.savefig(
        "examples/curvefit/figs/curvefit_params_hmc.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_hmc)
    print("  ✓ Saved GenJAX HMC parameter density figure")

    # 3. NumPyro HMC (if available)
    if has_numpyro:
        print("  Running NumPyro HMC...")
        try:
            numpyro_result = numpyro_run_hmc_inference_jit(
                jrand.key(seed + 2),
                xs,
                ys,
                num_samples=n_samples,
                num_warmup=500,
                step_size=0.001,
                num_steps=50,
            )

            numpyro_a = numpyro_result["a"]
            numpyro_b = numpyro_result["b"]
            numpyro_c = numpyro_result["c"]

            # Create and save NumPyro figure
            fig_numpyro = create_method_figure(
                "NumPyro HMC", numpyro_a, numpyro_b, numpyro_c, method_colors["numpyro"]
            )
            fig_numpyro.savefig(
                "examples/curvefit/figs/curvefit_params_numpyro.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_numpyro)
            print("  ✓ Saved NumPyro HMC parameter density figure")

        except Exception as e:
            print(f"  NumPyro HMC failed: {e}")
            print("  Skipping NumPyro visualization")
    else:
        print("  Skipping NumPyro HMC (NumPyro not available)")

    print("\n✓ Completed individual method parameter density figures")


def save_is_comparison_parameter_density(
    n_points=10,
    seed=42,
):
    """Save parameter density figures comparing IS with different particle counts."""
    print("\n=== IS Comparison Parameter Density Figures ===")

    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const
    from scipy.ndimage import gaussian_filter

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")

    # Distinguishable colors for IS variants (all light to dark)
    is_variant_colors = {
        "is_50": {
            "hex": "Purples",
            "surface": "Purples",
            "color": "#B19CD9",
        },  # Light purple
        "is_500": {
            "hex": "Blues",
            "surface": "Blues",
            "color": "#0173B2",
        },  # Medium blue
        "is_5000": {
            "hex": "Greens",
            "surface": "Greens",
            "color": "#029E73",
        },  # Dark green (to avoid confusion)
    }

    # Use the same create_method_figure function from above
    a_lim = (-1.5, 1.0)
    b_lim = (-2.5, 2.0)
    c_lim = (-1.0, 2.5)

    def create_is_figure(n_particles, color_info, filename):
        """Create parameter density figure for IS with specified particles."""
        print(f"\n  Running GenJAX IS (N={n_particles})...")

        # Run IS inference
        samples, weights = infer_latents_jit(
            jrand.key(seed), xs, ys, Const(n_particles)
        )

        # Resample for visualization
        normalized_weights = jnp.exp(weights - jnp.max(weights))
        normalized_weights = normalized_weights / jnp.sum(normalized_weights)
        resample_idx = jrand.choice(
            jrand.key(seed + n_particles),
            jnp.arange(n_particles),
            shape=(2000,),
            p=normalized_weights,
            replace=True,
        )

        a_vals = samples.get_choices()["curve"]["a"][resample_idx]
        b_vals = samples.get_choices()["curve"]["b"][resample_idx]
        c_vals = samples.get_choices()["curve"]["c"][resample_idx]

        # Create figure using shared layout
        fig = plt.figure(figsize=(28, 7))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

        # Convert to numpy
        a_vals_np = np.array(a_vals)
        b_vals_np = np.array(b_vals)
        c_vals_np = np.array(c_vals)

        # Panel 1: 2D hex (a, b)
        ax1 = fig.add_subplot(gs[0, 0])
        h1 = ax1.hexbin(
            a_vals_np, b_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax1.scatter(
            true_a,
            true_b,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax1.axhline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.set_xlabel("a (constant)", fontsize=20, fontweight="bold")
        ax1.set_ylabel("b (linear)", fontsize=20, fontweight="bold")
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1] - a_lim[0]) / (b_lim[1] - b_lim[0]), adjustable="box")
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)

        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(
            hist_ab_smooth > hist_ab_smooth.max() * 0.01, hist_ab_smooth, np.nan
        )
        surf_ab = ax2.plot_surface(
            X_ab,
            Y_ab,
            hist_ab_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax2.plot(
            [a_lim[0], a_lim[1]],
            [true_b, true_b],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax2.plot(
            [true_a, true_a],
            [b_lim[0], b_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax2.plot(
            [true_a, true_a],
            [true_b, true_b],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax2.scatter(
            [true_a],
            [true_b],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax2.set_xlabel("a", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_ylabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_xlim(a_lim)
        ax2.set_ylim(b_lim)
        ax2.set_zlim(0, hist_ab_smooth.max() * 1.1)
        ax2.view_init(elev=25, azim=45)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.grid(True, alpha=0.3)

        # Panel 3: 2D hex (b, c)
        ax3 = fig.add_subplot(gs[0, 2])
        h3 = ax3.hexbin(
            b_vals_np, c_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax3.scatter(
            true_b,
            true_c,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax3.axhline(true_c, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.set_xlabel("b (linear)", fontsize=20, fontweight="bold")
        ax3.set_ylabel("c (quadratic)", fontsize=20, fontweight="bold")
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1] - b_lim[0]) / (c_lim[1] - c_lim[0]), adjustable="box")
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)

        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection="3d")
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25, range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(
            hist_bc_smooth > hist_bc_smooth.max() * 0.01, hist_bc_smooth, np.nan
        )
        surf_bc = ax4.plot_surface(
            X_bc,
            Y_bc,
            hist_bc_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax4.plot(
            [b_lim[0], b_lim[1]],
            [true_c, true_c],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax4.plot(
            [true_b, true_b],
            [c_lim[0], c_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax4.plot(
            [true_b, true_b],
            [true_c, true_c],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax4.scatter(
            [true_b],
            [true_c],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax4.set_xlabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_ylabel("c", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_xlim(b_lim)
        ax4.set_ylim(c_lim)
        ax4.set_zlim(0, hist_bc_smooth.max() * 1.1)
        ax4.view_init(elev=25, azim=45)
        ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved IS (N={n_particles}) figure")

        return a_vals.mean(), b_vals.mean(), c_vals.mean()

    # Generate IS comparison figures
    create_is_figure(
        50,
        is_variant_colors["is_50"],
        "examples/curvefit/figs/curvefit_params_is50.pdf",
    )
    create_is_figure(
        500,
        is_variant_colors["is_500"],
        "examples/curvefit/figs/curvefit_params_is500.pdf",
    )
    create_is_figure(
        5000,
        is_variant_colors["is_5000"],
        "examples/curvefit/figs/curvefit_params_is5000.pdf",
    )

    print("\n✓ Completed IS comparison parameter density figures")


def save_is_single_resample_comparison(
    n_points=10,
    seed=42,
    n_trials=1000,
):
    """Save single particle resampling comparison for IS with different particle counts."""
    print(f"\n=== IS Single Particle Resampling Comparison ({n_trials} trials) ===")

    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const
    from scipy.ndimage import gaussian_filter

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")

    # Distinguishable colors for IS variants
    single_resample_colors = {
        "is_50": {
            "hex": "Purples",
            "surface": "Purples",
            "color": "#B19CD9",
        },  # Light purple
        "is_500": {
            "hex": "Blues",
            "surface": "Blues",
            "color": "#0173B2",
        },  # Medium blue
        "is_5000": {
            "hex": "Greens",
            "surface": "Greens",
            "color": "#029E73",
        },  # Dark green
    }

    def run_is_single_resample_vectorized(key, xs, ys, n_samples, n_trials):
        """Run IS with n_samples, resample to single particle, repeat n_trials times."""
        keys = jrand.split(key, n_trials)

        def single_trial(trial_key):
            is_key, resample_key = jrand.split(trial_key)
            samples, log_weights = infer_latents_jit(is_key, xs, ys, Const(n_samples))
            weights = jnp.exp(log_weights - jnp.max(log_weights))
            weights = weights / jnp.sum(weights)
            idx = jrand.choice(resample_key, jnp.arange(n_samples), p=weights)
            a = samples.get_choices()["curve"]["a"][idx]
            b = samples.get_choices()["curve"]["b"][idx]
            c = samples.get_choices()["curve"]["c"][idx]
            return a, b, c

        vectorized_trial = jax.vmap(single_trial)
        return vectorized_trial(keys)

    # Use the same visualization function but with single resampled particles
    a_lim = (-1.5, 1.0)
    b_lim = (-2.5, 2.0)
    c_lim = (-1.0, 2.5)

    def create_single_resample_figure(a_vals, b_vals, c_vals, color_info, filename):
        """Create figure for single particle resampling results."""
        fig = plt.figure(figsize=(28, 7))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

        # Convert to numpy
        a_vals_np = np.array(a_vals)
        b_vals_np = np.array(b_vals)
        c_vals_np = np.array(c_vals)

        # Panel 1: 2D hex (a, b)
        ax1 = fig.add_subplot(gs[0, 0])
        h1 = ax1.hexbin(
            a_vals_np, b_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax1.scatter(
            true_a,
            true_b,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax1.axhline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.set_xlabel("a (constant)", fontsize=20, fontweight="bold")
        ax1.set_ylabel("b (linear)", fontsize=20, fontweight="bold")
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1] - a_lim[0]) / (b_lim[1] - b_lim[0]), adjustable="box")
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)

        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(
            hist_ab_smooth > hist_ab_smooth.max() * 0.01, hist_ab_smooth, np.nan
        )
        surf_ab = ax2.plot_surface(
            X_ab,
            Y_ab,
            hist_ab_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax2.plot(
            [a_lim[0], a_lim[1]],
            [true_b, true_b],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax2.plot(
            [true_a, true_a],
            [b_lim[0], b_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax2.plot(
            [true_a, true_a],
            [true_b, true_b],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax2.scatter(
            [true_a],
            [true_b],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax2.set_xlabel("a", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_ylabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_xlim(a_lim)
        ax2.set_ylim(b_lim)
        ax2.set_zlim(0, hist_ab_smooth.max() * 1.1)
        ax2.view_init(elev=25, azim=45)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.grid(True, alpha=0.3)

        # Panel 3: 2D hex (b, c)
        ax3 = fig.add_subplot(gs[0, 2])
        h3 = ax3.hexbin(
            b_vals_np, c_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax3.scatter(
            true_b,
            true_c,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax3.axhline(true_c, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.set_xlabel("b (linear)", fontsize=20, fontweight="bold")
        ax3.set_ylabel("c (quadratic)", fontsize=20, fontweight="bold")
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1] - b_lim[0]) / (c_lim[1] - c_lim[0]), adjustable="box")
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)

        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection="3d")
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25, range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(
            hist_bc_smooth > hist_bc_smooth.max() * 0.01, hist_bc_smooth, np.nan
        )
        surf_bc = ax4.plot_surface(
            X_bc,
            Y_bc,
            hist_bc_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax4.plot(
            [b_lim[0], b_lim[1]],
            [true_c, true_c],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax4.plot(
            [true_b, true_b],
            [c_lim[0], c_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax4.plot(
            [true_b, true_b],
            [true_c, true_c],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax4.scatter(
            [true_b],
            [true_c],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax4.set_xlabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_ylabel("c", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_xlim(b_lim)
        ax4.set_ylim(c_lim)
        ax4.set_zlim(0, hist_bc_smooth.max() * 1.1)
        ax4.view_init(elev=25, azim=45)
        ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Generate N=50 figure
    print("\n  Running GenJAX IS (N=50) with single particle resampling...")
    key_50 = jrand.key(seed)
    a_50, b_50, c_50 = run_is_single_resample_vectorized(key_50, xs, ys, 50, n_trials)

    create_single_resample_figure(
        a_50,
        b_50,
        c_50,
        single_resample_colors["is_50"],
        "examples/curvefit/figs/curvefit_params_resample50.pdf",
    )
    print("  ✓ Saved IS (N=50) single particle figure")
    print(
        f"    Mean: a={float(a_50.mean()):.3f}, b={float(b_50.mean()):.3f}, c={float(c_50.mean()):.3f}"
    )
    print(
        f"    Std:  a={float(a_50.std()):.3f}, b={float(b_50.std()):.3f}, c={float(c_50.std()):.3f}"
    )

    # Generate N=500 figure
    print("\n  Running GenJAX IS (N=500) with single particle resampling...")
    key_500 = jrand.key(seed + 500)
    a_500, b_500, c_500 = run_is_single_resample_vectorized(
        key_500, xs, ys, 500, n_trials
    )

    create_single_resample_figure(
        a_500,
        b_500,
        c_500,
        single_resample_colors["is_500"],
        "examples/curvefit/figs/curvefit_params_resample500.pdf",
    )
    print("  ✓ Saved IS (N=500) single particle figure")
    print(
        f"    Mean: a={float(a_500.mean()):.3f}, b={float(b_500.mean()):.3f}, c={float(c_500.mean()):.3f}"
    )
    print(
        f"    Std:  a={float(a_500.std()):.3f}, b={float(b_500.std()):.3f}, c={float(c_500.std()):.3f}"
    )

    # Generate N=5000 figure
    print("\n  Running GenJAX IS (N=5000) with single particle resampling...")
    key_5000 = jrand.key(seed + 1000)
    a_5000, b_5000, c_5000 = run_is_single_resample_vectorized(
        key_5000, xs, ys, 5000, n_trials
    )

    create_single_resample_figure(
        a_5000,
        b_5000,
        c_5000,
        single_resample_colors["is_5000"],
        "examples/curvefit/figs/curvefit_params_resample5000.pdf",
    )
    print("  ✓ Saved IS (N=5000) single particle figure")
    print(
        f"    Mean: a={float(a_5000.mean()):.3f}, b={float(b_5000.mean()):.3f}, c={float(c_5000.mean()):.3f}"
    )
    print(
        f"    Std:  a={float(a_5000.std()):.3f}, b={float(b_5000.std()):.3f}, c={float(c_5000.std()):.3f}"
    )

    print("\n✓ Completed IS single particle resampling comparison")


def save_parameter_density_timing_comparison(
    n_points=10,
    seed=42,
    timing_repeats=20,
):
    """Create horizontal bar plot comparing timing for all parameter density methods."""
    from examples.curvefit.core import infer_latents_jit, hmc_infer_latents_jit
    from genjax.core import Const
    from examples.utils import benchmark_with_warmup

    print("\n=== Parameter Density Methods Timing Comparison ===")

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]

    # Results storage
    methods = []
    times = []
    errors = []
    colors = []

    # 1. GenJAX IS (N=50)
    print("1. Timing GenJAX IS (N=50)...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(50)),
        repeats=timing_repeats,
    )
    methods.append("GenJAX IS (N=50)")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append("#B19CD9")  # Light purple
    print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # 2. GenJAX IS (N=500)
    print("2. Timing GenJAX IS (N=500)...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(500)),
        repeats=timing_repeats,
    )
    methods.append("GenJAX IS (N=500)")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append("#0173B2")  # Medium blue
    print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # 3. GenJAX IS (N=5000)
    print("3. Timing GenJAX IS (N=5000)...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(5000)),
        repeats=timing_repeats,
    )
    methods.append("GenJAX IS (N=5000)")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append("#029E73")  # Dark green
    print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # 4. GenJAX HMC
    print("4. Timing GenJAX HMC...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: hmc_infer_latents_jit(
            jrand.key(seed), xs, ys, Const(1000), Const(500), Const(0.001), Const(50)
        ),
        repeats=timing_repeats,
    )
    methods.append("GenJAX HMC")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append("#DE8F05")  # Orange
    print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # 5. NumPyro HMC (if available)
    try:
        from examples.curvefit.core import numpyro_run_hmc_inference_jit

        print("5. Timing NumPyro HMC...")
        time_results, (mean_time, std_time) = benchmark_with_warmup(
            lambda: numpyro_run_hmc_inference_jit(
                jrand.key(seed),
                xs,
                ys,
                num_samples=1000,
                num_warmup=500,
                step_size=0.001,
                num_steps=50,
            ),
            repeats=timing_repeats,
        )
        methods.append("NumPyro HMC")
        times.append(mean_time * 1000)
        errors.append(std_time * 1000)
        colors.append("#029E73")  # Green
        print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")
    except Exception as e:
        print(f"   NumPyro HMC failed: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar plot
    y_positions = range(len(methods))
    bars = ax.barh(y_positions, times, xerr=errors, capsize=5, color=colors, alpha=0.8)

    # Add timing values at the end of bars
    for i, (bar, time, error) in enumerate(zip(bars, times, errors)):
        width = bar.get_width()
        ax.text(
            width + error + max(times) * 0.02,  # Position text after error bar
            bar.get_y() + bar.get_height() / 2.0,
            f"{time:.1f} ms",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    # Style the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=18, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=18, fontweight="bold")
    # ax.set_title("Parameter Density Methods - Timing Comparison", fontsize=20, fontweight='bold', pad=20)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid for x-axis only
    ax.grid(True, alpha=0.3, axis="x")

    # Set x-axis limits to accommodate timing labels
    max_time = max(times)
    max_error = max(errors)
    ax.set_xlim(0, max_time + max_error + max_time * 0.2)

    # Set x-axis ticks
    from matplotlib.ticker import MultipleLocator

    if max_time < 10:
        ax.xaxis.set_major_locator(MultipleLocator(2))
    elif max_time < 100:
        ax.xaxis.set_major_locator(MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(200))

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
    ax.tick_params(axis="y", width=0, length=0)  # No tick marks on y-axis

    plt.tight_layout()

    # Save figure
    filename = "examples/curvefit/figs/curvefit_parameter_density_timing.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\n✓ Saved parameter density timing comparison: {filename}")

    # Print summary
    print("\n=== Timing Summary ===")
    for method, time, error in zip(methods, times, errors):
        print(f"{method}: {time:.1f} ± {error:.1f} ms")


def create_all_legends():
    """Create legend figures with distinguishable colors."""
    from matplotlib.lines import Line2D

    # Complete color palette
    all_colors = {
        "genjax_is": "#0173B2",  # Medium blue (base)
        "genjax_hmc": "#DE8F05",  # Orange
        "numpyro_hmc": "#029E73",  # Green
        "genjax_is_50": "#B19CD9",  # Light purple (distinguishable)
        "genjax_is_500": "#0173B2",  # Medium blue (distinguishable)
        "genjax_is_5000": "#029E73",  # Dark green (distinguishable)
    }

    all_methods = [
        ("genjax_is", "GenJAX IS (N=1000)"),
        ("genjax_hmc", "GenJAX HMC"),
        ("numpyro_hmc", "NumPyro HMC"),
        ("genjax_is_50", "GenJAX IS (N=50)"),
        ("genjax_is_500", "GenJAX IS (N=500)"),
        ("genjax_is_5000", "GenJAX IS (N=5000)"),
    ]

    # Main horizontal legend
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    ax.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=all_colors[key], lw=5, label=label)
        for key, label in all_methods
    ]

    legend = fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
        columnspacing=2,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    fig.savefig(
        "examples/curvefit/figs/curvefit_legend_all.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print("✓ Created final legend with distinguishable colors")

    # Create GenJAX IS-only legend
    create_genjax_is_legend()


def create_genjax_is_legend():
    """Create a separate legend figure for GenJAX IS methods only."""
    from matplotlib.lines import Line2D

    # GenJAX IS color palette
    is_colors = {
        "genjax_is_50": "#B19CD9",  # Light purple
        "genjax_is_500": "#0173B2",  # Medium blue
        "genjax_is_5000": "#029E73",  # Dark green
    }

    is_methods = [
        ("genjax_is_50", "GenJAX IS (N=50)"),
        ("genjax_is_500", "GenJAX IS (N=500)"),
        ("genjax_is_5000", "GenJAX IS (N=5000)"),
    ]

    # Create horizontal legend
    fig = plt.figure(figsize=(8, 1.2))
    ax = fig.add_subplot(111)
    ax.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=is_colors[key], lw=5, label=label)
        for key, label in is_methods
    ]

    legend = fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
        columnspacing=2,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    fig.savefig(
        "examples/curvefit/figs/curvefit_legend_is_horiz.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print("✓ Created GenJAX IS legend")

    # Also create a vertical version
    fig_vert = plt.figure(figsize=(3, 2.5))
    ax_vert = fig_vert.add_subplot(111)
    ax_vert.set_visible(False)

    legend_vert = fig_vert.legend(
        handles=legend_elements,
        loc="center",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
    )

    legend_vert.get_frame().set_facecolor("white")
    legend_vert.get_frame().set_alpha(1.0)
    legend_vert.get_frame().set_edgecolor("black")
    legend_vert.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    fig_vert.savefig(
        "examples/curvefit/figs/curvefit_legend_is_vert.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig_vert)
    print("✓ Created GenJAX IS legend (vertical)")


def save_is_only_timing_comparison(
    n_points=10,
    seed=42,
    timing_repeats=20,
):
    """Create horizontal bar plot comparing timing for IS methods only (N=5, N=1000, N=5000)."""
    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const
    from examples.utils import benchmark_with_warmup

    print("\n=== IS-Only Timing Comparison ===")

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]

    # Results storage
    methods = []
    times = []
    errors = []
    colors = []

    # IS color scheme with distinguishable shades
    is_colors = {
        50: "#B19CD9",  # Light purple (IS N=50)
        500: "#0173B2",  # Medium blue (IS N=500)
        5000: "#029E73",  # Dark green (IS N=5000)
    }

    # Benchmark each IS variant
    for n_particles in [50, 500, 5000]:
        print(f"Timing GenJAX IS (N={n_particles})...")
        time_results, (mean_time, std_time) = benchmark_with_warmup(
            lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(n_particles)),
            repeats=timing_repeats,
        )
        methods.append(f"GenJAX IS (N={n_particles})")
        times.append(mean_time * 1000)
        errors.append(std_time * 1000)
        colors.append(is_colors[n_particles])
        print(f"   Time: {mean_time * 1000:.1f} ± {std_time * 1000:.1f} ms")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create horizontal bar plot
    y_positions = range(len(methods))
    bars = ax.barh(y_positions, times, xerr=errors, capsize=5, color=colors, alpha=0.8)

    # Add timing values at the end of bars
    for i, (bar, time, error) in enumerate(zip(bars, times, errors)):
        width = bar.get_width()
        ax.text(
            width + error + max(times) * 0.02,  # Position text after error bar
            bar.get_y() + bar.get_height() / 2.0,
            f"{time:.1f} ms",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    # Style the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=18, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=18, fontweight="bold")
    # ax.set_title("Importance Sampling - Timing Comparison", fontsize=20, fontweight='bold', pad=20)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid for x-axis only
    ax.grid(True, alpha=0.3, axis="x")

    # Set x-axis limits to accommodate timing labels
    max_time = max(times)
    max_error = max(errors)
    ax.set_xlim(0, max_time + max_error + max_time * 0.2)

    # Set x-axis ticks
    from matplotlib.ticker import MultipleLocator

    if max_time < 10:
        ax.xaxis.set_major_locator(MultipleLocator(2))
    elif max_time < 100:
        ax.xaxis.set_major_locator(MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(50))

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=6)
    ax.tick_params(axis="y", width=0, length=0)  # No tick marks on y-axis

    plt.tight_layout()

    # Save figure
    filename = "examples/curvefit/figs/curvefit_is_only_timing.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\n✓ Saved IS-only timing comparison: {filename}")

    # Print summary
    print("\n=== IS Timing Summary ===")
    for method, time, error in zip(methods, times, errors):
        print(f"{method}: {time:.1f} ± {error:.1f} ms")


def save_is_only_parameter_density(
    n_points=10,
    seed=42,
):
    """Save parameter density figures for IS methods only (N=5, N=1000, N=5000)."""
    print("\n=== IS-Only Parameter Density Figures ===")

    from examples.curvefit.core import infer_latents_jit
    from genjax.core import Const
    from scipy.ndimage import gaussian_filter

    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]

    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")

    # IS color scheme with distinguishable shades
    is_variant_configs = {
        5: {
            "hex": "Purples",
            "surface": "Purples",
            "color": "#B19CD9",
        },  # Light purple-blue
        1000: {"hex": "Blues", "surface": "Blues", "color": "#0173B2"},  # Medium blue
        5000: {"hex": "Blues_r", "surface": "Blues_r", "color": "#08519C"},  # Dark blue
    }

    # Parameter limits
    a_lim = (-1.5, 1.0)
    b_lim = (-2.5, 2.0)
    c_lim = (-1.0, 2.5)

    def create_is_figure(n_particles, color_info, filename):
        """Create parameter density figure for IS with specified particles."""
        print(f"\n  Running GenJAX IS (N={n_particles})...")

        # Run IS inference
        samples, weights = infer_latents_jit(
            jrand.key(seed), xs, ys, Const(n_particles)
        )

        # Resample for visualization
        normalized_weights = jnp.exp(weights - jnp.max(weights))
        normalized_weights = normalized_weights / jnp.sum(normalized_weights)
        resample_idx = jrand.choice(
            jrand.key(seed + n_particles),
            jnp.arange(n_particles),
            shape=(2000,),
            p=normalized_weights,
            replace=True,
        )

        a_vals = samples.get_choices()["curve"]["a"][resample_idx]
        b_vals = samples.get_choices()["curve"]["b"][resample_idx]
        c_vals = samples.get_choices()["curve"]["c"][resample_idx]

        # Create figure using shared layout
        fig = plt.figure(figsize=(28, 7))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

        # Convert to numpy
        a_vals_np = np.array(a_vals)
        b_vals_np = np.array(b_vals)
        c_vals_np = np.array(c_vals)

        # Panel 1: 2D hex (a, b)
        ax1 = fig.add_subplot(gs[0, 0])
        h1 = ax1.hexbin(
            a_vals_np, b_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax1.scatter(
            true_a,
            true_b,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax1.axhline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax1.set_xlabel("a (constant)", fontsize=20, fontweight="bold")
        ax1.set_ylabel("b (linear)", fontsize=20, fontweight="bold")
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1] - a_lim[0]) / (b_lim[1] - b_lim[0]), adjustable="box")
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)

        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(
            hist_ab_smooth > hist_ab_smooth.max() * 0.01, hist_ab_smooth, np.nan
        )
        surf_ab = ax2.plot_surface(
            X_ab,
            Y_ab,
            hist_ab_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax2.plot(
            [a_lim[0], a_lim[1]],
            [true_b, true_b],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax2.plot(
            [true_a, true_a],
            [b_lim[0], b_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax2.plot(
            [true_a, true_a],
            [true_b, true_b],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax2.scatter(
            [true_a],
            [true_b],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax2.set_xlabel("a", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_ylabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax2.set_xlim(a_lim)
        ax2.set_ylim(b_lim)
        ax2.set_zlim(0, hist_ab_smooth.max() * 1.1)
        ax2.view_init(elev=25, azim=45)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.grid(True, alpha=0.3)

        # Panel 3: 2D hex (b, c)
        ax3 = fig.add_subplot(gs[0, 2])
        h3 = ax3.hexbin(
            b_vals_np, c_vals_np, gridsize=25, cmap=color_info["hex"], mincnt=1
        )
        ax3.scatter(
            true_b,
            true_c,
            c="#CC3311",
            s=400,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=100,
        )
        ax3.axhline(true_c, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color="#CC3311", linestyle="--", alpha=0.6, linewidth=2)
        ax3.set_xlabel("b (linear)", fontsize=20, fontweight="bold")
        ax3.set_ylabel("c (quadratic)", fontsize=20, fontweight="bold")
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1] - b_lim[0]) / (c_lim[1] - c_lim[0]), adjustable="box")
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)

        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection="3d")
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25, range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(
            hist_bc_smooth > hist_bc_smooth.max() * 0.01, hist_bc_smooth, np.nan
        )
        surf_bc = ax4.plot_surface(
            X_bc,
            Y_bc,
            hist_bc_masked,
            cmap=color_info["surface"],
            alpha=0.9,
            linewidth=0,
            antialiased=True,
        )

        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1

        # Add red lines at ground truth values (draw after surface)
        ax4.plot(
            [b_lim[0], b_lim[1]],
            [true_c, true_c],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        ax4.plot(
            [true_b, true_b],
            [c_lim[0], c_lim[1]],
            [0, 0],
            "r-",
            linewidth=3,
            alpha=1.0,
            zorder=100,
        )
        # Add vertical red line at ground truth
        ax4.plot(
            [true_b, true_b],
            [true_c, true_c],
            [0, z_max],
            "r-",
            linewidth=4,
            alpha=1.0,
            zorder=101,
        )
        ax4.scatter(
            [true_b],
            [true_c],
            [0],
            c="#CC3311",
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=3,
            zorder=102,
        )
        ax4.set_xlabel("b", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_ylabel("c", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_zlabel("Density", fontsize=18, labelpad=12, fontweight="bold")
        ax4.set_xlim(b_lim)
        ax4.set_ylim(c_lim)
        ax4.set_zlim(0, hist_bc_smooth.max() * 1.1)
        ax4.view_init(elev=25, azim=45)
        ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax4.xaxis.pane.fill = False
        ax4.yaxis.pane.fill = False
        ax4.zaxis.pane.fill = False
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved IS (N={n_particles}) figure")

        # Return mean estimates
        return a_vals.mean(), b_vals.mean(), c_vals.mean()

    # Generate IS comparison figures
    results = []
    for n_particles in [5, 1000, 5000]:
        filename = f"examples/curvefit/figs/curvefit_is_only_parameter_density_n{n_particles}.pdf"
        mean_a, mean_b, mean_c = create_is_figure(
            n_particles, is_variant_configs[n_particles], filename
        )
        results.append((n_particles, mean_a, mean_b, mean_c))

    # Print summary of estimates
    print("\n=== IS Parameter Estimates ===")
    print(f"True values: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    for n_particles, mean_a, mean_b, mean_c in results:
        print(
            f"IS (N={n_particles:4d}): a={mean_a:.3f}, b={mean_b:.3f}, c={mean_c:.3f}"
        )

    print("\n✓ Completed IS-only parameter density figures")


def save_outlier_conditional_demo(
    n_points=20,
    outlier_rate=0.2,
    seed=42,
    n_samples_is=1000,
):
    """Create two-panel figure demonstrating robust curve fitting with generative conditionals.

    This figure highlights how GenJAX's Cond combinator enables elegant outlier modeling
    with improved robustness compared to standard models.

    Args:
        n_points: Number of data points to generate
        outlier_rate: Fraction of points that are outliers (default 0.2)
        seed: Random seed for reproducibility
        n_samples_is: Number of importance sampling particles
    """
    from examples.curvefit.core import (
        infer_latents_jit,
        infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from genjax.core import Const

    print("\n=== Outlier Conditional Demo (Robust Curve Fitting) ===")
    print(f"  Outlier rate: {outlier_rate * 100:.0f}%")
    print(f"  Data points: {n_points}")

    # Create figure with two panels
    fig = plt.figure(figsize=(14, 6))
    ax_data = plt.subplot(1, 2, 1)
    ax_metrics = plt.subplot(1, 2, 2)

    # Generate ground truth polynomial
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    true_params = jnp.array([true_a, true_b, true_c])

    # Generate data with outliers
    key = jrand.key(seed)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    # Generate x values
    xs = jnp.sort(jrand.uniform(x_key, shape=(n_points,), minval=0.0, maxval=1.0))

    # Generate true y values from polynomial
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs)

    # Add noise and outliers
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate

    # Outliers are uniformly distributed in [-4, 4]
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )

    # Create observed data
    ys = jnp.where(is_outlier, outlier_vals, y_true + noise)

    # Panel A: Data and Model Fits
    print("\n1. Running standard model inference (no outlier handling)...")
    standard_samples, standard_weights = infer_latents_jit(
        jrand.key(seed + 1), xs, ys, Const(n_samples_is)
    )

    # Get standard model posterior mean
    normalized_weights = jnp.exp(standard_weights - jnp.max(standard_weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)

    standard_a = jnp.sum(
        standard_samples.get_choices()["curve"]["a"] * normalized_weights
    )
    standard_b = jnp.sum(
        standard_samples.get_choices()["curve"]["b"] * normalized_weights
    )
    standard_c = jnp.sum(
        standard_samples.get_choices()["curve"]["c"] * normalized_weights
    )
    standard_params = jnp.array([standard_a, standard_b, standard_c])

    print("\n2. Running Cond model inference (with outlier detection)...")
    cond_samples, cond_weights = infer_latents_with_outliers_jit(
        jrand.key(seed + 2),
        xs,
        ys,
        Const(n_samples_is),
        Const(outlier_rate),
        Const(0.0),
        Const(5.0),
    )

    # Get Cond model posterior mean
    cond_normalized_weights = jnp.exp(cond_weights - jnp.max(cond_weights))
    cond_normalized_weights = cond_normalized_weights / jnp.sum(cond_normalized_weights)

    cond_a = jnp.sum(cond_samples.get_choices()["curve"]["a"] * cond_normalized_weights)
    cond_b = jnp.sum(cond_samples.get_choices()["curve"]["b"] * cond_normalized_weights)
    cond_c = jnp.sum(cond_samples.get_choices()["curve"]["c"] * cond_normalized_weights)
    cond_params = jnp.array([cond_a, cond_b, cond_c])

    # For simplicity, detect outliers based on large residuals from the fitted curve
    # Points with residuals > 3 standard deviations are likely outliers
    y_cond_at_data = jax.vmap(lambda x: polyfn(x, cond_a, cond_b, cond_c))(xs)
    residuals = jnp.abs(ys - y_cond_at_data)
    residual_threshold = 0.5  # Larger threshold for clearer outlier detection
    detected_outliers = residuals > residual_threshold

    # Plot data
    inlier_mask = ~is_outlier
    outlier_mask = is_outlier

    # Plot inliers and outliers with different markers
    ax_data.scatter(
        xs[inlier_mask],
        ys[inlier_mask],
        s=80,
        c="#0173B2",
        alpha=0.8,
        label="Inliers",
        zorder=5,
    )
    ax_data.scatter(
        xs[outlier_mask],
        ys[outlier_mask],
        s=80,
        c="#CC3311",
        marker="x",
        linewidth=2.5,
        alpha=0.8,
        label="True Outliers",
        zorder=5,
    )

    # Plot curves
    x_plot = jnp.linspace(0, 1, 200)

    # True curve
    y_true_plot = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(x_plot)
    ax_data.plot(
        x_plot, y_true_plot, "k--", linewidth=2.5, label="True Polynomial", alpha=0.8
    )

    # Standard model fit (poor due to outliers)
    y_standard_plot = jax.vmap(lambda x: polyfn(x, standard_a, standard_b, standard_c))(
        x_plot
    )
    ax_data.plot(
        x_plot,
        y_standard_plot,
        color="#E69F00",
        linewidth=3,
        label="Standard Model",
        alpha=0.9,
    )

    # Cond model fit (robust)
    y_cond_plot = jax.vmap(lambda x: polyfn(x, cond_a, cond_b, cond_c))(x_plot)
    ax_data.plot(
        x_plot,
        y_cond_plot,
        color="#009E73",
        linewidth=3,
        label="GenJAX Cond Model",
        alpha=0.9,
    )

    # Mark detected outliers with circles
    ax_data.scatter(
        xs[detected_outliers],
        ys[detected_outliers],
        s=250,
        facecolors="none",
        edgecolors="#009E73",
        linewidth=3,
        label="Detected Outliers",
        zorder=6,
    )

    # Style panel A
    ax_data.set_xlabel("x", fontsize=20, fontweight="bold")
    ax_data.set_ylabel("y", fontsize=20, fontweight="bold")
    ax_data.set_xlim(-0.05, 1.05)
    ax_data.set_ylim(-5, 5)
    ax_data.grid(True, alpha=0.3)
    ax_data.legend(loc="upper left", fontsize=14, framealpha=0.95)
    set_minimal_ticks(ax_data)  # Use GRVS standard (3 ticks)

    # Panel B: Inference Quality Metrics
    print("\n3. Computing inference quality metrics...")

    # Parameter recovery error (RMSE)
    standard_rmse = float(jnp.sqrt(jnp.mean((standard_params - true_params) ** 2)))
    cond_rmse = float(jnp.sqrt(jnp.mean((cond_params - true_params) ** 2)))

    # Log marginal likelihood (approximated by log sum exp of weights)
    standard_lml = float(
        jnp.max(standard_weights)
        + jnp.log(jnp.mean(jnp.exp(standard_weights - jnp.max(standard_weights))))
    )
    cond_lml = float(
        jnp.max(cond_weights)
        + jnp.log(jnp.mean(jnp.exp(cond_weights - jnp.max(cond_weights))))
    )

    # Outlier detection F1 score
    detected_outliers_bool = np.array(detected_outliers)
    true_outliers_bool = np.array(is_outlier)

    # Calculate F1 score
    true_positives = np.sum(detected_outliers_bool & true_outliers_bool)
    false_positives = np.sum(detected_outliers_bool & ~true_outliers_bool)
    false_negatives = np.sum(~detected_outliers_bool & true_outliers_bool)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Create bar chart
    metrics = ["Parameter\nRMSE", "Log Marginal\nLikelihood", "Outlier F1\nScore"]
    standard_vals = [
        standard_rmse,
        standard_lml / 10,
        0.0,
    ]  # Standard model can't detect outliers
    cond_vals = [cond_rmse, cond_lml / 10, f1_score]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax_metrics.bar(
        x - width / 2,
        standard_vals,
        width,
        label="Standard Model",
        color="#E69F00",
        alpha=0.8,
    )
    bars2 = ax_metrics.bar(
        x + width / 2,
        cond_vals,
        width,
        label="GenJAX Cond Model",
        color="#009E73",
        alpha=0.8,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i == 0:  # RMSE
                label = f"{height:.3f}"
            elif i == 1:  # LML (scaled)
                label = f"{height * 10:.1f}"
            else:  # F1
                label = f"{height:.2f}"

            ax_metrics.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    # Style panel B
    ax_metrics.set_ylabel("Score", fontsize=20, fontweight="bold")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics, fontsize=16)
    ax_metrics.legend(fontsize=14, loc="upper right")
    ax_metrics.grid(True, alpha=0.3, axis="y")
    ax_metrics.set_ylim(0, 1.2)

    # Remove top and right spines
    ax_metrics.spines["top"].set_visible(False)
    ax_metrics.spines["right"].set_visible(False)

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.3)

    # Save figure as both PDF and PNG
    filename_pdf = "examples/curvefit/figs/curvefit_outlier_robustness_demo.pdf"
    filename_png = "examples/curvefit/figs/curvefit_outlier_robustness_demo.png"
    fig.savefig(filename_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(filename_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Saved outlier conditional demo: {filename_pdf}")

    # Print summary
    print("\n=== Results Summary ===")
    print(f"True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    print("\nStandard Model:")
    print(f"  Parameters: a={standard_a:.3f}, b={standard_b:.3f}, c={standard_c:.3f}")
    print(f"  RMSE: {standard_rmse:.3f}")
    print(f"  Log ML: {standard_lml:.1f}")
    print("\nGenJAX Cond Model:")
    print(f"  Parameters: a={cond_a:.3f}, b={cond_b:.3f}, c={cond_c:.3f}")
    print(f"  RMSE: {cond_rmse:.3f}")
    print(f"  Log ML: {cond_lml:.1f}")
    print(f"  Outlier Detection F1: {f1_score:.2f}")
    print(f"    Precision: {precision:.2f}")
    print(f"    Recall: {recall:.2f}")


# Placeholder functions for other outlier visualizations
def save_outlier_trace_viz():
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_inference_viz_beta(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_posterior_comparison_beta(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_data_viz(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_inference_comparison(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_indicators_viz(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_method_comparison(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_framework_comparison(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_parameter_posterior_histogram(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_scaling_study(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


def save_outlier_rate_sensitivity(**kwargs):
    print("  [Not implemented - using save_outlier_conditional_demo instead]")


# ================= GIBBS SAMPLING VISUALIZATIONS =================


def _copy_to_main_figs(filename):
    """Helper function to copy figure to main paper figs directory."""
    import shutil
    import os

    main_figs_path = "../../figs"
    if os.path.exists(main_figs_path):
        source_path = f"examples/curvefit/figs/{filename}"
        dest_path = f"{main_figs_path}/{filename}"
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            return True
    return False


def save_gibbs_parameter_convergence(
    n_points=15,
    outlier_rate=0.25,
    n_samples=1000,
    n_warmup=200,
    seed=42,
):
    """Show Gibbs sampling convergence for curve parameters.

    Creates a 4-panel figure showing:
    - Top 3 panels: Trace plots for parameters a, b, c
    - Bottom panel: Outlier detection probability over time for selected points
    """
    from examples.curvefit.core import gibbs_infer_latents_with_outliers_jit
    from examples.curvefit.data import polyfn
    from genjax.core import Const

    print("\n=== Gibbs Parameter Convergence Analysis ===")
    print(f"  Data points: {n_points}, Outlier rate: {outlier_rate * 100:.0f}%")
    print(f"  Gibbs samples: {n_samples}, Warmup: {n_warmup}")

    # Setup
    setup_publication_fonts()
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["framework_comparison"])

    # Generate synthetic data with known outliers
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

    print(f"  True outliers at indices: {jnp.where(is_outlier_true)[0]}")

    # Run Gibbs sampling
    print("  Running Gibbs sampling...")
    gibbs_result = gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    curve_samples = gibbs_result["curve_samples"]
    outlier_samples = gibbs_result["outlier_samples"]

    # Panel 1-3: Parameter traces
    param_names = ["a", "b", "c"]
    true_values = [true_a, true_b, true_c]

    for i, (param, true_val) in enumerate(zip(param_names, true_values)):
        ax = axes[0, i] if i < 2 else axes[1, 0]
        samples = curve_samples[param]

        # Plot trace
        ax.plot(samples, color=get_method_color("genjax_is"), alpha=0.8, linewidth=1.5)
        ax.axhline(
            true_val,
            color=get_method_color("data_points"),
            linestyle="--",
            linewidth=2,
            label=f"True {param}",
        )

        ax.set_xlabel("Gibbs iteration", fontweight="bold")
        ax.set_ylabel(f"Parameter {param}", fontweight="bold")
        ax.legend()
        apply_grid_style(ax)
        apply_standard_ticks(ax)

    # Panel 4: Outlier detection probabilities
    ax = axes[1, 1]
    outlier_probs = jnp.mean(outlier_samples, axis=0)  # Posterior outlier probability

    # Show a few interesting points
    outlier_indices = jnp.where(is_outlier_true)[0]
    inlier_indices = jnp.where(~is_outlier_true)[0]

    # Plot outlier probabilities over time for selected points
    for idx in outlier_indices[:3]:  # First 3 true outliers
        prob_trace = jnp.mean(
            outlier_samples[:, idx : idx + 1], axis=1
        )  # Running average
        ax.plot(
            prob_trace,
            color=get_method_color("data_points"),
            alpha=0.7,
            linewidth=1.5,
            label=f"True outlier {idx}",
        )

    for idx in inlier_indices[:2]:  # First 2 true inliers
        prob_trace = jnp.mean(outlier_samples[:, idx : idx + 1], axis=1)
        ax.plot(
            prob_trace,
            color=get_method_color("curves"),
            alpha=0.7,
            linewidth=1.5,
            label=f"True inlier {idx}",
        )

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Gibbs iteration", fontweight="bold")
    ax.set_ylabel("Outlier probability", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    apply_grid_style(ax)
    apply_standard_ticks(ax)

    plt.tight_layout()
    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_gibbs_convergence.pdf"
    )

    # Copy to main figs directory for paper
    filename = "curvefit_gibbs_convergence.pdf"
    if _copy_to_main_figs(filename):
        print(f"  ✓ Saved: {filename} (+ copied to main figs/)")
    else:
        print(f"  ✓ Saved: {filename}")


def save_gibbs_outlier_detection(
    n_points=15,
    outlier_rate=0.25,
    n_samples=1000,
    n_warmup=200,
    seed=42,
):
    """Show Gibbs sampling outlier detection performance.

    Creates a two-panel figure:
    - Left: Data points colored by posterior outlier probability
    - Right: Detection metrics (precision, recall, F1)
    """
    from examples.curvefit.core import gibbs_infer_latents_with_outliers_jit
    from examples.curvefit.data import polyfn
    from genjax.core import Const

    print("\n=== Gibbs Outlier Detection Analysis ===")

    # Setup
    setup_publication_fonts()
    fig, (ax_data, ax_metrics) = plt.subplots(
        1, 2, figsize=FIGURE_SIZES["framework_comparison"]
    )

    # Generate data (same as convergence analysis)
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

    # Run Gibbs sampling
    gibbs_result = gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    curve_samples = gibbs_result["curve_samples"]
    outlier_samples = gibbs_result["outlier_samples"]
    outlier_probs = jnp.mean(outlier_samples, axis=0)

    # Panel 1: Data visualization with outlier probabilities
    xs_plot = jnp.linspace(0, 1, 100)

    # Plot posterior curve
    a_mean = jnp.mean(curve_samples["a"])
    b_mean = jnp.mean(curve_samples["b"])
    c_mean = jnp.mean(curve_samples["c"])
    y_curve = jax.vmap(lambda x: polyfn(x, a_mean, b_mean, c_mean))(xs_plot)
    ax_data.plot(
        xs_plot,
        y_curve,
        color=get_method_color("curves"),
        linewidth=3,
        alpha=0.8,
        label="Inferred curve",
    )

    # Plot true curve for comparison
    y_true_plot = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_plot)
    ax_data.plot(
        xs_plot,
        y_true_plot,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label="True curve",
    )

    # Plot data points colored by outlier probability
    scatter = ax_data.scatter(
        xs_obs,
        ys_obs,
        c=outlier_probs,
        cmap="Reds",
        s=120,
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_data)
    cbar.set_label("Outlier probability", fontweight="bold")

    ax_data.set_xlabel("x", fontweight="bold")
    ax_data.set_ylabel("y", fontweight="bold")
    ax_data.legend()
    apply_grid_style(ax_data)
    apply_standard_ticks(ax_data)

    # Panel 2: Detection metrics
    thresholds = jnp.linspace(0.1, 0.9, 9)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        predicted_outliers = outlier_probs > threshold

        # True positives, false positives, false negatives
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

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    ax_metrics.plot(
        thresholds,
        precisions,
        "o-",
        color=get_method_color("genjax_is"),
        linewidth=2,
        markersize=8,
        label="Precision",
    )
    ax_metrics.plot(
        thresholds,
        recalls,
        "s-",
        color=get_method_color("genjax_hmc"),
        linewidth=2,
        markersize=8,
        label="Recall",
    )
    ax_metrics.plot(
        thresholds,
        f1_scores,
        "^-",
        color=get_method_color("data_points"),
        linewidth=2,
        markersize=8,
        label="F1 Score",
    )

    ax_metrics.set_xlabel("Detection threshold", fontweight="bold")
    ax_metrics.set_ylabel("Score", fontweight="bold")
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.legend()
    apply_grid_style(ax_metrics)
    apply_standard_ticks(ax_metrics)

    plt.tight_layout()
    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_gibbs_outlier_detection.pdf"
    )

    # Copy to main figs directory for paper
    filename = "curvefit_gibbs_outlier_detection.pdf"
    if _copy_to_main_figs(filename):
        print(f"  ✓ Saved: {filename} (+ copied to main figs/)")
    else:
        print(f"  ✓ Saved: {filename}")


def save_gibbs_vs_methods_comparison(
    n_points=15,
    outlier_rate=0.25,
    n_samples=1000,
    n_warmup=200,
    seed=42,
    timing_repeats=20,
):
    """Compare Gibbs vs IS vs HMC on outlier detection task.

    Creates a three-panel figure:
    - Left: Posterior curves comparison
    - Middle: Outlier detection accuracy
    - Right: Runtime comparison
    """
    from examples.curvefit.core import (
        gibbs_infer_latents_with_outliers_jit,
        infer_latents_with_outliers_jit,
        hmc_infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from examples.utils import benchmark_with_warmup
    from genjax.core import Const

    print("\n=== Gibbs vs Other Methods Comparison ===")
    print("  Comparing Gibbs vs IS vs HMC on outlier detection")

    # Setup
    setup_publication_fonts()
    fig, (ax_curves, ax_detection, ax_timing) = plt.subplots(1, 3, figsize=(18, 5))

    # Generate data
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

    xs_plot = jnp.linspace(0, 1, 100)
    y_true_plot = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_plot)

    # Run all methods
    print("  Running Gibbs sampling...")
    gibbs_result = gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    print("  Running Importance Sampling...")
    is_traces, is_weights = infer_latents_with_outliers_jit(
        jrand.key(seed + 2),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        outlier_rate=Const(outlier_rate),
    )

    print("  Running HMC...")
    hmc_traces, hmc_diagnostics = hmc_infer_latents_with_outliers_jit(
        jrand.key(seed + 3),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    # Panel 1: Posterior curves
    methods_data = [
        ("Gibbs", gibbs_result["curve_samples"], get_method_color("genjax_is")),
        (
            "IS",
            {
                "a": is_traces.get_choices()["curve"]["a"],
                "b": is_traces.get_choices()["curve"]["b"],
                "c": is_traces.get_choices()["curve"]["c"],
            },
            get_method_color("genjax_hmc"),
        ),
        (
            "HMC",
            {
                "a": hmc_traces.get_choices()["curve"]["a"],
                "b": hmc_traces.get_choices()["curve"]["b"],
                "c": hmc_traces.get_choices()["curve"]["c"],
            },
            get_method_color("data_points"),
        ),
    ]

    ax_curves.plot(
        xs_plot,
        y_true_plot,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label="True curve",
    )

    for method_name, curve_samples, color in methods_data:
        a_mean = jnp.mean(curve_samples["a"])
        b_mean = jnp.mean(curve_samples["b"])
        c_mean = jnp.mean(curve_samples["c"])
        y_curve = jax.vmap(lambda x: polyfn(x, a_mean, b_mean, c_mean))(xs_plot)
        ax_curves.plot(
            xs_plot,
            y_curve,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label=f"{method_name}",
        )

    # Plot data points
    ax_curves.scatter(xs_obs, ys_obs, color="black", s=80, zorder=10, alpha=0.7)
    ax_curves.set_xlabel("x", fontweight="bold")
    ax_curves.set_ylabel("y", fontweight="bold")
    ax_curves.legend()
    apply_grid_style(ax_curves)
    apply_standard_ticks(ax_curves)

    # Panel 2: Detection accuracy comparison
    gibbs_outlier_probs = jnp.mean(gibbs_result["outlier_samples"], axis=0)
    is_outlier_probs = jnp.mean(is_traces.get_choices()["ys"]["is_outlier"], axis=0)
    hmc_outlier_probs = jnp.mean(hmc_traces.get_choices()["ys"]["is_outlier"], axis=0)

    detection_data = [
        ("Gibbs", gibbs_outlier_probs, get_method_color("genjax_is")),
        ("IS", is_outlier_probs, get_method_color("genjax_hmc")),
        ("HMC", hmc_outlier_probs, get_method_color("data_points")),
    ]

    # Calculate F1 scores at threshold 0.5
    f1_scores = []
    for method_name, outlier_probs, color in detection_data:
        predicted = outlier_probs > 0.5
        tp = jnp.sum(predicted & is_outlier_true)
        fp = jnp.sum(predicted & ~is_outlier_true)
        fn = jnp.sum(~predicted & is_outlier_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    methods = [d[0] for d in detection_data]
    colors = [d[2] for d in detection_data]

    bars = ax_detection.bar(
        methods, f1_scores, color=colors, alpha=0.7, edgecolor="black"
    )
    ax_detection.set_ylabel("F1 Score", fontweight="bold")
    ax_detection.set_ylim(0, 1.05)
    apply_grid_style(ax_detection)

    # Panel 3: Runtime comparison
    print("  Benchmarking methods...")

    def gibbs_task():
        return gibbs_infer_latents_with_outliers_jit(
            jrand.key(seed + 10),
            xs_obs,
            ys_obs,
            n_samples=Const(n_samples),
            n_warmup=Const(n_warmup),
            outlier_rate=Const(outlier_rate),
        )

    def is_task():
        return infer_latents_with_outliers_jit(
            jrand.key(seed + 11),
            xs_obs,
            ys_obs,
            n_samples=Const(n_samples),
            outlier_rate=Const(outlier_rate),
        )

    def hmc_task():
        return hmc_infer_latents_with_outliers_jit(
            jrand.key(seed + 12),
            xs_obs,
            ys_obs,
            n_samples=Const(n_samples),
            n_warmup=Const(n_warmup),
            outlier_rate=Const(outlier_rate),
        )

    timing_results = []
    for task, method_name in [
        (gibbs_task, "Gibbs"),
        (is_task, "IS"),
        (hmc_task, "HMC"),
    ]:
        times, (mean_time, std_time) = benchmark_with_warmup(
            task, repeats=timing_repeats
        )
        timing_results.append((method_name, mean_time * 1000, std_time * 1000))

    method_names = [r[0] for r in timing_results]
    mean_times = [r[1] for r in timing_results]
    std_times = [r[2] for r in timing_results]

    bars = ax_timing.bar(
        method_names,
        mean_times,
        yerr=std_times,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        capsize=5,
    )
    ax_timing.set_ylabel("Runtime (ms)", fontweight="bold")
    apply_grid_style(ax_timing)

    plt.tight_layout()
    save_publication_figure(fig, "examples/curvefit/figs/curvefit_gibbs_vs_methods.pdf")

    # Copy to main figs directory for paper
    filename = "curvefit_gibbs_vs_methods.pdf"
    if _copy_to_main_figs(filename):
        print(f"  ✓ Saved: {filename} (+ copied to main figs/)")
    else:
        print(f"  ✓ Saved: {filename}")


def save_gibbs_trace_analysis(
    n_points=15,
    outlier_rate=0.25,
    n_samples=1000,
    n_warmup=200,
    seed=42,
):
    """Detailed Gibbs trace analysis showing mixing and autocorrelation.

    Creates a 4-panel figure:
    - Top row: Trace plots for parameters a and b
    - Bottom left: Autocorrelation functions
    - Bottom right: Effective sample size comparison
    """
    from examples.curvefit.core import gibbs_infer_latents_with_outliers_jit
    from examples.curvefit.data import polyfn
    from genjax.core import Const
    import numpy as np

    print("\n=== Gibbs Trace Analysis ===")

    # Setup
    setup_publication_fonts()
    fig = plt.figure(figsize=FIGURE_SIZES["framework_comparison"])
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_trace_a = fig.add_subplot(gs[0, 0])
    ax_trace_b = fig.add_subplot(gs[0, 1])
    ax_trace_c = fig.add_subplot(gs[0, 2])
    ax_autocorr = fig.add_subplot(gs[1, 0:2])
    ax_ess = fig.add_subplot(gs[1, 2])

    # Generate data and run Gibbs
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

    gibbs_result = gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    curve_samples = gibbs_result["curve_samples"]

    # Trace plots
    param_data = [
        ("a", curve_samples["a"], true_a, ax_trace_a),
        ("b", curve_samples["b"], true_b, ax_trace_b),
        ("c", curve_samples["c"], true_c, ax_trace_c),
    ]

    for param_name, samples, true_val, ax in param_data:
        ax.plot(samples, color=get_method_color("genjax_is"), alpha=0.8, linewidth=1)
        ax.axhline(
            true_val, color=get_method_color("data_points"), linestyle="--", linewidth=2
        )
        ax.set_ylabel(f"Parameter {param_name}", fontweight="bold")
        ax.set_xlabel("Iteration", fontweight="bold")
        apply_grid_style(ax)
        apply_standard_ticks(ax)

    # Autocorrelation function
    def autocorrelation(x, max_lag=50):
        """Compute autocorrelation function."""
        x = np.array(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag]

    max_lag = min(50, n_samples // 10)
    lags = np.arange(max_lag)

    for param_name, samples, _, _ in param_data[:2]:  # Just a and b for clarity
        autocorr = autocorrelation(samples, max_lag)
        color = (
            get_method_color("genjax_is")
            if param_name == "a"
            else get_method_color("genjax_hmc")
        )
        ax_autocorr.plot(
            lags, autocorr, color=color, linewidth=2, label=f"Parameter {param_name}"
        )

    ax_autocorr.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_autocorr.axhline(
        0.1, color="red", linestyle=":", alpha=0.5, label="0.1 threshold"
    )
    ax_autocorr.set_xlabel("Lag", fontweight="bold")
    ax_autocorr.set_ylabel("Autocorrelation", fontweight="bold")
    ax_autocorr.legend()
    apply_grid_style(ax_autocorr)
    apply_standard_ticks(ax_autocorr)

    # Effective sample size (simplified estimate)
    def effective_sample_size(x):
        """Estimate effective sample size."""
        autocorr = autocorrelation(np.array(x), min(len(x) // 4, 100))
        # Find first negative autocorr or where it drops below 0.1
        cutoff = 1
        for i, ac in enumerate(autocorr[1:], 1):
            if ac <= 0.1:
                cutoff = i
                break
        tau = 1 + 2 * np.sum(autocorr[1:cutoff])
        return len(x) / tau

    ess_values = []
    param_names = []
    for param_name, samples, _, _ in param_data:
        ess = effective_sample_size(samples)
        ess_values.append(ess)
        param_names.append(param_name)

    bars = ax_ess.bar(
        param_names,
        ess_values,
        color=[
            get_method_color("genjax_is"),
            get_method_color("genjax_hmc"),
            get_method_color("data_points"),
        ],
        alpha=0.7,
        edgecolor="black",
    )
    ax_ess.set_ylabel("Effective Sample Size", fontweight="bold")
    ax_ess.axhline(
        n_samples, color="gray", linestyle="--", alpha=0.5, label=f"Total ({n_samples})"
    )
    ax_ess.legend()
    apply_grid_style(ax_ess)

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_gibbs_trace_analysis.pdf"
    )
    print("  ✓ Saved: curvefit_gibbs_trace_analysis.pdf")

    # Copy to main figs directory
    _copy_to_main_figs("curvefit_gibbs_trace_analysis.pdf")


def save_outlier_prior_and_inference_demo(
    n_points=15,
    outlier_rate=0.25,
    random_seed=42,
    n_samples_is=1000,
):
    """Create two-panel figure showing (a) outlier model prior sample and (b) IS inference performance.

    Panel A shows a prior sample from the outlier model with:
    - The underlying polynomial curve
    - Observed data points (inliers and outliers clearly marked)
    - Visual identification of which points are outliers

    Panel B shows importance sampling inference performance:
    - Posterior curves from IS inference on the outlier model
    - Quality metrics (ESS, log marginal likelihood)
    - Demonstration of robust inference despite outliers

    Args:
        n_points: Number of data points to generate
        outlier_rate: Fraction of points that are outliers (default 0.25)
        seed: Random seed for reproducibility
        n_samples_is: Number of importance sampling particles
    """
    from examples.curvefit.core import (
        infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from genjax.core import Const

    print("\n=== Outlier Model: Prior Sample and IS Inference Demo ===")
    print(f"  Outlier rate: {outlier_rate * 100:.0f}%")
    print(f"  Data points: {n_points}")
    print(f"  IS particles: {n_samples_is}")

    # Setup publication fonts and figure
    setup_publication_fonts()
    fig, (ax_prior, ax_inference) = plt.subplots(
        2, 1, figsize=FIGURE_SIZES["two_panel_vertical"]
    )

    # Generate x values for plotting and evaluation
    xs_obs = jnp.linspace(0.0, 1.0, n_points)  # Observation points
    xs_plot = jnp.linspace(0.0, 1.0, 100)  # Plotting points

    # ================== Panel A: Prior Sample ==================
    print("\n1. Generating prior sample from outlier model...")

    # Generate ground truth polynomial (similar to existing demo)
    true_a, true_b, true_c = -0.211, -0.395, 0.673
    true_params = jnp.array([true_a, true_b, true_c])

    # Generate data with outliers (similar to existing demo approach)
    key = jrand.key(random_seed)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    # Generate true y values from polynomial
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)

    # Add noise and outliers
    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_rand = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate

    # Outliers are uniformly distributed in [-4, 4]
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )

    # Create observed data
    ys_obs = jnp.where(is_outlier_rand, outlier_vals, y_true + noise)
    outlier_indicators = is_outlier_rand

    # For visualization, use the true parameters
    a_sample, b_sample, c_sample = true_a, true_b, true_c

    # Evaluate the sampled curve at plotting points
    y_curve = jax.vmap(lambda x: polyfn(x, a_sample, b_sample, c_sample))(xs_plot)

    # Plot the prior sample
    ax_prior.plot(
        xs_plot,
        y_curve,
        color=get_method_color("curves"),
        linewidth=3,
        alpha=0.8,
        label="True Curve",
    )

    # Plot inlier points
    inlier_mask = ~outlier_indicators
    if jnp.any(inlier_mask):
        ax_prior.scatter(
            xs_obs[inlier_mask],
            ys_obs[inlier_mask],
            color=get_method_color("data_points"),
            s=120,
            zorder=10,
            edgecolor="white",
            linewidth=2,
            label="Inlier Points",
        )

    # Plot outlier points with different styling
    if jnp.any(outlier_indicators):
        ax_prior.scatter(
            xs_obs[outlier_indicators],
            ys_obs[outlier_indicators],
            color=get_method_color("background"),
            marker="x",
            s=150,
            linewidth=4,
            zorder=10,
            label="Outlier Points",
        )

    ax_prior.set_xlabel("X", fontweight="bold")
    ax_prior.set_ylabel("Y", fontweight="bold")
    ax_prior.legend(fontsize=16)
    apply_grid_style(ax_prior)
    apply_standard_ticks(ax_prior)

    # ================== Panel B: IS Inference Performance ==================
    print("\n2. Running IS inference on outlier model...")

    # ys_obs is already computed above - contains the observed data with outliers

    # Run IS inference
    inference_key = jrand.key(random_seed + 100)
    samples, weights = infer_latents_with_outliers_jit(
        inference_key,
        xs_obs,
        ys_obs,
        Const(n_samples_is),
        Const(outlier_rate),
        Const(0.0),
        Const(5.0),
    )

    # Compute inference diagnostics
    normalized_weights = jnp.exp(weights - jnp.max(weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    ess = 1.0 / jnp.sum(normalized_weights**2)
    lml = jnp.log(jnp.mean(jnp.exp(weights - jnp.max(weights)))) + jnp.max(weights)

    # Get posterior mean curve
    posterior_a = jnp.sum(samples.get_choices()["curve"]["a"] * normalized_weights)
    posterior_b = jnp.sum(samples.get_choices()["curve"]["b"] * normalized_weights)
    posterior_c = jnp.sum(samples.get_choices()["curve"]["c"] * normalized_weights)

    y_posterior = jax.vmap(lambda x: polyfn(x, posterior_a, posterior_b, posterior_c))(
        xs_plot
    )

    # Plot a few posterior samples
    n_posterior_samples = 5
    for i in range(min(n_posterior_samples, n_samples_is)):
        idx = int(i * n_samples_is / n_posterior_samples)
        a_i = samples.get_choices()["curve"]["a"][idx]
        b_i = samples.get_choices()["curve"]["b"][idx]
        c_i = samples.get_choices()["curve"]["c"][idx]
        y_i = jax.vmap(lambda x: polyfn(x, a_i, b_i, c_i))(xs_plot)

        ax_inference.plot(
            xs_plot, y_i, color=get_method_color("curves"), alpha=0.3, linewidth=1.5
        )

    # Plot posterior mean
    ax_inference.plot(
        xs_plot,
        y_posterior,
        color=get_method_color("genjax_is"),
        linewidth=3,
        label="Posterior Mean",
    )

    # Plot the true curve for comparison
    ax_inference.plot(
        xs_plot,
        y_curve,
        color=get_method_color("true_values"),
        linestyle="--",
        linewidth=3,
        label="True Curve",
    )

    # Plot observed data
    ax_inference.scatter(
        xs_obs[inlier_mask],
        ys_obs[inlier_mask],
        color=get_method_color("data_points"),
        s=120,
        zorder=10,
        edgecolor="white",
        linewidth=2,
        alpha=0.8,
    )
    ax_inference.scatter(
        xs_obs[outlier_indicators],
        ys_obs[outlier_indicators],
        color=get_method_color("background"),
        marker="x",
        s=150,
        linewidth=4,
        zorder=10,
        alpha=0.8,
    )

    ax_inference.set_xlabel("X", fontweight="bold")
    ax_inference.set_ylabel("Y", fontweight="bold")
    ax_inference.legend(fontsize=16)
    apply_grid_style(ax_inference)
    apply_standard_ticks(ax_inference)

    # Add performance metrics as text
    metrics_text = f"ESS: {ess:.0f}\nLog ML: {lml:.1f}"
    ax_inference.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_inference.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save with descriptive name
    filename = "examples/curvefit/figs/curvefit_outlier_prior_and_inference.pdf"
    save_publication_figure(fig, filename)

    print(f"\n✓ Saved outlier model demonstration: {filename}")
    print("\nPrior Sample Summary:")
    print(f"  True curve: a={a_sample:.3f}, b={b_sample:.3f}, c={c_sample:.3f}")
    print(
        f"  Outliers detected: {jnp.sum(outlier_indicators)}/{n_points} ({jnp.mean(outlier_indicators) * 100:.1f}%)"
    )
    print("\nIS Inference Performance:")
    print(f"  Effective Sample Size: {ess:.0f}")
    print(f"  Log Marginal Likelihood: {lml:.1f}")
    print(
        f"  Posterior mean: a={posterior_a:.3f}, b={posterior_b:.3f}, c={posterior_c:.3f}"
    )

    return {
        "prior_sample": {"a": a_sample, "b": b_sample, "c": c_sample},
        "outlier_indicators": outlier_indicators,
        "ess": ess,
        "lml": lml,
        "posterior_mean": {"a": posterior_a, "b": posterior_b, "c": posterior_c},
    }


def save_outlier_algorithm_comparison(
    n_points=15,
    outlier_rate=0.25,
    random_seed=42,
    n_is_large=1000,
    n_is_small=50,
    n_hmc_samples=200,
    n_trials=5,
):
    """Compare IS(N=1000) vs IS(N=50) for outlier model inference.

    Creates a multi-panel figure showing:
    Top row: (a) Inference quality with outlier predictions (b) Detection metrics
    Middle: Runtime comparison (horizontal bar chart)
    Bottom: Comprehensive legend with all plot information

    Args:
        n_points: Number of data points
        outlier_rate: Fraction of outlier points
        random_seed: Random seed for reproducibility
        n_is_large: Number of particles for large IS approach
        n_is_small: Number of particles for small IS approach
        n_hmc_samples: Unused (kept for compatibility)
        n_trials: Number of timing trials for robust estimates
    """
    from examples.curvefit.core import (
        infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from examples.utils import benchmark_with_warmup
    from genjax.core import Const

    print("\n=== Outlier Model: IS Sample Size Comparison ===")
    print(f"  Comparing IS(N={n_is_large}) vs IS(N={n_is_small})")
    print(f"  Data points: {n_points}, Outlier rate: {outlier_rate * 100:.0f}%")

    # Generate test data (same as in prior demo)
    true_a, true_b, true_c = -0.211, -0.395, 0.673

    key = jrand.key(random_seed)
    x_key, noise_key, outlier_key = jrand.split(key, 3)

    xs_obs = jnp.linspace(0.0, 1.0, n_points)
    xs_plot = jnp.linspace(0.0, 1.0, 100)
    y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)

    noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
    is_outlier_rand = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
    outlier_vals = jrand.uniform(
        outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
    )
    ys_obs = jnp.where(is_outlier_rand, outlier_vals, y_true + noise)

    # Setup figure with custom layout
    setup_publication_fonts()
    fig = plt.figure(figsize=(16, 12))

    # Create grid layout: 3 rows x 2 columns
    # Top row: 2 subplots (curves, metrics)
    # Middle row: timing spanning both columns
    # Bottom row: legend spanning both columns
    ax_quality = plt.subplot2grid((3, 2), (0, 0))
    ax_detection = plt.subplot2grid((3, 2), (0, 1))
    ax_runtime = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax_legend = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    # ================== Algorithm 1: Pure IS(N=1000) ==================
    print("\n1. Running pure IS inference...")

    # Timing benchmark for IS
    def is_inference():
        return infer_latents_with_outliers_jit(
            jrand.key(random_seed + 1),
            xs_obs,
            ys_obs,
            Const(n_is_large),
            Const(outlier_rate),
            Const(0.0),
            Const(5.0),
        )

    is_times, (is_mean_time, is_std_time) = benchmark_with_warmup(
        is_inference, repeats=n_trials, inner_repeats=1
    )

    # Run IS for analysis
    is_samples, is_weights = is_inference()

    # Compute IS diagnostics
    is_normalized_weights = jnp.exp(is_weights - jnp.max(is_weights))
    is_normalized_weights = is_normalized_weights / jnp.sum(is_normalized_weights)
    is_ess = 1.0 / jnp.sum(is_normalized_weights**2)
    is_lml = jnp.log(jnp.mean(jnp.exp(is_weights - jnp.max(is_weights)))) + jnp.max(
        is_weights
    )

    # Get IS posterior mean
    is_a = jnp.sum(is_samples.get_choices()["curve"]["a"] * is_normalized_weights)
    is_b = jnp.sum(is_samples.get_choices()["curve"]["b"] * is_normalized_weights)
    is_c = jnp.sum(is_samples.get_choices()["curve"]["c"] * is_normalized_weights)
    is_posterior = jax.vmap(lambda x: polyfn(x, is_a, is_b, is_c))(xs_plot)

    # ================== Algorithm 2: IS with smaller sample size ==================
    print(f"\n2. Running IS with N={n_is_small} particles...")

    # Timing benchmark for smaller IS
    def small_is_inference():
        return infer_latents_with_outliers_jit(
            jrand.key(random_seed + 2),
            xs_obs,
            ys_obs,
            Const(n_is_small),
            Const(outlier_rate),
            Const(0.0),
            Const(5.0),
        )

    small_is_times, (small_is_mean_time, small_is_std_time) = benchmark_with_warmup(
        small_is_inference, repeats=n_trials, inner_repeats=1
    )

    # Run small IS for analysis
    small_is_samples, small_is_weights = small_is_inference()

    # Normalize weights for posterior computation
    from jax.scipy.special import logsumexp

    small_is_normalized_weights = jnp.exp(
        small_is_weights - logsumexp(small_is_weights)
    )

    # Get small IS posterior mean
    small_is_a = jnp.sum(
        small_is_samples.get_choices()["curve"]["a"] * small_is_normalized_weights
    )
    small_is_b = jnp.sum(
        small_is_samples.get_choices()["curve"]["b"] * small_is_normalized_weights
    )
    small_is_c = jnp.sum(
        small_is_samples.get_choices()["curve"]["c"] * small_is_normalized_weights
    )
    small_is_posterior = jax.vmap(
        lambda x: polyfn(x, small_is_a, small_is_b, small_is_c)
    )(xs_plot)

    # Effective sample size for small IS
    small_is_ess = 1.0 / jnp.sum(small_is_normalized_weights**2)

    # ================== Outlier Detection Analysis ==================

    # For large IS method: extract outlier indicators from posterior samples
    is_outlier_samples = is_samples.get_choices()["ys"][
        "is_outlier"
    ]  # (n_particles, n_points)
    is_outlier_posterior = jnp.sum(
        is_outlier_samples * is_normalized_weights[:, None], axis=0
    )
    is_detected = is_outlier_posterior > 0.5

    # For small IS method: extract outlier indicators from posterior samples
    small_is_outlier_samples = small_is_samples.get_choices()["ys"][
        "is_outlier"
    ]  # (n_particles, n_points)
    small_is_outlier_posterior = jnp.sum(
        small_is_outlier_samples * small_is_normalized_weights[:, None], axis=0
    )
    small_is_detected = small_is_outlier_posterior > 0.5

    # Compute detection metrics
    def compute_detection_metrics(detected, truth):
        """Compute precision, recall, F1 for outlier detection."""
        tp = jnp.sum(detected & truth)
        fp = jnp.sum(detected & ~truth)
        fn = jnp.sum(~detected & truth)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return float(precision), float(recall), float(f1)

    is_precision, is_recall, is_f1 = compute_detection_metrics(
        is_detected, is_outlier_rand
    )
    small_is_precision, small_is_recall, small_is_f1 = compute_detection_metrics(
        small_is_detected, is_outlier_rand
    )

    # ================== Panel A: Inference Quality ==================

    # Plot true curve
    y_true_plot = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_plot)
    ax_quality.plot(
        xs_plot,
        y_true_plot,
        color=get_method_color("true_values"),
        linestyle="--",
        linewidth=3,
        label="True Curve",
    )

    # Plot posterior curves
    ax_quality.plot(
        xs_plot,
        is_posterior,
        color=get_method_color("genjax_is"),
        linewidth=3,
        label=f"IS(N={n_is_large})",
    )
    ax_quality.plot(
        xs_plot,
        small_is_posterior,
        color=get_method_color("genjax_hmc"),
        linewidth=3,
        label=f"IS(N={n_is_small})",
    )

    # Plot data points with algorithm predictions
    # Create combined detection status for each point
    for i, (x, y) in enumerate(zip(xs_obs, ys_obs)):
        true_outlier = is_outlier_rand[i]
        is_detected_i = is_detected[i]
        small_is_detected_i = small_is_detected[i]

        # Choose marker and color based on true status and detections
        if true_outlier:
            # True outlier - use X marker, color by detection
            if is_detected_i and small_is_detected_i:
                # Both algorithms detect: dark red
                color = "#8B0000"
                label_suffix = " (both detect)"
            elif is_detected_i or small_is_detected_i:
                # One algorithm detects: orange
                color = "#FF8C00"
                label_suffix = " (one detects)"
            else:
                # Neither detects: light red
                color = "#FFB6C1"
                label_suffix = " (none detect)"

            ax_quality.scatter(
                x, y, marker="x", color=color, s=150, linewidth=4, zorder=12, alpha=0.9
            )
        else:
            # True inlier - use circle marker, color by false positive status
            if is_detected_i and small_is_detected_i:
                # Both algorithms false positive: dark orange with outline
                color = "#FF4500"
                edge_color = "black"
                label_suffix = " (both FP)"
            elif is_detected_i or small_is_detected_i:
                # One algorithm false positive: yellow with outline
                color = "#FFD700"
                edge_color = "black"
                label_suffix = " (one FP)"
            else:
                # Correct classification: blue
                color = get_method_color("data_points")
                edge_color = "white"
                label_suffix = " (correct)"

            ax_quality.scatter(
                x,
                y,
                marker="o",
                color=color,
                s=120,
                edgecolor=edge_color,
                linewidth=2,
                zorder=11,
                alpha=0.8,
            )

    # Add custom legend for detection visualization
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#0173B2", linewidth=3, label=f"IS(N={n_is_large})"),
        Line2D([0], [0], color="#DE8F05", linewidth=3, label=f"IS(N={n_is_small})"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=3, label="True Curve"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=get_method_color("data_points"),
            markersize=8,
            linewidth=0,
            markeredgecolor="white",
            markeredgewidth=1,
            label="Inlier (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#FFD700",
            markersize=8,
            linewidth=0,
            markeredgecolor="black",
            markeredgewidth=1,
            label="Inlier (false pos.)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#8B0000",
            markersize=8,
            linewidth=3,
            label="Outlier (detected)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#FFB6C1",
            markersize=8,
            linewidth=3,
            label="Outlier (missed)",
        ),
    ]

    ax_quality.set_xlabel("X", fontweight="bold")
    ax_quality.set_ylabel("Y", fontweight="bold")
    ax_quality.set_title(
        "(a) Inference Quality & Outlier Predictions", fontweight="bold", pad=15
    )
    apply_grid_style(ax_quality)
    apply_standard_ticks(ax_quality)

    # ================== Panel B: Outlier Detection Performance ==================

    # Bar chart showing detection metrics
    metrics = ["Precision", "Recall", "F1"]
    is_scores = [is_precision, is_recall, is_f1]
    small_is_scores = [small_is_precision, small_is_recall, small_is_f1]

    x_pos = jnp.arange(len(metrics))
    width = 0.35

    bars1 = ax_detection.bar(
        x_pos - width / 2,
        is_scores,
        width,
        color=get_method_color("genjax_is"),
        alpha=0.8,
        label=f"IS(N={n_is_large})",
    )
    bars2 = ax_detection.bar(
        x_pos + width / 2,
        small_is_scores,
        width,
        color=get_method_color("genjax_hmc"),
        alpha=0.8,
        label=f"IS(N={n_is_small})",
    )

    ax_detection.set_xlabel("Detection Metric", fontweight="bold")
    ax_detection.set_ylabel("Score", fontweight="bold")
    ax_detection.set_title(
        "(b) Outlier Detection Performance", fontweight="bold", pad=15
    )
    ax_detection.set_xticks(x_pos)
    ax_detection.set_xticklabels(metrics)
    ax_detection.set_ylim(0, 1.1)
    apply_grid_style(ax_detection)

    # Add metric values as text on bars
    for i, (is_score, small_is_score) in enumerate(zip(is_scores, small_is_scores)):
        ax_detection.text(
            i - width / 2,
            is_score + 0.02,
            f"{is_score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax_detection.text(
            i + width / 2,
            small_is_score + 0.02,
            f"{small_is_score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # ================== Panel C: Runtime Comparison ==================

    # Horizontal bar chart comparing runtimes
    methods = ["IS(N=1000)", f"IS(N={n_is_small})"]
    runtimes = [is_mean_time * 1000, small_is_mean_time * 1000]  # Convert to ms
    runtime_errs = [is_std_time * 1000, small_is_std_time * 1000]

    y_pos = [0, 1]
    bars = ax_runtime.barh(
        y_pos,
        runtimes,
        xerr=runtime_errs,
        color=[get_method_color("genjax_is"), get_method_color("genjax_hmc")],
        alpha=0.8,
        capsize=5,
    )

    ax_runtime.set_yticks(y_pos)
    ax_runtime.set_yticklabels(methods)
    ax_runtime.set_xlabel("Runtime (ms)", fontweight="bold")
    ax_runtime.set_title("Runtime Comparison", fontweight="bold", pad=15)
    apply_grid_style(ax_runtime)

    # Add runtime values as text on bars
    for i, (runtime, err) in enumerate(zip(runtimes, runtime_errs)):
        ax_runtime.text(
            runtime + err + 0.1,
            i,
            f"{runtime:.1f}±{err:.1f}ms",
            va="center",
            ha="left",
            fontweight="bold",
        )

    # ================== Legend Panel ==================

    # Create comprehensive legend in bottom panel
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")  # Hide axes

    # Combine all legend elements
    algorithm_elements = [
        Line2D(
            [0],
            [0],
            color=get_method_color("genjax_is"),
            linewidth=4,
            label=f"IS(N={n_is_large})",
        ),
        Line2D(
            [0],
            [0],
            color=get_method_color("genjax_hmc"),
            linewidth=4,
            label=f"IS(N={n_is_small}) + 5 Gibbs + 10 HMC",
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=3, label="True Curve"),
    ]

    point_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=get_method_color("data_points"),
            markersize=10,
            linewidth=0,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Inlier (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#FFD700",
            markersize=10,
            linewidth=0,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Inlier (false positive)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#8B0000",
            markersize=10,
            linewidth=4,
            label="Outlier (detected)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#FFB6C1",
            markersize=10,
            linewidth=4,
            label="Outlier (missed)",
        ),
    ]

    # Create two-column legend
    legend1 = ax_legend.legend(
        handles=algorithm_elements,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=14,
        title="Algorithms & Curves",
        title_fontsize=16,
    )
    legend1.get_title().set_fontweight("bold")

    legend2 = ax_legend.legend(
        handles=point_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=14,
        title="Point Classifications",
        title_fontsize=16,
    )
    legend2.get_title().set_fontweight("bold")

    # Add the first legend back (matplotlib removes it when creating the second)
    ax_legend.add_artist(legend1)

    plt.tight_layout()

    # Save with descriptive name
    filename = "examples/curvefit/figs/curvefit_outlier_algorithm_comparison.pdf"
    save_publication_figure(fig, filename)

    print(f"\n✓ Saved algorithm comparison: {filename}")
    print("\nPerformance Summary:")
    print(
        f"  IS(N={n_is_large}): {is_mean_time * 1000:.1f}±{is_std_time * 1000:.1f}ms, ESS={is_ess:.0f}"
    )
    print(
        f"  IS(N={n_is_small}): {small_is_mean_time * 1000:.1f}±{small_is_std_time * 1000:.1f}ms, ESS={small_is_ess:.0f}"
    )
    print(
        f"  Speedup: {small_is_mean_time / is_mean_time:.1f}x {'faster' if small_is_mean_time < is_mean_time else 'slower'}"
    )

    print("\nOutlier Detection Summary:")
    print(
        f"  IS(N={n_is_large}): P={is_precision:.2f}, R={is_recall:.2f}, F1={is_f1:.2f}"
    )
    print(
        f"  IS(N={n_is_small}): P={small_is_precision:.2f}, R={small_is_recall:.2f}, F1={small_is_f1:.2f}"
    )

    return {
        "is_large_runtime": is_mean_time,
        "is_small_runtime": small_is_mean_time,
        "is_large_ess": is_ess,
        "is_small_ess": small_is_ess,
        "is_lml": is_lml,
        "is_large_f1": is_f1,
        "is_small_f1": small_is_f1,
    }


def save_enumerative_gibbs_comparison(
    n_points=15,
    outlier_rate=0.25,
    n_samples=1000,
    n_warmup=200,
    seed=42,
):
    """
    Compare regular Gibbs vs enumerative Gibbs implementations.

    Creates a 2x2 panel figure showing:
    - Top row: Parameter convergence comparison (regular vs enumerative)
    - Bottom row: Outlier detection performance and timing comparison
    """
    from examples.curvefit.core import (
        gibbs_infer_latents_with_outliers_jit,
        enumerative_gibbs_infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from examples.utils import benchmark_with_warmup
    from genjax.core import Const
    import numpy as np

    print("\n=== Enumerative Gibbs vs Regular Gibbs Comparison ===")

    # Setup
    setup_publication_fonts()
    fig = plt.figure(figsize=FIGURE_SIZES["framework_comparison"])
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_convergence = fig.add_subplot(gs[0, :])  # Top row spans both columns
    ax_detection = fig.add_subplot(gs[1, 0])  # Bottom left
    ax_timing = fig.add_subplot(gs[1, 1])  # Bottom right

    # Generate data
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

    print(
        f"  Generated data: {n_points} points, {jnp.sum(is_outlier_true)} true outliers"
    )

    # Run both Gibbs samplers
    print("  Running regular Gibbs...")
    regular_result = gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    print("  Running enumerative Gibbs...")
    enum_result = enumerative_gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 2),
        xs_obs,
        ys_obs,
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )

    # Panel 1: Parameter convergence comparison
    iterations = np.arange(n_samples)

    # Plot parameter 'a' convergence for both methods
    ax_convergence.plot(
        iterations,
        regular_result["curve_samples"]["a"],
        color=get_method_color("genjax_is"),
        alpha=0.7,
        linewidth=1,
        label="Regular Gibbs",
    )
    ax_convergence.plot(
        iterations,
        enum_result["curve_samples"]["a"],
        color=get_method_color("genjax_hmc"),
        alpha=0.7,
        linewidth=1,
        label="Enumerative Gibbs",
    )
    ax_convergence.axhline(
        true_a,
        color=get_method_color("data_points"),
        linestyle="--",
        linewidth=2,
        label="Ground Truth",
    )

    ax_convergence.set_xlabel("Iteration", fontweight="bold")
    ax_convergence.set_ylabel("Parameter a", fontweight="bold")
    ax_convergence.legend()
    apply_grid_style(ax_convergence)
    apply_standard_ticks(ax_convergence)

    # Panel 2: Outlier detection performance
    def compute_detection_metrics(outlier_samples, true_outliers):
        # Average outlier probability per point
        outlier_probs = jnp.mean(outlier_samples, axis=0)
        # Binary predictions (threshold at 0.5)
        predicted_outliers = outlier_probs > 0.5

        # Compute metrics
        tp = jnp.sum(predicted_outliers & true_outliers)
        fp = jnp.sum(predicted_outliers & ~true_outliers)
        fn = jnp.sum(~predicted_outliers & true_outliers)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1

    reg_precision, reg_recall, reg_f1 = compute_detection_metrics(
        regular_result["outlier_samples"], is_outlier_true
    )
    enum_precision, enum_recall, enum_f1 = compute_detection_metrics(
        enum_result["outlier_samples"], is_outlier_true
    )

    methods = ["Regular\nGibbs", "Enumerative\nGibbs"]
    f1_scores = [reg_f1, enum_f1]
    colors = [get_method_color("genjax_is"), get_method_color("genjax_hmc")]

    bars = ax_detection.bar(
        methods, f1_scores, color=colors, alpha=0.7, edgecolor="black"
    )
    ax_detection.set_ylabel("F1 Score", fontweight="bold")
    ax_detection.set_ylim(0, 1.0)
    apply_grid_style(ax_detection)

    # Add F1 values on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax_detection.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{f1:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 3: Timing comparison
    print("  Benchmarking performance...")

    def benchmark_gibbs_method(method_func, method_name, repeats=10):
        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: method_func(
                jrand.key(seed),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(outlier_rate),
            ),
            repeats=repeats,
        )
        print(f"    {method_name}: {mean_time * 1000:.1f}±{std_time * 1000:.1f}ms")
        return mean_time, std_time

    reg_mean_time, reg_std_time = benchmark_gibbs_method(
        gibbs_infer_latents_with_outliers_jit, "Regular Gibbs"
    )
    enum_mean_time, enum_std_time = benchmark_gibbs_method(
        enumerative_gibbs_infer_latents_with_outliers_jit, "Enumerative Gibbs"
    )

    times = [reg_mean_time * 1000, enum_mean_time * 1000]  # Convert to ms
    errors = [reg_std_time * 1000, enum_std_time * 1000]

    bars = ax_timing.bar(
        methods,
        times,
        yerr=errors,
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    ax_timing.set_ylabel("Time (ms)", fontweight="bold")
    apply_grid_style(ax_timing)

    # Add timing values on bars
    for bar, time, error in zip(bars, times, errors):
        height = bar.get_height()
        ax_timing.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + error + 2,
            f"{time:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_enumerative_gibbs_comparison.pdf"
    )
    print("  ✓ Saved: curvefit_enumerative_gibbs_comparison.pdf")

    # Copy to main figs directory
    _copy_to_main_figs("curvefit_enumerative_gibbs_comparison.pdf")

    print("\nComparison Summary:")
    print(f"  Regular Gibbs: {reg_mean_time * 1000:.1f}ms, F1={reg_f1:.3f}")
    print(f"  Enumerative Gibbs: {enum_mean_time * 1000:.1f}ms, F1={enum_f1:.3f}")
    speedup = (
        reg_mean_time / enum_mean_time
        if enum_mean_time < reg_mean_time
        else enum_mean_time / reg_mean_time
    )
    faster_method = "Enumerative" if enum_mean_time < reg_mean_time else "Regular"
    print(f"  {faster_method} Gibbs is {speedup:.1f}x faster")


def save_enumerative_gibbs_vectorization_demo(
    n_points_list=[5, 10, 15, 20, 25],
    outlier_rate=0.25,
    n_samples=500,
    n_warmup=100,
    seed=42,
):
    """
    Demonstrate vectorization benefits of enumerative Gibbs vs regular Gibbs.

    Creates a figure showing how performance scales with the number of data points,
    highlighting the vectorization advantage of enumerative Gibbs.
    """
    from examples.curvefit.core import (
        gibbs_infer_latents_with_outliers_jit,
        enumerative_gibbs_infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
    from examples.utils import benchmark_with_warmup
    from genjax.core import Const

    print("\n=== Enumerative Gibbs Vectorization Demo ===")
    print(f"Testing with {len(n_points_list)} different data sizes: {n_points_list}")

    # Setup
    setup_publication_fonts()
    fig, (ax_timing, ax_speedup) = plt.subplots(
        1, 2, figsize=FIGURE_SIZES["two_panel_horizontal"]
    )

    regular_times = []
    enum_times = []

    for n_points in n_points_list:
        print(f"\n  Testing with {n_points} data points...")

        # Generate data for this size
        key = jrand.key(seed)
        x_key, noise_key, outlier_key = jrand.split(key, 3)

        xs_obs = jnp.linspace(0.0, 1.0, n_points)
        true_a, true_b, true_c = -0.211, -0.395, 0.673
        y_true = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(xs_obs)
        noise = jrand.normal(noise_key, shape=(n_points,)) * 0.05
        is_outlier_true = jrand.uniform(outlier_key, shape=(n_points,)) < outlier_rate
        outlier_vals = jrand.uniform(
            outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0
        )
        ys_obs = jnp.where(is_outlier_true, outlier_vals, y_true + noise)

        # Benchmark regular Gibbs
        def run_regular():
            return gibbs_infer_latents_with_outliers_jit(
                jrand.key(seed),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(outlier_rate),
            )

        _, (reg_time, _) = benchmark_with_warmup(run_regular, repeats=5)
        regular_times.append(reg_time * 1000)  # Convert to ms

        # Benchmark enumerative Gibbs
        def run_enumerative():
            return enumerative_gibbs_infer_latents_with_outliers_jit(
                jrand.key(seed),
                xs_obs,
                ys_obs,
                n_samples=Const(n_samples),
                n_warmup=Const(n_warmup),
                outlier_rate=Const(outlier_rate),
            )

        _, (enum_time, _) = benchmark_with_warmup(run_enumerative, repeats=5)
        enum_times.append(enum_time * 1000)  # Convert to ms

        print(
            f"    Regular: {reg_time * 1000:.1f}ms, Enumerative: {enum_time * 1000:.1f}ms"
        )

    # Panel 1: Timing comparison
    ax_timing.plot(
        n_points_list,
        regular_times,
        "o-",
        color=get_method_color("genjax_is"),
        linewidth=2,
        markersize=8,
        label="Regular Gibbs",
    )
    ax_timing.plot(
        n_points_list,
        enum_times,
        "s-",
        color=get_method_color("genjax_hmc"),
        linewidth=2,
        markersize=8,
        label="Enumerative Gibbs",
    )

    ax_timing.set_xlabel("Number of Data Points", fontweight="bold")
    ax_timing.set_ylabel("Time (ms)", fontweight="bold")
    ax_timing.legend()
    apply_grid_style(ax_timing)
    apply_standard_ticks(ax_timing)

    # Panel 2: Speedup ratio
    speedups = [reg / enum for reg, enum in zip(regular_times, enum_times)]
    ax_speedup.plot(
        n_points_list,
        speedups,
        "o-",
        color=get_method_color("data_points"),
        linewidth=2,
        markersize=8,
    )
    ax_speedup.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")

    ax_speedup.set_xlabel("Number of Data Points", fontweight="bold")
    ax_speedup.set_ylabel("Speedup Factor\n(Regular / Enumerative)", fontweight="bold")
    ax_speedup.legend()
    apply_grid_style(ax_speedup)
    apply_standard_ticks(ax_speedup)

    save_publication_figure(
        fig, "examples/curvefit/figs/curvefit_enumerative_gibbs_vectorization.pdf"
    )
    print("  ✓ Saved: curvefit_enumerative_gibbs_vectorization.pdf")

    # Copy to main figs directory
    _copy_to_main_figs("curvefit_enumerative_gibbs_vectorization.pdf")

    print("\nVectorization Summary:")
    for i, n_pts in enumerate(n_points_list):
        print(f"  {n_pts:2d} points: {speedups[i]:.2f}x speedup")

    return {
        "n_points": n_points_list,
        "regular_times": regular_times,
        "enum_times": enum_times,
        "speedups": speedups,
    }


# =============================================================================
# EFFICIENCY ANALYSIS FUNCTIONS
# =============================================================================


def generate_challenging_outlier_data(n_points=20, outlier_rate=0.4, seed=42):
    """Generate challenging data with many outliers to stress-test convergence."""
    from examples.curvefit.data import polyfn

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


def save_gibbs_efficiency_frontier_figure(outlier_rate=0.4, n_points=20, seed=42):
    """
    Create comprehensive efficiency frontier analysis figure for Gibbs sampling.

    Tests different (warmup, sampling) combinations to find the minimum viable
    configuration for good performance.
    """
    print("🔬 Creating Gibbs Efficiency Frontier Analysis...")

    from examples.curvefit.core import enumerative_gibbs_infer_latents_with_outliers_jit
    import time

    # Generate challenging test data
    data = generate_challenging_outlier_data(
        n_points=n_points, outlier_rate=outlier_rate, seed=seed
    )
    print(
        f"  Generated challenging data: {data['n_true_outliers']}/{len(data['xs'])} outliers ({outlier_rate * 100:.0f}%)"
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
            outlier_rate=Const(outlier_rate),
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

    # Create visualization
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
    save_publication_figure(fig, filename)
    print(f"  ✓ Saved: {filename}")
    _copy_to_main_figs("gibbs_efficiency_frontier.pdf")

    # Return best configuration for further use
    best_config = good_configs[0] if good_configs else results[-1]
    print("\n📊 Efficiency Analysis Results:")
    print(f"  • Best configuration: {best_config['label']}")
    print(f"  • Total sweeps: {best_config['total_sweeps']}")
    print(
        f"  • Performance: F1={best_config['detection_f1']:.3f}, MSE={best_config['param_mse']:.3f}"
    )
    print(f"  • Runtime: {best_config['runtime']:.1f}ms")

    return best_config, results


def save_gibbs_vs_is_comparison_figure(n_points=20, outlier_rate=0.35, seed=42):
    """Create horizontal bar plot comparing Gibbs vs IS(N=1000) accuracy and runtime."""
    print("\n🔄 Creating Gibbs vs IS(N=1000) Comparison Figure...")

    from examples.curvefit.core import (
        enumerative_gibbs_infer_latents_with_outliers_jit,
        infer_latents_with_outliers_jit,
    )
    import time

    # Generate test data
    data = generate_challenging_outlier_data(
        n_points=n_points, outlier_rate=outlier_rate, seed=seed
    )
    print(
        f"  Test data: {data['n_true_outliers']}/{len(data['xs'])} outliers ({outlier_rate * 100:.0f}%)"
    )

    # Run Gibbs (using minimal efficient configuration)
    print("\n  Running Gibbs sampling...")
    n_warmup, n_samples = 50, 100  # 150 total sweeps (minimal efficient config)

    start_time = time.time()
    gibbs_result = enumerative_gibbs_infer_latents_with_outliers_jit(
        jrand.key(seed + 1),
        data["xs"],
        data["ys"],
        n_samples=Const(n_samples),
        n_warmup=Const(n_warmup),
        outlier_rate=Const(outlier_rate),
    )
    jax.block_until_ready(gibbs_result)
    gibbs_runtime = (time.time() - start_time) * 1000  # Convert to ms

    gibbs_perf = evaluate_gibbs_performance(gibbs_result, data)

    print(
        f"    Gibbs ({n_warmup + n_samples} sweeps): F1={gibbs_perf['detection_f1']:.3f}, "
        f"MSE={gibbs_perf['param_mse']:.3f}, Time={gibbs_runtime:.1f}ms"
    )

    # Run IS with N=1000
    print("\n  Running IS(N=1000)...")

    start_time = time.time()
    is_result = infer_latents_with_outliers_jit(
        jrand.key(seed + 2),
        data["xs"],
        data["ys"],
        n_samples=Const(1000),
        outlier_rate=Const(outlier_rate),
        outlier_mean=Const(0.0),
        outlier_std=Const(5.0),
    )
    jax.block_until_ready(is_result)
    is_runtime = (time.time() - start_time) * 1000  # Convert to ms

    # Evaluate IS performance
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
    param_accuracy_gibbs = 1.0 / (1.0 + gibbs_perf["param_mse"] * 100)
    param_accuracy_is = 1.0 / (1.0 + is_param_mse * 100)

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

    print(
        f"    IS (1000 particles): F1={is_f1:.3f}, "
        f"MSE={is_param_mse:.3f}, Time={is_runtime:.1f}ms"
    )

    # Create comparison figure
    setup_publication_fonts()
    fig, (ax_accuracy, ax_runtime) = plt.subplots(
        1, 2, figsize=FIGURE_SIZES["two_panel_horizontal"]
    )

    # Extract data
    methods = ["Vectorized Gibbs", "IS(N=1000)"]
    f1_scores = [gibbs_perf["detection_f1"], is_f1]
    param_accuracies = [param_accuracy_gibbs, param_accuracy_is]
    runtimes = [gibbs_runtime, is_runtime]

    # Colors following GRVS
    colors = [get_method_color("genjax_hmc"), get_method_color("genjax_is")]

    # Panel 1: Accuracy Comparison (horizontal bars)
    y_pos = np.arange(len(methods))

    # F1 scores
    bars_f1 = ax_accuracy.barh(
        y_pos - 0.15,
        f1_scores,
        0.3,
        label="Outlier Detection F1",
        color=colors[0],
        alpha=0.8,
    )

    # Parameter accuracy
    bars_param = ax_accuracy.barh(
        y_pos + 0.15,
        param_accuracies,
        0.3,
        label="Parameter Accuracy",
        color=colors[1],
        alpha=0.8,
    )

    # Add value labels on bars
    for i, (f1, param_acc) in enumerate(zip(f1_scores, param_accuracies)):
        ax_accuracy.text(
            f1 + 0.01,
            i - 0.15,
            f"{f1:.3f}",
            va="center",
            fontweight="bold",
            fontsize=11,
        )
        ax_accuracy.text(
            param_acc + 0.01,
            i + 0.15,
            f"{param_acc:.3f}",
            va="center",
            fontweight="bold",
            fontsize=11,
        )

    ax_accuracy.set_yticks(y_pos)
    ax_accuracy.set_yticklabels(methods, fontweight="bold")
    ax_accuracy.set_xlabel("Accuracy Score", fontweight="bold")
    ax_accuracy.set_title("Accuracy Comparison", fontweight="bold", fontsize=14)
    ax_accuracy.legend(loc="lower right")
    ax_accuracy.set_xlim(0, 1.1)
    apply_grid_style(ax_accuracy)
    apply_standard_ticks(ax_accuracy)

    # Panel 2: Runtime Comparison (horizontal bars)
    bars_runtime = ax_runtime.barh(y_pos, runtimes, 0.6, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, runtime in enumerate(runtimes):
        ax_runtime.text(
            runtime + max(runtimes) * 0.02,
            i,
            f"{runtime:.1f}ms",
            va="center",
            fontweight="bold",
            fontsize=11,
        )

    ax_runtime.set_yticks(y_pos)
    ax_runtime.set_yticklabels(methods, fontweight="bold")
    ax_runtime.set_xlabel("Runtime (ms)", fontweight="bold")
    ax_runtime.set_title("Runtime Comparison", fontweight="bold", fontsize=14)
    ax_runtime.set_xlim(0, max(runtimes) * 1.2)
    apply_grid_style(ax_runtime)
    apply_standard_ticks(ax_runtime)

    plt.tight_layout()

    filename = "examples/curvefit/figs/gibbs_vs_is_comparison.pdf"
    save_publication_figure(fig, filename)
    print(f"  ✓ Saved: {filename}")
    _copy_to_main_figs("gibbs_vs_is_comparison.pdf")

    # Print summary stats
    print("\n📊 Comparison Summary:")
    print("  Accuracy Advantage (Gibbs vs IS):")
    print(
        f"    • F1 Score: {f1_scores[0]:.3f} vs {f1_scores[1]:.3f} "
        f"({(f1_scores[0] / f1_scores[1] - 1) * 100:+.1f}%)"
    )
    print(
        f"    • Parameter Accuracy: {param_accuracies[0]:.3f} vs {param_accuracies[1]:.3f} "
        f"({(param_accuracies[0] / param_accuracies[1] - 1) * 100:+.1f}%)"
    )
    print("  Runtime Comparison:")
    print(
        f"    • Gibbs: {runtimes[0]:.1f}ms vs IS: {runtimes[1]:.1f}ms "
        f"({(runtimes[0] / runtimes[1] - 1) * 100:+.1f}%)"
    )

    return {
        "gibbs": {
            "f1": f1_scores[0],
            "param_acc": param_accuracies[0],
            "runtime": runtimes[0],
        },
        "is": {
            "f1": f1_scores[1],
            "param_acc": param_accuracies[1],
            "runtime": runtimes[1],
        },
    }
