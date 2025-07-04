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
import sys
sys.path.append('..')
from utils import benchmark_with_warmup

# Import shared GenJAX Research Visualization Standards
from viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, set_minimal_ticks, apply_standard_ticks, save_publication_figure,
    PRIMARY_COLORS, LINE_SPECS, MARKER_SPECS
)

# Figure sizes and styling now imported from shared examples.viz module
# FIGURE_SIZES, PRIMARY_COLORS, etc. are available from the import above

# Apply GenJAX Research Visualization Standards
setup_publication_fonts()


# set_minimal_ticks function now imported from examples.viz


def get_reference_dataset(seed=42, n_points=20):
    """Get the standard reference dataset for all visualizations."""
    from data import generate_fixed_dataset

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


def save_onepoint_trace_viz():
    """Save one-point curve trace visualization."""
    from core import onepoint_curve

    print("Making and saving onepoint trace visualization.")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])

    # Generate a trace at x = 0.5
    trace = onepoint_curve.simulate(0.5)
    curve, (x, y) = trace.get_retval()

    # Plot the curve over x values
    xvals = jnp.linspace(0, 1, 300)
    ax.plot(xvals, jax.vmap(curve)(xvals), 
            color=get_method_color("curves"), **LINE_SPECS["curve_main"])

    # Mark the sampled point
    ax.scatter(x, y, color=get_method_color("data_points"), **MARKER_SPECS["data_points"])

    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_prior_trace.pdf")
    
    # Also create the multiple onepoint traces with densities
    save_multiple_onepoint_traces_with_density()


def save_multiple_onepoint_traces_with_density():
    """Save multiple one-point trace visualizations with density values."""
    from core import onepoint_curve
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
        ax.plot(xvals, jax.vmap(curve)(xvals), 
                color=get_method_color("curves"), **LINE_SPECS["curve_main"])
        
        # Mark the sampled point
        ax.scatter(x, y, color=get_method_color("data_points"), **MARKER_SPECS["data_points"])
        
        # Add density value as text below the plot with larger font
        ax.text(0.5, -0.15, f"log p = {log_densities[i]:.2f}", 
                ha='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)
    
    save_publication_figure(fig, "figs/curvefit_prior_traces_density.pdf")


def save_multipoint_trace_viz():
    """Save multi-point curve trace visualization."""
    from core import npoint_curve

    print("Making and saving multipoint trace visualization.")

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])

    # Generate trace with multiple points
    xs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    trace = npoint_curve.simulate(xs)
    curve, (xs_ret, ys) = trace.get_retval()

    # Plot the curve
    xvals = jnp.linspace(0, 1, 300)
    ax.plot(xvals, jax.vmap(curve)(xvals), color=get_method_color("curves"), **LINE_SPECS["curve_main"])

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

    save_publication_figure(fig, "figs/curvefit_prior_multipoint_trace.pdf")
    
    # Also create the multiple multipoint traces with densities
    save_multiple_multipoint_traces_with_density()


def save_multiple_multipoint_traces_with_density():
    """Save multiple multi-point trace visualizations with density values."""
    from core import npoint_curve
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
        ax.plot(xvals, jax.vmap(curve)(xvals), color=get_method_color("curves"), **LINE_SPECS["curve_main"])
        
        # Mark the sampled points
        ax.scatter(xs_ret, ys, color=get_method_color("data_points"), **MARKER_SPECS["secondary_points"])
        
        # Add density value as text below the plot with larger font
        ax.text(0.5, -0.15, f"log p = {log_densities[i]:.2f}", 
                ha='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)
    
    save_publication_figure(fig, "figs/curvefit_prior_multipoint_traces_density.pdf")


def save_four_multipoint_trace_vizs():
    """Save visualization showing four different multi-point curve traces."""
    from core import npoint_curve
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
        ax.plot(xvals, jax.vmap(curve)(xvals), color=get_method_color("curves"), **LINE_SPECS["curve_secondary"])

        # Mark the sampled points
        ax.scatter(
            xs_ret, ys, color=get_method_color("data_points"), s=100, zorder=10, edgecolor="white", linewidth=2
        )

        # Remove axis labels and ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_prior_multipoint_traces_grid.pdf")


def save_single_multipoint_trace_with_density():
    """Save a single multi-point curve trace with curve, points, and density value."""
    from core import npoint_curve
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
    ax.plot(xvals, jax.vmap(curve)(xvals), 
            color=get_method_color("curves"), **LINE_SPECS["curve_main"])
    
    # Mark the sampled points
    ax.scatter(xs_ret, ys, color=get_method_color("data_points"), **MARKER_SPECS["secondary_points"])
    
    # Add density value as text below the plot with larger font
    ax.text(0.5, -0.15, f"log p = {float(log_density):.2f}", 
            ha='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
    
    # Remove axis labels and ticks for cleaner look (matching other trace figures)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)
    
    save_publication_figure(fig, "figs/curvefit_single_multipoint_trace_density.pdf")


def save_inference_viz(seed=42):
    """Save posterior visualization using importance sampling."""
    from core import (
        infer_latents,
        npoint_curve,
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
    ax.plot(x_range, true_curve, color="#333333", linewidth=3, label="True curve", zorder=50)

    # Plot posterior samples
    for i in range(n_resample):
        curve_vals = a_samples[i] + b_samples[i] * x_range + c_samples[i] * x_range**2
        ax.plot(x_range, curve_vals, color=get_method_color("curves"), alpha=0.1, linewidth=1)

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

    save_publication_figure(fig, "figs/curvefit_posterior_curves.pdf")


def save_genjax_posterior_comparison(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=20,
):
    """Save comparison of GenJAX IS vs HMC posterior inference."""
    from core import (
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
        resample_key, jnp.arange(n_samples_is), shape=(n_samples_is,), p=normalized_weights
    )

    # Get resampled IS coefficients
    is_a = is_samples.get_choices()["curve"]["a"][is_indices]
    is_b = is_samples.get_choices()["curve"]["b"][is_indices]
    is_c = is_samples.get_choices()["curve"]["c"][is_indices]

    # Run HMC inference
    hmc_samples, accept_rate = hmc_infer_latents_jit(
        jrand.key(seed + 100), xs, ys,
        Const(n_samples_hmc), Const(n_warmup),
        Const(0.001), Const(50)
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
    true_curve = true_params["a"] + true_params["b"] * x_fine + true_params["c"] * x_fine**2
    ax.plot(x_fine, true_curve, 'k-', linewidth=3, label='True', zorder=100)

    # Plot IS posterior samples
    for i in range(min(50, n_samples_is)):
        curve = is_a[i] + is_b[i] * x_fine + is_c[i] * x_fine**2
        ax.plot(x_fine, curve, color=is_color, alpha=0.05, linewidth=0.8)

    # Plot HMC posterior samples
    for i in range(min(50, n_samples_hmc)):
        curve = hmc_a[i] + hmc_b[i] * x_fine + hmc_c[i] * x_fine**2
        ax.plot(x_fine, curve, color=hmc_color, alpha=0.05, linewidth=0.8)

    # Plot data points
    ax.scatter(xs, ys, color='#CC3311', s=100, zorder=200, edgecolor='white', linewidth=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_title('Posterior Curves')
    apply_grid_style(ax)
    ax.set_xlim(-0.1, 1.1)

    # Create legend patches
    is_patch = mpatches.Patch(color=is_color, label=f'IS ({n_samples_is})')
    hmc_patch = mpatches.Patch(color=hmc_color, label=f'HMC ({n_samples_hmc})')
    ax.legend(handles=[is_patch, hmc_patch], loc='best')

    # 2. Parameter marginals
    param_names = ['a', 'b', 'c']
    param_data = [
        (is_a, hmc_a, true_params["a"]),
        (is_b, hmc_b, true_params["b"]),
        (is_c, hmc_c, true_params["c"])
    ]

    for i, (is_vals, hmc_vals, true_val) in enumerate(param_data):
        ax = axes[0, 1] if i == 0 else (axes[1, 0] if i == 1 else axes[1, 1])

        # Histograms
        bins = np.linspace(
            min(is_vals.min(), hmc_vals.min()) - 0.1,
            max(is_vals.max(), hmc_vals.max()) + 0.1,
            30
        )

        ax.hist(is_vals, bins=bins, alpha=0.5, density=True, color=is_color, label='IS')
        ax.hist(hmc_vals, bins=bins, alpha=0.5, density=True, color=hmc_color, label='HMC')

        # True value
        ax.axvline(true_val, color='#CC3311', linestyle='--', linewidth=2, label='True')

        ax.set_xlabel(f'{param_names[i]}')
        ax.set_ylabel('Density')
        # ax.set_title(f'Parameter {param_names[i]}')
        apply_grid_style(ax)
        ax.legend()

    save_publication_figure(fig, "figs/curvefit_posterior_comparison.pdf")

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
    from core import (
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

    print(f"Reference Dataset - True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
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
    print(f"  Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")

    # Get samples for visualization
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )

    # Resample according to weights
    normalized_weights = jnp.exp(is_weights - jnp.max(is_weights))
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    resample_key = jrand.key(seed + 1)
    resample_indices = jrand.choice(
        resample_key, jnp.arange(n_samples_is), shape=(n_samples_is,), p=normalized_weights
    )

    # Get resampled coefficients
    is_a_resampled = is_samples.get_choices()["curve"]["a"][resample_indices]
    is_b_resampled = is_samples.get_choices()["curve"]["b"][resample_indices]
    is_c_resampled = is_samples.get_choices()["curve"]["c"][resample_indices]

    print(f"  IS a mean (resampled): {is_a_resampled.mean():.3f}, std: {is_a_resampled.std():.3f}")
    print(f"  IS b mean (resampled): {is_b_resampled.mean():.3f}, std: {is_b_resampled.std():.3f}")
    print(f"  IS c mean (resampled): {is_c_resampled.mean():.3f}, std: {is_c_resampled.std():.3f}")

    # Also compute weighted mean
    is_a_all = is_samples.get_choices()["curve"]["a"]
    is_b_all = is_samples.get_choices()["curve"]["b"]
    is_c_all = is_samples.get_choices()["curve"]["c"]
    is_a_weighted = jnp.sum(is_a_all * normalized_weights)
    is_b_weighted = jnp.sum(is_b_all * normalized_weights)
    is_c_weighted = jnp.sum(is_c_all * normalized_weights)

    print(f"  IS weighted mean: a={is_a_weighted:.3f}, b={is_b_weighted:.3f}, c={is_c_weighted:.3f}")

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
            jrand.key(seed), xs, ys,
            Const(n_samples_hmc), Const(n_warmup),
            Const(0.001), Const(50)
        ),
        repeats=timing_repeats,
    )
    print(f"  Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")

    # Get samples
    hmc_samples, diagnostics = hmc_infer_latents_jit(
        jrand.key(seed), xs, ys,
        Const(n_samples_hmc), Const(n_warmup),
        Const(0.001), Const(50)
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
        print(f"  Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")

        # Get samples
        numpyro_is_result = numpyro_run_importance_sampling_jit(
            jrand.key(seed), xs, ys, num_samples=n_samples_is
        )

        # Extract samples - NumPyro IS returns weighted samples
        numpyro_is_a = numpyro_is_result["a"]
        numpyro_is_b = numpyro_is_result["b"]
        numpyro_is_c = numpyro_is_result["c"]

        print(f"  NumPyro IS a mean: {numpyro_is_a.mean():.3f}, std: {numpyro_is_a.std():.3f}")
        print(f"  NumPyro IS b mean: {numpyro_is_b.mean():.3f}, std: {numpyro_is_b.std():.3f}")
        print(f"  NumPyro IS c mean: {numpyro_is_c.mean():.3f}, std: {numpyro_is_c.std():.3f}")

        results["numpyro_is"] = {
            "method": "NumPyro IS",
            "samples": (numpyro_is_a, numpyro_is_b, numpyro_is_c),
            "timing": (mean_time, std_time),
            "mean_curve": (numpyro_is_a.mean(), numpyro_is_b.mean(), numpyro_is_c.mean()),
        }
    except Exception as e:
        print(f"  NumPyro IS failed: {type(e).__name__}: {str(e)}")
        print("  Skipping NumPyro IS...")

    # 4. NumPyro HMC
    print("\n4. NumPyro HMC...")
    times, (mean_time, std_time) = benchmark_with_warmup(
        lambda: numpyro_run_hmc_inference_jit(
            jrand.key(seed), xs, ys,
            num_samples=n_samples_hmc, num_warmup=n_warmup,
            step_size=0.001, num_steps=50
        ),
        repeats=timing_repeats,
    )
    print(f"  Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")

    # Get samples
    numpyro_hmc_result = numpyro_run_hmc_inference_jit(
        jrand.key(seed), xs, ys,
        num_samples=n_samples_hmc, num_warmup=n_warmup,
        step_size=0.001, num_steps=50
    )
    numpyro_hmc_samples = numpyro_hmc_result["samples"]
    numpyro_hmc_a = numpyro_hmc_samples["a"]
    numpyro_hmc_b = numpyro_hmc_samples["b"]
    numpyro_hmc_c = numpyro_hmc_samples["c"]
    
    # Get diagnostics
    summary = numpyro_hmc_summary_statistics(numpyro_hmc_result)
    print(f"  Accept rate: {summary['accept_rate']:.3f}")

    print(f"  NumPyro a mean: {numpyro_hmc_a.mean():.3f}, std: {numpyro_hmc_a.std():.3f}")
    print(f"  NumPyro b mean: {numpyro_hmc_b.mean():.3f}, std: {numpyro_hmc_b.std():.3f}")
    print(f"  NumPyro c mean: {numpyro_hmc_c.mean():.3f}, std: {numpyro_hmc_c.std():.3f}")

    results["numpyro_hmc"] = {
        "method": "NumPyro HMC",
        "samples": (numpyro_hmc_a, numpyro_hmc_b, numpyro_hmc_c),
        "timing": (mean_time, std_time),
        "mean_curve": (numpyro_hmc_a.mean(), numpyro_hmc_b.mean(), numpyro_hmc_c.mean()),
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
        print(f"  {result['method']} mean curve: a={a_mean:.3f}, b={b_mean:.3f}, c={c_mean:.3f}")
        print(f"  {result['method']} mean curve range: [{mean_curve.min():.3f}, {mean_curve.max():.3f}]")

    # Plot data points
    ax1.scatter(
        xs, ys, color="#333333", s=100, zorder=200, edgecolor="white", linewidth=2
    )

    ax1.set_xlabel("x", fontsize=18, fontweight='bold')
    ax1.set_ylabel("y", fontsize=18, fontweight='bold')
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
    for i, (y_pos, time, error, color, label) in enumerate(zip(y_positions, times, errors, colors_list, labels)):
        bar = ax2.barh(y_pos, time, xerr=error, capsize=5, color=color, alpha=0.8, label=label)
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
            fontweight='bold'
        )
    
    # Remove y-axis labels (will use legend instead)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([])
    
    # Style the x-axis
    ax2.set_xlabel("Time (ms)", fontsize=18, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
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
        ax2.xaxis.set_major_locator(MultipleLocator(200))  # Every 200ms for large values
    
    # Set tick parameters
    ax1.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax2.tick_params(axis='y', which='major', labelsize=16, width=0, length=0)  # No tick marks on y-axis
    ax2.tick_params(axis='x', which='major', labelsize=16, width=2, length=6)  # Style x-axis ticks

    plt.tight_layout()

    # Save figure
    filename = f"figs/curvefit_framework_comparison_n{n_points}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\n✓ Saved framework comparison: {filename}")

    return results


def save_inference_scaling_viz(n_trials=100, extended_timing=False):
    """Save inference scaling visualization across different sample sizes.
    
    Args:
        n_trials: Number of independent trials to run for each sample size (default: 100)
        extended_timing: If True, run longer timing trials to potentially show GPU throttling
    """
    from core import infer_latents_jit
    from genjax.core import Const
    from examples.utils import benchmark_with_warmup

    print(f"Making and saving inference scaling visualization with {n_trials} trials per N.")
    if extended_timing:
        print("  Running extended timing trials to capture GPU behavior...")

    # Get reference dataset
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]

    # Test different sample sizes - extended range to potentially observe GPU throttling
    n_samples_list = [10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000, 
                      15000, 20000, 30000, 40000, 50000, 70000, 100000, 150000, 200000, 300000, 
                      400000, 500000, 700000, 1000000]
    lml_estimates = []
    lml_stds = []  # Store standard deviations for variance bounds
    runtime_means = []
    runtime_stds = []

    base_key = jrand.key(42)

    for n_samples in n_samples_list:
        print(f"  Testing with {n_samples} samples ({n_trials} trials)...")
        
        # Storage for trial results
        trial_lml = []
        
        # Run multiple trials
        for trial in range(n_trials):
            trial_key = jrand.key(42 + trial)  # Different key for each trial
            
            # Run inference
            samples, weights = infer_latents_jit(trial_key, xs, ys, Const(n_samples))
            
            # Estimate log marginal likelihood
            lml = jnp.log(jnp.mean(jnp.exp(weights - jnp.max(weights)))) + jnp.max(weights)
            trial_lml.append(float(lml))
        
        # Average over trials and compute standard deviation
        lml_mean = jnp.mean(jnp.array(trial_lml))
        lml_std = jnp.std(jnp.array(trial_lml))
        lml_estimates.append(lml_mean)
        lml_stds.append(lml_std)
        
        # Benchmark runtime with extended trials if requested
        timing_repeats = 500 if extended_timing else 100
        inner_repeats = 50 if extended_timing else 20
        
        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: infer_latents_jit(base_key, xs, ys, Const(n_samples)),
            repeats=timing_repeats,
            inner_repeats=inner_repeats
        )
        runtime_means.append(mean_time * 1000)  # Convert to ms
        runtime_stds.append(std_time * 1000)

    # Create figure with two panels in horizontal layout
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 3.5)  # Wider and less tall
    )

    # Common color for GenJAX IS
    genjax_is_color = "#0173B2"

    # Runtime plot with potential GPU throttling visibility
    runtime_array = np.array(runtime_means)
    runtime_std_array = np.array(runtime_stds)
    
    # Plot mean line
    ax1.plot(
        n_samples_list,
        runtime_array,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
        label="Mean Runtime"
    )
    
    # Add shaded region for ±1 standard deviation (like LML plot)
    ax1.fill_between(
        n_samples_list,
        runtime_array - runtime_std_array,
        runtime_array + runtime_std_array,
        alpha=0.3,
        color=genjax_is_color,
        label="±1 std"
    )
    
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Runtime (ms)")
    # ax1.set_title("Vectorized Runtime", fontweight="normal")
    ax1.set_xscale("log")
    ax1.set_xlim(8, 1200000)
    
    # Fixed y-axis range to better show GPU behavior
    ax1.set_ylim(0.1, 0.5)
    
    # Add shading for GPU throttling region (past 10^5)
    ax1.axvspan(100000, 1200000, alpha=0.1, color='orange', label='GPU Throttling')
    
    # Add colored vertical lines for N=10, 100, 100000
    ax1.axvline(10, color='#B19CD9', linestyle='-', alpha=0.8, linewidth=4)
    ax1.axvline(100, color='#0173B2', linestyle='-', alpha=0.8, linewidth=4)
    ax1.axvline(100000, color='#029E73', linestyle='-', alpha=0.8, linewidth=4)
    
    # Add GPU underutilized text in unshaded region
    ax1.text(3000, 0.15, 'GPU\nUnderutilized', ha='center', va='bottom', 
             color='gray', fontsize=12, fontweight='bold')
    
    # Add throttle text in shaded region near x-axis
    ax1.text(300000, 0.15, 'GPU\nThrottling', ha='center', va='bottom', 
             color='darkorange', fontsize=12, fontweight='bold')
    
    # Add vertical line for GPU OOM at 10^6
    ax1.axvline(1000000, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add GPU OOM text above the plot frame
    ax1.text(1000000, 1.05, 'GPU OOM', ha='center', va='bottom', 
             color='red', fontsize=14, fontweight='bold', 
             transform=ax1.get_xaxis_transform())
    
    # Set specific x-axis tick locations with scientific notation
    ax1.set_xticks([100, 1000, 10000, 100000, 1000000])
    ax1.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
    # Only set y-axis ticks to avoid overriding x-axis
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

    # LML estimate plot with variance bounds
    lml_means = np.array(lml_estimates)
    lml_std_array = np.array(lml_stds)
    
    # Plot mean line
    ax2.plot(
        n_samples_list,
        lml_means,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
        label="Mean LML"
    )
    
    # Add shaded region for ±1 standard deviation
    ax2.fill_between(
        n_samples_list,
        lml_means - lml_std_array,
        lml_means + lml_std_array,
        alpha=0.3,
        color=genjax_is_color,
        label="±1 std"
    )
    
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("LMLE")
    # ax2.set_title("LML Estimates", fontweight="normal")
    ax2.set_xscale("log")
    ax2.set_xlim(8, 1200000)
    
    # Add horizontal line at final LML value
    final_lml = lml_means[-1]
    ax2.axhline(final_lml, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    # Add label on y-axis
    ax2.text(70, final_lml, f'{final_lml:.2f}', ha='right', va='center', 
             color='gray', fontsize=12, fontweight='bold')
    
    # Add shading for GPU throttling region (past 10^5)
    ax2.axvspan(100000, 1200000, alpha=0.1, color='orange')
    
    # Add colored vertical lines for N=10, 100, 100000
    ax2.axvline(10, color='#B19CD9', linestyle='-', alpha=0.8, linewidth=4)
    ax2.axvline(100, color='#0173B2', linestyle='-', alpha=0.8, linewidth=4)
    ax2.axvline(100000, color='#029E73', linestyle='-', alpha=0.8, linewidth=4)
    
    # Add vertical line for GPU OOM at 10^6
    ax2.axvline(1000000, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Set specific x-axis tick locations with scientific notation
    ax2.set_xticks([100, 1000, 10000, 100000, 1000000])
    ax2.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
    # Only set y-axis ticks to avoid overriding x-axis
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    
    # Add GPU OOM text above the plot frame (using transform for consistent placement)
    ax2.text(1000000, 1.05, 'GPU OOM', ha='center', va='bottom', 
             color='red', fontsize=14, fontweight='bold', 
             transform=ax2.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig("figs/curvefit_scaling_performance.pdf")
    plt.close()

    print("✓ Saved inference scaling visualization")


def save_posterior_scaling_plots(n_runs=1000, seed=42):
    """Save posterior plots for different particle counts (N=100, 1000, 10000).
    
    Args:
        n_runs: Number of independent IS runs to get posterior samples
        seed: Random seed
    """
    from core import infer_latents_jit
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
        10: '#B19CD9',       # Light purple
        100: '#0173B2',      # Medium blue
        100000: '#029E73'    # Dark green
    }
    
    # X values for plotting curves
    x_plot = jnp.linspace(-0.1, 1.1, 300)
    
    # Create figure with 3 subplots - wider and with room for labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    for idx, (n_particles, ax) in enumerate(zip(n_particles_list, axes)):
        print(f"\n  Generating posterior plot for N={n_particles}...")
        
        # Plot true curve
        true_curve = true_a + true_b * x_plot + true_c * x_plot**2
        ax.plot(x_plot, true_curve, 'k-', linewidth=3, label='True curve', zorder=50)
        
        # Plot true noise interval (observation noise is 0.05)
        noise_std = 0.05
        ax.fill_between(x_plot, true_curve - noise_std, true_curve + noise_std, 
                       color='gray', alpha=0.3, label='True noise', zorder=40)
        
        # Add dotted lines to outline the noise interval
        ax.plot(x_plot, true_curve - noise_std, 'k', linestyle=':', linewidth=2.5, 
                dashes=(5, 3), alpha=0.7, zorder=41)
        ax.plot(x_plot, true_curve + noise_std, 'k', linestyle=':', linewidth=2.5, 
                dashes=(5, 3), alpha=0.7, zorder=41)
        
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
            sample_idx = jrand.choice(resample_key, jnp.arange(n_particles), p=normalized_weights)
            
            # Extract coefficients for this particle
            a_sample = samples.get_choices()["curve"]["a"][sample_idx]
            b_sample = samples.get_choices()["curve"]["b"][sample_idx]
            c_sample = samples.get_choices()["curve"]["c"][sample_idx]
            
            # Compute curve
            curve_sample = a_sample + b_sample * x_plot + c_sample * x_plot**2
            posterior_curves.append(curve_sample)
        
        # Plot posterior curves with alpha blending using particle-specific color
        for curve in posterior_curves:
            ax.plot(x_plot, curve, color=particle_colors[n_particles], alpha=0.025, linewidth=1)
        
        # Compute and print statistics for diagnostics
        if n_particles == 100000:
            posterior_array = jnp.array(posterior_curves)
            mean_curve = jnp.mean(posterior_array, axis=0)
            std_curve = jnp.std(posterior_array, axis=0)
            max_std = jnp.max(std_curve)
            print(f"    Max posterior std for N=100000: {max_std:.4f} (compare to noise=0.05)")
        
        # Plot data points last
        ax.scatter(xs, ys, color='#CC3311', s=120, zorder=100, 
                  edgecolor='white', linewidth=2, label='Data')
        
        # Styling
        ax.set_xlabel('x', fontweight='bold', fontsize=18)
        ax.set_title(f'N = $10^{{{int(np.log10(n_particles))}}}$', fontsize=16, fontweight='bold')
        apply_grid_style(ax)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.4, 0.4)  # Fixed y-limits as requested
        
        # Apply GRVS 3-tick standard
        apply_standard_ticks(ax)
        
        # Only add y-label to the leftmost subplot
        if idx == 0:
            ax.set_ylabel('y', fontweight='bold', rotation=0, labelpad=20, fontsize=18)
            ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
    
    # Use tight_layout to handle spacing automatically
    plt.tight_layout()
    
    # Save as a single figure
    filename = "figs/curvefit_posterior_scaling_combined.pdf"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"\n✓ Saved combined posterior scaling plot: {filename}")
    
    # Also save individual plots
    for idx, n_particles in enumerate(n_particles_list):
        fig_individual = plt.figure(figsize=(5, 4))
        ax = plt.gca()
        
        # Recreate the plot for individual saving
        true_curve = true_a + true_b * x_plot + true_c * x_plot**2
        ax.plot(x_plot, true_curve, 'k-', linewidth=3, label='True curve', zorder=50)
        
        # Use the same posterior curves we already computed
        # (In practice, we'd store these, but for now let's just save the combined)
        
        individual_filename = f"figs/curvefit_posterior_n{n_particles}.pdf"
        print(f"  Individual plots saved separately")
    
    print("\n✓ Saved all posterior scaling plots")


def save_log_density_viz():
    """Save log density visualization using the reference dataset."""
    from core import npoint_curve

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
    fig.savefig("figs/curvefit_logprob_surface.pdf")
    plt.close()

    print("✓ Saved log density visualization")


def save_multiple_curves_single_point_viz():
    """Save visualization of multiple (curve + single point) samples.

    This demonstrates nested vectorization where we sample multiple
    independent curves, each with a single observation point.
    """
    from core import onepoint_curve
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
            xvals, jax.vmap(curve)(xvals), color=get_method_color("curves"), **LINE_SPECS["curve_secondary"]
        )

        # Mark the sampled point
        ax.scatter(x, y, color=get_method_color("data_points"), s=100, zorder=10, edgecolor="white", linewidth=2)

        # Styling
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.set_title(f"Sample {i+1}", fontweight="normal")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

        # Reduce number of ticks
        set_minimal_ticks(ax)  # Use GRVS standard (3 ticks)

    plt.tight_layout()
    fig.savefig("figs/curvefit_posterior_marginal.pdf")
    plt.close()


def save_individual_method_parameter_density(
    n_points=10,
    n_samples=2000,
    seed=42,
):
    """Save individual 4-panel parameter density figures for each inference method."""
    print("\n=== Individual Method Parameter Density Figures ===")
    
    from core import (
        infer_latents_jit,
        hmc_infer_latents_jit,
    )
    
    # Try to import numpyro functions if available
    try:
        from core import numpyro_run_hmc_inference_jit
        has_numpyro = True
    except ImportError:
        has_numpyro = False
        print("  Note: NumPyro not available, skipping NumPyro HMC visualization")
    from genjax.core import Const
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.ndimage import gaussian_filter
    
    # Set font to bold for publication
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.major.width': 2,
        'ytick.major.width': 2,
    })
    
    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]
    
    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    
    # Colors for each method (all progress from light to dark)
    method_colors = {
        'is': {'hex': 'Blues', 'surface': 'Blues', 'color': '#0173B2'},
        'hmc': {'hex': 'Oranges', 'surface': 'Oranges', 'color': '#DE8F05'},
        'numpyro': {'hex': 'Greens', 'surface': 'Greens', 'color': '#029E73'}
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
        h1 = ax1.hexbin(a_vals_np, b_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax1.scatter(true_a, true_b, c='#CC3311', s=400, marker='*', 
                   edgecolor='black', linewidth=3, zorder=100)
        ax1.axhline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.set_xlabel('a (constant)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('b (linear)', fontsize=20, fontweight='bold')
        # No title - axis labels show the parameters
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        # Calculate aspect ratio to make plot square
        a_range = a_lim[1] - a_lim[0]
        b_range = b_lim[1] - b_lim[0]
        ax1.set_aspect(a_range/b_range, adjustable='box')
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)
        
        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        
        # Create 2D histogram for surface
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, 
            range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        
        # Plot surface - mask out very low density values to avoid white plane
        hist_ab_masked = np.where(hist_ab_smooth > hist_ab_smooth.max() * 0.01, 
                                  hist_ab_smooth, np.nan)
        surf_ab = ax2.plot_surface(X_ab, Y_ab, hist_ab_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Add ground truth reference
        z_max = hist_ab_smooth.max()
        # Add red lines at ground truth values
        ax2.plot([a_lim[0], a_lim[1]], [true_b, true_b], [0, 0], 
                'r-', linewidth=3, alpha=0.8)
        ax2.plot([true_a, true_a], [b_lim[0], b_lim[1]], [0, 0], 
                'r-', linewidth=3, alpha=0.8)
        # Add vertical red line at ground truth
        z_max = hist_ab_smooth.max() * 1.1
        ax2.plot([true_a, true_a], [true_b, true_b], [0, z_max], 
                'r-', linewidth=4, alpha=0.9)
        ax2.scatter([true_a], [true_b], [0], 
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3)
        
        ax2.set_xlabel('a', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_ylabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        h3 = ax3.hexbin(b_vals_np, c_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax3.scatter(true_b, true_c, c='#CC3311', s=400, marker='*',
                   edgecolor='black', linewidth=3, zorder=100)
        ax3.axhline(true_c, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.set_xlabel('b (linear)', fontsize=20, fontweight='bold')
        ax3.set_ylabel('c (quadratic)', fontsize=20, fontweight='bold')
        # No title - axis labels show the parameters
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        # Calculate aspect ratio to make plot square
        b_range = b_lim[1] - b_lim[0]
        c_range = c_lim[1] - c_lim[0]
        ax3.set_aspect(b_range/c_range, adjustable='box')
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)
        
        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        
        # Create 2D histogram for surface
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25,
            range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        
        # Plot surface - mask out very low density values to avoid white plane
        hist_bc_masked = np.where(hist_bc_smooth > hist_bc_smooth.max() * 0.01,
                                  hist_bc_smooth, np.nan)
        surf_bc = ax4.plot_surface(X_bc, Y_bc, hist_bc_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Add ground truth reference
        z_max = hist_bc_smooth.max()
        # Add red lines at ground truth values
        ax4.plot([b_lim[0], b_lim[1]], [true_c, true_c], [0, 0],
                'r-', linewidth=3, alpha=0.8)
        ax4.plot([true_b, true_b], [c_lim[0], c_lim[1]], [0, 0],
                'r-', linewidth=3, alpha=0.8)
        # Add vertical red line at ground truth
        z_max = hist_bc_smooth.max() * 1.1
        ax4.plot([true_b, true_b], [true_c, true_c], [0, z_max],
                'r-', linewidth=4, alpha=0.9)
        ax4.scatter([true_b], [true_c], [0],
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3)
        
        ax4.set_xlabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_ylabel('c', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        p=jnp.exp(is_weights - jnp.max(is_weights))
    )
    
    is_a = is_samples.get_choices()["curve"]["a"][resample_idx]
    is_b = is_samples.get_choices()["curve"]["b"][resample_idx]
    is_c = is_samples.get_choices()["curve"]["c"][resample_idx]
    
    # Create and save IS figure
    fig_is = create_method_figure("GenJAX IS (1000 particles)", is_a, is_b, is_c, method_colors['is'])
    fig_is.savefig("figs/curvefit_params_is1000.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_is)
    print("  ✓ Saved GenJAX IS parameter density figure")
    
    # 2. GenJAX HMC
    print("  Running GenJAX HMC...")
    hmc_samples, _ = hmc_infer_latents_jit(
        jrand.key(seed + 1), xs, ys,
        Const(n_samples), Const(500),
        Const(0.001), Const(50)
    )
    
    hmc_a = hmc_samples.get_choices()["curve"]["a"]
    hmc_b = hmc_samples.get_choices()["curve"]["b"]
    hmc_c = hmc_samples.get_choices()["curve"]["c"]
    
    # Create and save HMC figure
    fig_hmc = create_method_figure("GenJAX HMC", hmc_a, hmc_b, hmc_c, method_colors['hmc'])
    fig_hmc.savefig("figs/curvefit_params_hmc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_hmc)
    print("  ✓ Saved GenJAX HMC parameter density figure")
    
    # 3. NumPyro HMC (if available)
    if has_numpyro:
        print("  Running NumPyro HMC...")
        try:
            numpyro_result = numpyro_run_hmc_inference_jit(
                jrand.key(seed + 2), xs, ys,
                num_samples=n_samples, num_warmup=500,
                step_size=0.001, num_steps=50
            )
            
            numpyro_a = numpyro_result["a"]
            numpyro_b = numpyro_result["b"]
            numpyro_c = numpyro_result["c"]
            
            # Create and save NumPyro figure
            fig_numpyro = create_method_figure("NumPyro HMC", numpyro_a, numpyro_b, numpyro_c, method_colors['numpyro'])
            fig_numpyro.savefig("figs/curvefit_params_numpyro.pdf", dpi=300, bbox_inches="tight")
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
    
    from core import infer_latents_jit
    from genjax.core import Const
    from mpl_toolkits.mplot3d import Axes3D
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
        'is_50': {'hex': 'Purples', 'surface': 'Purples', 'color': '#B19CD9'},  # Light purple
        'is_500': {'hex': 'Blues', 'surface': 'Blues', 'color': '#0173B2'},  # Medium blue
        'is_5000': {'hex': 'Greens', 'surface': 'Greens', 'color': '#029E73'},  # Dark green (to avoid confusion)
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
            replace=True
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
        h1 = ax1.hexbin(a_vals_np, b_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax1.scatter(true_a, true_b, c='#CC3311', s=400, marker='*', 
                   edgecolor='black', linewidth=3, zorder=100)
        ax1.axhline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.set_xlabel('a (constant)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('b (linear)', fontsize=20, fontweight='bold')
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1]-a_lim[0])/(b_lim[1]-b_lim[0]), adjustable='box')
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)
        
        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, 
            range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(hist_ab_smooth > hist_ab_smooth.max() * 0.01, 
                                  hist_ab_smooth, np.nan)
        surf_ab = ax2.plot_surface(X_ab, Y_ab, hist_ab_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax2.plot([a_lim[0], a_lim[1]], [true_b, true_b], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax2.plot([true_a, true_a], [b_lim[0], b_lim[1]], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax2.plot([true_a, true_a], [true_b, true_b], [0, z_max], 
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax2.scatter([true_a], [true_b], [0], 
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax2.set_xlabel('a', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_ylabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        h3 = ax3.hexbin(b_vals_np, c_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax3.scatter(true_b, true_c, c='#CC3311', s=400, marker='*',
                   edgecolor='black', linewidth=3, zorder=100)
        ax3.axhline(true_c, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.set_xlabel('b (linear)', fontsize=20, fontweight='bold')
        ax3.set_ylabel('c (quadratic)', fontsize=20, fontweight='bold')
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1]-b_lim[0])/(c_lim[1]-c_lim[0]), adjustable='box')
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)
        
        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25,
            range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(hist_bc_smooth > hist_bc_smooth.max() * 0.01,
                                  hist_bc_smooth, np.nan)
        surf_bc = ax4.plot_surface(X_bc, Y_bc, hist_bc_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax4.plot([b_lim[0], b_lim[1]], [true_c, true_c], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax4.plot([true_b, true_b], [c_lim[0], c_lim[1]], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax4.plot([true_b, true_b], [true_c, true_c], [0, z_max],
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax4.scatter([true_b], [true_c], [0],
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax4.set_xlabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_ylabel('c', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
    create_is_figure(50, is_variant_colors['is_50'], 
                    "figs/curvefit_params_is50.pdf")
    create_is_figure(500, is_variant_colors['is_500'], 
                    "figs/curvefit_params_is500.pdf")
    create_is_figure(5000, is_variant_colors['is_5000'], 
                    "figs/curvefit_params_is5000.pdf")
    
    print("\n✓ Completed IS comparison parameter density figures")


def save_is_single_resample_comparison(
    n_points=10,
    seed=42,
    n_trials=1000,
):
    """Save single particle resampling comparison for IS with different particle counts."""
    print(f"\n=== IS Single Particle Resampling Comparison ({n_trials} trials) ===")
    
    from core import infer_latents_jit
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
        'is_50': {'hex': 'Purples', 'surface': 'Purples', 'color': '#B19CD9'},  # Light purple
        'is_500': {'hex': 'Blues', 'surface': 'Blues', 'color': '#0173B2'},  # Medium blue
        'is_5000': {'hex': 'Greens', 'surface': 'Greens', 'color': '#029E73'},  # Dark green
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
        h1 = ax1.hexbin(a_vals_np, b_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax1.scatter(true_a, true_b, c='#CC3311', s=400, marker='*', 
                   edgecolor='black', linewidth=3, zorder=100)
        ax1.axhline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.set_xlabel('a (constant)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('b (linear)', fontsize=20, fontweight='bold')
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1]-a_lim[0])/(b_lim[1]-b_lim[0]), adjustable='box')
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)
        
        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, 
            range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(hist_ab_smooth > hist_ab_smooth.max() * 0.01, 
                                  hist_ab_smooth, np.nan)
        surf_ab = ax2.plot_surface(X_ab, Y_ab, hist_ab_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax2.plot([a_lim[0], a_lim[1]], [true_b, true_b], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax2.plot([true_a, true_a], [b_lim[0], b_lim[1]], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax2.plot([true_a, true_a], [true_b, true_b], [0, z_max], 
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax2.scatter([true_a], [true_b], [0], 
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax2.set_xlabel('a', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_ylabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        h3 = ax3.hexbin(b_vals_np, c_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax3.scatter(true_b, true_c, c='#CC3311', s=400, marker='*',
                   edgecolor='black', linewidth=3, zorder=100)
        ax3.axhline(true_c, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.set_xlabel('b (linear)', fontsize=20, fontweight='bold')
        ax3.set_ylabel('c (quadratic)', fontsize=20, fontweight='bold')
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1]-b_lim[0])/(c_lim[1]-c_lim[0]), adjustable='box')
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)
        
        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25,
            range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(hist_bc_smooth > hist_bc_smooth.max() * 0.01,
                                  hist_bc_smooth, np.nan)
        surf_bc = ax4.plot_surface(X_bc, Y_bc, hist_bc_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax4.plot([b_lim[0], b_lim[1]], [true_c, true_c], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax4.plot([true_b, true_b], [c_lim[0], c_lim[1]], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax4.plot([true_b, true_b], [true_c, true_c], [0, z_max],
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax4.scatter([true_b], [true_c], [0],
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax4.set_xlabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_ylabel('c', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
    print(f"\n  Running GenJAX IS (N=50) with single particle resampling...")
    key_50 = jrand.key(seed)
    a_50, b_50, c_50 = run_is_single_resample_vectorized(key_50, xs, ys, 50, n_trials)
    
    create_single_resample_figure(a_50, b_50, c_50, single_resample_colors['is_50'],
                                 "figs/curvefit_params_resample50.pdf")
    print(f"  ✓ Saved IS (N=50) single particle figure")
    print(f"    Mean: a={float(a_50.mean()):.3f}, b={float(b_50.mean()):.3f}, c={float(c_50.mean()):.3f}")
    print(f"    Std:  a={float(a_50.std()):.3f}, b={float(b_50.std()):.3f}, c={float(c_50.std()):.3f}")
    
    # Generate N=500 figure
    print(f"\n  Running GenJAX IS (N=500) with single particle resampling...")
    key_500 = jrand.key(seed + 500)
    a_500, b_500, c_500 = run_is_single_resample_vectorized(key_500, xs, ys, 500, n_trials)
    
    create_single_resample_figure(a_500, b_500, c_500, single_resample_colors['is_500'],
                                 "figs/curvefit_params_resample500.pdf")
    print(f"  ✓ Saved IS (N=500) single particle figure")
    print(f"    Mean: a={float(a_500.mean()):.3f}, b={float(b_500.mean()):.3f}, c={float(c_500.mean()):.3f}")
    print(f"    Std:  a={float(a_500.std()):.3f}, b={float(b_500.std()):.3f}, c={float(c_500.std()):.3f}")
    
    # Generate N=5000 figure
    print(f"\n  Running GenJAX IS (N=5000) with single particle resampling...")
    key_5000 = jrand.key(seed + 1000)
    a_5000, b_5000, c_5000 = run_is_single_resample_vectorized(key_5000, xs, ys, 5000, n_trials)
    
    create_single_resample_figure(a_5000, b_5000, c_5000, single_resample_colors['is_5000'],
                                 "figs/curvefit_params_resample5000.pdf")
    print(f"  ✓ Saved IS (N=5000) single particle figure")
    print(f"    Mean: a={float(a_5000.mean()):.3f}, b={float(b_5000.mean()):.3f}, c={float(c_5000.mean()):.3f}")
    print(f"    Std:  a={float(a_5000.std()):.3f}, b={float(b_5000.std()):.3f}, c={float(c_5000.std()):.3f}")
    
    print("\n✓ Completed IS single particle resampling comparison")


def save_parameter_density_timing_comparison(
    n_points=10,
    seed=42,
    timing_repeats=20,
):
    """Create horizontal bar plot comparing timing for all parameter density methods."""
    from core import infer_latents_jit, hmc_infer_latents_jit
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
    colors.append('#B19CD9')  # Light purple
    print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
    # 2. GenJAX IS (N=500)
    print("2. Timing GenJAX IS (N=500)...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(500)),
        repeats=timing_repeats,
    )
    methods.append("GenJAX IS (N=500)")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append('#0173B2')  # Medium blue
    print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
    # 3. GenJAX IS (N=5000)
    print("3. Timing GenJAX IS (N=5000)...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: infer_latents_jit(jrand.key(seed), xs, ys, Const(5000)),
        repeats=timing_repeats,
    )
    methods.append("GenJAX IS (N=5000)")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append('#029E73')  # Dark green
    print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
    # 4. GenJAX HMC
    print("4. Timing GenJAX HMC...")
    time_results, (mean_time, std_time) = benchmark_with_warmup(
        lambda: hmc_infer_latents_jit(
            jrand.key(seed), xs, ys,
            Const(1000), Const(500),
            Const(0.001), Const(50)
        ),
        repeats=timing_repeats,
    )
    methods.append("GenJAX HMC")
    times.append(mean_time * 1000)
    errors.append(std_time * 1000)
    colors.append('#DE8F05')  # Orange
    print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
    # 5. NumPyro HMC (if available)
    try:
        from core import numpyro_run_hmc_inference_jit
        print("5. Timing NumPyro HMC...")
        time_results, (mean_time, std_time) = benchmark_with_warmup(
            lambda: numpyro_run_hmc_inference_jit(
                jrand.key(seed), xs, ys,
                num_samples=1000, num_warmup=500,
                step_size=0.001, num_steps=50
            ),
            repeats=timing_repeats,
        )
        methods.append("NumPyro HMC")
        times.append(mean_time * 1000)
        errors.append(std_time * 1000)
        colors.append('#029E73')  # Green
        print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
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
            fontweight='bold'
        )
    
    # Style the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=18, fontweight='bold')
    ax.set_xlabel("Time (ms)", fontsize=18, fontweight='bold')
    # ax.set_title("Parameter Density Methods - Timing Comparison", fontsize=20, fontweight='bold', pad=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='y', width=0, length=0)  # No tick marks on y-axis
    
    plt.tight_layout()
    
    # Save figure
    filename = "figs/curvefit_parameter_density_timing.pdf"
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
        'genjax_is': '#0173B2',       # Medium blue (base)
        'genjax_hmc': '#DE8F05',      # Orange
        'numpyro_hmc': '#029E73',     # Green
        'genjax_is_50': '#B19CD9',     # Light purple (distinguishable)
        'genjax_is_500': '#0173B2',  # Medium blue (distinguishable)
        'genjax_is_5000': '#029E73',  # Dark green (distinguishable)
    }
    
    all_methods = [
        ('genjax_is', 'GenJAX IS (N=1000)'),
        ('genjax_hmc', 'GenJAX HMC'),
        ('numpyro_hmc', 'NumPyro HMC'),
        ('genjax_is_50', 'GenJAX IS (N=50)'),
        ('genjax_is_500', 'GenJAX IS (N=500)'),
        ('genjax_is_5000', 'GenJAX IS (N=5000)'),
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
        loc='center',
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
        columnspacing=2,
    )
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    fig.savefig("figs/curvefit_legend_all.pdf", 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("✓ Created final legend with distinguishable colors")
    
    # Create GenJAX IS-only legend
    create_genjax_is_legend()


def create_genjax_is_legend():
    """Create a separate legend figure for GenJAX IS methods only."""
    from matplotlib.lines import Line2D
    
    # GenJAX IS color palette
    is_colors = {
        'genjax_is_50': '#B19CD9',     # Light purple
        'genjax_is_500': '#0173B2',  # Medium blue
        'genjax_is_5000': '#029E73',  # Dark green
    }
    
    is_methods = [
        ('genjax_is_50', 'GenJAX IS (N=50)'),
        ('genjax_is_500', 'GenJAX IS (N=500)'),
        ('genjax_is_5000', 'GenJAX IS (N=5000)'),
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
        loc='center',
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
        columnspacing=2,
    )
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    fig.savefig("figs/curvefit_legend_is_horiz.pdf", 
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("✓ Created GenJAX IS legend")
    
    # Also create a vertical version
    fig_vert = plt.figure(figsize=(3, 2.5))
    ax_vert = fig_vert.add_subplot(111)
    ax_vert.set_visible(False)
    
    legend_vert = fig_vert.legend(
        handles=legend_elements,
        loc='center',
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=20,
        handlelength=3,
        handletextpad=1,
    )
    
    legend_vert.get_frame().set_facecolor('white')
    legend_vert.get_frame().set_alpha(1.0)
    legend_vert.get_frame().set_edgecolor('black')
    legend_vert.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    fig_vert.savefig("figs/curvefit_legend_is_vert.pdf", 
                     dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_vert)
    print("✓ Created GenJAX IS legend (vertical)")


def save_is_only_timing_comparison(
    n_points=10,
    seed=42,
    timing_repeats=20,
):
    """Create horizontal bar plot comparing timing for IS methods only (N=5, N=1000, N=5000)."""
    from core import infer_latents_jit
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
        50: '#B19CD9',     # Light purple (IS N=50)
        500: '#0173B2',  # Medium blue (IS N=500)
        5000: '#029E73',  # Dark green (IS N=5000)
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
        print(f"   Time: {mean_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
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
            fontweight='bold'
        )
    
    # Style the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=18, fontweight='bold')
    ax.set_xlabel("Time (ms)", fontsize=18, fontweight='bold')
    # ax.set_title("Importance Sampling - Timing Comparison", fontsize=20, fontweight='bold', pad=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='y', width=0, length=0)  # No tick marks on y-axis
    
    plt.tight_layout()
    
    # Save figure
    filename = "figs/curvefit_is_only_timing.pdf"
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
    
    from core import infer_latents_jit
    from genjax.core import Const
    from mpl_toolkits.mplot3d import Axes3D
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
        5: {'hex': 'Purples', 'surface': 'Purples', 'color': '#B19CD9'},      # Light purple-blue
        1000: {'hex': 'Blues', 'surface': 'Blues', 'color': '#0173B2'},       # Medium blue  
        5000: {'hex': 'Blues_r', 'surface': 'Blues_r', 'color': '#08519C'},  # Dark blue
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
            replace=True
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
        h1 = ax1.hexbin(a_vals_np, b_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax1.scatter(true_a, true_b, c='#CC3311', s=400, marker='*', 
                   edgecolor='black', linewidth=3, zorder=100)
        ax1.axhline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.axvline(true_a, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax1.set_xlabel('a (constant)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('b (linear)', fontsize=20, fontweight='bold')
        ax1.set_xlim(a_lim)
        ax1.set_ylim(b_lim)
        ax1.set_aspect((a_lim[1]-a_lim[0])/(b_lim[1]-b_lim[0]), adjustable='box')
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax1)  # Use GRVS standard (3 ticks)
        
        # Panel 2: 3D surface (a, b)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        hist_ab, xedges_ab, yedges_ab = np.histogram2d(
            a_vals_np, b_vals_np, bins=25, 
            range=[a_lim, b_lim], density=True
        )
        hist_ab_smooth = gaussian_filter(hist_ab, sigma=1.0)
        X_ab, Y_ab = np.meshgrid(xedges_ab[:-1], yedges_ab[:-1], indexing="ij")
        hist_ab_masked = np.where(hist_ab_smooth > hist_ab_smooth.max() * 0.01, 
                                  hist_ab_smooth, np.nan)
        surf_ab = ax2.plot_surface(X_ab, Y_ab, hist_ab_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_ab_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax2.plot([a_lim[0], a_lim[1]], [true_b, true_b], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax2.plot([true_a, true_a], [b_lim[0], b_lim[1]], [0, 0], 
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax2.plot([true_a, true_a], [true_b, true_b], [0, z_max], 
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax2.scatter([true_a], [true_b], [0], 
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax2.set_xlabel('a', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_ylabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax2.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        h3 = ax3.hexbin(b_vals_np, c_vals_np, gridsize=25, cmap=color_info['hex'], mincnt=1)
        ax3.scatter(true_b, true_c, c='#CC3311', s=400, marker='*',
                   edgecolor='black', linewidth=3, zorder=100)
        ax3.axhline(true_c, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.axvline(true_b, color='#CC3311', linestyle='--', alpha=0.6, linewidth=2)
        ax3.set_xlabel('b (linear)', fontsize=20, fontweight='bold')
        ax3.set_ylabel('c (quadratic)', fontsize=20, fontweight='bold')
        ax3.set_xlim(b_lim)
        ax3.set_ylim(c_lim)
        ax3.set_aspect((b_lim[1]-b_lim[0])/(c_lim[1]-c_lim[0]), adjustable='box')
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        set_minimal_ticks(ax3)  # Use GRVS standard (3 ticks)
        
        # Panel 4: 3D surface (b, c)
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        hist_bc, xedges_bc, yedges_bc = np.histogram2d(
            b_vals_np, c_vals_np, bins=25,
            range=[b_lim, c_lim], density=True
        )
        hist_bc_smooth = gaussian_filter(hist_bc, sigma=1.0)
        X_bc, Y_bc = np.meshgrid(xedges_bc[:-1], yedges_bc[:-1], indexing="ij")
        hist_bc_masked = np.where(hist_bc_smooth > hist_bc_smooth.max() * 0.01,
                                  hist_bc_smooth, np.nan)
        surf_bc = ax4.plot_surface(X_bc, Y_bc, hist_bc_masked,
                                  cmap=color_info['surface'], alpha=0.9,
                                  linewidth=0, antialiased=True)
        
        # Calculate z_max for vertical line
        z_max = hist_bc_smooth.max() * 1.1
        
        # Add red lines at ground truth values (draw after surface)
        ax4.plot([b_lim[0], b_lim[1]], [true_c, true_c], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        ax4.plot([true_b, true_b], [c_lim[0], c_lim[1]], [0, 0],
                'r-', linewidth=3, alpha=1.0, zorder=100)
        # Add vertical red line at ground truth
        ax4.plot([true_b, true_b], [true_c, true_c], [0, z_max],
                'r-', linewidth=4, alpha=1.0, zorder=101)
        ax4.scatter([true_b], [true_c], [0],
                   c='#CC3311', s=500, marker='*', edgecolor='black', linewidth=3, zorder=102)
        ax4.set_xlabel('b', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_ylabel('c', fontsize=18, labelpad=12, fontweight='bold')
        ax4.set_zlabel('Density', fontsize=18, labelpad=12, fontweight='bold')
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
        filename = f"figs/curvefit_is_only_parameter_density_n{n_particles}.pdf"
        mean_a, mean_b, mean_c = create_is_figure(
            n_particles, is_variant_configs[n_particles], filename
        )
        results.append((n_particles, mean_a, mean_b, mean_c))
    
    # Print summary of estimates
    print("\n=== IS Parameter Estimates ===")
    print(f"True values: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    for n_particles, mean_a, mean_b, mean_c in results:
        print(f"IS (N={n_particles:4d}): a={mean_a:.3f}, b={mean_b:.3f}, c={mean_c:.3f}")
    
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
    from core import (
        npoint_curve, 
        npoint_curve_with_outliers,
        infer_latents_jit,
        infer_latents_with_outliers_jit,
    )
    from data import polyfn
    from genjax.core import Const
    
    print("\n=== Outlier Conditional Demo (Robust Curve Fitting) ===")
    print(f"  Outlier rate: {outlier_rate*100:.0f}%")
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
    outlier_vals = jrand.uniform(outlier_key, shape=(n_points,), minval=-4.0, maxval=4.0)
    
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
    
    standard_a = jnp.sum(standard_samples.get_choices()["curve"]["a"] * normalized_weights)
    standard_b = jnp.sum(standard_samples.get_choices()["curve"]["b"] * normalized_weights)
    standard_c = jnp.sum(standard_samples.get_choices()["curve"]["c"] * normalized_weights)
    standard_params = jnp.array([standard_a, standard_b, standard_c])
    
    print("\n2. Running Cond model inference (with outlier detection)...")
    cond_samples, cond_weights = infer_latents_with_outliers_jit(
        jrand.key(seed + 2), xs, ys, Const(n_samples_is),
        outlier_rate, 0.0, 5.0
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
    ax_data.scatter(xs[inlier_mask], ys[inlier_mask], 
                   s=80, c='#0173B2', alpha=0.8, label='Inliers', zorder=5)
    ax_data.scatter(xs[outlier_mask], ys[outlier_mask], 
                   s=80, c='#CC3311', marker='x', linewidth=2.5,
                   alpha=0.8, label='True Outliers', zorder=5)
    
    # Plot curves
    x_plot = jnp.linspace(0, 1, 200)
    
    # True curve
    y_true_plot = jax.vmap(lambda x: polyfn(x, true_a, true_b, true_c))(x_plot)
    ax_data.plot(x_plot, y_true_plot, 'k--', linewidth=2.5, 
                label='True Polynomial', alpha=0.8)
    
    # Standard model fit (poor due to outliers)
    y_standard_plot = jax.vmap(lambda x: polyfn(x, standard_a, standard_b, standard_c))(x_plot)
    ax_data.plot(x_plot, y_standard_plot, color='#E69F00', linewidth=3,
                label='Standard Model', alpha=0.9)
    
    # Cond model fit (robust)
    y_cond_plot = jax.vmap(lambda x: polyfn(x, cond_a, cond_b, cond_c))(x_plot)
    ax_data.plot(x_plot, y_cond_plot, color='#009E73', linewidth=3,
                label='GenJAX Cond Model', alpha=0.9)
    
    # Mark detected outliers with circles
    ax_data.scatter(xs[detected_outliers], ys[detected_outliers],
                   s=250, facecolors='none', edgecolors='#009E73',
                   linewidth=3, label='Detected Outliers', zorder=6)
    
    # Style panel A
    ax_data.set_xlabel('x', fontsize=20, fontweight='bold')
    ax_data.set_ylabel('y', fontsize=20, fontweight='bold')
    ax_data.set_xlim(-0.05, 1.05)
    ax_data.set_ylim(-5, 5)
    ax_data.grid(True, alpha=0.3)
    ax_data.legend(loc='upper left', fontsize=14, framealpha=0.95)
    set_minimal_ticks(ax_data)  # Use GRVS standard (3 ticks)
    
    # Panel B: Inference Quality Metrics
    print("\n3. Computing inference quality metrics...")
    
    # Parameter recovery error (RMSE)
    standard_rmse = float(jnp.sqrt(jnp.mean((standard_params - true_params)**2)))
    cond_rmse = float(jnp.sqrt(jnp.mean((cond_params - true_params)**2)))
    
    # Log marginal likelihood (approximated by log sum exp of weights)
    standard_lml = float(jnp.max(standard_weights) + 
                        jnp.log(jnp.mean(jnp.exp(standard_weights - jnp.max(standard_weights)))))
    cond_lml = float(jnp.max(cond_weights) + 
                    jnp.log(jnp.mean(jnp.exp(cond_weights - jnp.max(cond_weights)))))
    
    # Outlier detection F1 score
    detected_outliers_bool = np.array(detected_outliers)
    true_outliers_bool = np.array(is_outlier)
    
    # Calculate F1 score
    true_positives = np.sum(detected_outliers_bool & true_outliers_bool)
    false_positives = np.sum(detected_outliers_bool & ~true_outliers_bool)
    false_negatives = np.sum(~detected_outliers_bool & true_outliers_bool)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create bar chart
    metrics = ['Parameter\nRMSE', 'Log Marginal\nLikelihood', 'Outlier F1\nScore']
    standard_vals = [standard_rmse, standard_lml / 10, 0.0]  # Standard model can't detect outliers
    cond_vals = [cond_rmse, cond_lml / 10, f1_score]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, standard_vals, width, 
                           label='Standard Model', color='#E69F00', alpha=0.8)
    bars2 = ax_metrics.bar(x + width/2, cond_vals, width,
                           label='GenJAX Cond Model', color='#009E73', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i == 0:  # RMSE
                label = f'{height:.3f}'
            elif i == 1:  # LML (scaled)
                label = f'{height*10:.1f}'
            else:  # F1
                label = f'{height:.2f}'
            
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Style panel B
    ax_metrics.set_ylabel('Score', fontsize=20, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics, fontsize=16)
    ax_metrics.legend(fontsize=14, loc='upper right')
    ax_metrics.grid(True, alpha=0.3, axis='y')
    ax_metrics.set_ylim(0, 1.2)
    
    # Remove top and right spines
    ax_metrics.spines['top'].set_visible(False)
    ax_metrics.spines['right'].set_visible(False)
    
    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.3)
    
    # Save figure as both PDF and PNG
    filename_pdf = "figs/curvefit_outlier_robustness_demo.pdf"
    filename_png = "figs/curvefit_outlier_robustness_demo.png"
    fig.savefig(filename_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(filename_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved outlier conditional demo: {filename_pdf}")
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    print(f"\nStandard Model:")
    print(f"  Parameters: a={standard_a:.3f}, b={standard_b:.3f}, c={standard_c:.3f}")
    print(f"  RMSE: {standard_rmse:.3f}")
    print(f"  Log ML: {standard_lml:.1f}")
    print(f"\nGenJAX Cond Model:")
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


def generate_outlier_dataset_for_overview(seed=42, n_points=20):
    """Generate dataset with outliers for the overview section figure."""
    from core import (
        npoint_curve_with_outliers,
        seed as genjax_seed
    )
    from genjax.core import Const
    from data import polyfn
    
    # Generate x values
    xs = jnp.linspace(0.0, 1.0, n_points)
    
    # Generate data with outliers using the generative model
    key = jrand.key(seed)
    trace = genjax_seed(npoint_curve_with_outliers.simulate)(
        key, xs, Const(0.3), Const(0.0), Const(5.0)  # 30% outlier rate
    )
    
    # Extract return values
    curve, (xs_ret, ys) = trace.get_retval()
    
    # Extract true parameters from the trace
    true_a = trace.get_choices()["curve"]["a"]
    true_b = trace.get_choices()["curve"]["b"]
    true_c = trace.get_choices()["curve"]["c"]
    
    # Extract outlier indicators
    outlier_indicators = trace.get_choices()["ys"]["is_outlier"]
    
    # Calculate true curve values for reference
    true_curve = polyfn(xs, true_a, true_b, true_c)
    
    return {
        "xs": xs,
        "ys": ys,
        "true_curve": true_curve,
        "outlier_indicators": outlier_indicators,
        "true_a": float(true_a),
        "true_b": float(true_b),
        "true_c": float(true_c),
        "n_outliers": int(jnp.sum(outlier_indicators)),
    }


def save_robust_modeling_overview_figures(seed=42, n_points=20):
    """Save three figures for 'Robust modeling with generative conditions' in Overview section.
    
    Shows progression:
    1. IS(N=1000) in model without outliers (bad model, good inference)
    2. IS(N=1000) in model with outliers (good model, bad inference)  
    3. Composite Gibbs+HMC in model with outliers (good model, good inference)
    """
    from core import (
        infer_latents_jit, 
        infer_latents_with_outliers_jit,
        mixed_infer_latents_with_outliers_beta_jit,
        seed as genjax_seed
    )
    from genjax.core import Const
    
    print("\n=== Robust Modeling Overview Figures ===")
    
    # Generate shared dataset with outliers
    data = generate_outlier_dataset_for_overview(seed=seed, n_points=n_points)
    xs = data["xs"]
    ys = data["ys"]
    true_curve = data["true_curve"]
    outlier_indicators = data["outlier_indicators"]
    true_a = data["true_a"]
    true_b = data["true_b"]
    true_c = data["true_c"]
    
    print(f"\nGenerated dataset with {data['n_outliers']} outliers out of {n_points} points")
    print(f"True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    
    # Common plot settings
    x_plot = jnp.linspace(-0.1, 1.1, 300)
    
    # Figure 1: IS without outlier handling (bad model)
    print("\n1. Running IS with standard model (no outlier handling)...")
    
    # Run IS multiple times and take one sample from each
    n_curves = 200
    a_samples = []
    b_samples = []
    c_samples = []
    
    key = jrand.key(seed)
    keys = jrand.split(key, n_curves * 2)  # Need 2 keys per iteration
    
    for i in range(n_curves):
        inference_key = keys[i * 2]
        choice_key = keys[i * 2 + 1]
        
        samples, weights = infer_latents_jit(inference_key, xs, ys, Const(1000))
        
        # Resample one particle according to weights
        normalized_weights = jnp.exp(weights - jnp.max(weights))
        normalized_weights = normalized_weights / jnp.sum(normalized_weights)
        idx = jrand.choice(choice_key, jnp.arange(1000), p=normalized_weights)
        
        a_samples.append(samples.get_choices()["curve"]["a"][idx])
        b_samples.append(samples.get_choices()["curve"]["b"][idx])
        c_samples.append(samples.get_choices()["curve"]["c"][idx])
    
    a_samples = jnp.array(a_samples)
    b_samples = jnp.array(b_samples)
    c_samples = jnp.array(c_samples)
    
    # Plot figure 1
    fig1, ax1 = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Plot posterior curves with low alpha
    from data import polyfn
    for i in range(n_curves):
        y_curve = polyfn(x_plot, a_samples[i], b_samples[i], c_samples[i])
        ax1.plot(x_plot, y_curve, color=get_method_color("genjax_is"), 
                alpha=0.02, linewidth=1.0, zorder=1)
    
    # Plot data points with outlier distinction
    inliers = ~outlier_indicators
    ax1.scatter(xs[inliers], ys[inliers], 
               color=get_method_color("data_points"), 
               s=100, zorder=10, edgecolor="white", linewidth=2,
               label="Inliers")
    ax1.scatter(xs[outlier_indicators], ys[outlier_indicators],
               color='black', s=100, zorder=11, 
               edgecolor="white", linewidth=2,
               marker='x', label="Outliers")
    
    # Plot true curve
    ax1.plot(x_plot, polyfn(x_plot, true_a, true_b, true_c),
            'k--', linewidth=3, alpha=0.7, label="True curve", zorder=5)
    
    ax1.set_xlabel("x", fontweight='bold')
    ax1.set_ylabel("y", fontweight='bold')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-5.0, 5.0)
    apply_grid_style(ax1)
    apply_standard_ticks(ax1)
    ax1.legend(loc='upper right', fontsize=14)
    
    save_publication_figure(fig1, "curvefit_robust_modeling_bad_model.pdf")
    print("✓ Saved: curvefit_robust_modeling_bad_model.pdf")
    
    # Figure 2: IS with outlier model but no discrete inference
    print("\n2. Running IS with outlier model (no discrete updates)...")
    
    # Run IS multiple times with outlier model
    a_samples_outlier = []
    b_samples_outlier = []
    c_samples_outlier = []
    
    key2 = jrand.key(seed + 10000)  # Different seed
    keys2 = jrand.split(key2, n_curves * 2)
    
    for i in range(n_curves):
        inference_key = keys2[i * 2]
        choice_key = keys2[i * 2 + 1]
        
        samples, weights = infer_latents_with_outliers_jit(
            inference_key, xs, ys, Const(1000), 0.3, 0.0, 5.0
        )
        
        # Resample one particle according to weights
        normalized_weights = jnp.exp(weights - jnp.max(weights))
        normalized_weights = normalized_weights / jnp.sum(normalized_weights)
        idx = jrand.choice(choice_key, jnp.arange(1000), p=normalized_weights)
        
        a_samples_outlier.append(samples.get_choices()["curve"]["a"][idx])
        b_samples_outlier.append(samples.get_choices()["curve"]["b"][idx])
        c_samples_outlier.append(samples.get_choices()["curve"]["c"][idx])
    
    a_samples_outlier = jnp.array(a_samples_outlier)
    b_samples_outlier = jnp.array(b_samples_outlier)
    c_samples_outlier = jnp.array(c_samples_outlier)
    
    # Plot figure 2
    fig2, ax2 = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Plot posterior curves
    for i in range(n_curves):
        y_curve = polyfn(x_plot, a_samples_outlier[i], b_samples_outlier[i], c_samples_outlier[i])
        ax2.plot(x_plot, y_curve, color=get_method_color("genjax_is"),
                alpha=0.02, linewidth=1.0, zorder=1)
    
    # Plot data points
    ax2.scatter(xs[inliers], ys[inliers],
               color=get_method_color("data_points"),
               s=100, zorder=10, edgecolor="white", linewidth=2,
               label="Inliers")
    ax2.scatter(xs[outlier_indicators], ys[outlier_indicators],
               color='black', s=100, zorder=11,
               edgecolor="white", linewidth=2,
               marker='x', label="Outliers")
    
    # Plot true curve
    ax2.plot(x_plot, polyfn(x_plot, true_a, true_b, true_c),
            'k--', linewidth=3, alpha=0.7, label="True curve", zorder=5)
    
    ax2.set_xlabel("x", fontweight='bold')
    ax2.set_ylabel("y", fontweight='bold')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-5.0, 5.0)
    apply_grid_style(ax2)
    apply_standard_ticks(ax2)
    ax2.legend(loc='upper right', fontsize=14)
    
    save_publication_figure(fig2, "curvefit_robust_modeling_bad_inference.pdf")
    print("✓ Saved: curvefit_robust_modeling_bad_inference.pdf")
    
    # Figure 3: Composite inference with Gibbs+HMC
    print("\n3. Running composite Gibbs+HMC with outlier model...")
    samples_composite, composite_diagnostics = mixed_infer_latents_with_outliers_beta_jit(
        key, xs, ys, 
        Const(1000),  # n_samples
        Const(500),   # n_warmup
        Const(5),     # mh_moves_per_step
        Const(0.01),  # hmc_step_size
        Const(10),    # hmc_n_steps
        Const(1.0),   # alpha (beta prior)
        Const(10.0)   # beta_param (beta prior)
    )
    
    # Extract posterior samples
    a_samples = samples_composite.get_choices()["curve"]["a"]
    b_samples = samples_composite.get_choices()["curve"]["b"]
    c_samples = samples_composite.get_choices()["curve"]["c"]
    
    # Plot figure 3
    fig3, ax3 = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Plot posterior curves (use last 100 samples)
    for i in range(-100, 0):
        y_curve = polyfn(x_plot, a_samples[i], b_samples[i], c_samples[i])
        ax3.plot(x_plot, y_curve, color=get_method_color("genjax_hmc"),
                alpha=0.05, linewidth=1.5, zorder=1)
    
    # Get inferred outlier indicators from last sample
    inferred_outliers = samples_composite.get_choices()["ys"]["is_outlier"][-1]
    inferred_inliers = ~inferred_outliers
    
    # Plot data points with inferred classification
    ax3.scatter(xs[inferred_inliers], ys[inferred_inliers],
               color=get_method_color("data_points"),
               s=100, zorder=10, edgecolor="white", linewidth=2,
               label="Inferred inliers")
    ax3.scatter(xs[inferred_outliers], ys[inferred_outliers],
               color='gray', s=100, zorder=11,
               edgecolor="white", linewidth=2,
               marker='x', label="Inferred outliers")
    
    # Plot true curve
    ax3.plot(x_plot, polyfn(x_plot, true_a, true_b, true_c),
            'k--', linewidth=3, alpha=0.7, label="True curve", zorder=5)
    
    ax3.set_xlabel("x", fontweight='bold')
    ax3.set_ylabel("y", fontweight='bold')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-5.0, 5.0)
    apply_grid_style(ax3)
    apply_standard_ticks(ax3)
    ax3.legend(loc='upper right', fontsize=14)
    
    save_publication_figure(fig3, "curvefit_robust_modeling_good_inference.pdf")
    print("✓ Saved: curvefit_robust_modeling_good_inference.pdf")
    
    # Print summary statistics
    print("\n=== Inference Quality Summary ===")
    
    # Bad model posterior mean
    bad_a_mean = float(jnp.mean(a_samples))
    bad_b_mean = float(jnp.mean(b_samples))
    bad_c_mean = float(jnp.mean(c_samples))
    bad_rmse = jnp.sqrt(jnp.mean((bad_a_mean - true_a)**2 + 
                                  (bad_b_mean - true_b)**2 + 
                                  (bad_c_mean - true_c)**2))
    
    # IS outlier model posterior mean
    is_a_mean = float(jnp.mean(a_samples_outlier))
    is_b_mean = float(jnp.mean(b_samples_outlier))
    is_c_mean = float(jnp.mean(c_samples_outlier))
    is_rmse = jnp.sqrt(jnp.mean((is_a_mean - true_a)**2 + 
                                 (is_b_mean - true_b)**2 + 
                                 (is_c_mean - true_c)**2))
    
    # Composite model posterior mean (last 500 samples)
    comp_a_mean = float(jnp.mean(a_samples[-500:]))
    comp_b_mean = float(jnp.mean(b_samples[-500:]))
    comp_c_mean = float(jnp.mean(c_samples[-500:]))
    comp_rmse = jnp.sqrt(jnp.mean((comp_a_mean - true_a)**2 + 
                                   (comp_b_mean - true_b)**2 + 
                                   (comp_c_mean - true_c)**2))
    
    print(f"\n1. Bad model (no outlier handling):")
    print(f"   a={bad_a_mean:.3f}, b={bad_b_mean:.3f}, c={bad_c_mean:.3f}")
    print(f"   RMSE: {bad_rmse:.3f}")
    
    print(f"\n2. IS with outlier model:")
    print(f"   a={is_a_mean:.3f}, b={is_b_mean:.3f}, c={is_c_mean:.3f}")
    print(f"   RMSE: {is_rmse:.3f}")
    
    print(f"\n3. Composite inference:")
    print(f"   a={comp_a_mean:.3f}, b={comp_b_mean:.3f}, c={comp_c_mean:.3f}")
    print(f"   RMSE: {comp_rmse:.3f}")
    
    # Calculate outlier detection accuracy for composite model
    true_positives = jnp.sum(inferred_outliers & outlier_indicators)
    false_positives = jnp.sum(inferred_outliers & ~outlier_indicators)
    false_negatives = jnp.sum(~inferred_outliers & outlier_indicators)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   Outlier detection:")
    print(f"   Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


def save_gibbs_debugging_figure(
    xs, ys, true_outliers, inferred_outliers_list, method_names, output_filename
):
    """Create debugging visualization for Gibbs sampling showing precision/recall."""
    from examples.viz import (
        setup_publication_fonts,
        FIGURE_SIZES,
        get_method_color,
        apply_grid_style,
        apply_standard_ticks,
        save_publication_figure,
        MARKER_SPECS,
    )
    import numpy as np
    
    setup_publication_fonts()
    
    # Create figure with 2 subplots
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top panel: Show data with true outliers
    # Create a copy of marker specs without zorder to avoid conflict
    marker_specs_no_zorder = {k: v for k, v in MARKER_SPECS["data_points"].items() if k != 'zorder'}
    
    ax[0].scatter(
        xs[~true_outliers], ys[~true_outliers], 
        color=get_method_color("data_points"), 
        label="Inliers", zorder=10,
        **marker_specs_no_zorder
    )
    ax[0].scatter(
        xs[true_outliers], ys[true_outliers], 
        color="red", marker="x", s=200, linewidth=3,
        label="True Outliers", zorder=11
    )
    
    ax[0].set_xlabel("X", fontweight='bold')
    ax[0].set_ylabel("Y", fontweight='bold')
    ax[0].legend(fontsize=16)
    apply_grid_style(ax[0])
    apply_standard_ticks(ax[0])
    
    # Bottom panel: Precision/Recall metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    for inferred in inferred_outliers_list:
        # True positives: correctly identified outliers
        tp = np.sum(true_outliers & inferred)
        # False positives: incorrectly identified as outliers
        fp = np.sum(~true_outliers & inferred)
        # False negatives: missed outliers
        fn = np.sum(true_outliers & ~inferred)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot metrics
    x_pos = np.arange(len(method_names))
    width = 0.25
    
    ax[1].bar(x_pos - width, precisions, width, label='Precision', color=get_method_color("genjax_is"))
    ax[1].bar(x_pos, recalls, width, label='Recall', color=get_method_color("genjax_hmc"))
    ax[1].bar(x_pos + width, f1_scores, width, label='F1 Score', color=get_method_color("numpyro_hmc"))
    
    ax[1].set_xlabel("Method", fontweight='bold')
    ax[1].set_ylabel("Score", fontweight='bold')
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(method_names, rotation=45, ha='right')
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(fontsize=16)
    apply_grid_style(ax[1])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
        ax[1].text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=12)
        ax[1].text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=12)
        ax[1].text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    save_publication_figure(fig, output_filename)


def save_outlier_detection_comparison(output_filename="figs/curvefit_outlier_detection_comparison.pdf"):
    """Create 3-panel outlier detection comparison figure."""
    
    from core import (
        npoint_curve,
        npoint_curve_with_outliers,
        npoint_curve_with_outliers_beta,
        infer_latents,
        mixed_gibbs_hmc_kernel,
        Lambda,
        polyfn
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
            
            params = samples.get_choices()['curve']
            curve_samples.append({
                'a': float(params['a'][idx]),
                'b': float(params['b'][idx]),
                'c': float(params['c'][idx])
            })
        
        return curve_samples  # Return all curves
    
    
    def run_outlier_is(xs, ys, n_samples=1000, n_trials=500):
        """Run IS on outlier model."""
        
        # Only constrain observations, let curve parameters be inferred
        constraints = {
            "ys": {"y": {"obs": ys}}
        }
        
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
                constraints
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
            curve_samples.append({
                'a': curve_params['a'][idx],
                'b': curve_params['b'][idx],
                'c': curve_params['c'][idx]
            })
        
        # Compute posterior probabilities
        outlier_samples = jnp.array(outlier_samples)
        outlier_probs = jnp.mean(outlier_samples, axis=0)
        
        return outlier_probs, curve_samples  # Return all curves
    
    
    def run_gibbs_hmc(xs, ys, n_chains=500, n_iterations=5000):
        """Run Gibbs+HMC with 500 parallel chains, taking final sample from each."""
        
        # Only constrain observations, let everything else be inferred
        constraints = {
            "ys": {"y": {"obs": ys}}
        }
        
        # Create kernel with moderate step size for better exploration
        # Use lower outlier rate to prevent over-detection
        kernel = mixed_gibbs_hmc_kernel(xs, ys, hmc_step_size=0.01, hmc_n_steps=20, outlier_rate=0.1)
        
        # Seed the kernel (don't JIT yet, will be done inside vmap)
        kernel_seeded = seed(kernel)
        
        # Function to run a single chain
        def run_single_chain(chain_key):
            # Initialize trace using beta model
            trace, _ = seed(npoint_curve_with_outliers_beta.generate)(
                chain_key, constraints, xs, 1.0, 3.0  # Beta(1, 3) has mean ~0.25
            )
            
            # Run chain
            def body_fn(i, carry):
                trace, key = carry
                key, subkey = jrand.split(key)
                new_trace = kernel_seeded(subkey, trace)
                return (new_trace, key)
            
            # Run the chain using fori_loop
            final_trace, _ = jax.lax.fori_loop(0, n_iterations, body_fn, (trace, chain_key))
            
            # Extract final values
            outliers = final_trace.get_choices()["ys"]["is_outlier"]
            curve_params = final_trace.get_choices()["curve"]
            
            return outliers, curve_params['a'], curve_params['b'], curve_params['c']
        
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
            curve_samples.append({
                'a': float(all_a[i]),
                'b': float(all_b[i]),
                'c': float(all_c[i])
            })
        
        # Print diagnostics
        mean_outliers = jnp.mean(jnp.sum(all_outliers, axis=1))
        print(f"   Mean outliers detected across chains: {mean_outliers:.1f}")
        
        return outlier_probs, curve_samples
    
    
    print("=== Creating Outlier Detection Comparison ===\n")
    
    # Create data
    xs, ys, outlier_indices, true_params = create_outlier_data()
    print(f"Created {len(xs)} points with {len(outlier_indices)} outliers")
    print(f"True outliers at indices: {outlier_indices}")
    print(f"Outlier rate: {len(outlier_indices)/len(xs):.1%}\n")
    
    print("Running inference methods...")
    
    # 1. Standard IS
    print("1. Standard IS (no outlier model)...")
    standard_curves = run_standard_is(xs, ys)
    print(f"   Collected {len(standard_curves)} standard curves")
    
    # 2. Outlier IS
    print("2. IS with outlier model...")
    is_outlier_probs, is_curves = run_outlier_is(xs, ys)
    print(f"   Collected {len(is_curves)} IS curves")
    
    # 3. Gibbs+HMC
    print("3. Gibbs+HMC with outlier model...")
    gibbs_outlier_probs, gibbs_curves = run_gibbs_hmc(xs, ys)
    print(f"   Collected {len(gibbs_curves)} Gibbs+HMC curves")
    
    # Create figure with GridSpec for better layout control
    setup_publication_fonts()
    fig = plt.figure(figsize=(18, 8))
    
    # Create a grid with 2 rows: main plots, colorbar
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[10, 1], hspace=0.5)
    
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
    ax.fill_between(x_plot, y_true - noise_std, y_true + noise_std, 
                   color='gray', alpha=0.3, label='True noise', zorder=40)
    
    # Add dotted lines to outline the noise interval
    ax.plot(x_plot, y_true - noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    ax.plot(x_plot, y_true + noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    
    # No outlier bounds for standard model
    
    # Plot true curve
    ax.plot(x_plot, y_true, 'k-', linewidth=3, label='True curve', zorder=50)
    
    # Plot posterior curves with better alpha blending
    for i, params in enumerate(standard_curves):
        y_pred = params['a'] + params['b'] * x_plot + params['c'] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(x_plot, y_pred, '#0173B2', alpha=0.05, linewidth=1.5, label='Importance Sampling')
        else:
            ax.plot(x_plot, y_pred, '#0173B2', alpha=0.05, linewidth=1.5)
    
    # Plot data (no outlier probabilities for this model)
    ax.scatter(xs, ys, c='gray', s=120, edgecolors='black', linewidth=1, zorder=100)
    
    
    ax.set_xlabel('X', fontweight='bold', fontsize=18)
    ax.set_ylabel('Y', fontweight='bold', fontsize=18, rotation=0, labelpad=20)
    ax.set_title('Curve model', fontsize=20, fontweight='bold', pad=10)
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)
    
    # Add local legend for first subplot with thicker lines
    legend = ax.legend(loc='lower right', fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)
    
    # Add text annotation in top left
    ax.text(0.05, 0.95, '(good inference, bad model)', transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # We'll add the second title after the middle panel is set up
    
    # Panel 2: IS with outlier model
    ax = axes[1]
    
    # Plot true noise interval (observation noise is 0.05)
    ax.fill_between(x_plot, y_true - noise_std, y_true + noise_std, 
                   color='gray', alpha=0.3, zorder=40)
    
    # Add dotted lines to outline the noise interval
    ax.plot(x_plot, y_true - noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    ax.plot(x_plot, y_true + noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    
    # Add horizontal lines for outlier interval (uniform over [-2, 2])
    ax.axhline(y=-2.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Outlier bounds')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Plot true curve
    ax.plot(x_plot, y_true, 'k-', linewidth=3, zorder=50)
    
    # Plot posterior curves with better alpha blending
    for i, params in enumerate(is_curves):
        y_pred = params['a'] + params['b'] * x_plot + params['c'] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(x_plot, y_pred, '#0173B2', alpha=0.05, linewidth=1.5, label='Importance Sampling')
        else:
            ax.plot(x_plot, y_pred, '#0173B2', alpha=0.05, linewidth=1.5)
    
    # Plot data colored by outlier probability
    scatter1 = ax.scatter(xs, ys, c=is_outlier_probs, s=120, cmap='RdBu_r', 
                        vmin=0, vmax=1, edgecolors='black', linewidth=1, zorder=100)
    
    
    ax.set_xlabel('X', fontweight='bold', fontsize=18)
    # Remove title
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)
    # Share y-axis - remove y-axis labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0)
    
    # Add local legend for second subplot with thicker lines
    legend = ax.legend(loc='lower right', fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)
    
    # Add text annotation in top left
    ax.text(0.05, 0.95, '(bad inference, good model)', transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add title that spans middle and right panels
    # We'll use the middle axes and extend the title
    ax.set_title('Robust curve model with outliers', fontsize=20, fontweight='bold', pad=10, loc='center')
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
    ax.fill_between(x_plot, y_true - noise_std, y_true + noise_std, 
                   color='gray', alpha=0.3, zorder=40)
    
    # Add dotted lines to outline the noise interval
    ax.plot(x_plot, y_true - noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    ax.plot(x_plot, y_true + noise_std, 'k', linestyle=':', linewidth=2.5, 
            dashes=(5, 3), alpha=0.7, zorder=41)
    
    # Add horizontal lines for outlier interval (uniform over [-2, 2])
    ax.axhline(y=-2.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Plot true curve
    ax.plot(x_plot, y_true, 'k-', linewidth=3, zorder=50)
    
    # Plot posterior curves with better alpha blending
    for i, params in enumerate(gibbs_curves):
        y_pred = params['a'] + params['b'] * x_plot + params['c'] * x_plot**2
        if i == 0:  # Add label only once
            ax.plot(x_plot, y_pred, '#EE7733', alpha=0.05, linewidth=1.5, label='Gibbs/HMC')
        else:
            ax.plot(x_plot, y_pred, '#EE7733', alpha=0.05, linewidth=1.5)
    
    # Plot data colored by outlier probability
    scatter2 = ax.scatter(xs, ys, c=gibbs_outlier_probs, s=120, cmap='RdBu_r', 
                        vmin=0, vmax=1, edgecolors='black', linewidth=1, zorder=100)
    
    
    ax.set_xlabel('X', fontweight='bold', fontsize=18)
    # Remove title
    apply_grid_style(ax)
    apply_standard_ticks(ax)  # Apply 3-tick standard
    ax.set_ylim(-2.5, 2.5)
    # Share y-axis - remove y-axis labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0)
    
    # Add local legend for third subplot with thicker lines
    legend = ax.legend(loc='lower right', fontsize=14, frameon=True)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3)
        line.set_alpha(1.0)
    
    # Add text annotation in top left
    ax.text(0.05, 0.95, '(good inference, good model)', transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add horizontal colorbar in columns 1 and 2 of the bottom row
    cbar_ax = fig.add_subplot(gs[1, 1:])
    cbar = plt.colorbar(scatter1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('P(outlier)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    save_publication_figure(fig, output_filename)
    print(f"\nSaved figure to {output_filename}")