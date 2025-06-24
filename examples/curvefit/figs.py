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

# Import shared GenJAX Research Visualization Standards
from examples.viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, set_minimal_ticks, apply_standard_ticks, save_publication_figure,
    PRIMARY_COLORS, LINE_SPECS, MARKER_SPECS
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
    
    save_publication_figure(fig, "examples/curvefit/figs/curvefit_prior_traces_density.pdf")


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
    
    save_publication_figure(fig, "examples/curvefit/figs/curvefit_posterior_traces_density.pdf")


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

    save_publication_figure(fig, "examples/curvefit/figs/curvefit_posterior_traces_grid.pdf")


def save_inference_viz(seed=42):
    """Save posterior visualization using importance sampling."""
    from examples.curvefit.core import (
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

    save_publication_figure(fig, "examples/curvefit/figs/curvefit_posterior_comparison.pdf")

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

    print(f"Making and saving inference scaling visualization with {n_trials} trials per N.")

    # Get reference dataset
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]

    # Test different sample sizes - more points for smoother curves
    n_samples_list = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000]
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
            lml = jnp.log(jnp.mean(jnp.exp(weights - jnp.max(weights)))) + jnp.max(weights)
            trial_lml.append(float(lml))
        
        # Average over trials
        ess_values.append(jnp.mean(jnp.array(trial_ess)))
        lml_estimates.append(jnp.mean(jnp.array(trial_lml)))
        
        # Benchmark runtime with more trials for stability
        times, (mean_time, std_time) = benchmark_with_warmup(
            lambda: infer_latents_jit(base_key, xs, ys, Const(n_samples)),
            repeats=100,  # Match the number of trials for consistency
            inner_repeats=20  # More inner repeats for accurate timing
        )
        runtime_means.append(mean_time * 1000)  # Convert to ms
        runtime_stds.append(std_time * 1000)

    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=FIGURE_SIZES["inference_scaling"]
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
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Runtime (ms)")
    # ax1.set_title("Vectorized Runtime", fontweight="normal")
    ax1.set_xscale("log")
    ax1.set_xlim(80, 12000)
    ax1.set_ylim(0.2, 0.3)  # Set runtime axis limits
    # Add a horizontal line showing the mean runtime to emphasize flatness
    mean_runtime = jnp.mean(jnp.array(runtime_means))
    ax1.axhline(mean_runtime, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    # Set specific x-axis tick locations with scientific notation
    ax1.set_xticks([100, 1000, 10000])
    ax1.set_xticklabels(['$10^2$', '$10^3$', '$10^4$'])
    # Only set y-axis ticks to avoid overriding x-axis
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

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
    ax2.set_xticklabels(['$10^2$', '$10^3$', '$10^4$'])
    # Only set y-axis ticks to avoid overriding x-axis
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

    # ESS plot
    ax3.plot(
        n_samples_list,
        ess_values,
        marker="o",
        linewidth=3,
        markersize=8,
        color=genjax_is_color,
    )
    ax3.set_xlabel("Number of Samples")
    ax3.set_ylabel("Effective Sample Size")
    # ax3.set_title("ESS Scaling", fontweight="normal")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(80, 12000)
    # Set specific x-axis tick locations with scientific notation
    ax3.set_xticks([100, 1000, 10000])
    ax3.set_xticklabels(['$10^2$', '$10^3$', '$10^4$'])
    # Keep y-axis as is for ESS
    ax3.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=3))

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
    fig_is.savefig("examples/curvefit/figs/curvefit_params_is1000.pdf", dpi=300, bbox_inches="tight")
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
    fig_hmc.savefig("examples/curvefit/figs/curvefit_params_hmc.pdf", dpi=300, bbox_inches="tight")
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
            fig_numpyro.savefig("examples/curvefit/figs/curvefit_params_numpyro.pdf", dpi=300, bbox_inches="tight")
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
                    "examples/curvefit/figs/curvefit_params_is50.pdf")
    create_is_figure(500, is_variant_colors['is_500'], 
                    "examples/curvefit/figs/curvefit_params_is500.pdf")
    create_is_figure(5000, is_variant_colors['is_5000'], 
                    "examples/curvefit/figs/curvefit_params_is5000.pdf")
    
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
                                 "examples/curvefit/figs/curvefit_params_resample50.pdf")
    print(f"  ✓ Saved IS (N=50) single particle figure")
    print(f"    Mean: a={float(a_50.mean()):.3f}, b={float(b_50.mean()):.3f}, c={float(c_50.mean()):.3f}")
    print(f"    Std:  a={float(a_50.std()):.3f}, b={float(b_50.std()):.3f}, c={float(c_50.std()):.3f}")
    
    # Generate N=500 figure
    print(f"\n  Running GenJAX IS (N=500) with single particle resampling...")
    key_500 = jrand.key(seed + 500)
    a_500, b_500, c_500 = run_is_single_resample_vectorized(key_500, xs, ys, 500, n_trials)
    
    create_single_resample_figure(a_500, b_500, c_500, single_resample_colors['is_500'],
                                 "examples/curvefit/figs/curvefit_params_resample500.pdf")
    print(f"  ✓ Saved IS (N=500) single particle figure")
    print(f"    Mean: a={float(a_500.mean()):.3f}, b={float(b_500.mean()):.3f}, c={float(c_500.mean()):.3f}")
    print(f"    Std:  a={float(a_500.std()):.3f}, b={float(b_500.std()):.3f}, c={float(c_500.std()):.3f}")
    
    # Generate N=5000 figure
    print(f"\n  Running GenJAX IS (N=5000) with single particle resampling...")
    key_5000 = jrand.key(seed + 1000)
    a_5000, b_5000, c_5000 = run_is_single_resample_vectorized(key_5000, xs, ys, 5000, n_trials)
    
    create_single_resample_figure(a_5000, b_5000, c_5000, single_resample_colors['is_5000'],
                                 "examples/curvefit/figs/curvefit_params_resample5000.pdf")
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
        from examples.curvefit.core import numpyro_run_hmc_inference_jit
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
    fig.savefig("examples/curvefit/figs/curvefit_legend_all.pdf", 
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
    fig.savefig("examples/curvefit/figs/curvefit_legend_is_horiz.pdf", 
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
    fig_vert.savefig("examples/curvefit/figs/curvefit_legend_is_vert.pdf", 
                     dpi=300, bbox_inches='tight', pad_inches=0.05)
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
        filename = f"examples/curvefit/figs/curvefit_is_only_parameter_density_n{n_particles}.pdf"
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
    from examples.curvefit.core import (
        npoint_curve, 
        npoint_curve_with_outliers,
        infer_latents_jit,
        infer_latents_with_outliers_jit,
    )
    from examples.curvefit.data import polyfn
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
        Const(outlier_rate), Const(0.0), Const(5.0)
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
    filename_pdf = "examples/curvefit/figs/curvefit_outlier_robustness_demo.pdf"
    filename_png = "examples/curvefit/figs/curvefit_outlier_robustness_demo.png"
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