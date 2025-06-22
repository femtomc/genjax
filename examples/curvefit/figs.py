"""
Figure generation for GenJAX curvefit case study.

This module provides all visualization functions for the curvefit case study,
organized into logical sections with consistent styling throughout.

Sections:
1. Configuration and Constants
2. Helper Functions
3. Trace Visualizations (onepoint and multipoint)
4. Density Visualizations
5. Inference Visualizations
6. Framework Comparisons
7. Overview Figures for Paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import jax
from examples.utils import benchmark_with_warmup
from scipy.ndimage import gaussian_filter  # For smoothing 3D surfaces

# ============================================================================
# SECTION 1: CONFIGURATION AND CONSTANTS
# ============================================================================

# Standardized figure sizes for consistent layout
FIGURE_SIZES = {
    # Single-panel figures - square format (1:1 ratio)
    "single_small": (3, 3),           # Compact individual trace, square
    "single_medium": (5, 5),          # Standard individual trace, square
    "single_large": (7, 7),           # Larger individual trace, square
    # Multi-panel figures
    "two_panel_horizontal": (12, 5),  # Full textwidth
    "two_panel_vertical": (6.5, 8),   # Half textwidth
    "four_panel_grid": (10, 10),      # 2x2 grid for trace collection, square
    # Custom sizes
    "framework_comparison": (12, 8),   # Two stacked panels
    "parameter_posterior": (15, 10),   # 3x2 grid with 3D plots
}

# Consistent color scheme across all figures
COLORS = {
    "curve": "#333333",           # Dark gray for curves
    "points": "#CC3311",          # Red for data points
    "points_edge": "#882255",     # Purple edge for points
    "true_params": "#CC3311",     # Red for true parameters
    "grid": "#666666",            # Light gray for grids
    "text_box": "#666666",        # Gray for text box edges
}

# Publication-quality plot settings
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.linewidth": 1.5,
    "axes.grid": False,
    "axes.spines.top": True,      # Show top spine for box frame
    "axes.spines.right": True,     # Show right spine for box frame
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def set_minimal_ticks(ax, x_ticks=3, y_ticks=3):
    """Set minimal number of ticks on both axes for cleaner plots."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, prune="both"))


def add_log_density_text(ax, log_density):
    """Add log density text below the frame in large font."""
    ax.text(
        0.5, -0.15,  # Centered below the frame
        f"log p = {log_density:.2f}",
        transform=ax.transAxes,
        fontsize=28,  # Much larger font
        fontweight='bold',
        verticalalignment="top",
        horizontalalignment="center",
        color="#333333",
    )


def get_reference_dataset(seed=42, n_points=10):
    """
    Generate the reference dataset used across visualizations.
    This ensures consistency when comparing different figures.
    """
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed
    
    xs = jnp.linspace(0, 1, n_points)
    seeded_simulate = genjax_seed(npoint_curve.simulate)
    
    key = jrand.key(seed)
    keys = jrand.split(key, 4)
    
    # Use first key for reference dataset
    trace = seeded_simulate(keys[0], xs)
    
    curve, (xs_ret, ys) = trace.get_retval()
    choices = trace.get_choices()
    
    true_params = {
        "a": float(choices["curve"]["a"]),
        "b": float(choices["curve"]["b"]),
        "c": float(choices["curve"]["c"]),
        "noise_std": 0.2,
    }
    
    return {
        "xs": xs,
        "ys": ys,
        "true_params": true_params,
        "trace": trace,
    }


# ============================================================================
# SECTION 3: TRACE VISUALIZATIONS
# ============================================================================

# --- Onepoint Trace Functions ---

def visualize_onepoint_trace(trace, ylim=(-1.5, 1.5), figsize=None, ax=None):
    """Visualize a single point trace with consistent styling."""
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)
    
    if ax is None:
        if figsize is None:
            figsize = FIGURE_SIZES["single_small"]
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
    
    # Plot curve with thicker line
    ax.plot(xvals, jax.vmap(curve)(xvals), 
            color=COLORS["curve"], linewidth=3.5)
    
    # Plot point with larger size
    ax.scatter(
        pt[0], pt[1],
        color=COLORS["points"],
        s=120,  # Larger scatter size
        zorder=10,
        edgecolor=COLORS["points_edge"],
        linewidth=2.0,
    )
    
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if fig is not None:
        fig.tight_layout(pad=0.5)
    return fig


def save_onepoint_trace_viz(key=None):
    """Save single onepoint trace visualization (010)."""
    from examples.curvefit.core import onepoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving onepoint trace visualization.")
    if key is None:
        key = jrand.key(42)
    
    seeded_simulate = genjax_seed(onepoint_curve.simulate)
    trace = seeded_simulate(key, 0.0)
    fig = visualize_onepoint_trace(trace)
    fig.savefig("examples/curvefit/figs/010_onepoint_trace.pdf")
    plt.close()


def save_four_separate_onepoint_traces(key=None):
    """Save four separate onepoint trace visualizations (011-014)."""
    from examples.curvefit.core import onepoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving four separate onepoint trace visualizations.")
    
    if key is None:
        key = jrand.key(100)
    
    seeded_simulate = genjax_seed(onepoint_curve.simulate)
    keys = jrand.split(key, 4)
    
    x_positions = [0.2, 0.4, 0.6, 0.8]
    
    for i, (subkey, x_pos) in enumerate(zip(keys, x_positions)):
        trace = seeded_simulate(subkey, x_pos)
        log_density = -float(trace.get_score())
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
        visualize_onepoint_trace(trace, ax=ax)
        add_log_density_text(ax, log_density)
        
        filename = f"examples/curvefit/figs/01{i+1}_onepoint_trace.pdf"
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {filename} (x = {x_pos}, log p = {log_density:.2f})")


# --- Multipoint Trace Functions ---

def visualize_multipoint_trace(trace, figsize=None, yrange=None, ax=None):
    """Visualize a multipoint trace with consistent styling."""
    curve, (xs, ys) = trace.get_retval()
    xvals = jnp.linspace(0, 1, 300)
    
    if ax is None:
        if figsize is None:
            figsize = FIGURE_SIZES["single_medium"]
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    
    # Plot curve with thicker line
    ax.plot(xvals, jax.vmap(curve)(xvals), 
            color=COLORS["curve"], linewidth=3.5)
    
    # Plot points with larger size
    ax.scatter(
        xs, ys,
        color=COLORS["points"],
        s=120,  # Larger scatter size
        zorder=10,
        edgecolor=COLORS["points_edge"],
        linewidth=2.0
    )
    
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if fig is not None:
        fig.tight_layout(pad=0.5)
    return fig


def save_multipoint_trace_viz(key=None):
    """Save single multipoint trace visualization (020)."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving multipoint trace visualization.")
    xs = jnp.linspace(0, 1, 10)
    
    if key is None:
        key = jrand.key(42)
    
    seeded_simulate = genjax_seed(npoint_curve.simulate)
    trace = seeded_simulate(key, xs)
    
    log_density = -float(trace.get_score())
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    visualize_multipoint_trace(trace, ax=ax)
    add_log_density_text(ax, log_density)
    
    fig.savefig("examples/curvefit/figs/020_multipoint_trace.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"  Saved multipoint trace (log p = {log_density:.2f})")


def save_four_multipoint_trace_vizs(key=None):
    """Save four multipoint traces in 2x2 grid (030)."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving four multipoint trace visualizations.")
    
    if key is None:
        key = jrand.key(42)
    
    xs = jnp.linspace(0, 1, 10)
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["four_panel_grid"])
    axes = axes.flatten()
    
    seeded_simulate = genjax_seed(npoint_curve.simulate)
    keys = jrand.split(key, 4)
    
    for i, (ax, subkey) in enumerate(zip(axes, keys)):
        trace = seeded_simulate(subkey, xs)
        log_density = -float(trace.get_score())
        
        visualize_multipoint_trace(trace, ax=ax)
        
        # Special styling for reference dataset
        if i == 0:
            ax.set_title(
                f"Sample {i + 1} (Reference Dataset)",
                fontsize=16, pad=8,
                color=COLORS["points"],
                fontweight="bold",
            )
        else:
            ax.set_title(f"Sample {i + 1}", fontsize=16, pad=8)
        
        # Clean axes for grid layout
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Box styling
        for spine in ax.spines.values():
            if i == 0:
                spine.set_linewidth(3.0)
                spine.set_color(COLORS["points"])
            else:
                spine.set_linewidth(1.5)
                spine.set_color("black")
        
        add_log_density_text(ax, log_density)
    
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/030_four_multipoint_traces.pdf", bbox_inches='tight')
    plt.close()


def save_four_separate_batched_multipoint_traces(key=None):
    """Save four separate batched multipoint traces (031-034)."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving four separate batched multipoint trace visualizations.")
    
    if key is None:
        key = jrand.key(42)
    
    xs = jnp.linspace(0, 1, 10)
    seeded_simulate = genjax_seed(npoint_curve.simulate)
    keys = jrand.split(key, 4)
    
    for i, subkey in enumerate(keys):
        trace = seeded_simulate(subkey, xs)
        log_density = -float(trace.get_score())
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
        visualize_multipoint_trace(trace, ax=ax)
        add_log_density_text(ax, log_density)
        
        filename = f"examples/curvefit/figs/03{i+1}_batched_multipoint_trace.pdf"
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {filename} (log p = {log_density:.2f})")


# ============================================================================
# SECTION 4: DENSITY VISUALIZATIONS
# ============================================================================

def compute_density_grid_vectorized(model, constraints_fn, grid_params):
    """
    Vectorized density computation helper.
    TODO: Implement proper JAX vectorization to replace Python loops.
    """
    # This is a placeholder for the vectorized implementation
    # Currently using Python loops in the individual functions
    pass


def save_four_onepoint_trace_densities(key=None):
    """Save four onepoint trace density visualizations (051-054)."""
    from examples.curvefit.core import onepoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving four onepoint trace density visualizations.")
    
    if key is None:
        key = jrand.key(42)
    
    x_positions = [0.2, 0.4, 0.6, 0.8]
    keys = jrand.split(key, 4)
    
    for i, (subkey, x_pos) in enumerate(zip(keys, x_positions)):
        # Generate reference trace
        seeded_simulate = genjax_seed(onepoint_curve.simulate)
        ref_trace = seeded_simulate(subkey, x_pos)
        choices = ref_trace.get_choices()
        true_a = float(choices["curve"]["a"])
        true_b = float(choices["curve"]["b"])
        true_c = float(choices["curve"]["c"])
        _, (x_obs, y_obs) = ref_trace.get_retval()
        
        # Define parameter grid
        n_grid = 20  # Reduced for performance
        a_range = jnp.linspace(true_a - 2.0, true_a + 2.0, n_grid)
        b_range = jnp.linspace(true_b - 2.0, true_b + 2.0, n_grid)
        
        # Compute densities (TODO: vectorize this)
        log_densities = jnp.zeros((n_grid, n_grid))
        for j, a in enumerate(a_range):
            for k, b in enumerate(b_range):
                constraints = {
                    "curve": {"a": a, "b": b, "c": true_c},
                    "pt": {"obs": y_obs}
                }
                _, log_weight = onepoint_curve.generate(constraints, x_pos)
                log_densities = log_densities.at[j, k].set(log_weight)
        
        # Create figure
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
        
        # Plot density heatmap
        im = ax.imshow(
            log_densities.T,
            origin="lower",
            aspect="auto",
            extent=[a_range.min(), a_range.max(), b_range.min(), b_range.max()],
            cmap="viridis",
            interpolation="bilinear"
        )
        
        # Mark true parameters
        ax.scatter(
            true_a, true_b,
            c=COLORS["true_params"],
            s=150,
            marker="*",
            edgecolor="white",
            linewidth=2,
            zorder=10
        )
        
        ax.set_xlabel("a (constant term)")
        ax.set_ylabel("b (linear coefficient)")
        ax.set_title(f"Log Density at x = {x_pos} (c fixed)", fontsize=16)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log Density", rotation=270, labelpad=20)
        
        set_minimal_ticks(ax, x_ticks=4, y_ticks=4)
        plt.tight_layout()
        
        filename = f"examples/curvefit/figs/05{i+1}_onepoint_trace_density.pdf"
        fig.savefig(filename)
        plt.close()
        
        print(f"  Saved {filename} (x = {x_pos})")


def save_four_batched_multipoint_trace_densities(key=None):
    """Save four batched multipoint trace density visualizations (041-044)."""
    from examples.curvefit.core import npoint_curve
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving four batched multipoint trace density visualizations.")
    
    if key is None:
        key = jrand.key(42)
    
    xs = jnp.linspace(0, 1, 10)
    keys = jrand.split(key, 4)
    
    for i, subkey in enumerate(keys):
        # Generate reference trace
        seeded_simulate = genjax_seed(npoint_curve.simulate)
        ref_trace = seeded_simulate(subkey, xs)
        choices = ref_trace.get_choices()
        true_a = float(choices["curve"]["a"])
        true_b = float(choices["curve"]["b"])
        true_c = float(choices["curve"]["c"])
        _, (xs_ret, ys) = ref_trace.get_retval()
        
        # Define parameter grid
        n_grid = 20  # Reduced for performance
        a_range = jnp.linspace(true_a - 2.0, true_a + 2.0, n_grid)
        b_range = jnp.linspace(true_b - 2.0, true_b + 2.0, n_grid)
        
        # Compute densities (TODO: vectorize this)
        log_densities = jnp.zeros((n_grid, n_grid))
        for j, a in enumerate(a_range):
            for k, b in enumerate(b_range):
                constraints = {
                    "curve": {"a": a, "b": b, "c": true_c},
                    "ys": {"obs": ys}
                }
                _, log_weight = npoint_curve.generate(constraints, xs)
                log_densities = log_densities.at[j, k].set(log_weight)
        
        # Create figure
        fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
        
        # Plot density heatmap
        im = ax.imshow(
            log_densities.T,
            origin="lower",
            aspect="auto",
            extent=[a_range.min(), a_range.max(), b_range.min(), b_range.max()],
            cmap="viridis",
            interpolation="bilinear"
        )
        
        # Mark true parameters
        ax.scatter(
            true_a, true_b,
            c=COLORS["true_params"],
            s=150,
            marker="*",
            edgecolor="white",
            linewidth=2,
            zorder=10
        )
        
        ax.set_xlabel("a (constant term)")
        ax.set_ylabel("b (linear coefficient)")
        ax.set_title(f"Log Density - Multipoint Sample {i+1} (c fixed)", fontsize=16)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log Density", rotation=270, labelpad=20)
        
        set_minimal_ticks(ax, x_ticks=4, y_ticks=4)
        plt.tight_layout()
        
        filename = f"examples/curvefit/figs/04{i+1}_batched_multipoint_trace_density.pdf"
        fig.savefig(filename)
        plt.close()
        
        print(f"  Saved {filename}")


# ============================================================================
# SECTION 5: INFERENCE VISUALIZATIONS
# ============================================================================

def save_inference_viz(n_curves_to_plot=200, seed=42):
    """Save inference visualization using reference dataset (050)."""
    from examples.curvefit.core import infer_latents
    from genjax.core import Const
    from genjax.pjax import seed as genjax_seed
    
    print("Making and saving inference visualization.")
    
    # Use reference dataset
    data = get_reference_dataset(seed=seed)
    xs = data["xs"]
    ys = data["ys"]
    
    # Run inference
    seeded_infer = genjax_seed(infer_latents)
    samples, weights = seeded_infer(jrand.key(seed), xs, ys, Const(1000))
    
    # Resample for visualization
    n_resample = min(n_curves_to_plot, 1000)
    indices = jrand.categorical(jrand.key(seed + 1), weights, shape=(n_resample,))
    
    # Extract curves
    curves = []
    choices = samples.get_choices()
    for idx in indices:
        a = choices["curve"]["a"][idx]
        b = choices["curve"]["b"][idx]
        c = choices["curve"]["c"][idx]
        curve_fn = lambda x, coeffs=jnp.array([a, b, c]): (
            coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
        )
        curves.append(curve_fn)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Plot posterior curves
    xvals = jnp.linspace(0, 1, 100)
    for curve_fn in curves:
        yvals = jax.vmap(curve_fn)(xvals)
        ax.plot(xvals, yvals, color="blue", alpha=0.05, linewidth=0.5)
    
    # Plot data points
    ax.scatter(
        xs, ys,
        color=COLORS["points"],
        s=100,
        zorder=10,
        edgecolor=COLORS["points_edge"],
        linewidth=2,
        label="Observations"
    )
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Predictive Distribution", fontsize=18)
    ax.legend()
    set_minimal_ticks(ax)
    
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/050_inference_viz.pdf")
    plt.close()
    
    print("✓ Saved inference visualization")


def save_inference_scaling_viz():
    """Save inference scaling visualization (060)."""
    from examples.curvefit.core import infer_latents
    from genjax.core import Const
    from genjax.pjax import seed as genjax_seed
    import time
    
    print("Making and saving inference scaling visualization.")
    
    # Test different numbers of particles
    n_particles_list = [10, 50, 100, 500, 1000, 2000]
    times = []
    
    # Generate test data
    data = get_reference_dataset()
    xs, ys = data["xs"], data["ys"]
    
    seeded_infer = genjax_seed(infer_latents)
    key = jrand.key(42)
    
    for n_particles in n_particles_list:
        # Warm up
        _ = seeded_infer(key, xs, ys, Const(n_particles))
        
        # Time multiple runs
        start = time.time()
        for _ in range(5):
            _ = seeded_infer(key, xs, ys, Const(n_particles))
        elapsed = (time.time() - start) / 5
        times.append(elapsed)
        
        print(f"  {n_particles} particles: {elapsed*1000:.1f} ms")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    ax.plot(n_particles_list, times, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Inference Scaling with Particle Count")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/060_inference_scaling.pdf")
    plt.close()
    
    print("✓ Saved inference scaling visualization")


def save_log_density_viz():
    """Save log density visualization (070)."""
    from examples.curvefit.core import npoint_curve
    
    print("Making and saving log density visualization.")
    
    # Use reference dataset
    data = get_reference_dataset()
    xs = data["xs"]
    ys = data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]
    
    # Define parameter grid
    n_grid = 50
    a_range = jnp.linspace(-3.0, 3.0, n_grid)
    b_range = jnp.linspace(-4.5, 4.5, n_grid)
    
    # Compute log densities
    log_densities = jnp.zeros((n_grid, n_grid))
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            constraints = {"curve": {"a": a, "b": b, "c": true_c}, "ys": {"obs": ys}}
            _, log_weight = npoint_curve.generate(constraints, xs)
            log_densities = log_densities.at[i, j].set(log_weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    im = ax.imshow(
        log_densities.T,
        origin="lower",
        aspect="auto",
        extent=[a_range.min(), a_range.max(), b_range.min(), b_range.max()],
        cmap="viridis",
    )
    
    ax.scatter(
        true_a, true_b,
        c=COLORS["true_params"],
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
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log Density", rotation=270, labelpad=20)
    
    set_minimal_ticks(ax, x_ticks=4, y_ticks=4)
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/070_log_density.pdf")
    plt.close()
    
    print("✓ Saved log density visualization")


# ============================================================================
# SECTION 6: FRAMEWORK COMPARISONS
# ============================================================================

def save_framework_comparison_figure(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=20,
):
    """Save framework comparison figure (080)."""
    try:
        from examples.curvefit.core import (
            infer_latents_jit,
            hmc_infer_latents_jit,
            numpyro_run_importance_sampling_jit,
            numpyro_run_hmc_inference_jit,
        )
    except ImportError:
        print("Warning: Some imports failed, skipping framework comparison")
        return {}
    
    from genjax.core import Const
    
    print("=== Framework Comparison ===")
    print(f"Data points: {n_points}")
    print(f"IS samples: {n_samples_is} (fixed)")
    print(f"HMC samples: {n_samples_hmc}")
    
    # Use reference dataset
    data = get_reference_dataset(seed=seed)
    xs, ys = data["xs"], data["ys"]
    
    results = {}
    
    # 1. GenJAX IS
    print(f"\n1. GenJAX IS ({n_samples_is} particles)...")
    
    def is_task():
        return infer_latents_jit(jrand.key(seed), xs, ys, Const(n_samples_is))
    
    is_times, is_timing_stats = benchmark_with_warmup(
        is_task, warmup_runs=2, repeats=timing_repeats, inner_repeats=10
    )
    is_samples, is_weights = is_task()
    
    results["genjax_is"] = {
        "method": f"GenJAX IS (N={n_samples_is})",
        "timing": is_timing_stats,
    }
    
    print(f"  Time: {is_timing_stats[0] * 1000:.1f} ± {is_timing_stats[1] * 1000:.1f} ms")
    
    # 2. GenJAX Vectorized HMC
    print("\n2. GenJAX Vectorized HMC (4 chains)...")
    
    def hmc_task():
        from examples.curvefit.core import hmc_infer_latents_vectorized_jit
        return hmc_infer_latents_vectorized_jit(
            jrand.key(seed), xs, ys,
            Const(n_samples_hmc), Const(n_warmup),
            Const(0.001), Const(50),
            Const(4),  # 4 chains
        )
    
    hmc_times, hmc_timing_stats = benchmark_with_warmup(
        hmc_task, warmup_runs=2, repeats=max(5, timing_repeats // 10), inner_repeats=2
    )
    
    results["genjax_hmc"] = {
        "method": f"GenJAX HMC (K=4, L={n_samples_hmc})",
        "timing": hmc_timing_stats,
    }
    
    print(f"  Time: {hmc_timing_stats[0] * 1000:.1f} ± {hmc_timing_stats[1] * 1000:.1f} ms")
    
    # 3. NumPyro HMC (if available)
    try:
        print("\n3. NumPyro HMC (4 chains)...")
        
        def numpyro_hmc_task():
            return numpyro_run_hmc_inference_jit(
                jrand.key(seed), xs, ys,
                n_samples_hmc, n_warmup,
                step_size=0.001, num_steps=50,
                num_chains=4
            )
        
        numpyro_hmc_times, numpyro_hmc_timing_stats = benchmark_with_warmup(
            numpyro_hmc_task, warmup_runs=2, repeats=max(5, timing_repeats // 10), inner_repeats=2
        )
        
        results["numpyro_hmc"] = {
            "method": f"NumPyro HMC (K=4, L={n_samples_hmc})",
            "timing": numpyro_hmc_timing_stats,
        }
        
        print(f"  Time: {numpyro_hmc_timing_stats[0] * 1000:.1f} ± {numpyro_hmc_timing_stats[1] * 1000:.1f} ms")
    except Exception as e:
        print(f"  NumPyro HMC failed: {e}")
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES["framework_comparison"],
                                    height_ratios=[3, 1])
    
    # Top panel: Posterior curves (placeholder)
    ax1.text(0.5, 0.5, "Posterior Curves\n(Implementation needed)",
             ha='center', va='center', transform=ax1.transAxes)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Framework Comparison: Posterior Inference")
    
    # Bottom panel: Timing comparison
    methods = list(results.keys())
    means = [r["timing"][0] * 1000 for r in results.values()]
    stds = [r["timing"][1] * 1000 for r in results.values()]
    
    x_pos = np.arange(len(methods))
    ax2.bar(x_pos, means, yerr=stds, capsize=5, width=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r["method"] for r in results.values()])
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Inference Time Comparison")
    
    plt.tight_layout()
    filename = f"examples/curvefit/figs/080_framework_comparison_n{n_points}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Saved framework comparison: {filename}")
    return results


def save_genjax_posterior_comparison(
    n_points=10,
    n_samples_is=1000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
    timing_repeats=10,
):
    """Save GenJAX IS vs Vectorized HMC posterior comparison (090)."""
    from examples.curvefit.core import infer_latents_jit, hmc_infer_latents_vectorized_jit
    from genjax.core import Const
    
    print("\n=== GenJAX Posterior Comparison ===")
    
    # Generate data
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    
    # Run IS
    print(f"Running IS with {n_samples_is} particles...")
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )
    
    # Run Vectorized HMC
    print(f"Running Vectorized HMC with {n_samples_hmc} samples per chain (4 chains)...")
    hmc_samples, _ = hmc_infer_latents_vectorized_jit(
        jrand.key(seed), xs, ys,
        Const(n_samples_hmc), Const(n_warmup),
        Const(0.001), Const(50),
        Const(4),  # 4 chains
    )
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract parameters
    is_a = is_samples.get_choices()["curve"]["a"]
    is_b = is_samples.get_choices()["curve"]["b"]
    is_c = is_samples.get_choices()["curve"]["c"]
    
    # For vectorized HMC, flatten across chains
    hmc_a = hmc_samples.get_choices()["curve"]["a"].reshape(-1)
    hmc_b = hmc_samples.get_choices()["curve"]["b"].reshape(-1)
    hmc_c = hmc_samples.get_choices()["curve"]["c"].reshape(-1)
    
    # Plot histograms
    params = [("a", is_a, hmc_a), ("b", is_b, hmc_b), ("c", is_c, hmc_c)]
    
    for ax, (param_name, is_vals, hmc_vals) in zip(axes, params):
        ax.hist(is_vals, bins=30, alpha=0.5, label=f"IS (N={n_samples_is})", density=True)
        ax.hist(hmc_vals, bins=30, alpha=0.5, label=f"HMC (K=4, L={n_samples_hmc})", density=True)
        ax.set_xlabel(f"Parameter {param_name}")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Posterior: {param_name}")
    
    plt.suptitle("GenJAX: IS vs HMC Posterior Comparison", fontsize=16)
    plt.tight_layout()
    
    fig.savefig("examples/curvefit/figs/090_genjax_posterior_comparison.pdf")
    plt.close()
    
    print("✓ Saved GenJAX posterior comparison")


def save_parameter_posterior_methods_comparison(
    n_points=10,
    n_samples_is=5000,
    n_samples_hmc=1000,
    n_warmup=500,
    seed=42,
):
    """Save 3D parameter posterior comparison across methods (100)."""
    print("\n=== 3D Parameter Posterior Methods Comparison ===")
    
    try:
        from examples.curvefit.core import (
            infer_latents_jit,
            hmc_infer_latents_vectorized_jit,
            numpyro_run_hmc_inference_jit,
            get_points_for_inference,
        )
    except ImportError:
        print("Warning: Some imports failed, skipping 3D parameter comparison")
        return
    
    from genjax.core import Const
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap
    
    # Get reference dataset
    data = get_reference_dataset(seed=seed, n_points=n_points)
    xs, ys = data["xs"], data["ys"]
    true_a = data["true_params"]["a"]
    true_b = data["true_params"]["b"]
    true_c = data["true_params"]["c"]
    
    print(f"  True parameters: a={true_a:.3f}, b={true_b:.3f}, c={true_c:.3f}")
    
    # 1. GenJAX IS
    print(f"\n  1. GenJAX IS ({n_samples_is} particles)...")
    is_samples, is_weights = infer_latents_jit(
        jrand.key(seed), xs, ys, Const(n_samples_is)
    )
    
    # Resample according to weights
    n_resample = 500  # Reduced for better visualization
    resample_key = jrand.key(seed + 100)
    resample_idx = jrand.choice(
        resample_key,
        jnp.arange(n_samples_is),
        shape=(n_resample,),
        p=jnp.exp(is_weights - jnp.max(is_weights))
    )
    
    is_a_resampled = is_samples.get_choices()["curve"]["a"][resample_idx]
    is_b_resampled = is_samples.get_choices()["curve"]["b"][resample_idx]
    is_c_resampled = is_samples.get_choices()["curve"]["c"][resample_idx]
    
    # 2. GenJAX HMC
    print(f"\n  2. GenJAX HMC ({n_samples_hmc} samples x 4 chains)...")
    hmc_samples, _ = hmc_infer_latents_vectorized_jit(
        jrand.key(seed), xs, ys,
        Const(n_samples_hmc), Const(n_warmup),
        Const(0.001), Const(50),
        Const(4),  # 4 chains
    )
    
    # Flatten chains and subsample for visualization
    hmc_a = hmc_samples.get_choices()["curve"]["a"].reshape(-1)[:500]
    hmc_b = hmc_samples.get_choices()["curve"]["b"].reshape(-1)[:500]
    hmc_c = hmc_samples.get_choices()["curve"]["c"].reshape(-1)[:500]
    
    # 3. NumPyro HMC
    print(f"\n  3. NumPyro HMC ({n_samples_hmc} samples x 4 chains)...")
    numpyro_result = numpyro_run_hmc_inference_jit(
        jrand.key(seed), xs, ys, 
        n_samples_hmc, n_warmup, 0.001, 50, 4  # 4 chains
    )
    numpyro_a = numpyro_result["samples"]["a"][:500]
    numpyro_b = numpyro_result["samples"]["b"][:500]
    numpyro_c = numpyro_result["samples"]["c"][:500]
    
    # Create figure with 3D plots
    fig = plt.figure(figsize=(18, 6))
    
    # Define method colors
    method_colors = {
        "genjax_is": "#117733",    # Green
        "genjax_hmc": "#332288",   # Blue
        "numpyro_hmc": "#AA4499",  # Purple
    }
    
    methods_data = [
        ("GenJAX IS (5000)", is_a_resampled, is_b_resampled, is_c_resampled, method_colors["genjax_is"]),
        ("GenJAX HMC", hmc_a, hmc_b, hmc_c, method_colors["genjax_hmc"]),
        ("NumPyro HMC", numpyro_a, numpyro_b, numpyro_c, method_colors["numpyro_hmc"]),
    ]
    
    # Calculate consistent axis limits for all three parameters
    all_a = np.concatenate([is_a_resampled, hmc_a, numpyro_a])
    all_b = np.concatenate([is_b_resampled, hmc_b, numpyro_b])
    all_c = np.concatenate([is_c_resampled, hmc_c, numpyro_c])
    
    a_min, a_max = all_a.min(), all_a.max()
    b_min, b_max = all_b.min(), all_b.max()
    c_min, c_max = all_c.min(), all_c.max()
    
    # Include ground truth in limits
    a_min = min(a_min, true_a)
    a_max = max(a_max, true_a)
    b_min = min(b_min, true_b)
    b_max = max(b_max, true_b)
    c_min = min(c_min, true_c)
    c_max = max(c_max, true_c)
    
    # Add padding
    a_range = a_max - a_min
    b_range = b_max - b_min
    c_range = c_max - c_min
    pad = 0.15
    a_lim = (a_min - pad * a_range, a_max + pad * a_range)
    b_lim = (b_min - pad * b_range, b_max + pad * b_range)
    c_lim = (c_min - pad * c_range, c_max + pad * c_range)
    
    # Create 3D scatter plots with density-based sizing and coloring
    for i, (method_name, a_samples, b_samples, c_samples, color) in enumerate(methods_data):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        
        # Calculate density using KDE
        points = np.vstack([a_samples, b_samples, c_samples])
        try:
            kde = gaussian_kde(points, bw_method='scott')
            density = kde(points)
            # Normalize density for visualization
            density_norm = (density - density.min()) / (density.max() - density.min())
        except:
            # Fallback if KDE fails
            density_norm = np.ones_like(a_samples) * 0.5
        
        # Define colormap
        cmap = plt.cm.viridis if i == 0 else plt.cm.plasma if i == 1 else plt.cm.inferno
        
        # Sort by density to plot low density first (so high density is on top)
        sort_idx = np.argsort(density_norm)
        
        # Create scatter plot with density-based coloring and sizing
        # Size inversely proportional to density (larger spheres for lower density)
        sizes = 30 + (1 - density_norm[sort_idx]) * 70
        
        # Use square markers and increased transparency
        scatter = ax.scatter(
            a_samples[sort_idx], b_samples[sort_idx], c_samples[sort_idx],
            c=density_norm[sort_idx],
            cmap=cmap,
            s=sizes,
            alpha=0.2,  # Reduced alpha for better visibility of ground truth
            edgecolors='none',
            marker='s'  # Square marker
        )
        
        # Plot ground truth elements LAST for maximum visibility
        # Add projection lines from ground truth (with higher alpha and thicker lines)
        # X projection (along a-axis)
        ax.plot([a_lim[0], true_a], [true_b, true_b], [true_c, true_c],
                'r-', linewidth=3, alpha=1.0, zorder=2000)
        # Y projection (along b-axis)
        ax.plot([true_a, true_a], [b_lim[0], true_b], [true_c, true_c],
                'r-', linewidth=3, alpha=1.0, zorder=2001)
        # Z projection (along c-axis)
        ax.plot([true_a, true_a], [true_b, true_b], [c_lim[0], true_c],
                'r-', linewidth=3, alpha=1.0, zorder=2002)
        
        # Add projection points on the walls (plotted after lines)
        ax.scatter([a_lim[0]], [true_b], [true_c], 
                   c='red', s=150, marker='o', alpha=1.0, zorder=2003,
                   edgecolors='black', linewidth=2)
        ax.scatter([true_a], [b_lim[0]], [true_c], 
                   c='red', s=150, marker='o', alpha=1.0, zorder=2004,
                   edgecolors='black', linewidth=2)
        ax.scatter([true_a], [true_b], [c_lim[0]], 
                   c='red', s=150, marker='o', alpha=1.0, zorder=2005,
                   edgecolors='black', linewidth=2)
        
        # Add ground truth point as a large red star (plotted last)
        ax.scatter(
            [true_a], [true_b], [true_c],
            c='red', s=800, marker='*',
            edgecolors='black', linewidth=3,
            zorder=2006
        )
        
        # Labels and title
        ax.set_xlabel('a (constant)', labelpad=10)
        ax.set_ylabel('b (linear)', labelpad=10)
        ax.set_zlabel('c (quadratic)', labelpad=10)
        ax.set_title(method_name, fontsize=16, pad=20)
        
        # Set axis limits
        ax.set_xlim(a_lim)
        ax.set_ylim(b_lim)
        ax.set_zlim(c_lim)
        
        # Reduce number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Set tick label size
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        
        # Set view angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        # Add a subtle grid
        ax.grid(True, alpha=0.3)
        
        # Add a colorbar showing density levels
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Density', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=10)
    
    # Add a super title
    fig.suptitle('3D Parameter Posterior Comparison', fontsize=20, y=0.98)
    
    # Add legend for ground truth
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=20, label='Ground Truth', linestyle='none'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2,
               label='Projections')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=14)
    
    plt.tight_layout()
    fig.savefig("examples/curvefit/figs/100_parameter_posterior_3d_comparison.pdf", 
                bbox_inches="tight", dpi=300)
    plt.close()
    
    print("✓ Saved 3D parameter posterior comparison")


# ============================================================================
# ADDITIONAL FUNCTIONS AVAILABLE IN OTHER FILES (NOT CURRENTLY USED)
# ============================================================================
# The following functions are available in other figure files but are not
# currently being used in the main workflow. They are preserved for potential
# future use or reference:
#
# From figs_backup.py:
# - save_multiple_curves_single_point_viz(): Multiple independent curves visualization
# - save_single_curve_multiple_points_viz(): Single curve with varying observation counts
# - save_programming_with_generative_functions_figure(): Code + visualization figure
# - save_vectorization_patterns_figure(): Complex vectorization patterns visualization
#
# From figs_overview_is.py:
# - generate_overview_figures(): Overview figures with importance sampling comparisons
#
# From figs_density_3d.py:
# - save_genjax_density_comparison(): 3D density surface comparisons
#
# From figs_3d_voxels.py:
# - save_parameter_posterior_3d_voxels(): 3D voxel visualization of posteriors
#
# From figs_3d_sphere.py:
# - save_parameter_posterior_3d_sphere(): 3D sphere visualization of posteriors
#
# From figs_overview.py:
# - generate_trace_tree_figure(): Trace tree visualization
# - generate_trace_visualization(): Basic trace visualization
# - generate_multiple_traces(): Multiple trace comparison
# - generate_vectorized_comparison(): Vectorization comparison figures
#
# These functions were part of exploratory visualizations that didn't make it
# into the final paper. They can be found in their respective files if needed.
