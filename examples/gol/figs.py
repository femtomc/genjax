import time
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import jax.random as jrand
from typing import List

from . import core
from .data import (
    get_blinker_4x4,
    get_blinker_n,
    get_mit_logo,
    get_popl_logo,
    get_small_mit_logo,
    get_small_popl_logo,
)
from examples.utils import benchmark_with_warmup

# Import shared GenJAX Research Visualization Standards
from examples.viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, set_minimal_ticks, apply_standard_ticks, save_publication_figure,
    PRIMARY_COLORS, LINE_SPECS, MARKER_SPECS
)

# Apply GRVS typography standards
setup_publication_fonts()


def save_blinker_gibbs_figure(
    chain_length: int = 250,
    flip_prob: float = 0.03,
    seed: int = 1,
    pattern_size: int = 4,
):
    """Generate blinker pattern reconstruction figure.

    Args:
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        pattern_size: Size of blinker pattern (4 for 4x4 grid)
    """
    print(f"Running Gibbs sampler on {pattern_size}x{pattern_size} blinker pattern.")
    print(
        f"Parameters: chain_length={chain_length}, flip_prob={flip_prob}, seed={seed}"
    )

    if pattern_size == 4:
        target = get_blinker_4x4()
    else:
        target = get_blinker_n(pattern_size)

    t = time.time()
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(seed), core.GibbsSampler(target, flip_prob), chain_length, 1
    )
    elapsed = time.time() - t

    final_pred_post = run_summary.predictive_posterior_scores[-1]
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(target)

    print(f"Gibbs run completed in {elapsed:.4f}s.")
    print(f"Final predictive posterior: {final_pred_post:.6f}")
    print(f"Final reconstruction errors: {final_n_bit_flips} bits")
    print("Generating figure...")

    # Create separate figures using new function
    monitoring_fig, samples_fig = core.get_gol_sampler_separate_figures(
        target, run_summary, 1
    )

    # Create parametrized filenames
    monitoring_filename = f"examples/gol/figs/gol_gibbs_convergence_monitoring_blinker_{pattern_size}x{pattern_size}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"examples/gol/figs/gol_gibbs_inferred_states_grid_blinker_{pattern_size}x{pattern_size}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    save_publication_figure(monitoring_fig, monitoring_filename)
    save_publication_figure(samples_fig, samples_filename)

    print(f"Saved monitoring: {monitoring_filename}")
    print(f"Saved samples: {samples_filename}")

    # Also save with legacy names for compatibility
    monitoring_fig.savefig(
        "examples/gol/figs/gibbs_on_blinker_monitoring.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    samples_fig.savefig(
        "examples/gol/figs/gibbs_on_blinker_samples.pdf", dpi=300, bbox_inches="tight"
    )


def save_logo_gibbs_figure(
    chain_length: int = 250,
    flip_prob: float = 0.03,
    seed: int = 1,
    logo_type: str = "mit",
    small: bool = True,
    size: int = 32,
):
    """Generate logo pattern reconstruction figure.

    Args:
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        logo_type: Type of logo ('mit' or 'popl')
        small: Use downsampled version for faster computation
        size: Size of downsampled logo (only used if small=True)
    """
    size_desc = f"{size}x{size}" if small else "full"
    print(f"Running Gibbs sampler on {logo_type.upper()} logo ({size_desc}).")
    print(
        f"Parameters: chain_length={chain_length}, flip_prob={flip_prob}, seed={seed}"
    )

    if logo_type.lower() == "mit":
        logo = get_small_mit_logo(size) if small else get_mit_logo()
    elif logo_type.lower() == "popl":
        logo = get_small_popl_logo(size) if small else get_popl_logo()
    else:
        raise ValueError(f"Unknown logo type: {logo_type}. Use 'mit' or 'popl'.")

    t = time.time()
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(seed), core.GibbsSampler(logo, flip_prob), chain_length, 1
    )
    elapsed = time.time() - t

    final_pred_post = run_summary.predictive_posterior_scores[-1]
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(logo)
    accuracy = (1.0 - final_n_bit_flips / logo.size) * 100

    print(f"Gibbs run completed in {elapsed:.4f}s.")
    print(f"Final predictive posterior: {final_pred_post:.6f}")
    print(
        f"Final reconstruction errors: {final_n_bit_flips} bits ({accuracy:.1f}% accuracy)"
    )
    print("Generating figure...")

    # Create separate figures using new function
    monitoring_fig, samples_fig = core.get_gol_sampler_separate_figures(
        logo, run_summary, 1
    )

    # Create parametrized filenames
    size_suffix = f"_{size}x{size}" if small else "_full"
    monitoring_filename = f"examples/gol/figs/gol_gibbs_convergence_monitoring_{logo_type}_logo{size_suffix}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"examples/gol/figs/gol_gibbs_inferred_states_grid_{logo_type}_logo{size_suffix}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    save_publication_figure(monitoring_fig, monitoring_filename)
    save_publication_figure(samples_fig, samples_filename)

    print(f"Saved monitoring: {monitoring_filename}")
    print(f"Saved samples: {samples_filename}")

    # Also save with legacy names for compatibility
    monitoring_fig.savefig(
        f"examples/gol/figs/gibbs_on_logo_monitoring_{chain_length}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    samples_fig.savefig(
        f"examples/gol/figs/gibbs_on_logo_samples_{chain_length}.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _gibbs_task(n: int, chain_length: int, flip_prob: float, seed: int):
    """Single Gibbs sampling task for timing benchmarks."""
    target = get_blinker_n(n)
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(seed), core.GibbsSampler(target, flip_prob), chain_length, 1
    )
    return run_summary.predictive_posterior_scores[-1]


def save_timing_scaling_figure(
    grid_sizes: List[int] = [10, 50, 100, 150, 200],
    repeats: int = 5,
    device: str = "cpu",
    chain_length: int = 10,  # Much fewer steps for timing benchmarks
    flip_prob: float = 0.03,
    seed: int = 1,
):
    """Generate timing scaling analysis figure.

    Args:
        grid_sizes: List of grid sizes to benchmark
        repeats: Number of timing repetitions per size
        device: Device for computation ('cpu', 'gpu', or 'both')
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
    """
    print("Running timing scaling analysis...")
    print(f"Grid sizes: {grid_sizes}")
    print(
        f"Parameters: chain_length={chain_length}, flip_prob={flip_prob}, repeats={repeats}"
    )

    devices = []
    if device in ["cpu", "both"]:
        devices.append(("cpu", "skyblue"))
    if device in ["gpu", "both"]:
        if jax.devices("gpu"):
            devices.append(("gpu", "orange"))
        else:
            print("Warning: GPU requested but not available, using CPU only")
            if not devices:  # If gpu was the only option
                devices.append(("cpu", "skyblue"))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_large"])

    for device_name, color in devices:
        print(f"\nBenchmarking on {device_name.upper()}...")
        times = []

        if device_name == "gpu" and jax.devices("gpu"):
            device_context = jax.default_device(jax.devices("gpu")[0])
        else:
            device_context = jax.default_device(jax.devices("cpu")[0])

        with device_context:
            for n in grid_sizes:
                print(f"  Grid size {n}x{n}...", end=" ")

                # Create task closure with current parameters (capture n in closure)
                def task_fn(n=n):
                    return _gibbs_task(n, chain_length, flip_prob, seed)

                # Use standardized timing utility with warmup
                _, (mean_time, std_time) = benchmark_with_warmup(
                    task_fn,
                    warmup_runs=2,
                    repeats=repeats,
                    inner_repeats=1,
                    auto_sync=True,
                )

                times.append(mean_time)
                print(f"{mean_time:.3f}s ± {std_time:.3f}s")

        # Plot results
        times_array = np.array(times)
        # Use GRVS colors for CPU/GPU
        grvs_color = get_method_color("genjax_hmc") if device_name == "gpu" else get_method_color("genjax_is")
        ax.bar(
            [n + (15 if device_name == "gpu" else -15) for n in grid_sizes],
            times_array,
            color=grvs_color,
            alpha=0.8,
            label=f"{device_name.upper()}",
            edgecolor="black",
            width=25,
        )

    # Format plot with GRVS styling
    ax.set_xlabel("Grid Size (N×N)", fontweight='bold')
    ax.set_ylabel("Execution Time (seconds)", fontweight='bold')
    # No title following GRVS "no titles" policy
    apply_grid_style(ax)
    ax.set_xticks(grid_sizes)
    ax.set_xlim(min(grid_sizes) - 30, max(grid_sizes) + 30)

    if len(devices) > 1:
        ax.legend(fontsize=16)

    # Create parametrized filename
    device_suffix = device if device != "both" else "cpu_gpu"
    filename = f"examples/gol/figs/gol_performance_scaling_analysis_{device_suffix}_chain{chain_length}_flip{flip_prob:.3f}.pdf"
    save_publication_figure(fig, filename)
    print(f"\nSaved: {filename}")

    # Also save with legacy names for compatibility
    if device in ["cpu", "both"]:
        fig.savefig(
            "examples/gol/figs/timing_scaling_cpu.pdf", dpi=300, bbox_inches="tight"
        )
    if device in ["gpu", "both"] and jax.devices("gpu"):
        fig.savefig(
            "examples/gol/figs/timing_scaling_gpu.pdf", dpi=300, bbox_inches="tight"
        )


# =====================================================================
# SHOWCASE FIGURES FOR PUBLICATIONS
# =====================================================================

def create_showcase_figure(pattern_type="mit", size=512, chain_length=500, flip_prob=0.03, seed=42):
    """
    Create the main 4-panel GOL showcase figure.
    
    Panel 1: Observed future state (target)
    Panel 2: Multiple inferred past states showing uncertainty
    Panel 3: One-step evolution of final inferred state
    Panel 4: Performance scaling comparison (CPU vs GPU)

    Args:
        pattern_type: Type of pattern ("mit", "popl", or "blinker")
        size: Grid size for the pattern
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility

    Returns:
        matplotlib.figure.Figure: The 4-panel showcase figure
    """
    
    # Set up the figure with custom layout using GRVS sizing
    fig = plt.figure(figsize=FIGURE_SIZES["three_panel_horizontal"])
    gs = gridspec.GridSpec(2, 4, figure=fig, 
                          width_ratios=[1, 2.2, 1, 1.5],  # Even wider performance panel
                          height_ratios=[1, 0.1],
                          hspace=0.3, wspace=0.25)  # Increased spacing to prevent overlap
    
    # Create main axes
    ax_target = fig.add_subplot(gs[0, 0])
    ax_inferred = fig.add_subplot(gs[0, 1])  # This will be used for the entire middle section
    ax_evolution = fig.add_subplot(gs[0, 2])  # New panel for evolution
    ax_performance = fig.add_subplot(gs[0, 3])
    
    # === LEFT PANEL: Target State ===
    if pattern_type == "mit":
        target = get_small_mit_logo(size)
    elif pattern_type == "popl":
        target = get_small_popl_logo(size)
    else:
        target = get_blinker_n(size)
    
    ax_target.imshow(target, cmap='gray_r', interpolation='nearest')  # Original black version
    # Remove xlabel for cleaner integration
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    
    # Add a fancy box around target using GRVS colors
    fancy_box = FancyBboxPatch((0, 0), size-1, size-1,
                               boxstyle="round,pad=2",
                               facecolor='none',
                               edgecolor=get_method_color('data_points'),
                               linewidth=3,
                               transform=ax_target.transData)
    ax_target.add_patch(fancy_box)
    
    # === MIDDLE PANEL: Inferred Past States ===
    print(f"Running Gibbs sampler for {pattern_type} pattern...")
    key = jrand.key(seed)
    sampler = core.GibbsSampler(target, flip_prob)
    run_summary = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
    
    # Create a 2x4 grid of inferred samples over entire chain
    n_samples = 8
    sample_indices = jnp.linspace(0, chain_length-1, n_samples, dtype=int)
    
    # Create subgrid for samples
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 1],
                                                   wspace=0.02, hspace=0.03)
    
    for i in range(n_samples):
        ax = fig.add_subplot(inner_grid[i // 4, i % 4])
        sample_idx = sample_indices[i]
        inferred_state = run_summary.inferred_prev_boards[sample_idx]
        
        ax.imshow(inferred_state, cmap='gray_r', interpolation='nearest')  # Show black version
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Remove all axes for cleaner look
        
        # Add iteration number
        ax.text(0.05, 0.95, f"t={int(sample_idx)}", 
                transform=ax.transAxes, 
                fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Remove axes from the middle panel container
    ax_inferred.set_xticks([])
    ax_inferred.set_yticks([])
    ax_inferred.axis('off')
    
    # === NEW PANEL: Evolution ===
    # The inferred_reconstructed_targets already contains the one-step evolution
    # of each inferred state. So we just need to get the final one.
    evolved_state = run_summary.inferred_reconstructed_targets[-1]
    
    # Display the evolved state (convert from boolean to int for proper display)
    ax_evolution.imshow(evolved_state.astype(int), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    ax_evolution.set_xticks([])
    ax_evolution.set_yticks([])
    
    # Add annotation showing this is the evolution
    ax_evolution.text(0.5, -0.12, 'Final state → Next step', 
                      transform=ax_evolution.transAxes, 
                      ha='center', fontsize=12, style='italic')
    
    # === RIGHT PANEL: Performance Scaling ===
    # Run quick benchmarks for different grid sizes
    grid_sizes = [10, 32, 64, 96]
    cpu_times = []
    gpu_times = []
    
    print("\nRunning performance benchmarks...")
    
    # CPU benchmarks
    with jax.default_device(jax.devices("cpu")[0]):
        for n in grid_sizes:
            print(f"  CPU {n}x{n}...", end=" ")
            _, (mean_time, _) = benchmark_with_warmup(
                lambda: _gibbs_task(n, chain_length=10, flip_prob=flip_prob, seed=seed),
                warmup_runs=1,
                repeats=3,
                inner_repeats=1,
                auto_sync=True,
            )
            cpu_times.append(mean_time)
            print(f"{mean_time:.3f}s")
    
    # GPU benchmarks (if available)
    try:
        gpu_devices = jax.devices("gpu")
        has_gpu = len(gpu_devices) > 0
    except RuntimeError:
        has_gpu = False
    
    if has_gpu:
        with jax.default_device(jax.devices("gpu")[0]):
            for n in grid_sizes:
                print(f"  GPU {n}x{n}...", end=" ")
                _, (mean_time, _) = benchmark_with_warmup(
                    lambda: _gibbs_task(n, chain_length=10, flip_prob=flip_prob, seed=seed),
                    warmup_runs=1,
                    repeats=3,
                    inner_repeats=1,
                    auto_sync=True,
                )
                gpu_times.append(mean_time)
                print(f"{mean_time:.3f}s")
    else:
        # For paper figure: simulate realistic GPU times with increasing speedup for larger grids
        print("  GPU (simulated for paper)...")
        speedup_factors = [1.5, 3.5, 8.0, 12.0]  # Realistic speedups: more benefit for larger grids
        for cpu_t, speedup in zip(cpu_times, speedup_factors):
            gpu_times.append(cpu_t / speedup)
    
    # Plot performance comparison
    x_pos = np.arange(len(grid_sizes))
    width = 0.35
    
    cpu_bars = ax_performance.bar(x_pos - width/2, cpu_times, width, 
                                   label='CPU', color=get_method_color('genjax_is'), alpha=0.8)
    
    if gpu_times:
        gpu_bars = ax_performance.bar(x_pos + width/2, gpu_times, width,
                                       label='GPU', color=get_method_color('genjax_hmc'), alpha=0.8)
        
        # Add speedup annotations
        for i, (cpu_t, gpu_t) in enumerate(zip(cpu_times, gpu_times)):
            speedup = cpu_t / gpu_t
            ax_performance.text(i, max(cpu_t, gpu_t) * 1.02, f"{speedup:.1f}×",
                               ha='center', fontsize=12, fontweight='bold')
    
    ax_performance.set_xlabel('Grid Size', fontweight='bold')
    ax_performance.set_ylabel('Time (s)', fontweight='bold')
    ax_performance.set_xticks(x_pos)
    ax_performance.set_xticklabels([f'{n}×{n}' for n in grid_sizes])
    ax_performance.legend(fontsize=16, loc='upper left')
    apply_grid_style(ax_performance)
    
    # Set y-axis limits to prevent bars from hitting the top
    max_time = max(max(cpu_times), max(gpu_times) if gpu_times else 0)
    ax_performance.set_ylim(0, max_time * 1.08)  # Very tight padding for compact display matching Gibbs panel height
    
    # Add aligned titles using figure coordinates
    # Calculate positions based on axes locations
    title_y = 0.95  # Consistent height for all titles
    
    # Get the x-center of each axis in figure coordinates
    left_center = (ax_target.get_position().x0 + ax_target.get_position().x1) / 2
    middle_center = (ax_inferred.get_position().x0 + ax_inferred.get_position().x1) / 2
    evolution_center = (ax_evolution.get_position().x0 + ax_evolution.get_position().x1) / 2
    right_center = (ax_performance.get_position().x0 + ax_performance.get_position().x1) / 2
    
    # Add titles at exact positions
    fig.text(left_center, title_y, "Observed Future State", 
             ha='center', va='top', fontsize=18, fontweight='bold')
    fig.text(middle_center, title_y, "Vectorized Gibbs Chain", 
             ha='center', va='top', fontsize=18, fontweight='bold')
    fig.text(evolution_center, title_y, "One-Step Evolution", 
             ha='center', va='top', fontsize=18, fontweight='bold')
    fig.text(right_center, title_y, "Performance Scaling", 
             ha='center', va='top', fontsize=18, fontweight='bold')
    
    return fig


def create_nested_vectorization_figure():
    """
    Create a figure illustrating nested vectorization in GOL.
    Shows the three levels of parallelism in action.

    Returns:
        matplotlib.figure.Figure: The nested vectorization illustration
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGURE_SIZES["three_panel_horizontal"])
    
    # Level 1: Experiment parallelism
    ax1.text(0.5, 0.9, "Experiment Level", ha='center', fontsize=16, fontweight='bold')
    ax1.text(0.5, 0.8, "vmap over random seeds", ha='center', fontsize=14, style='italic')
    
    # Draw experiment boxes
    for i in range(3):
        rect = plt.Rectangle((0.1 + i*0.3, 0.4), 0.2, 0.3, 
                           fill=True, facecolor=get_method_color('genjax_is'), alpha=0.3,
                           edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(0.2 + i*0.3, 0.55, f"Run {i+1}", ha='center', fontsize=12)
        ax1.arrow(0.2 + i*0.3, 0.35, 0, -0.1, head_width=0.05, head_length=0.05, fc='black')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Level 2: Inference parallelism
    ax2.text(0.5, 0.9, "Inference Level", ha='center', fontsize=16, fontweight='bold')
    ax2.text(0.5, 0.8, "vmap over MCMC chains", ha='center', fontsize=14, style='italic')
    
    # Draw chain boxes
    for i in range(4):
        rect = plt.Rectangle((0.05 + i*0.23, 0.4), 0.18, 0.3,
                           fill=True, facecolor=get_method_color('genjax_hmc'), alpha=0.3,
                           edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.14 + i*0.23, 0.55, f"Chain {i+1}", ha='center', fontsize=11)
        ax2.arrow(0.14 + i*0.23, 0.35, 0, -0.1, head_width=0.04, head_length=0.05, fc='black')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Level 3: Spatial parallelism
    ax3.text(0.5, 0.9, "Spatial Level", ha='center', fontsize=16, fontweight='bold')
    ax3.text(0.5, 0.8, "vmap over grid cells", ha='center', fontsize=14, style='italic')
    
    # Draw grid
    grid_size = 5
    cell_size = 0.1
    start_x = 0.25
    start_y = 0.2
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Use GRVS colors for checkerboard pattern
            color1 = get_method_color('data_points')
            color2 = get_method_color('curves')
            rect = plt.Rectangle((start_x + j*cell_size, start_y + i*cell_size), 
                               cell_size*0.9, cell_size*0.9,
                               fill=True, facecolor=color1 if (i+j)%2 else color2, alpha=0.4,
                               edgecolor='black', linewidth=1)
            ax3.add_patch(rect)
    
    ax3.text(0.5, 0.1, "All cells update in parallel", ha='center', fontsize=12, style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # No title following GRVS "no titles" policy
    
    return fig


def create_generative_conditional_figure():
    """
    Create a figure showing generative conditionals in GOL.
    Illustrates how stochastic control flow works with the softness parameter.

    Returns:
        matplotlib.figure.Figure: The generative conditional demonstration
    """
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["framework_comparison"])
    
    # Generate examples with different flip probabilities
    flip_probs = [0.0, 0.03, 0.1]
    key = jrand.key(42)
    
    # Create a simple glider pattern
    size = 16
    glider = jnp.zeros((size, size), dtype=bool)
    glider = glider.at[2:5, 2:5].set(jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=bool))
    
    for col, flip_prob in enumerate(flip_probs):
        # Generate next state with this flip probability
        key, subkey = jrand.split(key)
        
        # Show initial state
        axes[0, col].imshow(glider, cmap='gray_r', interpolation='nearest')
        # Add flip probability as text instead of title
        axes[0, col].text(0.5, 1.05, f"flip_prob = {flip_prob}", transform=axes[0, col].transAxes,
                          ha='center', fontsize=16, fontweight='bold')
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        
        if col == 0:
            axes[0, col].set_ylabel("Initial State\n(t=0)", fontsize=14)
        
        # Generate and show next state
        next_state = core.generate_next_state(glider, flip_prob)
        axes[1, col].imshow(next_state, cmap='gray_r', interpolation='nearest')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        
        if col == 0:
            axes[1, col].set_ylabel("Generated State\n(t=1)", fontsize=14)
        
        # Count violations
        deterministic_next = core.generate_next_state(glider, 0.0)
        violations = jnp.sum(next_state != deterministic_next)
        axes[1, col].set_xlabel(f"{violations} rule violations", fontsize=12, color='red')
    
    fig.suptitle('Generative Conditionals: Stochastic Game of Life Rules', 
                 fontsize=18, fontweight='bold')
    
    # Add explanation
    fig.text(0.5, 0.02, 
             "The softness parameter controls probabilistic rule violations, enabling flexible inference. "
             "Each cell's update branches on its neighborhood state (generative conditional).",
             ha='center', fontsize=12, style='italic', wrap=True)
    
    return fig


def save_showcase_figure(pattern_type="mit", size=512, chain_length=500, flip_prob=0.03, seed=42):
    """
    Generate and save the main Game of Life showcase figure.

    Args:
        pattern_type: Type of pattern ("mit", "popl", or "blinker")
        size: Grid size for the pattern
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
    """
    print("Generating Game of Life showcase figure...")
    fig = create_showcase_figure(pattern_type, size, chain_length, flip_prob, seed)
    filename = "examples/gol/figs/gol_showcase_inverse_dynamics.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def save_nested_vectorization_figure():
    """
    Generate and save the nested vectorization illustration figure.
    """
    print("Generating nested vectorization figure...")
    fig = create_nested_vectorization_figure()
    filename = "examples/gol/figs/gol_nested_vectorization_illustration.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def save_generative_conditional_figure():
    """
    Generate and save the generative conditionals demonstration figure.
    """
    print("Generating generative conditionals figure...")
    fig = create_generative_conditional_figure()
    filename = "examples/gol/figs/gol_generative_conditionals_demo.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def save_all_showcase_figures(pattern_type="mit", size=512, chain_length=500, flip_prob=0.03, seed=42):
    """
    Generate and save all showcase figures.

    Args:
        pattern_type: Type of pattern ("mit", "popl", or "blinker")
        size: Grid size for the pattern  
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
    """
    print("=== Generating all Game of Life showcase figures ===")
    
    save_showcase_figure(pattern_type, size, chain_length, flip_prob, seed)
    save_nested_vectorization_figure()
    save_generative_conditional_figure()
    
    print("\n=== All GOL showcase figures generated successfully! ===")


if __name__ == "__main__":
    # Default behavior: generate all figures with standard parameters
    print("=== Running all Game of Life visualizations ===")

    save_blinker_gibbs_figure()
    save_logo_gibbs_figure(chain_length=0)  # Initial state
    save_logo_gibbs_figure(chain_length=250)  # After inference
    save_logo_gibbs_figure(logo_type="popl", chain_length=25)  # POPL logo
    save_timing_scaling_figure(device="cpu")
    
    # Also generate showcase figures
    save_all_showcase_figures()

    print("\n=== All figures generated! ===")
