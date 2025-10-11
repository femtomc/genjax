import time
import json
import jax
import jax.numpy as jnp
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
    get_popl_logo_white_lambda,
    get_small_popl_logo_white_lambda,
    get_hermes_logo,
    get_small_hermes_logo,
    get_wizards_logo,
    get_small_wizards_logo,
)
from genjax.timing import benchmark_with_warmup

# Import shared GenJAX Research Visualization Standards
from genjax.viz.standard import (
    setup_publication_fonts,
    FIGURE_SIZES,
    get_method_color,
    apply_grid_style,
    set_minimal_ticks,
    apply_standard_ticks,
    save_publication_figure,
    PRIMARY_COLORS,
    LINE_SPECS,
    MARKER_SPECS,
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
    monitoring_filename = f"figs/gol_gibbs_convergence_monitoring_blinker_{pattern_size}x{pattern_size}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"figs/gol_gibbs_inferred_states_grid_blinker_{pattern_size}x{pattern_size}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    save_publication_figure(monitoring_fig, monitoring_filename)
    save_publication_figure(samples_fig, samples_filename)

    print(f"Saved monitoring: {monitoring_filename}")
    print(f"Saved samples: {samples_filename}")

    # Also save with legacy names for compatibility
    monitoring_fig.savefig(
        "figs/gibbs_on_blinker_monitoring.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    samples_fig.savefig(
        "figs/gibbs_on_blinker_samples.pdf", dpi=300, bbox_inches="tight"
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
    monitoring_filename = f"figs/gol_gibbs_convergence_monitoring_{logo_type}_logo{size_suffix}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"figs/gol_gibbs_inferred_states_grid_{logo_type}_logo{size_suffix}_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    save_publication_figure(monitoring_fig, monitoring_filename)
    save_publication_figure(samples_fig, samples_filename)

    print(f"Saved monitoring: {monitoring_filename}")
    print(f"Saved samples: {samples_filename}")

    # Also save with legacy names for compatibility
    monitoring_fig.savefig(
        f"figs/gibbs_on_logo_monitoring_{chain_length}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    samples_fig.savefig(
        f"figs/gibbs_on_logo_samples_{chain_length}.pdf",
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
        grvs_color = (
            get_method_color("genjax_hmc")
            if device_name == "gpu"
            else get_method_color("genjax_is")
        )
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
    ax.set_xlabel("Grid Size (N×N)", fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontweight="bold")
    # No title following GRVS "no titles" policy
    apply_grid_style(ax)
    ax.set_xticks(grid_sizes)
    ax.set_xlim(min(grid_sizes) - 30, max(grid_sizes) + 30)

    if len(devices) > 1:
        ax.legend(fontsize=16)

    # Create parametrized filename
    device_suffix = device if device != "both" else "cpu_gpu"
    filename = f"figs/gol_performance_scaling_analysis_{device_suffix}_chain{chain_length}_flip{flip_prob:.3f}.pdf"
    save_publication_figure(fig, filename)
    print(f"\nSaved: {filename}")

    # Also save with legacy names for compatibility
    if device in ["cpu", "both"]:
        fig.savefig(
            "figs/timing_scaling_cpu.pdf", dpi=300, bbox_inches="tight"
        )
    if device in ["gpu", "both"] and jax.devices("gpu"):
        fig.savefig(
            "figs/timing_scaling_gpu.pdf", dpi=300, bbox_inches="tight"
        )


# =====================================================================
# SHOWCASE FIGURES FOR PUBLICATIONS
# =====================================================================


def create_showcase_figure(
    pattern_type="mit", size=256, chain_length=150, flip_prob=0.03, seed=42,
    white_lambda=False, load_from_file=None
):
    """
    Create the main 3-panel GOL showcase figure.

    Panel 1: Observed future state (target)
    Panel 2: Multiple inferred past states showing uncertainty
    Panel 3: One-step evolution of final inferred state

    Args:
        pattern_type: Type of pattern ("mit", "popl", "blinker", "hermes", "wizards")
        size: Grid size for the pattern (default 256x256)
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        white_lambda: Whether to use white lambda version of POPL logo
        load_from_file: Path to saved experiment data (if None, runs new experiment)

    Returns:
        matplotlib.figure.Figure: The 3-panel showcase figure
    """

    # Set up the figure with 3 panels in a single row
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1, 1, 1],
        wspace=0.15,
    )

    # Create main axes
    ax_target = fig.add_subplot(gs[0, 0])
    ax_inferred = fig.add_subplot(gs[0, 1])
    ax_evolution = fig.add_subplot(gs[0, 2])

    # Load data from file or run new experiment
    import json
    
    if load_from_file:
        print(f"Loading experiment data from: {load_from_file}")
        with open(load_from_file, 'r') as f:
            exp_data = json.load(f)
        
        # Extract data from saved experiment
        target = jnp.array(exp_data["target"])
        chain_length = exp_data["metadata"]["chain_length"]
        
        # Create a mock run_summary object with the loaded data
        class MockRunSummary:
            def __init__(self, data):
                self.predictive_posterior_scores = jnp.array(data["predictive_posterior_scores"])
                self.inferred_prev_boards = jnp.array(data["inferred_prev_boards"])
                self.inferred_reconstructed_targets = jnp.array(data["inferred_reconstructed_targets"])
                self._final_n_bit_flips = data["metadata"]["final_n_bit_flips"]
            
            def n_incorrect_bits_in_reconstructed_image(self, target):
                return self._final_n_bit_flips
        
        run_summary = MockRunSummary(exp_data)
        final_pred_post = exp_data["metadata"]["final_pred_post"]
        accuracy = exp_data["metadata"]["final_accuracy"]
        
    else:
        # === LEFT PANEL: Target State ===
        if pattern_type == "mit":
            target = get_small_mit_logo(size)
        elif pattern_type == "popl":
            if white_lambda:
                target = get_small_popl_logo_white_lambda(size)
            else:
                target = get_small_popl_logo(size)
        elif pattern_type == "hermes":
            target = get_small_hermes_logo(size)
        elif pattern_type == "wizards":
            target = get_small_wizards_logo(size)
        else:
            target = get_blinker_n(size)
            
        # Run new experiment
        print(f"Running Gibbs sampler for {pattern_type} pattern...")
        key = jrand.key(seed)
        sampler = core.GibbsSampler(target, flip_prob)
        run_summary = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
        
        # Calculate metrics
        final_pred_post = run_summary.predictive_posterior_scores[-1]
        final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(target)
        accuracy = (1 - final_n_bit_flips / target.size) * 100

    ax_target.imshow(
        target, cmap="gray_r", interpolation="nearest"
    )  # Original black version
    # Remove xlabel for cleaner integration
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    ax_target.set_aspect('equal', 'box')

    # Add thick red border to highlight this is the observed state (like in schematic)
    for spine in ax_target.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)

    # === MIDDLE PANEL: Inferred Past States ===

    # Create a 2x2 grid of inferred samples over entire chain
    n_samples = 4
    sample_indices = jnp.linspace(0, chain_length - 1, n_samples, dtype=int)

    # Create subgrid for samples with padding
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 2, 
        subplot_spec=gs[0, 1], 
        wspace=0.08, hspace=0.08  # Tighter spacing for 2x2 grid
    )

    for i in range(n_samples):
        ax = fig.add_subplot(inner_grid[i // 2, i % 2])
        sample_idx = sample_indices[i]
        inferred_state = run_summary.inferred_prev_boards[sample_idx]

        ax.imshow(
            inferred_state, cmap="gray_r", interpolation="nearest"
        )  # Show black version
        
        # Shrink the cells by adding padding
        padding = 0.08  # Fraction of image size to pad
        img_size = inferred_state.shape[0]
        pad_size = img_size * padding
        ax.set_xlim(-pad_size, img_size + pad_size)
        ax.set_ylim(img_size + pad_size, -pad_size)  # Inverted for image coordinates
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")  # Remove all axes for cleaner look

        # Add iteration number
        ax.text(
            0.05,
            0.95,
            f"t={int(sample_idx)}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        
        # Add green border to the final state (t=499)
        if i == n_samples - 1:  # Last sample in the 2x2 grid
            # Create a rectangle patch for the border with dashed line
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fill=False, edgecolor='green', linewidth=4,
                           linestyle='--')
            ax.add_patch(rect)

    # Remove axes from the middle panel container
    ax_inferred.set_xticks([])
    ax_inferred.set_yticks([])
    ax_inferred.axis("off")
    ax_inferred.set_aspect('equal', 'box')

    # === NEW PANEL: Evolution ===
    # The inferred_reconstructed_targets already contains the one-step evolution
    # of each inferred state. So we just need to get the final one.
    evolved_state = run_summary.inferred_reconstructed_targets[-1]

    # Display the evolved state (convert from boolean to int for proper display)
    ax_evolution.imshow(
        evolved_state.astype(int),
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_evolution.set_xticks([])
    ax_evolution.set_yticks([])
    ax_evolution.set_aspect('equal', 'box')

    # Add annotation showing this is the evolution
    ax_evolution.text(
        0.5,
        -0.12,
        "Final state → Next step",
        transform=ax_evolution.transAxes,
        ha="center",
        fontsize=12,
        style="italic",
    )

    # Print summary statistics (already calculated above)
    print(f"\nFinal predictive posterior: {final_pred_post:.6f}")
    print(f"Final reconstruction errors: {final_n_bit_flips} bits ({accuracy:.1f}% accuracy)")
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(target)

    # Add aligned titles using figure coordinates
    # Calculate positions based on axes locations
    title_y = 0.93  # Tighter to subplots for better integration

    # Get the x-center of each axis in figure coordinates
    left_center = (ax_target.get_position().x0 + ax_target.get_position().x1) / 2
    middle_center = (ax_inferred.get_position().x0 + ax_inferred.get_position().x1) / 2
    evolution_center = (
        ax_evolution.get_position().x0 + ax_evolution.get_position().x1
    ) / 2

    # Add titles at exact positions
    fig.text(
        left_center,
        title_y,
        "Observed State",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        middle_center,
        title_y,
        "Inversion via Gibbs",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        evolution_center,
        title_y,
        "One-Step Evolution",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )

    return fig




def save_timing_bar_plot(grid_sizes=[64, 128, 256, 512], chain_length=10, flip_prob=0.03, repeats=3):
    """
    Generate timing data and save the standalone timing bar plot.

    Args:
        grid_sizes: List of grid sizes to benchmark
        chain_length: Number of Gibbs steps (use fewer for timing)
        flip_prob: Probability of rule violations
        repeats: Number of timing repetitions
    """
    print("Generating timing bar plot...")

    setup_publication_fonts()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Run actual timing benchmarks
    print(f"  Benchmarking CPU at grid sizes: {grid_sizes}")
    cpu_times_ms = []

    for n in grid_sizes:
        def task_fn():
            return _gibbs_task(n, chain_length, flip_prob, seed=1)

        _, (mean_time, std_time) = benchmark_with_warmup(
            task_fn,
            warmup_runs=2,
            repeats=repeats,
            inner_repeats=1,
            auto_sync=True,
        )
        cpu_times_ms.append(mean_time * 1000)  # Convert to ms
        print(f"    {n}×{n}: {mean_time*1000:.1f} ms")

    # Check if GPU available
    try:
        gpu_available = len(jax.devices("gpu")) > 0
    except RuntimeError:
        gpu_available = False
    gpu_times_ms = []

    if gpu_available:
        print(f"  Benchmarking GPU at grid sizes: {grid_sizes}")
        with jax.default_device(jax.devices("gpu")[0]):
            for n in grid_sizes:
                def task_fn():
                    return _gibbs_task(n, chain_length, flip_prob, seed=1)

                _, (mean_time, std_time) = benchmark_with_warmup(
                    task_fn,
                    warmup_runs=2,
                    repeats=repeats,
                    inner_repeats=1,
                    auto_sync=True,
                )
                gpu_times_ms.append(mean_time * 1000)
                print(f"    {n}×{n}: {mean_time*1000:.1f} ms")

    # Reverse order - smallest at top, largest at bottom
    sizes = list(reversed(grid_sizes))
    cpu_times_ms = list(reversed(cpu_times_ms))
    if gpu_available:
        gpu_times_ms = list(reversed(gpu_times_ms))

    # Create horizontal bar plot
    y_pos = np.arange(len(sizes))
    bar_height = 0.35

    if gpu_available:
        bars_gpu = ax.barh(y_pos + bar_height/2, gpu_times_ms, bar_height,
                          label='GPU', color=get_method_color("genjax_hmc"), alpha=0.8)
        bars_cpu = ax.barh(y_pos - bar_height/2, cpu_times_ms, bar_height,
                          label='CPU', color=get_method_color("genjax_is"), alpha=0.8)

        # Add speedup annotations
        for i, (bar_cpu, bar_gpu) in enumerate(zip(bars_cpu, bars_gpu)):
            speedup = cpu_times_ms[i] / gpu_times_ms[i]
            ax.text(bar_cpu.get_width() + max(cpu_times_ms)*0.02, bar_cpu.get_y() + bar_cpu.get_height()/2,
                   f'{speedup:.1f}×', ha='left', va='center',
                   fontsize=16, fontweight='bold', color=get_method_color("data_points"))
    else:
        bars_cpu = ax.barh(y_pos, cpu_times_ms, bar_height,
                          label='CPU', color=get_method_color("genjax_is"), alpha=0.8)

    # Format axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{s}×{s}' for s in sizes], fontsize=16)
    ax.set_xlabel('Time per Gibbs sweep (ms)', fontsize=18, fontweight='bold')
    ax.set_xlim(0, max(cpu_times_ms) * (1.25 if gpu_available else 1.15))

    if gpu_available:
        ax.legend(loc='upper center', fontsize=16, ncol=2, frameon=True,
                 bbox_to_anchor=(0.5, 1.05), columnspacing=2)

    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.95, "Game of Life Gibbs Sampling Performance",
            ha='center', fontsize=20, fontweight='bold')

    plt.tight_layout()

    filename = "figs/gol_gibbs_timing_bar_plot.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return filename


def create_nested_vectorization_figure():
    """
    Create a figure illustrating nested vectorization in GOL.
    Shows the three levels of parallelism in action.

    Returns:
        matplotlib.figure.Figure: The nested vectorization illustration
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=FIGURE_SIZES["three_panel_horizontal"]
    )

    # Level 1: Experiment parallelism
    ax1.text(0.5, 0.9, "Experiment Level", ha="center", fontsize=16, fontweight="bold")
    ax1.text(
        0.5, 0.8, "vmap over random seeds", ha="center", fontsize=14, style="italic"
    )

    # Draw experiment boxes
    for i in range(3):
        rect = plt.Rectangle(
            (0.1 + i * 0.3, 0.4),
            0.2,
            0.3,
            fill=True,
            facecolor=get_method_color("genjax_is"),
            alpha=0.3,
            edgecolor="black",
            linewidth=2,
        )
        ax1.add_patch(rect)
        ax1.text(0.2 + i * 0.3, 0.55, f"Run {i + 1}", ha="center", fontsize=12)
        ax1.arrow(
            0.2 + i * 0.3, 0.35, 0, -0.1, head_width=0.05, head_length=0.05, fc="black"
        )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Level 2: Inference parallelism
    ax2.text(0.5, 0.9, "Inference Level", ha="center", fontsize=16, fontweight="bold")
    ax2.text(
        0.5, 0.8, "vmap over MCMC chains", ha="center", fontsize=14, style="italic"
    )

    # Draw chain boxes
    for i in range(4):
        rect = plt.Rectangle(
            (0.05 + i * 0.23, 0.4),
            0.18,
            0.3,
            fill=True,
            facecolor=get_method_color("genjax_hmc"),
            alpha=0.3,
            edgecolor="black",
            linewidth=2,
        )
        ax2.add_patch(rect)
        ax2.text(0.14 + i * 0.23, 0.55, f"Chain {i + 1}", ha="center", fontsize=11)
        ax2.arrow(
            0.14 + i * 0.23,
            0.35,
            0,
            -0.1,
            head_width=0.04,
            head_length=0.05,
            fc="black",
        )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Level 3: Spatial parallelism
    ax3.text(0.5, 0.9, "Spatial Level", ha="center", fontsize=16, fontweight="bold")
    ax3.text(0.5, 0.8, "vmap over grid cells", ha="center", fontsize=14, style="italic")

    # Draw grid
    grid_size = 5
    cell_size = 0.1
    start_x = 0.25
    start_y = 0.2

    for i in range(grid_size):
        for j in range(grid_size):
            # Use GRVS colors for checkerboard pattern
            color1 = get_method_color("data_points")
            color2 = get_method_color("curves")
            rect = plt.Rectangle(
                (start_x + j * cell_size, start_y + i * cell_size),
                cell_size * 0.9,
                cell_size * 0.9,
                fill=True,
                facecolor=color1 if (i + j) % 2 else color2,
                alpha=0.4,
                edgecolor="black",
                linewidth=1,
            )
            ax3.add_patch(rect)

    ax3.text(
        0.5,
        0.1,
        "All cells update in parallel",
        ha="center",
        fontsize=12,
        style="italic",
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # No title following GRVS "no titles" policy

    return fig


def create_generative_conditional_figure():
    """
    Create a figure showing generative conditionals in GOL.
    Illustrates how generative conditionals works with the softness parameter.

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
    glider = glider.at[2:5, 2:5].set(
        jnp.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool)
    )

    for col, flip_prob in enumerate(flip_probs):
        # Generate next state with this flip probability
        key, subkey = jrand.split(key)

        # Show initial state
        axes[0, col].imshow(glider, cmap="gray_r", interpolation="nearest")
        # Add flip probability as text instead of title
        axes[0, col].text(
            0.5,
            1.05,
            f"flip_prob = {flip_prob}",
            transform=axes[0, col].transAxes,
            ha="center",
            fontsize=16,
            fontweight="bold",
        )
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        if col == 0:
            axes[0, col].set_ylabel("Initial State\n(t=0)", fontsize=14)

        # Generate and show next state
        next_state = core.generate_next_state(glider, flip_prob)
        axes[1, col].imshow(next_state, cmap="gray_r", interpolation="nearest")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        if col == 0:
            axes[1, col].set_ylabel("Generated State\n(t=1)", fontsize=14)

        # Count violations
        deterministic_next = core.generate_next_state(glider, 0.0)
        violations = jnp.sum(next_state != deterministic_next)
        axes[1, col].set_xlabel(
            f"{violations} rule violations", fontsize=12, color="red"
        )

    fig.suptitle(
        "Generative Conditionals: Stochastic Game of Life Rules",
        fontsize=18,
        fontweight="bold",
    )

    # Add explanation
    fig.text(
        0.5,
        0.02,
        "The softness parameter controls probabilistic rule violations, enabling flexible inference. "
        "Each cell's update branches on its neighborhood state (generative conditional).",
        ha="center",
        fontsize=12,
        style="italic",
        wrap=True,
    )

    return fig


def save_showcase_figure(
    pattern_type="mit",
    size=256,
    chain_length=150,
    flip_prob=0.03,
    seed=42,
    load_from_file=None,
    output_label=None,
):
    """
    Generate and save the main Game of Life showcase figure.

    Args:
        pattern_type: Type of pattern ("mit", "popl", "blinker", "hermes", "wizards")
        size: Grid size for the pattern
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        load_from_file: Path to saved experiment data (if None, runs new experiment)
        output_label: Optional label to use in the output filename
    """
    print("Generating Game of Life showcase figure...")
    fig = create_showcase_figure(
        pattern_type, size, chain_length, flip_prob, seed, load_from_file=load_from_file
    )
    label = output_label if output_label is not None else size
    filename = f"figs/gol_integrated_showcase_{pattern_type}_{label}.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def save_nested_vectorization_figure():
    """
    Generate and save the nested vectorization illustration figure.
    """
    print("Generating nested vectorization figure...")
    fig = create_nested_vectorization_figure()
    filename = "figs/gol_nested_vectorization_illustration.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def save_generative_conditional_figure():
    """
    Generate and save the generative conditionals demonstration figure.
    """
    print("Generating generative conditionals figure...")
    fig = create_generative_conditional_figure()
    filename = "figs/gol_generative_conditionals_demo.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def create_forward_inverse_schematic(flip_prob=0.03, gibbs_steps=250, seed=42):
    """
    Create a schematic diagram illustrating the forward and inverse problems in Game of Life.
    Shows the probabilistic forward model and uses Gibbs sampling to find multiple preimages.
    
    Args:
        flip_prob: Probability of rule violations in forward model
        gibbs_steps: Number of Gibbs steps to run for finding preimages
        seed: Random seed for reproducibility
        
    Returns:
        matplotlib.figure.Figure: The schematic diagram
    """
    fig = plt.figure(figsize=(14, 5))
    
    # Create main grid for single-row layout
    gs = gridspec.GridSpec(1, 7, figure=fig, wspace=0.25,
                          width_ratios=[1.2, 0.8, 1.2, 0.8, 1.2, 0.4, 2.2])
    
    # Main row elements
    ax_initial = fig.add_subplot(gs[0, 0])
    ax_forward_arrow = fig.add_subplot(gs[0, 1])
    ax_final = fig.add_subplot(gs[0, 2])
    ax_inverse_arrow = fig.add_subplot(gs[0, 3])
    ax_inferred = fig.add_subplot(gs[0, 4])
    ax_spacer = fig.add_subplot(gs[0, 5])
    ax_alternatives = fig.add_subplot(gs[0, 6])
    
    # Use a more interesting pattern - let's use a 6x6 pattern with more structure
    # Create a small glider gun or R-pentomino pattern
    initial_state = jnp.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=int)
    
    # Apply GoL rules with noise to get observed state
    key = jrand.key(seed)
    key, subkey = jrand.split(key)
    from genjax import seed as genjax_seed
    observed_trace = genjax_seed(core.generate_next_state.simulate)(subkey, initial_state, flip_prob)
    observed_state = observed_trace.get_retval()
    
    # Count flipped bits
    deterministic_next = core.generate_next_state(initial_state, 0.0)
    num_flips = jnp.sum(observed_state != deterministic_next)
    
    # === INITIAL STATE ===
    ax_initial.imshow(initial_state, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_initial.set_xticks([])
    ax_initial.set_yticks([])
    # Add solid black frame
    for spine in ax_initial.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2.5)
    
    # === FORWARD ARROW ===
    ax_forward_arrow.axis('off')
    ax_forward_arrow.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                             arrowprops=dict(arrowstyle='->', lw=5, color='green'),
                             xycoords='axes fraction')
    ax_forward_arrow.text(0.5, 0.25, 'forward', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='green',
                         transform=ax_forward_arrow.transAxes)
    
    # === OBSERVED STATE ===
    ax_final.imshow(observed_state, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_final.set_xticks([])
    ax_final.set_yticks([])
    
    # Highlight cells that were flipped due to noise
    if num_flips > 0:
        flipped_mask = observed_state != deterministic_next
        for i in range(initial_state.shape[0]):
            for j in range(initial_state.shape[1]):
                if flipped_mask[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       fill=False, edgecolor='orange', 
                                       linewidth=2.5, linestyle='--')
                    ax_final.add_patch(rect)
    
    # Add thick red frame to highlight this is the observed state
    for spine in ax_final.spines.values():
        spine.set_color('red')
        spine.set_linewidth(3.5)
    
    # === INVERSE ARROW ===
    ax_inverse_arrow.axis('off')
    ax_inverse_arrow.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                             arrowprops=dict(arrowstyle='->', lw=5, color='red'),
                             xycoords='axes fraction')
    ax_inverse_arrow.text(0.5, 0.25, 'inference', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='red',
                         transform=ax_inverse_arrow.transAxes)
    
    # === RUN GIBBS SAMPLING ===
    # Use the same approach as in save_blinker_gibbs_figure
    key, subkey = jrand.split(key)
    sampler = core.GibbsSampler(observed_state, flip_prob)
    run_summary = core.run_sampler_and_get_summary(
        subkey, sampler, gibbs_steps, n_steps_per_summary_frame=1
    )
    
    # Check convergence by evolving the final state forward
    final_inferred = run_summary.inferred_prev_boards[-1]
    evolved_state = core.generate_next_state(final_inferred, 0.0)
    reconstruction_error = jnp.sum(evolved_state != observed_state)
    
    # === INFERRED STATE ===
    ax_inferred.imshow(final_inferred, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_inferred.set_xticks([])
    ax_inferred.set_yticks([])
    
    # Add solid frame (matching initial state style)
    for spine in ax_inferred.spines.values():
        spine.set_color('darkred')
        spine.set_linewidth(2.5)
    
    # === SPACER ===
    ax_spacer.axis('off')
    
    # === ALTERNATIVE SOLUTIONS FROM GIBBS ===
    ax_alternatives.axis('off')
    
    # Add title in the correct position
    ax_alternatives.text(0.5, 0.9, 'Many possible\ninitial states', 
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', color='red',
                        transform=ax_alternatives.transAxes)
    
    # Create a 2x3 grid of alternative states
    grid_rows = 2
    grid_cols = 3
    cell_size = 0.28
    spacing = 0.08
    
    # Calculate positions to center the grid
    total_width = grid_cols * cell_size + (grid_cols - 1) * spacing
    total_height = grid_rows * cell_size + (grid_rows - 1) * spacing
    start_x = 0.5 - total_width / 2
    start_y = 0.4 - total_height / 2
    
    # Get 6 different states from different points in the Gibbs chain
    # Make sure they actually lead to the observed state
    n_samples = len(run_summary.inferred_prev_boards)
    
    # Select states from later in the chain (better convergence)
    # Focus on the last half of the chain where convergence is better
    start_idx = n_samples // 2
    end_idx = n_samples
    
    # Sample 6 states evenly from the converged portion
    sample_indices = []
    for i in range(6):
        idx = start_idx + int(i * (end_idx - start_idx) / 6)
        sample_indices.append(min(idx, n_samples - 1))
    
    # Verify and display the alternative states
    for idx in range(6):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Position for this cell
        x = start_x + col * (cell_size + spacing)
        y = start_y + (grid_rows - 1 - row) * (cell_size + spacing)
        
        # Create axes for this alternative
        mini_ax = fig.add_axes([
            ax_alternatives.get_position().x0 + ax_alternatives.get_position().width * x,
            ax_alternatives.get_position().y0 + ax_alternatives.get_position().height * y,
            ax_alternatives.get_position().width * cell_size,
            ax_alternatives.get_position().height * cell_size
        ])
        
        # Get the state
        state = run_summary.inferred_prev_boards[sample_indices[idx]]
        
        # Verify it leads to observed state (approximately)
        evolved = core.generate_next_state(state, 0.0)
        error = jnp.sum(evolved != observed_state)
        
        # Show the state
        mini_ax.imshow(state, cmap='gray_r', interpolation='nearest')
        mini_ax.set_xticks([])
        mini_ax.set_yticks([])
        
        # Add border with color indicating quality
        # Green-ish for good reconstruction, red-ish for poor
        border_color = 'darkgreen' if error <= 2 else 'darkred'
        for spine in mini_ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color(border_color)
    
    return fig


def save_forward_inverse_schematic():
    """Save the forward/inverse problem schematic diagram."""
    fig = create_forward_inverse_schematic()
    filename = "figs/gol_forward_inverse_problem_schematic.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved forward/inverse schematic: {filename}")
    return fig


def create_simple_inference_schematic(pattern=None, pattern_size=256):
    """
    Create a simple schematic diagram showing the inference problem:
    ? ⇄ observed state (with straight arrows for GoL forward and inference backward)
    
    Args:
        pattern: The observed pattern to show (if None, uses a simple example)
        pattern_size: Size of the pattern if loading from data
        
    Returns:
        matplotlib.figure.Figure: The schematic diagram
    """
    # Match showcase figure size and layout
    fig = plt.figure(figsize=(14, 4))  # Same width as showcase, single row height
    
    # Create grid with 3 equal columns to match showcase
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.15,
                          width_ratios=[1, 1, 1])  # Equal widths like showcase
    
    ax_unknown = fig.add_subplot(gs[0, 0])
    ax_arrows = fig.add_subplot(gs[0, 1])
    ax_observed = fig.add_subplot(gs[0, 2])
    
    # === UNKNOWN STATE (?) ===
    ax_unknown.set_xlim(0, 1)
    ax_unknown.set_ylim(0, 1)
    ax_unknown.text(0.5, 0.5, '?', fontsize=80, ha='center', va='center',
                    fontweight='bold', color='darkgray')
    ax_unknown.set_xticks([])
    ax_unknown.set_yticks([])
    ax_unknown.set_aspect('equal')
    
    # Add green dashed border to indicate unknown (will be inferred)
    for spine in ax_unknown.spines.values():
        spine.set_color('green')
        spine.set_linewidth(3)
        spine.set_linestyle('--')
    
    # === STRAIGHT ARROWS ===
    ax_arrows.axis('off')
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(0, 1)
    
    # Top arrow (inference - from observed to initial)
    ax_arrows.annotate('', xy=(0.15, 0.65), xytext=(0.85, 0.65),
                      arrowprops=dict(arrowstyle='->', lw=4, color='green'),
                      xycoords='axes fraction')
    
    # Bottom arrow (GoL forward - from initial to observed)
    ax_arrows.annotate('', xy=(0.85, 0.35), xytext=(0.15, 0.35),
                      arrowprops=dict(arrowstyle='->', lw=4, color='black'),
                      xycoords='axes fraction')
    
    # Add labels for arrows
    ax_arrows.text(0.5, 0.75, 'inference', ha='center', va='center',
                  fontsize=16, fontweight='bold', color='green',
                  transform=ax_arrows.transAxes)
    ax_arrows.text(0.5, 0.25, '@gen GoL', ha='center', va='center',
                  fontsize=16, fontweight='bold', color='black',
                  transform=ax_arrows.transAxes)
    
    # === OBSERVED STATE ===
    if pattern is None:
        # Use a simple example pattern if none provided
        pattern = jnp.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=int)
    elif isinstance(pattern, str):
        # Load pattern by name
        if pattern == "wizards":
            pattern = get_wizards_logo() if pattern_size > 256 else get_small_wizards_logo(pattern_size)
        elif pattern == "mit":
            pattern = get_mit_logo() if pattern_size > 256 else get_small_mit_logo(pattern_size)
        elif pattern == "popl":
            pattern = get_popl_logo() if pattern_size > 256 else get_small_popl_logo(pattern_size)
        elif pattern == "hermes":
            pattern = get_hermes_logo() if pattern_size > 256 else get_small_hermes_logo(pattern_size)
        else:
            pattern = get_blinker_n(pattern_size)
    
    ax_observed.imshow(pattern, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_observed.set_xticks([])
    ax_observed.set_yticks([])
    
    # Add thick red border to highlight this is the observed state
    for spine in ax_observed.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)
    
    # Add titles at top to match showcase figure style
    title_y = 0.93  # Match showcase title position
    
    # Get x-center of each axis in figure coordinates
    left_center = (ax_unknown.get_position().x0 + ax_unknown.get_position().x1) / 2
    middle_center = (ax_arrows.get_position().x0 + ax_arrows.get_position().x1) / 2
    right_center = (ax_observed.get_position().x0 + ax_observed.get_position().x1) / 2
    
    # Add titles using figure text like showcase
    fig.text(left_center, title_y, "Previous State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(middle_center, title_y, "Game of Life", ha="center", va="top", 
             fontsize=20, fontweight="bold")
    fig.text(right_center, title_y, "Observed State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    
    return fig


def save_simple_inference_schematic(pattern="wizards", pattern_size=256):
    """
    Generate and save the simple inference schematic diagram.
    
    Args:
        pattern: Pattern type or array to use as observed state
        pattern_size: Size of the pattern
    """
    print("Generating simple inference schematic...")
    fig = create_simple_inference_schematic(pattern, pattern_size)
    filename = f"figs/gol_inference_schematic_{pattern}_{pattern_size}.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return fig


def create_integrated_showcase_figure(
    pattern_type="wizards", size=256, chain_length=150, flip_prob=0.03, seed=42
):
    """
    Create an integrated single-row showcase figure with question mark, arrows, observed state, and Gibbs results.
    
    Args:
        pattern_type: Type of pattern to use
        size: Grid size for the pattern
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        
    Returns:
        matplotlib.figure.Figure: The integrated showcase figure
    """
    # Create figure with single row, 5 panels
    fig = plt.figure(figsize=(20, 4))  # Wider to accommodate 5 panels
    
    # Create grid with 1 row and 5 columns
    gs = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[1, 0.4, 1, 1, 1],  # Question mark, arrows (much smaller), observed, Gibbs grid, evolution
        wspace=0.15,  # Horizontal spacing
    )
    
    # Create axes
    ax_unknown = fig.add_subplot(gs[0, 0])      # Previous state (?)
    ax_arrows = fig.add_subplot(gs[0, 1])       # Arrows
    ax_observed = fig.add_subplot(gs[0, 2])     # Observed state
    ax_gibbs = fig.add_subplot(gs[0, 3])        # Gibbs results
    ax_evolution = fig.add_subplot(gs[0, 4])    # One-step evolution
    
    # Load pattern
    if pattern_type == "wizards":
        pattern = get_wizards_logo() if size > 256 else get_small_wizards_logo(size)
    elif pattern_type == "mit":
        pattern = get_mit_logo() if size > 256 else get_small_mit_logo(size)
    elif pattern_type == "popl":
        pattern = get_popl_logo() if size > 256 else get_small_popl_logo(size)
    elif pattern_type == "hermes":
        pattern = get_hermes_logo() if size > 256 else get_small_hermes_logo(size)
    elif pattern_type == "blinker":
        pattern = get_blinker_n(size)
    else:
        pattern = jnp.array(np.random.randint(0, 2, (size, size)))
    
    # === PANEL 1: UNKNOWN STATE (?) ===
    ax_unknown.set_xlim(0, 1)
    ax_unknown.set_ylim(0, 1)
    ax_unknown.text(0.5, 0.5, '?', fontsize=80, ha='center', va='center',
                    fontweight='bold', color='darkgray')
    ax_unknown.set_xticks([])
    ax_unknown.set_yticks([])
    ax_unknown.set_aspect('equal')
    
    # Add green dashed border
    for spine in ax_unknown.spines.values():
        spine.set_color('green')
        spine.set_linewidth(3)
        spine.set_linestyle('--')
    
    # === PANEL 2: ARROWS ===
    ax_arrows.axis('off')
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(0, 1)
    
    # Top arrow (inference - from observed to initial)
    ax_arrows.annotate('', xy=(0.15, 0.55), xytext=(0.85, 0.55),
                      arrowprops=dict(arrowstyle='->', lw=4, color='green'),
                      xycoords='axes fraction')
    
    # Bottom arrow (GoL forward - from initial to observed)
    ax_arrows.annotate('', xy=(0.85, 0.45), xytext=(0.15, 0.45),
                      arrowprops=dict(arrowstyle='->', lw=4, color='black'),
                      xycoords='axes fraction')
    
    # Arrow labels
    ax_arrows.text(0.5, 0.65, 'inference', ha='center', va='center',
                  fontsize=16, fontweight='bold', color='green',
                  transform=ax_arrows.transAxes)
    ax_arrows.text(0.5, 0.35, '@gen GoL', ha='center', va='center',
                  fontsize=16, fontweight='bold', color='black',
                  transform=ax_arrows.transAxes)
    
    # === PANEL 3: OBSERVED STATE ===
    ax_observed.imshow(pattern, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_observed.set_xticks([])
    ax_observed.set_yticks([])
    
    # Red border for observed state
    for spine in ax_observed.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)
    
    # === PANEL 4: GIBBS SAMPLING RESULTS ===
    # Run Gibbs sampling
    print(f"Running Gibbs sampler for {pattern_type} pattern...")
    sampler = core.GibbsSampler(pattern, p_flip=flip_prob)
    key = jrand.key(seed)
    run_summary = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
    
    print(f"\nFinal predictive posterior: {run_summary.predictive_posterior_scores[-1]:.6f}")
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(pattern)
    accuracy = (1 - final_n_bit_flips / pattern.size) * 100
    print(f"Final reconstruction errors: {final_n_bit_flips} bits ({accuracy:.1f}% accuracy)")
    
    # Create 2x2 grid within the Gibbs panel
    n_samples = 4
    sample_indices = np.linspace(0, chain_length - 1, n_samples, dtype=int)
    
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig, ax_gibbs.get_position().bounds,
                    nrows_ncols=(2, 2), axes_pad=0.12, share_all=True)
    
    # Remove the original axis
    ax_gibbs.remove()
    
    for i, (ax, idx) in enumerate(zip(grid, sample_indices)):
        ax.imshow(run_summary.inferred_prev_boards[idx], cmap='gray_r', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add time label
        ax.text(0.02, 0.98, f't={idx}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Highlight final sample
        if i == len(sample_indices) - 1:
            for spine in ax.spines.values():
                spine.set_color('green')
                spine.set_linewidth(3)
                spine.set_linestyle('--')
    
    # === PANEL 5: ONE-STEP EVOLUTION ===
    # Get the final inferred state and evolve it one step
    final_state = run_summary.inferred_prev_boards[-1]
    evolved_state = run_summary.inferred_reconstructed_targets[-1].astype(int)
    
    ax_evolution.imshow(evolved_state, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_evolution.set_xticks([])
    ax_evolution.set_yticks([])
    
    # Add annotation below (moved up)
    ax_evolution.text(0.5, -0.08, 'Final Gibbs state → Next step',
                     ha='center', va='top', fontsize=14, fontweight='bold',
                     transform=ax_evolution.transAxes)
    
    # === ADD TITLES ===
    title_y = 0.95  # Position for titles (lifted higher)
    
    # Get actual x-center positions of each axis
    left_pos = (ax_unknown.get_position().x0 + ax_unknown.get_position().x1) / 2
    arrow_pos = (ax_arrows.get_position().x0 + ax_arrows.get_position().x1) / 2
    obs_pos = (ax_observed.get_position().x0 + ax_observed.get_position().x1) / 2
    gibbs_pos = (ax_gibbs.get_position().x0 + ax_gibbs.get_position().x1) / 2
    evol_pos = (ax_evolution.get_position().x0 + ax_evolution.get_position().x1) / 2
    
    # Add properly centered titles (no title for arrows column)
    fig.text(left_pos, title_y, "Previous State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    # Skip title for arrows column
    fig.text(obs_pos, title_y, "Observed State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(gibbs_pos, title_y, "Inversion via Gibbs", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(evol_pos, title_y, "One-Step Evolution", ha="center", va="top",
             fontsize=20, fontweight="bold")
    
    return fig


def save_integrated_showcase_figure(
    pattern_type="wizards",
    size=256,
    chain_length=150,
    flip_prob=0.03,
    seed=42,
    output_label=None,
):
    """Save the integrated showcase figure."""
    fig = create_integrated_showcase_figure(
        pattern_type, size, chain_length, flip_prob, seed
    )
    label = output_label if output_label is not None else size
    filename = f"figs/gol_integrated_showcase_{pattern_type}_{label}.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved integrated showcase figure: {filename}")
    return fig
    
    # === SCHEMATIC ROW ===
    # Unknown state (?)
    ax_schematic_unknown.set_xlim(0, 1)
    ax_schematic_unknown.set_ylim(0, 1)
    ax_schematic_unknown.text(0.5, 0.5, '?', fontsize=80, ha='center', va='center',
                              fontweight='bold', color='darkgray')
    ax_schematic_unknown.set_xticks([])
    ax_schematic_unknown.set_yticks([])
    ax_schematic_unknown.set_aspect('equal')
    
    # Add green dashed border
    for spine in ax_schematic_unknown.spines.values():
        spine.set_color('green')
        spine.set_linewidth(3)
        spine.set_linestyle('--')
    
    # Arrows panel
    ax_schematic_arrows.axis('off')
    ax_schematic_arrows.set_xlim(0, 1)
    ax_schematic_arrows.set_ylim(0, 1)
    
    # Top arrow (inference)
    ax_schematic_arrows.annotate('', xy=(0.15, 0.65), xytext=(0.85, 0.65),
                                arrowprops=dict(arrowstyle='->', lw=4, color='green'),
                                xycoords='axes fraction')
    
    # Bottom arrow (GoL forward)
    ax_schematic_arrows.annotate('', xy=(0.85, 0.35), xytext=(0.15, 0.35),
                                arrowprops=dict(arrowstyle='->', lw=4, color='black'),
                                xycoords='axes fraction')
    
    # Arrow labels
    ax_schematic_arrows.text(0.5, 0.75, 'inference', ha='center', va='center',
                            fontsize=16, fontweight='bold', color='green',
                            transform=ax_schematic_arrows.transAxes)
    ax_schematic_arrows.text(0.5, 0.25, '@gen GoL', ha='center', va='center',
                            fontsize=16, fontweight='bold', color='black',
                            transform=ax_schematic_arrows.transAxes)
    
    # Observed state in schematic
    ax_schematic_observed.imshow(pattern, cmap='gray_r', interpolation='nearest', aspect='equal')
    ax_schematic_observed.set_xticks([])
    ax_schematic_observed.set_yticks([])
    
    # Red border for observed state
    for spine in ax_schematic_observed.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)
    
    # === SHOWCASE ROW ===
    # Run Gibbs sampling
    print(f"Running Gibbs sampler for {pattern_type} pattern...")
    sampler = core.GibbsSampler(pattern, p_flip=flip_prob)
    key = jrand.key(seed)
    run_summary = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
    
    print(f"\nFinal predictive posterior: {run_summary.predictive_posterior_scores[-1]:.6f}")
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(pattern)
    accuracy = (1 - final_n_bit_flips / pattern.size) * 100
    print(f"Final reconstruction errors: {final_n_bit_flips} bits ({accuracy:.1f}% accuracy)")
    
    # Observed state in showcase (same as schematic)
    ax_showcase_observed.imshow(pattern, cmap='gray_r', interpolation='nearest')
    ax_showcase_observed.set_xticks([])
    ax_showcase_observed.set_yticks([])
    ax_showcase_observed.set_aspect('equal', 'box')
    
    # Red border
    for spine in ax_showcase_observed.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)
    
    # Gibbs sampling progression
    n_samples = 4
    sample_indices = np.linspace(0, chain_length - 1, n_samples, dtype=int)
    
    # Create 2x2 grid within the middle panel
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig, ax_showcase_gibbs.get_position().bounds,
                    nrows_ncols=(2, 2), axes_pad=0.05, share_all=True)
    
    # Remove the original axis
    ax_showcase_gibbs.remove()
    
    for i, (ax, idx) in enumerate(zip(grid, sample_indices)):
        ax.imshow(run_summary.inferred_prev_boards[idx], cmap='gray_r', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add time label
        ax.text(0.02, 0.98, f't={idx}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Highlight final sample
        if i == len(sample_indices) - 1:
            for spine in ax.spines.values():
                spine.set_color('green')
                spine.set_linewidth(3)
                spine.set_linestyle('--')
    
    # One-step evolution
    final_state = run_summary.inferred_prev_boards[-1]
    evolved_state = run_summary.inferred_reconstructed_targets[-1].astype(int)
    
    ax_showcase_evolution.imshow(evolved_state, cmap='gray_r', interpolation='nearest')
    ax_showcase_evolution.set_xticks([])
    ax_showcase_evolution.set_yticks([])
    ax_showcase_evolution.set_aspect('equal', 'box')
    
    # Add annotation below
    ax_showcase_evolution.text(0.5, -0.12, 'Final state → Next step',
                              ha='center', va='top', fontsize=14, fontweight='bold',
                              transform=ax_showcase_evolution.transAxes)
    
    # === ADD TITLES ===
    title_y = 0.95  # High position for titles
    
    # Top row titles
    fig.text(0.2, title_y, "Previous State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(0.5, title_y, "Game of Life", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(0.8, title_y, "Observed State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    
    # Bottom row titles
    fig.text(0.2, 0.48, "Observed State", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(0.5, 0.48, "Inversion via Gibbs", ha="center", va="top",
             fontsize=20, fontweight="bold")
    fig.text(0.8, 0.48, "One-Step Evolution", ha="center", va="top",
             fontsize=20, fontweight="bold")
    
    return fig


def save_combined_schematic_showcase_figure(
    pattern_type="wizards",
    size=256,
    chain_length=150,
    flip_prob=0.03,
    seed=42,
    output_label=None,
):
    """Save the combined schematic and showcase figure."""
    return save_integrated_showcase_figure(
        pattern_type,
        size=size,
        chain_length=chain_length,
        flip_prob=flip_prob,
        seed=seed,
        output_label=output_label,
    )


def save_all_showcase_figures(
    pattern_type="wizards",
    size=512,
    chain_length=150,
    flip_prob=0.03,
    seed=42,
    output_label=None,
):
    """Generate the showcase and timing figures used in the paper."""
    print("=== Generating Game of Life showcase figures ===")
    save_showcase_figure(
        pattern_type,
        size,
        chain_length,
        flip_prob,
        seed,
        output_label=output_label,
    )
    save_timing_bar_plot()
    print("\n=== Game of Life showcase figures generated successfully! ===")



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
