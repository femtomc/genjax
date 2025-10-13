import json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.random as jrand

from . import core
from .data import (
    get_blinker_n,
    get_small_wizards_logo,
)
from genjax.timing import benchmark_with_warmup

# Import shared GenJAX Research Visualization Standards
from genjax.viz.standard import (
    setup_publication_fonts,
    get_method_color,
    save_publication_figure,
)

# Apply GRVS typography standards
setup_publication_fonts()


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


def save_timing_bar_plot(
    grid_sizes=[64, 128, 256, 512], chain_length=1, flip_prob=0.03, repeats=3
):
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
    cpu_device = jax.devices("cpu")[0]

    for n in grid_sizes:
        # Place data on CPU device BEFORE timing
        target = jax.device_put(get_blinker_n(n), cpu_device)
        key = jax.device_put(jrand.key(1), cpu_device)
        sampler = core.GibbsSampler(target, flip_prob)

        def task_fn():
            result = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
            return result.predictive_posterior_scores[-1]

        _, (mean_time, std_time) = benchmark_with_warmup(
            task_fn,
            warmup_runs=2,
            repeats=repeats,
            inner_repeats=1,
            auto_sync=True,
        )
        cpu_times_ms.append(mean_time * 1000)  # Convert to ms
        print(f"    {n}×{n}: {mean_time * 1000:.1f} ms")

    # Check if GPU available
    try:
        gpu_available = len(jax.devices("gpu")) > 0
    except RuntimeError:
        gpu_available = False
    gpu_times_ms = []

    if gpu_available:
        print(f"  Benchmarking GPU at grid sizes: {grid_sizes}")
        gpu_device = jax.devices("gpu")[0]
        for n in grid_sizes:
            # Place data on GPU device BEFORE timing
            target = jax.device_put(get_blinker_n(n), gpu_device)
            key = jax.device_put(jrand.key(1), gpu_device)
            sampler = core.GibbsSampler(target, flip_prob)

            def task_fn():
                result = core.run_sampler_and_get_summary(key, sampler, chain_length, 1)
                return result.predictive_posterior_scores[-1]

            _, (mean_time, std_time) = benchmark_with_warmup(
                task_fn,
                warmup_runs=2,
                repeats=repeats,
                inner_repeats=1,
                auto_sync=True,
            )
            gpu_times_ms.append(mean_time * 1000)
            print(f"    {n}×{n}: {mean_time * 1000:.1f} ms")

    # Reverse order - smallest at top, largest at bottom
    sizes = list(reversed(grid_sizes))
    cpu_times_ms = list(reversed(cpu_times_ms))
    if gpu_available:
        gpu_times_ms = list(reversed(gpu_times_ms))

    # Create horizontal bar plot
    y_pos = np.arange(len(sizes))
    bar_height = 0.35

    if gpu_available:
        bars_gpu = ax.barh(
            y_pos + bar_height / 2,
            gpu_times_ms,
            bar_height,
            label="GPU",
            color=get_method_color("genjax_hmc"),
            alpha=0.8,
        )
        bars_cpu = ax.barh(
            y_pos - bar_height / 2,
            cpu_times_ms,
            bar_height,
            label="CPU",
            color=get_method_color("genjax_is"),
            alpha=0.8,
        )

        # Add speedup annotations
        for i, (bar_cpu, bar_gpu) in enumerate(zip(bars_cpu, bars_gpu)):
            speedup = cpu_times_ms[i] / gpu_times_ms[i]
            ax.text(
                bar_cpu.get_width() + max(cpu_times_ms) * 0.02,
                bar_cpu.get_y() + bar_cpu.get_height() / 2,
                f"{speedup:.1f}×",
                ha="left",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=get_method_color("data_points"),
            )
    else:
        bars_cpu = ax.barh(
            y_pos,
            cpu_times_ms,
            bar_height,
            label="CPU",
            color=get_method_color("genjax_is"),
            alpha=0.8,
        )

    # Format axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s}×{s}" for s in sizes], fontsize=16)
    ax.set_xlabel("Time per Gibbs sweep (ms)", fontsize=18, fontweight="bold")
    ax.set_xlim(0, max(cpu_times_ms) * (1.25 if gpu_available else 1.15))

    if gpu_available:
        ax.legend(
            loc="upper center",
            fontsize=16,
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 1.05),
            columnspacing=2,
        )

    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.95,
        "Game of Life Gibbs Sampling Performance",
        ha="center",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    filename = "figs/gol_gibbs_timing_bar_plot.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved: {filename}")
    return filename


def _gibbs_task(n: int, chain_length: int, flip_prob: float, seed: int):
    """Single Gibbs sampling task for timing benchmarks."""
    target = get_blinker_n(n)
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(seed), core.GibbsSampler(target, flip_prob), chain_length, 1
    )
    return run_summary.predictive_posterior_scores[-1]


def create_showcase_figure(
    pattern_type="mit",
    size=256,
    chain_length=150,
    flip_prob=0.03,
    seed=42,
    white_lambda=False,
    load_from_file=None,
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

    if load_from_file:
        print(f"Loading experiment data from: {load_from_file}")
        with open(load_from_file, "r") as f:
            exp_data = json.load(f)

        # Extract data from saved experiment
        target = jnp.array(exp_data["target"])
        chain_length = exp_data["metadata"]["chain_length"]

        # Create a mock run_summary object with the loaded data
        class MockRunSummary:
            def __init__(self, data):
                self.predictive_posterior_scores = jnp.array(
                    data["predictive_posterior_scores"]
                )
                self.inferred_prev_boards = jnp.array(data["inferred_prev_boards"])
                self.inferred_reconstructed_targets = jnp.array(
                    data["inferred_reconstructed_targets"]
                )
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
    ax_target.set_aspect("equal", "box")

    # Add thick red border to highlight this is the observed state (like in schematic)
    for spine in ax_target.spines.values():
        spine.set_color(get_method_color("data_points"))
        spine.set_linewidth(4)

    # === MIDDLE PANEL: Inferred Past States ===

    # Create a 2x2 grid of inferred samples over entire chain
    n_samples = 4
    sample_indices = jnp.round(jnp.linspace(0, chain_length - 1, n_samples)).astype(int)

    # Create subgrid for samples with padding
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs[0, 1],
        wspace=0.08,
        hspace=0.08,  # Tighter spacing for 2x2 grid
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

            rect = Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor="green",
                linewidth=4,
                linestyle="--",
            )
            ax.add_patch(rect)

    # Remove axes from the middle panel container
    ax_inferred.set_xticks([])
    ax_inferred.set_yticks([])
    ax_inferred.axis("off")
    ax_inferred.set_aspect("equal", "box")

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
    ax_evolution.set_aspect("equal", "box")

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
    print(
        f"Final reconstruction errors: {final_n_bit_flips} bits ({accuracy:.1f}% accuracy)"
    )
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
