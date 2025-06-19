import time
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

# Set matplotlib defaults for research-quality figures
matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    }
)


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
    monitoring_filename = f"examples/gol/figs/blinker_{pattern_size}x{pattern_size}_monitoring_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"examples/gol/figs/blinker_{pattern_size}x{pattern_size}_samples_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    monitoring_fig.savefig(monitoring_filename, dpi=300, bbox_inches="tight")
    samples_fig.savefig(samples_filename, dpi=300, bbox_inches="tight")

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
    monitoring_filename = f"examples/gol/figs/{logo_type}_logo{size_suffix}_monitoring_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"
    samples_filename = f"examples/gol/figs/{logo_type}_logo{size_suffix}_samples_chain{chain_length}_flip{flip_prob:.3f}_seed{seed}.pdf"

    monitoring_fig.savefig(monitoring_filename, dpi=300, bbox_inches="tight")
    samples_fig.savefig(samples_filename, dpi=300, bbox_inches="tight")

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
    chain_length: int = 250,
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

    fig, ax = plt.subplots(figsize=(10, 6))

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
        ax.bar(
            [n + (15 if device_name == "gpu" else -15) for n in grid_sizes],
            times_array,
            color=color,
            alpha=0.7,
            label=f"{device_name.upper()}",
            edgecolor="black",
            width=25,
        )

    # Format plot
    ax.set_xlabel("Grid Size (N×N)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title(
        f"Game of Life Inference Scaling\n({chain_length} Gibbs steps, flip_prob={flip_prob})"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xticks(grid_sizes)
    ax.set_xlim(min(grid_sizes) - 30, max(grid_sizes) + 30)

    if len(devices) > 1:
        ax.legend()

    plt.tight_layout(pad=2.0)

    # Create parametrized filename
    device_suffix = device if device != "both" else "cpu_gpu"
    filename = f"examples/gol/figs/timing_scaling_{device_suffix}_chain{chain_length}_flip{flip_prob:.3f}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
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


if __name__ == "__main__":
    # Default behavior: generate all figures with standard parameters
    print("=== Running all Game of Life visualizations ===")

    save_blinker_gibbs_figure()
    save_logo_gibbs_figure(chain_length=0)  # Initial state
    save_logo_gibbs_figure(chain_length=250)  # After inference
    save_logo_gibbs_figure(logo_type="popl", chain_length=25)  # POPL logo
    save_timing_scaling_figure(device="cpu")

    print("\n=== All figures generated! ===")
