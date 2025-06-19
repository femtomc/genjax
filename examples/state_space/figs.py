"""
Visualization utilities for rejuvenation SMC case study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax.scipy.special
from typing import Dict, Any, List
import numpy as np


def generate_figure_name(
    figure_type: str,
    model_name: str,
    scenario_name: str = None,
    strategy: str = None,
    n_particles: int = None,
    frames: List[int] = None,
    additional_info: str = None,
) -> str:
    """
    Generate descriptive figure filename with model, particles, and SMC variant details.

    Args:
        figure_type: Type of figure (e.g., "rejuvenation_smc", "difficulty_comparison")
        model_name: Model name (e.g., "linear_gaussian_2d", "discrete_hmm")
        scenario_name: Scenario name (e.g., "standard", "challenging", "extreme")
        strategy: Strategy name (e.g., "mh_1", "mala_5")
        n_particles: Number of particles
        frames: List of time frames
        additional_info: Additional descriptive information

    Returns:
        Descriptive filename string
    """
    parts = [figure_type, model_name]

    if scenario_name:
        parts.append(scenario_name)

    if strategy:
        parts.append(strategy)

    if n_particles:
        parts.append(f"n{n_particles}")

    if frames:
        frames_str = "_".join(map(str, frames))
        parts.append(f"frames{frames_str}")

    if additional_info:
        parts.append(additional_info)

    return "_".join(parts) + ".pdf"


def setup_plotting():
    """Set up plotting style for research paper quality figures."""
    sns.set_style("white")
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.grid"] = False


def plot_convergence_comparison(
    hmm_results: Dict[str, Any], lg_results: Dict[str, Any], save_path: str = None
):
    """
    Plot convergence comparison between HMM and Linear Gaussian models.
    """
    setup_plotting()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract particle counts
    n_particles = [r["n_particles"] for r in hmm_results["results"]]

    # HMM convergence
    errors = [r["error"] for r in hmm_results["results"]]
    relative_errors = [r["relative_error"] for r in hmm_results["results"]]

    ax1.loglog(
        n_particles, errors, "o-", linewidth=2, markersize=8, label="Absolute Error"
    )
    ax1.loglog(
        n_particles,
        relative_errors,
        "s--",
        linewidth=2,
        markersize=8,
        label="Relative Error",
    )

    # Add 1/sqrt(N) reference line
    n_ref = np.array(n_particles)
    ref_line = errors[0] * np.sqrt(n_particles[0]) / np.sqrt(n_ref)
    ax1.loglog(n_ref, ref_line, "k:", alpha=0.5, label=r"$\propto 1/\sqrt{N}$")

    ax1.set_xlabel("Number of Particles")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    # Linear Gaussian convergence
    errors = [r["error"] for r in lg_results["results"]]
    relative_errors = [r["relative_error"] for r in lg_results["results"]]

    ax2.loglog(
        n_particles, errors, "o-", linewidth=2, markersize=8, label="Absolute Error"
    )
    ax2.loglog(
        n_particles,
        relative_errors,
        "s--",
        linewidth=2,
        markersize=8,
        label="Relative Error",
    )

    # Add 1/sqrt(N) reference line
    n_ref = np.array(n_particles)
    ref_line = errors[0] * np.sqrt(n_particles[0]) / np.sqrt(n_ref)
    ax2.loglog(n_ref, ref_line, "k:", alpha=0.5, label=r"$\propto 1/\sqrt{N}$")

    ax2.set_xlabel("Number of Particles")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_log_marginal_comparison(
    hmm_results: Dict[str, Any], lg_results: Dict[str, Any], save_path: str = None
):
    """
    Plot log marginal likelihood estimates vs exact values.
    """
    setup_plotting()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Extract data
    n_particles_list = [r["n_particles"] for r in hmm_results["results"]]
    colors = sns.color_palette("husl", 2)

    # Create grouped bar chart
    lg_results["exact_log_marginal"]

    for i, n_p in enumerate(n_particles_list):
        hmm_lm = [
            r["log_marginal"] for r in hmm_results["results"] if r["n_particles"] == n_p
        ][0]
        lg_lm = [
            r["log_marginal"] for r in lg_results["results"] if r["n_particles"] == n_p
        ][0]

        ax.bar(i - 0.2, hmm_lm, 0.4, label="HMM" if i == 0 else "", color=colors[0])
        ax.bar(
            i + 0.2,
            lg_lm,
            0.4,
            label="Linear Gaussian" if i == 0 else "",
            color=colors[1],
        )

    # Add exact baselines
    ax.axhline(
        y=hmm_results["exact_log_marginal"],
        color=colors[0],
        linestyle="--",
        alpha=0.7,
        label="HMM Exact",
    )
    ax.axhline(
        y=lg_results["exact_log_marginal"],
        color=colors[1],
        linestyle="--",
        alpha=0.7,
        label="LG Exact",
    )

    ax.set_xticks(range(len(n_particles_list)))
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Log Marginal Likelihood")
    ax.legend()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ess_comparison(
    hmm_results: Dict[str, Any], lg_results: Dict[str, Any], save_path: str = None
):
    """
    Plot effective sample size comparison.
    """
    setup_plotting()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Extract data
    n_particles = [r["n_particles"] for r in hmm_results["results"]]
    hmm_ess = [r["ess"] for r in hmm_results["results"]]
    lg_ess = [r["ess"] for r in lg_results["results"]]

    # Plot ESS for each model
    colors = sns.color_palette("husl", 2)
    width = 0.35
    x = np.arange(len(n_particles))

    ax.bar(x - width / 2, hmm_ess, width, label="HMM", color=colors[0])
    ax.bar(x + width / 2, lg_ess, width, label="Linear Gaussian", color=colors[1])

    # Add reference line at n_particles
    for i, n in enumerate(n_particles):
        ax.axhline(
            y=n,
            xmin=(i - 0.4) / len(n_particles),
            xmax=(i + 0.4) / len(n_particles),
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Effective Sample Size")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set y-axis limit
    max_particles = max(n_particles)
    ax.set_ylim(0, max_particles * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_summary_table(
    hmm_results: Dict[str, Any], lg_results: Dict[str, Any], save_path: str = None
):
    """
    Create summary table of results.
    """
    setup_plotting()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Define columns
    columns = ["Particles", "Log ML", "Exact", "Error", "Rel Error (%)", "ESS"]

    # HMM table
    hmm_data = []
    for r in hmm_results["results"]:
        hmm_data.append(
            [
                f"{r['n_particles']:,}",
                f"{r['log_marginal']:.4f}",
                f"{hmm_results['exact_log_marginal']:.4f}",
                f"{r['error']:.4f}",
                f"{r['relative_error'] * 100:.2f}%",
                f"{r['ess']:.1f}",
            ]
        )

    ax1.axis("tight")
    ax1.axis("off")
    table1 = ax1.table(
        cellText=hmm_data, colLabels=columns, cellLoc="center", loc="center"
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 1.8)

    # Style the header
    for i in range(len(columns)):
        table1[(0, i)].set_facecolor("#4CAF50")
        table1[(0, i)].set_text_props(weight="bold", color="white")

    # Linear Gaussian table
    lg_data = []
    for r in lg_results["results"]:
        lg_data.append(
            [
                f"{r['n_particles']:,}",
                f"{r['log_marginal']:.4f}",
                f"{lg_results['exact_log_marginal']:.4f}",
                f"{r['error']:.4f}",
                f"{r['relative_error'] * 100:.2f}%",
                f"{r['ess']:.1f}",
            ]
        )

    ax2.axis("tight")
    ax2.axis("off")
    table2 = ax2.table(
        cellText=lg_data, colLabels=columns, cellLoc="center", loc="center"
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 1.8)

    # Style the header
    for i in range(len(columns)):
        table2[(0, i)].set_facecolor("#2196F3")
        table2[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_2d_particle_evolution(
    particle_data: Dict[str, Any],
    save_path: str = None,
    model_name: str = "linear_gaussian_2d",
    scenario_name: str = "standard",
):
    """
    Plot 2D particle evolution in a (4,4) grid for 16 timesteps.
    """
    setup_plotting()

    # Extract data
    all_particles = particle_data["all_particles"]
    true_states = particle_data["true_states"]
    observations = particle_data["observations"]
    T = particle_data["T"]

    # Create (4,4) grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    # Get axis limits from data
    all_states = []
    for t in range(T):
        # Extract particle states at time t
        particle_choices = all_particles.traces.get_choices()
        states_t = particle_choices["state"][t]  # Shape: (n_particles, d_state)
        all_states.append(states_t)

    # Concatenate all states for limits
    all_states_concat = jnp.concatenate(
        all_states + [true_states, observations], axis=0
    )
    xlim = (all_states_concat[:, 0].min() - 0.5, all_states_concat[:, 0].max() + 0.5)
    ylim = (all_states_concat[:, 1].min() - 0.5, all_states_concat[:, 1].max() + 0.5)

    for t in range(min(T, 16)):
        ax = axes[t]

        # Extract particle states and weights at time t
        states_t = all_states[t]
        log_weights_t = all_particles.log_weights[t]
        weights_t = jnp.exp(log_weights_t - jax.scipy.special.logsumexp(log_weights_t))

        # Plot particles with opacity proportional to weight
        # Normalize weights to alpha values between 0.1 and 0.9
        alphas = 0.1 + 0.8 * (weights_t - weights_t.min()) / (
            weights_t.max() - weights_t.min() + 1e-10
        )

        # Plot each particle individually with its opacity
        for i in range(len(states_t)):
            ax.scatter(
                states_t[i, 0],
                states_t[i, 1],
                s=200,  # Fixed size for all particles
                alpha=float(alphas[i]),
                c="blue",
                edgecolors="darkblue",
                linewidth=0.5,
            )

        # Plot true state
        ax.plot(
            true_states[t, 0],
            true_states[t, 1],
            "r*",
            markersize=15,
            label="True State" if t == 0 else None,
        )

        # Plot observation
        ax.plot(
            observations[t, 0],
            observations[t, 1],
            "go",
            markersize=10,
            label="Observation" if t == 0 else None,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"t = {t}", fontsize=16)
        ax.grid(False)

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add legend only to first subplot
        if t == 0:
            ax.legend(loc="upper right", fontsize=12)

    # Remove empty subplots if T < 16
    for i in range(T, 16):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Auto-generate descriptive filename if save_path not provided
    if save_path is None:
        T = particle_data.get("T", 16)
        n_particles = particle_data.get("n_particles", "unknown")

        # Generate descriptive filename in figs directory
        import os

        figs_dir = os.path.join(os.path.dirname(__file__), "figs")
        os.makedirs(figs_dir, exist_ok=True)

        filename = generate_figure_name(
            figure_type="particle_evolution_2d",
            model_name=model_name,
            scenario_name=scenario_name,
            n_particles=n_particles,
            frames=list(range(T)),
            additional_info="grid4x4",
        )
        save_path = os.path.join(figs_dir, filename)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved particle evolution: {save_path}")
    else:
        plt.show()


def plot_particle_density_evolution(
    particle_data: Dict[str, Any], save_path: str = None
):
    """
    Plot particle density evolution in a 4x4 grid.
    """
    setup_plotting()

    # Extract data
    all_particles = particle_data["all_particles"]
    true_states = particle_data["true_states"]
    observations = particle_data["observations"]
    T = particle_data["T"]

    # Create 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    # Get axis limits from data
    all_states = []
    for t in range(T):
        # Extract particle states at time t
        particle_choices = all_particles.traces.get_choices()
        states_t = particle_choices["state"][t]  # Shape: (n_particles, d_state)
        all_states.append(states_t)

    # Concatenate all states for limits
    all_states_concat = jnp.concatenate(
        all_states + [true_states, observations], axis=0
    )
    xlim = (all_states_concat[:, 0].min() - 1, all_states_concat[:, 0].max() + 1)
    ylim = (all_states_concat[:, 1].min() - 1, all_states_concat[:, 1].max() + 1)

    for t in range(min(T, 16)):
        ax = axes[t]

        # Extract particle states and weights at time t
        states_t = all_states[t]
        log_weights_t = all_particles.log_weights[t]
        weights_t = jnp.exp(log_weights_t - jax.scipy.special.logsumexp(log_weights_t))

        # Create 2D histogram with weights
        H, xedges, yedges = np.histogram2d(
            states_t[:, 0],
            states_t[:, 1],
            bins=30,
            range=[xlim, ylim],
            weights=weights_t,
        )

        # Plot density
        ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="Blues",
            alpha=0.8,
        )

        # Plot true state
        ax.plot(
            true_states[t, 0],
            true_states[t, 1],
            "r*",
            markersize=20,
            label="True State" if t == 0 else None,
        )

        # Plot observation
        ax.plot(
            observations[t, 0],
            observations[t, 1],
            "go",
            markersize=15,
            label="Observation" if t == 0 else None,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"t = {t}", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

        # Remove all tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if t == 0:
            ax.legend(loc="upper right", fontsize=12)

    # Remove empty subplots if T < 16
    for i in range(T, 16):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rejuvenation_comparison(
    rejuvenation_results: Dict[str, Dict[str, Any]],
    frames: List[int] = [0, 4, 8, 12],
    save_path: str = None,
    scenario_name: str = "standard",
    model_name: str = "linear_gaussian_2d",
):
    """
    Compare different rejuvenation strategies at specific time frames.

    Args:
        rejuvenation_results: Dictionary with keys:
            - "mh_1": MH with K=1 rejuvenation move
            - "mh_5": MH with K=5 rejuvenation moves
            - "mala_1": MALA with K=1 rejuvenation move
            - "mala_5": MALA with K=5 rejuvenation moves
        frames: List of time indices to display
        save_path: Path to save figure (if None, auto-generate descriptive name)
        scenario_name: Name of scenario (e.g., "standard", "challenging", "extreme")
        model_name: Name of model (e.g., "linear_gaussian_2d", "discrete_hmm")
    """
    setup_plotting()

    # Create grid based on available strategies, but default to 4x4 if all are present
    available_strategies = [
        s for s in ["mh_1", "mh_5", "mala_1", "mala_5"] if s in rejuvenation_results
    ]
    n_strategies = len(available_strategies)
    n_frames = len(frames)

    if n_strategies == 4 and n_frames == 4:
        # Standard 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    else:
        # Dynamic sizing
        fig, axes = plt.subplots(
            n_strategies, n_frames, figsize=(4 * n_frames, 4 * n_strategies)
        )
        if n_strategies == 1:
            axes = axes.reshape(1, -1)
        elif n_frames == 1:
            axes = axes.reshape(-1, 1)

    # Strategy labels - only include strategies that are present
    all_strategies = ["mh_1", "mh_5", "mala_1", "mala_5"]
    all_labels = ["MH (K=1)", "MH (K=5)", "MALA (K=1)", "MALA (K=5)"]

    # Filter to only present strategies
    strategies = []
    strategy_labels = []
    for strat, label in zip(all_strategies, all_labels):
        if strat in rejuvenation_results:
            strategies.append(strat)
            strategy_labels.append(label)

    # Get axis limits from all data
    all_states_list = []
    true_states = None
    observations = None

    for strategy in strategies:
        if strategy in rejuvenation_results:
            data = rejuvenation_results[strategy]
            all_particles = data["all_particles"]

            if true_states is None:
                true_states = data["true_states"]
                observations = data["observations"]

            # Extract all particle states
            particle_choices = all_particles.traces.get_choices()
            states = particle_choices["state"]  # Shape: (T, n_particles, d_state)

            for t in frames:
                if t < states.shape[0]:
                    all_states_list.append(states[t])

    # Compute axis limits
    if all_states_list and true_states is not None:
        # Use proper array indexing for JAX
        frames_array = jnp.array(frames)
        all_states_concat = jnp.concatenate(
            all_states_list + [true_states[frames_array], observations[frames_array]],
            axis=0,
        )
        xlim = (
            all_states_concat[:, 0].min() - 0.5,
            all_states_concat[:, 0].max() + 0.5,
        )
        ylim = (
            all_states_concat[:, 1].min() - 0.5,
            all_states_concat[:, 1].max() + 0.5,
        )
    else:
        xlim = ylim = (-5, 5)

    # Plot each strategy and time frame
    for row, (strategy, label) in enumerate(zip(strategies, strategy_labels)):
        if strategy not in rejuvenation_results:
            continue

        data = rejuvenation_results[strategy]
        all_particles = data["all_particles"]

        # Extract particle states and weights
        particle_choices = all_particles.traces.get_choices()
        states = particle_choices["state"]  # Shape: (T, n_particles, d_state)
        log_weights = all_particles.log_weights  # Shape: (T, n_particles)

        for col, t in enumerate(frames):
            ax = axes[row, col]

            if t < states.shape[0]:
                # Get states and weights at time t
                states_t = states[t]
                log_weights_t = log_weights[t]
                weights_t = jnp.exp(
                    log_weights_t - jax.scipy.special.logsumexp(log_weights_t)
                )

                # Plot particles with opacity proportional to weight
                alphas = 0.1 + 0.8 * (weights_t - weights_t.min()) / (
                    weights_t.max() - weights_t.min() + 1e-10
                )

                # Plot each particle
                for i in range(len(states_t)):
                    ax.scatter(
                        states_t[i, 0],
                        states_t[i, 1],
                        s=100,
                        alpha=float(alphas[i]),
                        c="blue",
                        edgecolors="darkblue",
                        linewidth=0.5,
                    )

                # Plot true state
                ax.plot(true_states[t, 0], true_states[t, 1], "r*", markersize=15)

                # Plot observation
                ax.plot(observations[t, 0], observations[t, 1], "go", markersize=10)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Add labels
            if row == 0:
                ax.set_title(f"t = {t}", fontsize=16)
            if col == 0:
                ax.set_ylabel(label, fontsize=16)

    # Add legend to first subplot
    if axes.size > 0:
        ax = axes[0, 0]
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                alpha=0.5,
                label="Particles",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="r",
                markersize=15,
                linestyle="",
                label="True State",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="g",
                markersize=10,
                linestyle="",
                label="Observation",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.tight_layout()

    # Auto-generate descriptive filename if save_path not provided
    if save_path is None:
        # Extract metadata from results
        if rejuvenation_results:
            sample_data = next(iter(rejuvenation_results.values()))
            n_particles = sample_data.get("n_particles", None)

            # Generate descriptive filename in figs directory
            import os

            figs_dir = os.path.join(os.path.dirname(__file__), "figs")
            os.makedirs(figs_dir, exist_ok=True)

            filename = generate_figure_name(
                figure_type="rejuvenation_smc",
                model_name=model_name,
                scenario_name=scenario_name,
                n_particles=n_particles,
                frames=frames,
                additional_info="all_strategies",
            )
            save_path = os.path.join(figs_dir, filename)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved rejuvenation comparison: {save_path}")
    else:
        plt.show()


def plot_difficulty_comparison(
    easy_results: Dict[str, Dict[str, Any]],
    challenging_results: Dict[str, Dict[str, Any]],
    extreme_results: Dict[str, Dict[str, Any]],
    frames: List[int] = [0, 4, 8, 12],
    save_path: str = None,
    model_name: str = "linear_gaussian_2d",
    strategy: str = "mala_5",
):
    """
    Compare algorithm performance across difficulty levels at specific time frames.

    Creates a 3x4 grid: Easy/Challenging/Extreme (rows) × frames 0,4,8,12 (columns)

    Args:
        easy_results: Results from standard scenario
        challenging_results: Results from challenging scenario
        extreme_results: Results from extreme scenario
        frames: List of time indices to display
        save_path: Path to save figure (if None, auto-generate descriptive name)
        model_name: Name of model for filename
        strategy: Which strategy to visualize (default: "mala_5")
    """
    setup_plotting()

    # Create 3x4 grid for difficulty comparison
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    scenario_data = [
        ("Easy", easy_results),
        ("Challenging", challenging_results),
        ("Extreme", extreme_results),
    ]

    # Use MALA with K=5 as representative algorithm for comparison
    strategy = "mala_5"

    # Get axis limits from all data
    all_states_list = []
    true_states = None
    observations = None

    for scenario_name, results in scenario_data:
        if strategy in results:
            data = results[strategy]
            all_particles = data["all_particles"]

            if true_states is None:
                true_states = data["true_states"]
                observations = data["observations"]

            # Extract particle states
            particle_choices = all_particles.traces.get_choices()
            states = particle_choices["state"]  # Shape: (T, n_particles, d_state)

            for t in frames:
                if t < states.shape[0]:
                    all_states_list.append(states[t])

    # Compute axis limits
    if all_states_list and true_states is not None:
        frames_array = jnp.array(frames)
        all_states_concat = jnp.concatenate(
            all_states_list + [true_states[frames_array], observations[frames_array]],
            axis=0,
        )
        xlim = (
            all_states_concat[:, 0].min() - 1.0,
            all_states_concat[:, 0].max() + 1.0,
        )
        ylim = (
            all_states_concat[:, 1].min() - 1.0,
            all_states_concat[:, 1].max() + 1.0,
        )
    else:
        xlim = ylim = (-8, 8)

    # Plot each scenario and time frame
    for row, (scenario_name, results) in enumerate(scenario_data):
        if strategy not in results:
            continue

        data = results[strategy]
        all_particles = data["all_particles"]

        # Extract particle states and weights
        particle_choices = all_particles.traces.get_choices()
        states = particle_choices["state"]  # Shape: (T, n_particles, d_state)
        log_weights = all_particles.log_weights  # Shape: (T, n_particles)

        for col, t in enumerate(frames):
            ax = axes[row, col]

            if t < states.shape[0]:
                # Get states and weights at time t
                states_t = states[t]
                log_weights_t = log_weights[t]
                weights_t = jnp.exp(
                    log_weights_t - jax.scipy.special.logsumexp(log_weights_t)
                )

                # Plot particles with opacity proportional to weight
                alphas = 0.1 + 0.8 * (weights_t - weights_t.min()) / (
                    weights_t.max() - weights_t.min() + 1e-10
                )

                # Plot each particle
                for i in range(len(states_t)):
                    ax.scatter(
                        states_t[i, 0],
                        states_t[i, 1],
                        s=50,
                        alpha=float(alphas[i]),
                        c="blue",
                        edgecolors="darkblue",
                        linewidth=0.3,
                    )

                # Plot true state
                ax.plot(true_states[t, 0], true_states[t, 1], "r*", markersize=12)

                # Plot observation
                ax.plot(observations[t, 0], observations[t, 1], "go", markersize=8)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Add labels
            if row == 0:
                ax.set_title(f"t = {t}", fontsize=14)
            if col == 0:
                ax.set_ylabel(scenario_name, fontsize=14, fontweight="bold")

    # Add legend to first subplot
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            alpha=0.5,
            label="Particles",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="r",
            markersize=12,
            linestyle="",
            label="True State",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="g",
            markersize=8,
            linestyle="",
            label="Observation",
        ),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    # Auto-generate descriptive filename if save_path not provided
    if save_path is None:
        # Extract metadata from easy results
        if strategy in easy_results:
            sample_data = easy_results[strategy]
            n_particles = sample_data.get("n_particles", None)

            # Generate descriptive filename in figs directory
            import os

            figs_dir = os.path.join(os.path.dirname(__file__), "figs")
            os.makedirs(figs_dir, exist_ok=True)

            filename = generate_figure_name(
                figure_type="difficulty_comparison",
                model_name=model_name,
                strategy=strategy,
                n_particles=n_particles,
                frames=frames,
                additional_info="easy_challenging_extreme",
            )
            save_path = os.path.join(figs_dir, filename)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved difficulty comparison: {save_path}")
    else:
        plt.show()


def create_all_figures(
    hmm_results: Dict[str, Any], lg_results: Dict[str, Any], output_dir: str = "figs"
):
    """
    Create all figures for the case study.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Convergence comparison
    plot_convergence_comparison(
        hmm_results,
        lg_results,
        save_path=os.path.join(output_dir, "convergence_comparison.pdf"),
    )
    print(f"✓ Created {output_dir}/convergence_comparison.pdf")

    # Log marginal comparison
    plot_log_marginal_comparison(
        hmm_results,
        lg_results,
        save_path=os.path.join(output_dir, "log_marginal_comparison.pdf"),
    )
    print(f"✓ Created {output_dir}/log_marginal_comparison.pdf")

    # ESS comparison
    plot_ess_comparison(
        hmm_results,
        lg_results,
        save_path=os.path.join(output_dir, "ess_comparison.pdf"),
    )
    print(f"✓ Created {output_dir}/ess_comparison.pdf")

    # Summary table
    plot_summary_table(
        hmm_results, lg_results, save_path=os.path.join(output_dir, "summary_table.pdf")
    )
    print(f"✓ Created {output_dir}/summary_table.pdf")

    print(f"\nAll figures saved to {output_dir}/")
