"""Visualization utilities for intuitive physics case study."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from core import *


def create_physics_visualization():
    """Create visualization of the physics environment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Environment without wall
    ax1.set_title("Environment: No Wall", fontsize=16)
    ax1.add_patch(
        patches.Circle(
            AGENT_START, AGENT_RADIUS, color="blue", alpha=0.7, label="Agent"
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (GOAL_CENTER[0] - GOAL_SIZE[0] / 2, GOAL_CENTER[1] - GOAL_SIZE[1] / 2),
            GOAL_SIZE[0],
            GOAL_SIZE[1],
            color="green",
            alpha=0.3,
            label="Goal",
        )
    )
    ax1.axhline(y=0, color="black", linewidth=2, label="Ground")
    ax1.set_xlim(-0.2, 2.5)
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Environment with wall
    ax2.set_title("Environment: With Wall", fontsize=16)
    ax2.add_patch(
        patches.Circle(
            AGENT_START, AGENT_RADIUS, color="blue", alpha=0.7, label="Agent"
        )
    )
    ax2.add_patch(
        patches.Rectangle(
            (GOAL_CENTER[0] - GOAL_SIZE[0] / 2, GOAL_CENTER[1] - GOAL_SIZE[1] / 2),
            GOAL_SIZE[0],
            GOAL_SIZE[1],
            color="green",
            alpha=0.3,
            label="Goal",
        )
    )
    ax2.add_patch(
        patches.Rectangle(
            (WALL_CENTER[0] - WALL_SIZE[0] / 2, WALL_CENTER[1] - WALL_SIZE[1] / 2),
            WALL_SIZE[0],
            WALL_SIZE[1],
            color="red",
            alpha=0.7,
            label="Wall",
        )
    )
    ax2.axhline(y=0, color="black", linewidth=2, label="Ground")
    ax2.set_xlim(-0.2, 2.5)
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_trajectory(theta, impulse, show_both=True):
    """Visualize agent trajectory with and without wall."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw environment
    ax.add_patch(
        patches.Rectangle(
            (GOAL_CENTER[0] - GOAL_SIZE[0] / 2, GOAL_CENTER[1] - GOAL_SIZE[1] / 2),
            GOAL_SIZE[0],
            GOAL_SIZE[1],
            color="green",
            alpha=0.3,
            label="Goal",
        )
    )
    if show_both:
        ax.add_patch(
            patches.Rectangle(
                (WALL_CENTER[0] - WALL_SIZE[0] / 2, WALL_CENTER[1] - WALL_SIZE[1] / 2),
                WALL_SIZE[0],
                WALL_SIZE[1],
                color="red",
                alpha=0.7,
                label="Wall",
            )
        )
    ax.axhline(y=0, color="black", linewidth=2, label="Ground")

    # Simulate and plot trajectories
    def get_trajectory_points(wall_present):
        initial_vx = impulse * jnp.cos(theta)
        initial_vy = impulse * jnp.sin(theta)
        state = jnp.array([AGENT_START[0], AGENT_START[1], initial_vx, initial_vy])

        positions = [state[:2]]
        for _ in range(TIME_STEPS):
            state = physics_step(state, wall_present)
            positions.append(state[:2])
            if state[1] <= AGENT_RADIUS + 0.001:  # Hit ground
                break

        return jnp.array(positions)

    # Plot trajectory without wall
    traj_no_wall = get_trajectory_points(False)
    ax.plot(
        traj_no_wall[:, 0],
        traj_no_wall[:, 1],
        "b-",
        linewidth=2,
        alpha=0.7,
        label="No Wall",
    )
    ax.scatter(traj_no_wall[-1, 0], traj_no_wall[-1, 1], color="blue", s=50, zorder=5)

    if show_both:
        # Plot trajectory with wall
        traj_with_wall = get_trajectory_points(True)
        ax.plot(
            traj_with_wall[:, 0],
            traj_with_wall[:, 1],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="With Wall",
        )
        ax.scatter(
            traj_with_wall[-1, 0], traj_with_wall[-1, 1], color="red", s=50, zorder=5
        )

    # Starting position
    ax.scatter(
        AGENT_START[0],
        AGENT_START[1],
        color="black",
        s=100,
        marker="o",
        zorder=10,
        label="Start",
    )

    ax.set_xlim(-0.2, 2.0)
    ax.set_ylim(-0.1, 0.8)
    ax.set_xlabel("X Position", fontsize=14)
    ax.set_ylabel("Y Position", fontsize=14)
    ax.set_title(f"Agent Trajectory: θ={theta:.3f}, ι={impulse:.3f}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def action_space_heatmap():
    """Create comprehensive action space analysis with utilities and preferences."""
    fig = plt.figure(figsize=(16, 12))

    # Create custom layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])

    # Compute utilities and action preferences for different agent types
    print("Computing action space utilities...")

    # Vectorized utility computation for all scenarios
    theta_mesh, impulse_mesh = jnp.meshgrid(ANGLE_GRID, IMPULSE_GRID, indexing="ij")

    # Create vectorized utility function using flattened approach
    theta_flat = theta_mesh.flatten()
    impulse_flat = impulse_mesh.flatten()

    def compute_utilities_vectorized(wall_present, goal_weight):
        vectorized_fn = jax.vmap(
            lambda t, i: compute_utility(t, i, wall_present, goal_weight),
            in_axes=(0, 0),
        )
        utilities_flat = vectorized_fn(theta_flat, impulse_flat)
        return utilities_flat.reshape(theta_mesh.shape)

    utilities_no_wall_high = compute_utilities_vectorized(False, 0.8)
    utilities_with_wall_high = compute_utilities_vectorized(True, 0.8)
    utilities_no_wall_low = compute_utilities_vectorized(False, 0.2)
    utilities_with_wall_low = compute_utilities_vectorized(True, 0.2)

    # Convert to real-world units for axis labels
    angle_ticks = [0, 6, 12, 18, 24]
    angle_labels = [f"{ANGLE_GRID[i] * 180 / jnp.pi:.0f}°" for i in angle_ticks]
    impulse_ticks = [0, 6, 12, 18, 24]
    impulse_labels = [f"{IMPULSE_GRID[i]:.1f}" for i in impulse_ticks]

    # Top row: High goal weight agent utilities
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        utilities_no_wall_high, aspect="auto", origin="lower", cmap="viridis"
    )
    ax1.set_title("Goal-Focused Agent\n(No Wall)", fontsize=14, fontweight="bold")
    ax1.set_xticks(impulse_ticks)
    ax1.set_xticklabels(impulse_labels)
    ax1.set_yticks(angle_ticks)
    ax1.set_yticklabels(angle_labels)
    ax1.set_ylabel("Launch Angle", fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        utilities_with_wall_high, aspect="auto", origin="lower", cmap="viridis"
    )
    ax2.set_title("Goal-Focused Agent\n(With Wall)", fontsize=14, fontweight="bold")
    ax2.set_xticks(impulse_ticks)
    ax2.set_xticklabels(impulse_labels)
    ax2.set_yticks(angle_ticks)
    ax2.set_yticklabels(angle_labels)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Utility difference for goal-focused agent
    ax3 = fig.add_subplot(gs[0, 2])
    utility_diff = utilities_no_wall_high - utilities_with_wall_high
    im3 = ax3.imshow(
        utility_diff,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-jnp.max(jnp.abs(utility_diff)),
        vmax=jnp.max(jnp.abs(utility_diff)),
    )
    ax3.set_title(
        "Utility Difference\n(No Wall - With Wall)", fontsize=14, fontweight="bold"
    )
    ax3.set_xticks(impulse_ticks)
    ax3.set_xticklabels(impulse_labels)
    ax3.set_yticks(angle_ticks)
    ax3.set_yticklabels(angle_labels)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Middle row: Low goal weight agent utilities
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(
        utilities_no_wall_low, aspect="auto", origin="lower", cmap="viridis"
    )
    ax4.set_title("Effort-Focused Agent\n(No Wall)", fontsize=14, fontweight="bold")
    ax4.set_xticks(impulse_ticks)
    ax4.set_xticklabels(impulse_labels)
    ax4.set_yticks(angle_ticks)
    ax4.set_yticklabels(angle_labels)
    ax4.set_ylabel("Launch Angle", fontsize=12)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(
        utilities_with_wall_low, aspect="auto", origin="lower", cmap="viridis"
    )
    ax5.set_title("Effort-Focused Agent\n(With Wall)", fontsize=14, fontweight="bold")
    ax5.set_xticks(impulse_ticks)
    ax5.set_xticklabels(impulse_labels)
    ax5.set_yticks(angle_ticks)
    ax5.set_yticklabels(angle_labels)
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Action preferences (softmax probabilities)
    ax6 = fig.add_subplot(gs[1, 2])
    temperature = 3.0
    probs_high_wall = jax.nn.softmax(
        utilities_with_wall_high.flatten() / temperature
    ).reshape(utilities_with_wall_high.shape)
    im6 = ax6.imshow(probs_high_wall, aspect="auto", origin="lower", cmap="plasma")
    ax6.set_title(
        "Action Preferences\n(Goal Agent + Wall)", fontsize=14, fontweight="bold"
    )
    ax6.set_xticks(impulse_ticks)
    ax6.set_xticklabels(impulse_labels)
    ax6.set_yticks(angle_ticks)
    ax6.set_yticklabels(angle_labels)
    plt.colorbar(im6, ax=ax6, shrink=0.8)

    # Bottom row: Final positions and trajectory analysis
    ax7 = fig.add_subplot(gs[2, :])

    # Show final positions for different scenarios
    # Use a representative angle (30 degrees)
    repr_angle_idx = len(ANGLE_GRID) // 2
    repr_angle = ANGLE_GRID[repr_angle_idx]

    # Vectorized trajectory simulation
    vectorized_simulate = jax.vmap(
        lambda impulse, wall: simulate_trajectory(repr_angle, impulse, wall),
        in_axes=(0, None),
    )
    final_pos_no_wall = vectorized_simulate(IMPULSE_GRID, False)
    final_pos_with_wall = vectorized_simulate(IMPULSE_GRID, True)

    ax7.plot(
        IMPULSE_GRID, final_pos_no_wall, "b-", linewidth=3, label="No Wall", alpha=0.8
    )
    ax7.plot(
        IMPULSE_GRID,
        final_pos_with_wall,
        "r--",
        linewidth=3,
        label="With Wall",
        alpha=0.8,
    )

    # Mark key regions
    ax7.axhspan(
        GOAL_CENTER[0] - GOAL_SIZE[0] / 2,
        GOAL_CENTER[0] + GOAL_SIZE[0] / 2,
        alpha=0.3,
        color="green",
        label="Goal Region",
    )
    ax7.axvline(
        WALL_CENTER[0],
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Wall Position",
    )

    ax7.set_xlabel("Launch Impulse", fontsize=14)
    ax7.set_ylabel("Final X Position", fontsize=14)
    ax7.set_title(
        f"Final Positions vs. Impulse (Launch Angle = {repr_angle * 180 / jnp.pi:.0f}°)",
        fontsize=14,
        fontweight="bold",
    )
    ax7.legend(fontsize=12)
    ax7.grid(True, alpha=0.3)

    # Add overall xlabel
    fig.text(0.5, 0.02, "Launch Impulse Magnitude", ha="center", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    return fig


def sample_actions_visualization(num_samples=5):
    """Visualize several sample action selections."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Sample actions from agent model
    sample_configs = [
        (True, 0.8, "Wall + High Goal Weight"),
        (False, 0.8, "No Wall + High Goal Weight"),
        (True, 0.2, "Wall + Low Goal Weight"),
        (False, 0.2, "No Wall + Low Goal Weight"),
    ]

    for i, (wall_present, goal_weight, title) in enumerate(sample_configs):
        ax = axes[i]

        # Sample multiple actions for this configuration
        colors = ["red", "blue", "green", "purple", "orange"]
        for sample_idx in range(min(num_samples, len(colors))):
            trace = rational_agent.simulate(const(wall_present), const(goal_weight))
            theta_idx, impulse_idx = trace.get_retval()
            theta = ANGLE_GRID[theta_idx]
            impulse = IMPULSE_GRID[impulse_idx]

            # Simulate trajectory
            initial_vx = impulse * jnp.cos(theta)
            initial_vy = impulse * jnp.sin(theta)
            state = jnp.array([AGENT_START[0], AGENT_START[1], initial_vx, initial_vy])

            positions = [state[:2]]
            for _ in range(TIME_STEPS):
                state = physics_step(state, wall_present)
                positions.append(state[:2])
                if state[1] <= AGENT_RADIUS + 0.001:
                    break

            traj = jnp.array(positions)
            ax.plot(
                traj[:, 0], traj[:, 1], color=colors[sample_idx], alpha=0.7, linewidth=2
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[sample_idx], s=30)

        # Draw environment
        ax.add_patch(
            patches.Rectangle(
                (GOAL_CENTER[0] - GOAL_SIZE[0] / 2, GOAL_CENTER[1] - GOAL_SIZE[1] / 2),
                GOAL_SIZE[0],
                GOAL_SIZE[1],
                color="green",
                alpha=0.3,
            )
        )
        if wall_present:
            ax.add_patch(
                patches.Rectangle(
                    (
                        WALL_CENTER[0] - WALL_SIZE[0] / 2,
                        WALL_CENTER[1] - WALL_SIZE[1] / 2,
                    ),
                    WALL_SIZE[0],
                    WALL_SIZE[1],
                    color="gray",
                    alpha=0.7,
                )
            )

        ax.axhline(y=0, color="black", linewidth=2)
        ax.scatter(
            AGENT_START[0], AGENT_START[1], color="black", s=100, marker="o", zorder=10
        )

        ax.set_xlim(-0.2, 2.0)
        ax.set_ylim(-0.1, 0.8)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(len(sample_configs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig


def inference_demonstration_fig():
    """Create visualization showing how actions reveal hidden environmental information."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    print("Computing inference demonstration...")

    # Sample some representative actions and show their "informativeness"
    representative_actions = [
        (2, 5, "Low angle, low impulse"),  # Doesn't reach wall
        (8, 15, "Medium angle, medium impulse"),  # Moderate evidence
        (2, 20, "Low angle, high impulse"),  # Strong evidence (collision likely)
        (15, 12, "High angle, medium impulse"),  # Trajectory shape evidence
    ]

    wall_probs = []
    action_labels = []

    for theta_idx, impulse_idx, label in representative_actions:
        # For demonstration, we'll compute a simplified version
        # In practice, this would use the full importance sampling
        theta = ANGLE_GRID[theta_idx]
        impulse = IMPULSE_GRID[impulse_idx]

        # Simulate both scenarios
        final_no_wall = simulate_trajectory(theta, impulse, False)
        final_with_wall = simulate_trajectory(theta, impulse, True)

        # Simple heuristic: more difference suggests wall was more likely to matter
        difference = abs(final_no_wall - final_with_wall)
        # Convert to rough probability (this is a simplified version)
        wall_prob = 1 / (1 + jnp.exp(-5 * (difference - 0.2)))  # Sigmoid mapping

        wall_probs.append(float(wall_prob))
        action_labels.append(
            f"{label}\n(θ={theta * 180 / jnp.pi:.0f}°, ι={impulse:.1f})"
        )

    # Plot 1: Action informativeness
    bars = ax1.bar(
        range(len(wall_probs)),
        wall_probs,
        color=["lightcoral", "gold", "darkred", "lightblue"],
        alpha=0.8,
    )
    ax1.set_ylabel("Inferred Wall Probability", fontsize=12)
    ax1.set_title(
        "Action Informativeness\nHow much does each action reveal about wall presence?",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(range(len(wall_probs)))
    ax1.set_xticklabels(
        [f"Action {i + 1}" for i in range(len(wall_probs))], fontsize=10
    )
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add probability labels on bars
    for i, (bar, prob) in enumerate(zip(bars, wall_probs)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{prob:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Trajectory comparison for most informative action
    most_informative_idx = jnp.argmax(jnp.array(wall_probs))
    theta_idx, impulse_idx, label = representative_actions[most_informative_idx]
    theta = ANGLE_GRID[theta_idx]
    impulse = IMPULSE_GRID[impulse_idx]

    # Simulate detailed trajectories
    def get_trajectory_points(wall_present):
        initial_vx = impulse * jnp.cos(theta)
        initial_vy = impulse * jnp.sin(theta)
        state = jnp.array([AGENT_START[0], AGENT_START[1], initial_vx, initial_vy])

        positions = [state[:2]]
        for _ in range(TIME_STEPS):
            state = physics_step(state, wall_present)
            positions.append(state[:2])
            if state[1] <= AGENT_RADIUS + 0.001:  # Hit ground
                break

        return jnp.array(positions)

    traj_no_wall = get_trajectory_points(False)
    traj_with_wall = get_trajectory_points(True)

    # Draw environment
    ax2.add_patch(
        patches.Rectangle(
            (GOAL_CENTER[0] - GOAL_SIZE[0] / 2, GOAL_CENTER[1] - GOAL_SIZE[1] / 2),
            GOAL_SIZE[0],
            GOAL_SIZE[1],
            color="green",
            alpha=0.3,
            label="Goal",
        )
    )
    ax2.add_patch(
        patches.Rectangle(
            (WALL_CENTER[0] - WALL_SIZE[0] / 2, WALL_CENTER[1] - WALL_SIZE[1] / 2),
            WALL_SIZE[0],
            WALL_SIZE[1],
            color="red",
            alpha=0.7,
            label="Wall",
        )
    )
    ax2.axhline(y=0, color="black", linewidth=2)

    # Plot trajectories
    ax2.plot(
        traj_no_wall[:, 0],
        traj_no_wall[:, 1],
        "b-",
        linewidth=3,
        alpha=0.8,
        label="No Wall",
    )
    ax2.plot(
        traj_with_wall[:, 0],
        traj_with_wall[:, 1],
        "r--",
        linewidth=3,
        alpha=0.8,
        label="With Wall",
    )
    ax2.scatter(
        [traj_no_wall[-1, 0], traj_with_wall[-1, 0]],
        [traj_no_wall[-1, 1], traj_with_wall[-1, 1]],
        c=["blue", "red"],
        s=100,
        zorder=5,
    )
    ax2.scatter(
        AGENT_START[0],
        AGENT_START[1],
        color="black",
        s=150,
        marker="o",
        zorder=10,
        label="Start",
    )

    ax2.set_xlim(-0.1, 2.0)
    ax2.set_ylim(-0.05, 0.6)
    ax2.set_xlabel("X Position", fontsize=12)
    ax2.set_ylabel("Y Position", fontsize=12)
    ax2.set_title(
        f"Most Informative Action\n{action_labels[most_informative_idx]}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Plot 3: Agent preference differences
    # Show how different agent types prefer different actions given wall presence
    goal_focused_prefs = jax.nn.softmax(
        jnp.array(
            [
                compute_utility(ANGLE_GRID[a[0]], IMPULSE_GRID[a[1]], True, 0.8)
                for a in [act[:2] for act in representative_actions]
            ]
        )
        / 3.0
    )

    effort_focused_prefs = jax.nn.softmax(
        jnp.array(
            [
                compute_utility(ANGLE_GRID[a[0]], IMPULSE_GRID[a[1]], True, 0.2)
                for a in [act[:2] for act in representative_actions]
            ]
        )
        / 3.0
    )

    x = np.arange(len(representative_actions))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        goal_focused_prefs,
        width,
        label="Goal-Focused Agent",
        color="darkgreen",
        alpha=0.7,
    )
    bars2 = ax3.bar(
        x + width / 2,
        effort_focused_prefs,
        width,
        label="Effort-Focused Agent",
        color="orange",
        alpha=0.7,
    )

    ax3.set_ylabel("Action Probability", fontsize=12)
    ax3.set_title(
        "Agent Type Preferences\n(Given Wall is Present)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [f"Action {i + 1}" for i in range(len(representative_actions))], fontsize=10
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Summary insight
    ax4.text(
        0.5,
        0.8,
        "Key Insight",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.5,
        0.6,
        "Rational agents adapt their actions to environmental constraints.\nBy observing actions, we can infer hidden aspects of the environment.",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    ax4.text(
        0.5,
        0.35,
        "Examples:",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    insight_text = """• High-impulse, low-angle trajectories reveal wall collisions
• Different agent types show distinct action preferences  
• Action informativeness varies dramatically across action space
• Model-inference co-design enables tractable 'inverse psychology'"""

    ax4.text(
        0.5,
        0.15,
        insight_text,
        ha="center",
        va="center",
        fontsize=11,
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    return fig


def timing_comparison_fig():
    """Create timing comparison visualization."""
    print("Running timing benchmarks...")

    # Run timing tests
    genjax_times, (genjax_mean, genjax_std) = genjax_timing(num_samples=100, repeats=20)
    physics_times, (physics_mean, physics_std) = physics_timing(
        num_trajectories=100, repeats=20
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Timing comparison
    methods = ["GenJAX Inference", "Physics Simulation"]
    means = [genjax_mean * 1000, physics_mean * 1000]  # Convert to ms
    stds = [genjax_std * 1000, physics_std * 1000]

    bars = ax1.bar(
        methods, means, yerr=stds, capsize=5, alpha=0.7, color=["blue", "orange"]
    )
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Performance Comparison", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.1f}±{std:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Timing distribution
    ax2.hist(
        genjax_times * 1000, bins=15, alpha=0.7, label="GenJAX Inference", color="blue"
    )
    ax2.hist(
        physics_times * 1000,
        bins=15,
        alpha=0.7,
        label="Physics Simulation",
        color="orange",
    )
    ax2.set_xlabel("Time (ms)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Timing Distributions", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(fig, filename, dpi=300):
    """Save figure with standardized settings."""
    import os

    # Create figs directory in the current module directory
    current_dir = os.path.dirname(__file__)
    figs_dir = os.path.join(current_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Save as PDF
    filepath = os.path.join(figs_dir, f"{filename}.pdf")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", format="pdf")
    print(f"Saved figure: {filepath}")

    return filepath
