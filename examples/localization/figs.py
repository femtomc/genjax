"""
Visualization functions for the localization case study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
from core import Pose, World

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


def plot_world(world: World, ax=None):
    """Plot the world boundaries."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw world boundaries
    rect = plt.Rectangle(
        (0, 0), world.width, world.height, fill=False, edgecolor="black", linewidth=2
    )
    ax.add_patch(rect)

    ax.set_xlim(-0.5, world.width + 0.5)
    ax.set_ylim(-0.5, world.height + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    return ax


def plot_pose(pose: Pose, ax=None, color="red", label=None, arrow_length=0.5):
    """Plot a single robot pose with position and heading."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot position
    ax.scatter(pose.x, pose.y, color=color, s=100, label=label, zorder=5)

    # Plot heading arrow
    dx = arrow_length * jnp.cos(pose.theta)
    dy = arrow_length * jnp.sin(pose.theta)
    ax.arrow(
        pose.x,
        pose.y,
        dx,
        dy,
        head_width=0.1,
        head_length=0.1,
        fc=color,
        ec=color,
        alpha=0.7,
        zorder=4,
    )

    return ax


def plot_trajectory(
    poses,
    world: World,
    ax=None,
    color="blue",
    label="Trajectory",
    show_arrows=True,
    arrow_interval=2,
):
    """Plot a trajectory of poses."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_world(world, ax)

    # Handle both single vectorized pose and list of poses
    if hasattr(poses, "x") and hasattr(poses.x, "__len__"):
        # Vectorized pose from JAX Scan - poses is a single Pose with arrays
        xs = poses.x
        ys = poses.y
        thetas = poses.theta
        pose_list = [Pose(xs[i], ys[i], thetas[i]) for i in range(len(xs))]
    else:
        # List of individual poses
        xs = [pose.x for pose in poses]
        ys = [pose.y for pose in poses]
        pose_list = poses

    # Plot trajectory line
    ax.plot(xs, ys, color=color, linewidth=2, label=label, alpha=0.8)

    # Plot poses with arrows
    if show_arrows:
        for i, pose in enumerate(pose_list):
            if i % arrow_interval == 0:
                plot_pose(pose, ax, color=color, arrow_length=0.3)

    # Mark start and end
    ax.scatter(xs[0], ys[0], color="green", s=150, marker="o", label="Start", zorder=6)
    ax.scatter(xs[-1], ys[-1], color="red", s=150, marker="s", label="End", zorder=6)

    return ax


def plot_particles(
    particles, world: World, weights=None, ax=None, alpha=0.6, size_scale=100
):
    """Plot particle distribution with optional weights."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_world(world, ax)

    # Extract positions
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]

    # Size particles by weight if provided
    if weights is not None:
        # Normalize weights for visualization
        normalized_weights = weights / jnp.max(weights)
        sizes = size_scale * normalized_weights
    else:
        sizes = size_scale / 10  # Default small size

    # Plot particles
    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        alpha=alpha,
        c=range(len(particles)),
        cmap="viridis",
        label=f"Particles (n={len(particles)})",
        zorder=3,
    )

    return ax, scatter


def plot_particle_filter_step(
    true_pose, particles, world: World, weights=None, step_num=0, save_path=None
):
    """Plot a single step of particle filter with true pose and particles."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot world
    plot_world(world, ax)

    # Plot particles
    plot_particles(particles, world, weights, ax, alpha=0.7)

    # Plot true pose
    plot_pose(true_pose, ax, color="red", label="True Pose", arrow_length=0.4)

    # Compute and display weighted mean
    if weights is not None:
        weights_norm = weights / jnp.sum(weights)
        mean_x = jnp.sum(jnp.array([p.x for p in particles]) * weights_norm)
        mean_y = jnp.sum(jnp.array([p.y for p in particles]) * weights_norm)
        mean_theta = jnp.arctan2(
            jnp.sum(jnp.sin(jnp.array([p.theta for p in particles])) * weights_norm),
            jnp.sum(jnp.cos(jnp.array([p.theta for p in particles])) * weights_norm),
        )
        mean_pose = Pose(mean_x, mean_y, mean_theta)
        plot_pose(
            mean_pose, ax, color="orange", label="Estimated Pose", arrow_length=0.4
        )

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_title(f"Particle Filter - Step {step_num}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_particle_filter_evolution(
    particle_history, weight_history, true_poses, world: World, save_dir=None
):
    """Plot evolution of particle filter over time."""
    n_steps = len(particle_history)

    # Create subplot grid
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_steps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for t in range(n_steps):
        ax = axes[t] if n_steps > 1 else axes[0]

        # Plot world
        plot_world(world, ax)

        # Plot particles at time t
        particles = particle_history[t]
        weights = weight_history[t]
        plot_particles(particles, world, weights, ax, alpha=0.6, size_scale=50)

        # Plot true pose
        if t < len(true_poses):
            plot_pose(true_poses[t], ax, color="red", arrow_length=0.3)

        ax.set_title(f"Step {t + 1}")
        ax.legend().set_visible(False)  # Hide legend for cleaner look

    # Hide unused subplots
    for t in range(n_steps, len(axes)):
        axes[t].set_visible(False)

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/particle_evolution.png", dpi=150, bbox_inches="tight")

    return fig, axes


def plot_estimation_error(true_poses, estimated_poses, save_path=None):
    """Plot estimation error over time."""
    # Compute position errors
    position_errors = []
    for true_pose, est_pose in zip(true_poses, estimated_poses):
        error = jnp.sqrt(
            (true_pose.x - est_pose.x) ** 2 + (true_pose.y - est_pose.y) ** 2
        )
        position_errors.append(error)

    # Compute heading errors (angular difference)
    heading_errors = []
    for true_pose, est_pose in zip(true_poses, estimated_poses):
        angle_diff = jnp.abs(true_pose.theta - est_pose.theta)
        # Wrap to [-pi, pi]
        angle_diff = jnp.where(angle_diff > jnp.pi, 2 * jnp.pi - angle_diff, angle_diff)
        heading_errors.append(angle_diff)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Position error
    ax1.plot(position_errors, "b-", linewidth=2, label="Position Error")
    ax1.set_ylabel("Position Error (units)")
    ax1.set_title("Localization Estimation Error")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Heading error
    ax2.plot(
        np.array(heading_errors) * 180 / jnp.pi,
        "r-",
        linewidth=2,
        label="Heading Error",
    )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Heading Error (degrees)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_sensor_observations(observations, true_distances=None, save_path=None):
    """Plot sensor observations over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot observations
    ax.plot(observations, "b-", linewidth=2, label="Observed Distance", marker="o")

    # Plot true distances if available
    if true_distances is not None:
        ax.plot(true_distances, "r--", linewidth=2, label="True Distance")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Distance to Wall")
    ax.set_title("Sensor Observations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
