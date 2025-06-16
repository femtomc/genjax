"""
Visualization functions for the localization case study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
from .core import Pose, World

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


def plot_world(world: World, ax=None):
    """Plot the world boundaries and internal walls."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Draw world boundaries
    rect = plt.Rectangle(
        (0, 0), world.width, world.height, fill=False, edgecolor="black", linewidth=3
    )
    ax.add_patch(rect)

    # Draw internal walls
    if world.num_walls > 0:
        for i in range(world.num_walls):
            ax.plot(
                [world.walls_x1[i], world.walls_x2[i]],
                [world.walls_y1[i], world.walls_y2[i]],
                color="black",
                linewidth=2,
                alpha=0.8,
            )

    # Add room labels for clarity
    if world.num_walls > 0:
        # Assuming the 3-room layout we designed
        ax.text(
            2,
            8,
            "Room 1",
            fontsize=14,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax.text(
            6,
            8,
            "Room 2",
            fontsize=14,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
        )
        ax.text(
            10,
            8,
            "Room 3",
            fontsize=14,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3),
        )

        # Mark doorways
        ax.text(4, 4, "Door", fontsize=10, ha="center", rotation=90, alpha=0.6)
        ax.text(8, 5, "Door", fontsize=10, ha="center", rotation=90, alpha=0.6)

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
    """Plot evolution of particle filter over time with enhanced grid layout."""
    n_steps = len(particle_history)

    # Show more steps with a larger grid layout
    if n_steps <= 8:
        # For smaller numbers of steps, use 4 columns
        n_cols = 4
        n_rows = (n_steps + n_cols - 1) // n_cols
    elif n_steps <= 16:
        # For medium numbers, use optimal square-ish layout
        n_cols = 4
        n_rows = 4
        n_steps_to_show = min(16, n_steps)  # Show up to 16 steps
    else:
        # For large numbers, use 5x4 layout
        n_cols = 5
        n_rows = 4
        n_steps_to_show = min(20, n_steps)  # Show up to 20 steps

    # If we have more steps than we can show, select them strategically
    if n_steps > n_cols * n_rows:
        n_steps_to_show = n_cols * n_rows
        # Select steps evenly distributed across the trajectory
        step_indices = jnp.linspace(0, n_steps - 1, n_steps_to_show, dtype=int)
    else:
        n_steps_to_show = n_steps
        step_indices = list(range(n_steps))

    # Create larger figure with more space
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Handle different subplot array structures
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, step_idx in enumerate(step_indices):
        ax = axes[i]

        # Plot world
        plot_world(world, ax)

        # Plot particles at time step_idx
        particles = particle_history[step_idx]
        weights = weight_history[step_idx]
        plot_particles(particles, world, weights, ax, alpha=0.7, size_scale=30)

        # Plot true pose
        if step_idx < len(true_poses):
            plot_pose(true_poses[step_idx], ax, color="red", arrow_length=0.25)
        elif step_idx == 0 and len(true_poses) > 0:
            # For initial step, show first true pose
            plot_pose(true_poses[0], ax, color="red", arrow_length=0.25)

        ax.set_title(f"Step {step_idx + 1}", fontsize=10)
        ax.legend().set_visible(False)  # Hide legend for cleaner look

        # Remove axis frame and labels for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused subplots
    for i in range(len(step_indices), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(pad=1.0)

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
    """Plot sensor observations over time.

    Args:
        observations: List of observations. Each can be:
                     - Single float (backward compatibility)
                     - Array of 8 LIDAR distances
        true_distances: Optional true distances for comparison
        save_path: Optional path to save figure
    """
    # Check if we have vector observations (LIDAR) or scalar observations
    first_obs = observations[0]
    is_vector_obs = hasattr(first_obs, "__len__") and len(first_obs) > 1

    if is_vector_obs:
        # Plot LIDAR observations - each ray separately
        n_rays = len(first_obs)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # Define ray angles for labeling (relative to robot heading)
        ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        ray_labels = [
            f"Ray {i} ({angle:.1f} rad)" for i, angle in enumerate(ray_angles)
        ]

        # Colors for each ray
        colors = plt.cm.tab10(np.linspace(0, 1, n_rays))

        for ray_idx in range(n_rays):
            ax = axes[ray_idx]

            # Extract distances for this ray across all time steps
            ray_observations = [
                obs[ray_idx] if hasattr(obs, "__len__") else obs for obs in observations
            ]

            # Plot observed distances for this ray
            ax.plot(
                ray_observations,
                color=colors[ray_idx],
                linewidth=2,
                label="Observed",
                marker="o",
                markersize=4,
            )

            # Plot true distances if available
            if true_distances is not None:
                if hasattr(true_distances[0], "__len__") and len(true_distances[0]) > 1:
                    # True distances are also vectors
                    true_ray_distances = [
                        true_dist[ray_idx] for true_dist in true_distances
                    ]
                    ax.plot(
                        true_ray_distances, "r--", linewidth=2, label="True", alpha=0.7
                    )
                else:
                    # True distances are scalars - use the minimum for all rays
                    ax.plot(
                        true_distances,
                        "r--",
                        linewidth=2,
                        label="True (min)",
                        alpha=0.7,
                    )

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Distance")
            ax.set_title(ray_labels[ray_idx])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Set consistent y-axis limits across all subplots
            ax.set_ylim(0, max(10, max(ray_observations) * 1.1))

        plt.suptitle("LIDAR Sensor Observations (8 Directional Rays)", fontsize=14)
        plt.tight_layout()

    else:
        # Backward compatibility: scalar observations
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

    return fig, axes if is_vector_obs else ax


def plot_ground_truth_trajectory(
    initial_pose, controls, poses, observations, world, save_path=None
):
    """Plot detailed ground truth trajectory with control commands and observations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top left: Trajectory with world
    ax1 = axes[0, 0]
    plot_world(world, ax1)

    # Plot full trajectory including initial pose
    all_poses = [initial_pose] + poses
    plot_trajectory(
        all_poses, world, ax1, color="blue", label="Ground Truth", arrow_interval=1
    )

    # Add step numbers
    for i, pose in enumerate(all_poses[::2]):  # Every other step
        ax1.text(
            pose.x + 0.1,
            pose.y + 0.1,
            str(i * 2),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax1.set_title("Ground Truth Trajectory")
    ax1.legend()

    # Top right: Control commands over time
    ax2 = axes[0, 1]
    velocities = [c.velocity for c in controls]
    angular_velocities = [c.angular_velocity for c in controls]

    ax2_twin = ax2.twinx()

    line1 = ax2.plot(velocities, "b-", marker="o", label="Velocity", linewidth=2)
    line2 = ax2_twin.plot(
        angular_velocities, "r-", marker="s", label="Angular Velocity", linewidth=2
    )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Velocity", color="b")
    ax2_twin.set_ylabel("Angular Velocity (rad/s)", color="r")
    ax2.set_title("Control Commands")
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left")

    # Bottom left: Position over time
    ax3 = axes[1, 0]
    xs = [initial_pose.x] + [p.x for p in poses]
    ys = [initial_pose.y] + [p.y for p in poses]
    thetas = [initial_pose.theta] + [p.theta for p in poses]

    ax3.plot(xs, "b-", marker="o", label="X Position", linewidth=2)
    ax3.plot(ys, "g-", marker="s", label="Y Position", linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(thetas, "r-", marker="^", label="Heading (rad)", linewidth=2)

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Position")
    ax3_twin.set_ylabel("Heading (rad)", color="r")
    ax3.set_title("Position and Heading Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    # Bottom right: Sensor observations
    ax4 = axes[1, 1]
    ax4.plot(observations, "b-", marker="o", label="Observed Distance", linewidth=2)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Distance to Wall")
    ax4.set_title("Sensor Observations")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_multiple_trajectories(trajectory_data_list, world, save_path=None):
    """Plot multiple ground truth trajectories for comparison."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot world
    plot_world(world, ax)

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (traj_type, (initial_pose, controls, poses, observations)) in enumerate(
        trajectory_data_list
    ):
        color = colors[i % len(colors)]
        all_poses = [initial_pose] + poses

        # Plot trajectory
        plot_trajectory(
            all_poses,
            world,
            ax,
            color=color,
            label=f"{traj_type} ({len(poses)} steps)",
            show_arrows=False,
        )

        # Mark start and end specifically for each trajectory
        ax.scatter(
            initial_pose.x,
            initial_pose.y,
            color=color,
            s=200,
            marker="o",
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.scatter(
            poses[-1].x,
            poses[-1].y,
            color=color,
            s=200,
            marker="s",
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )

    ax.set_title("Multiple Ground Truth Trajectories")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
