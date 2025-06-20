"""
Visualization functions for the localization case study.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from .core import Pose, World

# Set up plotting style
plt.style.use("default")


def plot_world(world: World, ax=None, room_label_fontsize=14):
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
            fontsize=room_label_fontsize,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax.text(
            6,
            8,
            "Room 2",
            fontsize=room_label_fontsize,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
        )
        ax.text(
            10,
            8,
            "Room 3",
            fontsize=room_label_fontsize,
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


def plot_lidar_rays(
    pose: Pose,
    world,
    distances=None,
    ax=None,
    n_angles=128,
    max_range=10.0,
    show_measurements=True,
    ray_color="cyan",
    measurement_color="orange",
    show_rays=True,
    ray_alpha=0.25,
):
    """Plot LIDAR measurement rays from robot pose.

    Args:
        pose: Robot pose (position and heading)
        world: World geometry for ray intersection
        distances: Optional array of measured distances (if None, computes true distances)
        ax: Matplotlib axis to plot on
        n_angles: Number of LIDAR rays
        max_range: Maximum LIDAR range
        show_measurements: Whether to show measurement endpoints
        ray_color: Color for the rays
        measurement_color: Color for measurement endpoints
        show_rays: Whether to show the ray lines (default True)
        ray_alpha: Transparency level for ray lines (default 0.25)

    Returns:
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Remove axis frame and grid for clean LIDAR visualization
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Import distance computation function
    from .core import distance_to_wall_lidar

    # Get true distances if not provided
    if distances is None:
        distances = distance_to_wall_lidar(pose, world, n_angles, max_range)

    # Create angular grid around robot (relative to robot's heading)
    angles = jnp.linspace(0, 2 * jnp.pi, n_angles, endpoint=False)
    world_angles = pose.theta + angles

    # Plot each ray
    for i, (angle, distance) in enumerate(zip(world_angles, distances)):
        # Ray endpoint
        end_x = pose.x + distance * jnp.cos(angle)
        end_y = pose.y + distance * jnp.sin(angle)

        # Plot ray line only if requested
        if show_rays:
            ax.plot(
                [pose.x, end_x],
                [pose.y, end_y],
                color=ray_color,
                linewidth=1.0,
                alpha=ray_alpha,
                zorder=2,
            )

        # Plot measurement endpoint
        if show_measurements:
            ax.scatter(
                end_x,
                end_y,
                color=measurement_color,
                s=20,
                alpha=0.8,
                zorder=3,
                edgecolor="black",
                linewidth=0.3,
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
    true_pose,
    particles,
    world: World,
    weights=None,
    step_num=0,
    save_path=None,
    show_lidar_rays=True,
    observations=None,
    n_rays=8,
):
    """Plot a single step of particle filter with true pose and particles."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot world
    plot_world(world, ax)

    # Remove grid and axis frame for clean LIDAR visualization
    if show_lidar_rays:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot particles
    plot_particles(particles, world, weights, ax, alpha=0.7)

    # Plot LIDAR rays from true pose (rays to walls, but show observed measurement points)
    if show_lidar_rays:
        from .core import distance_to_wall_lidar

        true_distances = distance_to_wall_lidar(true_pose, world, n_angles=n_rays)
        # Plot rays to true wall intersections but don't show the true measurement points
        plot_lidar_rays(
            true_pose,
            world,
            distances=true_distances,
            ax=ax,
            n_angles=n_rays,
            ray_color="cyan",
            measurement_color="black",
            show_rays=True,
            show_measurements=False,
        )

        # Show observed measurement points if available
        if observations is not None:
            plot_lidar_rays(
                true_pose,
                world,
                distances=observations,
                ax=ax,
                n_angles=n_rays,
                ray_color="cyan",
                measurement_color="orange",
                show_rays=False,
                show_measurements=True,
            )

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

    # Add LIDAR measurements to legend if shown
    if show_lidar_rays:
        ax.plot([], [], color="cyan", linewidth=1.0, alpha=0.25, label="LIDAR Rays")
        ax.scatter(
            [],
            [],
            color="orange",
            s=20,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
            label=f"Observed Measurements ({n_rays} rays)",
        )

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_title(f"Particle Filter - Step {step_num}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_particle_filter_evolution(
    particle_history,
    weight_history,
    true_poses,
    world: World,
    save_dir=None,
    show_lidar_rays=True,
    observations_history=None,
    n_rays=8,
):
    """Plot evolution of particle filter over time with enhanced grid layout."""
    n_steps = len(particle_history)

    # Fixed layout: 1 row, 4 columns showing specific timesteps
    n_rows = 1
    n_cols = 4

    # Show specific timesteps: 1, 6, 11, and 16 (0-indexed: 0, 5, 10, 15)
    step_indices = [0, 5, 10, 15]  # These correspond to steps 1, 6, 11, 16

    # Filter out indices that don't exist
    step_indices = [idx for idx in step_indices if idx < n_steps]
    n_steps_to_show = len(step_indices)

    # Create figure with horizontal layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle different subplot array structures for (1, 4) layout
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

    for i, step_idx in enumerate(step_indices):
        ax = axes[i]

        # Plot world with smaller room labels for compact grid view
        plot_world(world, ax, room_label_fontsize=8)

        # Plot particles at time step_idx
        particles = particle_history[step_idx]
        weights = weight_history[step_idx]
        plot_particles(particles, world, weights, ax, alpha=0.7, size_scale=30)

        # Plot LIDAR rays if requested (rays to walls, but show observed measurement points)
        if show_lidar_rays and step_idx < len(true_poses):
            from .core import distance_to_wall_lidar

            true_distances = distance_to_wall_lidar(
                true_poses[step_idx], world, n_angles=n_rays
            )
            # Plot rays to true wall intersections but don't show the true measurement points
            plot_lidar_rays(
                true_poses[step_idx],
                world,
                distances=true_distances,
                ax=ax,
                n_angles=n_rays,
                ray_color="cyan",
                measurement_color="black",
                show_rays=True,
                show_measurements=False,
            )

            # Show observed measurement points if available
            if observations_history is not None and step_idx < len(
                observations_history
            ):
                observations = observations_history[step_idx]
                plot_lidar_rays(
                    true_poses[step_idx],
                    world,
                    distances=observations,
                    ax=ax,
                    n_angles=n_rays,
                    ray_color="cyan",
                    measurement_color="orange",
                    show_rays=False,
                    show_measurements=True,
                )

        # Plot true pose
        # step_idx corresponds to particle timestep after incorporating observations[step_idx]
        # So we want to show the pose where observations[step_idx] was taken: true_poses[step_idx]
        if step_idx < len(true_poses):
            plot_pose(true_poses[step_idx], ax, color="red", arrow_length=0.25)

        # Remove default title and legend for cleaner look
        ax.set_title("")
        ax.legend().set_visible(False)

        # Add bolded step number in upper left corner of the room
        ax.text(
            0.5,
            world.height - 0.5,
            f"{step_idx + 1}",
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="top",
        )

        # Remove axis frame, grid, and labels for cleaner appearance
        ax.grid(False)
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
        plt.savefig(f"{save_dir}/particle_evolution.pdf", dpi=150, bbox_inches="tight")

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

        # For many rays, show only a subset or create a different visualization
        if n_rays > 16:
            # For large numbers of rays, show aggregate plot instead of individual rays
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot all observations as lines
            for t, obs in enumerate(observations):
                ax.plot(obs, alpha=0.3, color="blue", linewidth=1)

            # Plot mean and std envelope
            obs_array = np.array(observations)
            obs_mean = np.mean(obs_array, axis=0)
            obs_std = np.std(obs_array, axis=0)
            ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

            ax.plot(
                ray_angles,
                obs_mean,
                "b-",
                linewidth=3,
                label=f"Mean Distance ({n_rays} rays)",
            )
            ax.fill_between(
                ray_angles,
                obs_mean - obs_std,
                obs_mean + obs_std,
                alpha=0.2,
                color="blue",
                label="± 1 std",
            )

            # Plot true distances if available
            if true_distances is not None and len(true_distances) > 0:
                true_array = np.array(true_distances)
                true_mean = np.mean(true_array, axis=0)
                ax.plot(
                    ray_angles,
                    true_mean,
                    "r--",
                    linewidth=2,
                    label="True Mean Distance",
                )

            ax.set_xlabel("Ray Angle (radians)")
            ax.set_ylabel("Distance")
            ax.set_title(f"LIDAR Sensor Observations ({n_rays} rays)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            return fig, ax
        else:
            # For smaller numbers of rays, use individual subplot approach
            n_cols = min(4, n_rays)
            n_rows = (n_rays + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
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
                    obs[ray_idx] if hasattr(obs, "__len__") else obs
                    for obs in observations
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
                    if (
                        hasattr(true_distances[0], "__len__")
                        and len(true_distances[0]) > 1
                    ):
                        # True distances are also vectors
                        true_ray_distances = [
                            true_dist[ray_idx] for true_dist in true_distances
                        ]
                        ax.plot(
                            true_ray_distances,
                            "r--",
                            linewidth=2,
                            label="True",
                            alpha=0.7,
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

            # Hide unused subplots
            for i in range(n_rays, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(
                f"LIDAR Sensor Observations ({n_rays} Directional Rays)", fontsize=14
            )
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


def plot_lidar_demo(pose: Pose, world: World, save_path=None, n_rays=8):
    """Demo plot showing LIDAR measurements from a robot pose."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot world
    plot_world(world, ax)

    # Remove axis frames and grid for clean room-only visualization
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Plot robot pose
    plot_pose(pose, ax, color="red", label="Robot", arrow_length=0.5)

    # Plot LIDAR measurements with both true and noisy observations
    from .core import distance_to_wall_lidar, sensor_model_single_ray
    import jax.random as jrand

    # Get true distances
    true_distances = distance_to_wall_lidar(pose, world, n_angles=n_rays)

    # Generate noisy observations (simulating sensor noise)
    key = jrand.PRNGKey(42)
    keys = jrand.split(key, n_rays)
    noisy_distances = []
    for i, (true_dist, subkey) in enumerate(zip(true_distances, keys)):
        # Simulate sensor noise
        from genjax import seed

        seeded_sensor = seed(sensor_model_single_ray.simulate)
        trace = seeded_sensor(subkey, true_dist, i)
        noisy_dist = trace.get_retval()
        noisy_distances.append(noisy_dist)

    # Plot true LIDAR measurements (black points with transparent rays)
    plot_lidar_rays(
        pose,
        world,
        distances=true_distances,
        ax=ax,
        n_angles=n_rays,
        ray_color="darkgray",
        measurement_color="black",
        show_rays=True,
        ray_alpha=0.15,
    )

    # Plot noisy observations (orange points with transparent rays)
    plot_lidar_rays(
        pose,
        world,
        distances=jnp.array(noisy_distances),
        ax=ax,
        n_angles=n_rays,
        ray_color="orange",
        measurement_color="orange",
        show_rays=True,
        ray_alpha=0.3,
        show_measurements=True,
    )

    # Add legend entries
    ax.scatter(
        [],
        [],
        color="black",
        s=20,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        label=f"True Measurements ({n_rays} rays)",
    )
    ax.scatter(
        [],
        [],
        color="orange",
        s=20,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        label=f"Noisy Measurements ({n_rays} rays)",
    )

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_title(
        f"LIDAR Sensor Demonstration\n{n_rays}-Ray Distance Measurements with Noise"
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_weight_evolution(weight_history, save_path=None):
    """Plot evolution of particle weight histograms over time.

    Args:
        weight_history: List of weight arrays for each timestep
        save_path: Optional path to save figure

    Returns:
        fig, axes: Figure and axes objects
    """
    n_timesteps = len(weight_history)

    # Select timesteps to show (max 12 for readability)
    if n_timesteps <= 12:
        timesteps_to_show = list(range(n_timesteps))
    else:
        # Show first, last, and evenly distributed middle steps
        timesteps_to_show = (
            [0]
            + list(range(2, n_timesteps - 2, max(1, (n_timesteps - 4) // 10)))
            + [n_timesteps - 1]
        )
        timesteps_to_show = timesteps_to_show[:12]  # Limit to 12

    n_cols = 4
    n_rows = (len(timesteps_to_show) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    # Handle different subplot array structures
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Compute global weight range for consistent x-axis
    all_weights = jnp.concatenate([weights for weights in weight_history])
    weight_min, weight_max = jnp.min(all_weights), jnp.max(all_weights)

    # Use log scale if weights span many orders of magnitude
    if weight_max / weight_min > 1000:
        use_log = True
        log_min, log_max = jnp.log10(weight_min + 1e-12), jnp.log10(weight_max + 1e-12)
        jnp.logspace(log_min, log_max, 50)
    else:
        use_log = False
        jnp.linspace(weight_min, weight_max, 50)

    for i, timestep in enumerate(timesteps_to_show):
        ax = axes[i]
        weights = weight_history[timestep]

        # Normalize weights
        weights_norm = weights / jnp.sum(weights)

        # Compute effective sample size
        ess = 1.0 / jnp.sum(weights_norm**2)

        # Plot histogram - handle uniform weights case
        weight_std = jnp.std(weights_norm)
        if weight_std < 1e-10:  # All weights are essentially uniform
            # For uniform weights, just show a single bar
            uniform_weight = 1.0 / len(weights_norm)
            ax.bar(
                [uniform_weight],
                [len(weights_norm)],
                width=uniform_weight * 0.1,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            # Normal histogram for non-uniform weights
            # Create bins that work with the weight range
            weight_min, weight_max = jnp.min(weights_norm), jnp.max(weights_norm)
            if weight_min == weight_max:
                # All weights are the same but not exactly uniform
                weight_bins = jnp.array(
                    [
                        weight_min - weight_min * 0.1,
                        weight_min,
                        weight_min + weight_min * 0.1,
                    ]
                )
            else:
                weight_bins = jnp.linspace(weight_min, weight_max, 50)
            ax.hist(
                weights_norm,
                bins=weight_bins,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                linewidth=0.5,
            )

        if use_log:
            ax.set_xscale("log")

        # Add statistics as text
        ax.text(
            0.02,
            0.98,
            f"Step {timestep}\nESS: {ess:.1f}\nMax: {jnp.max(weights_norm):.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Normalized Weight")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        # Color code by ESS quality
        if ess > len(weights) * 0.5:  # Good diversity
            ax.set_facecolor((0.9, 1.0, 0.9))  # Light green
        elif ess > len(weights) * 0.1:  # Moderate diversity
            ax.set_facecolor((1.0, 1.0, 0.9))  # Light yellow
        else:  # Poor diversity
            ax.set_facecolor((1.0, 0.9, 0.9))  # Light red

    # Hide unused subplots
    for i in range(len(timesteps_to_show), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Particle Weight Distribution Evolution", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_weight_flow(weight_data, save_path=None):
    """Plot particle weight distribution evolution as horizontal raincloud plots.

    Shows weight distribution characteristics over time with:
    - X-axis: Weight magnitude
    - Y-axis: Timestep
    - Raincloud plots (violin + box + scatter) for each timestep

    Args:
        weight_data: Either:
                    - List of diagnostic weight arrays from state API
                    - List of regular weight arrays (backward compatibility)
                    - None (uses uniform weights placeholder)
        save_path: Optional path to save figure

    Returns:
        fig, ax: Figure and axis objects
    """
    # Import our custom raincloud plot
    from genjax.viz.raincloud import horizontal_raincloud

    if weight_data is None:
        # Handle case where no diagnostic weights available
        print(
            "No diagnostic weights available. Using uniform weights for visualization."
        )
        # Create placeholder uniform weights for visualization
        n_timesteps = 5
        n_particles = 100
        weight_history = [
            jnp.ones(n_particles) / n_particles for _ in range(n_timesteps)
        ]
    elif hasattr(weight_data, "shape") and len(weight_data.shape) == 2:
        # We have a 2D array (timesteps, particles) - this is diagnostic weights
        T, N = weight_data.shape
        print(f"Processing diagnostic weights: {T} timesteps, {N} particles")

        # These are log normalized weights, convert to linear
        weight_history = []
        for t in range(T):
            # Convert log weights to linear weights
            log_weights_t = weight_data[t]
            linear_weights_t = jnp.exp(log_weights_t)
            # Normalize to sum to 1
            linear_weights_t = linear_weights_t / jnp.sum(linear_weights_t)
            weight_history.append(linear_weights_t)
    elif isinstance(weight_data, list) and len(weight_data) > 0:
        # Check if we have log weights (from diagnostics) or linear weights
        first_weights = weight_data[0]
        if jnp.min(first_weights) < 0:  # Likely log weights
            # Convert log weights to linear for plotting
            weight_history = [jnp.exp(log_weights) for log_weights in weight_data]
        else:
            # Already linear weights
            weight_history = weight_data
    else:
        # Fallback to uniform weights
        print("Invalid weight data. Using uniform weights for visualization.")
        weight_history = [jnp.ones(100) / 100 for _ in range(5)]

    n_timesteps = len(weight_history)

    # Prepare labels for timesteps
    labels = [f"t={t}" for t in range(n_timesteps)]

    # Create colors with ESS-based coloring
    colors = []
    ess_values = []

    for weights in weight_history:
        # Normalize weights if not already normalized
        if jnp.abs(jnp.sum(weights) - 1.0) > 1e-6:
            weights_norm = weights / jnp.sum(weights)
        else:
            weights_norm = weights

        # Compute ESS
        ess = 1.0 / jnp.sum(weights_norm**2)
        ess_values.append(ess)

        # Color based on ESS quality
        n_particles = len(weights)
        if ess < n_particles * 0.1:
            colors.append("lightcoral")  # Poor ESS
        elif ess < n_particles * 0.5:
            colors.append("orange")  # Fair ESS
        else:
            colors.append("lightgreen")  # Good ESS

    # Create the raincloud plot (squeezed vertically)
    fig, ax = plt.subplots(figsize=(14, 5))

    horizontal_raincloud(
        data=weight_history,
        labels=labels,
        ax=ax,
        colors=colors,
        width_violin=0.3,
        width_box=0.1,
        jitter=0.05,
        point_size=8,
        alpha=0.7,
        orient="v",  # Changed to vertical to make time flow horizontally
    )

    # Add ESS annotations (adjusted for horizontal time flow)
    for t, ess in enumerate(ess_values):
        weight_max = jnp.max(weight_history[t])
        ax.text(
            t,
            weight_max * 1.1,
            f"ESS: {ess:.1f}",
            fontsize=10,
            va="bottom",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Customize the plot for horizontal time flow
    ax.set_ylabel("Particle Weight (normalized)", fontsize=12)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add ESS quality legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightcoral", alpha=0.7, label="Poor ESS (<10%)"),
        Patch(facecolor="orange", alpha=0.7, label="Fair ESS (10-50%)"),
        Patch(facecolor="lightgreen", alpha=0.7, label="Good ESS (>50%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_multiple_trajectories(trajectory_data_list, world, save_path=None):
    """Plot multiple ground truth trajectories for comparison."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot world
    plot_world(world, ax)

    # Remove axis frames and grid for clean room-only visualization
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (traj_type, (all_poses, controls, observations)) in enumerate(
        trajectory_data_list
    ):
        color = colors[i % len(colors)]

        # Plot trajectory
        plot_trajectory(
            all_poses,
            world,
            ax,
            color=color,
            label=f"{traj_type} ({len(all_poses)} steps)",
            show_arrows=False,
        )

        # Mark start and end specifically for each trajectory
        ax.scatter(
            all_poses[0].x,
            all_poses[0].y,
            color=color,
            s=200,
            marker="o",
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )
        ax.scatter(
            all_poses[-1].x,
            all_poses[-1].y,
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


def plot_smc_timing_comparison(
    benchmark_results, save_path=None, n_particles=200, K=10
):
    """Create horizontal bar chart comparing SMC method timing."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme for different SMC methods
    colors = {
        "smc_basic": "#1f77b4",  # Blue - Bootstrap filter
        "smc_mh": "#ff7f0e",  # Orange - MH rejuvenation
        "smc_hmc": "#2ca02c",  # Green - HMC rejuvenation
        "smc_locally_optimal": "#d62728",  # Red - Locally optimal proposal
    }

    method_labels = {
        "smc_basic": f"Bootstrap filter (N={n_particles})",
        "smc_mh": f"SMC (N={n_particles}) + MH (K={K})",
        "smc_hmc": f"SMC (N={n_particles}) + HMC (K={K})",
        "smc_locally_optimal": f"SMC (N={n_particles}) + Locally Optimal (K={K})",
    }

    # Extract timing data
    method_names = []
    timing_means = []
    timing_stds = []
    method_colors = []

    for method_name, result in benchmark_results.items():
        if method_name in colors:
            method_names.append(method_labels[method_name])
            timing_mean = (
                float(result["timing_stats"][0]) * 1000
            )  # Convert to milliseconds
            timing_std = (
                float(result["timing_stats"][1]) * 1000
            )  # Convert to milliseconds
            timing_means.append(timing_mean)
            timing_stds.append(timing_std)
            method_colors.append(colors[method_name])

    # Create horizontal bar chart
    y_pos = jnp.arange(len(method_names))
    bars = ax.barh(
        y_pos, timing_means, xerr=timing_stds, color=method_colors, alpha=0.7, capsize=5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names, fontsize=18)
    ax.set_xlabel("Time (milliseconds)", fontsize=16)
    ax.set_title("SMC Method Timing Comparison", fontsize=20, fontweight="bold", pad=20)
    ax.tick_params(labelsize=16)
    ax.grid(True, axis="x", alpha=0.3)

    # Add timing values as text on bars
    for i, (bar, mean, std) in enumerate(zip(bars, timing_means, timing_stds)):
        width = bar.get_width()
        mean_val = float(mean)
        std_val = float(std)
        ax.text(
            width + std_val + max(timing_means) * 0.01,  # Position after error bar
            bar.get_y() + bar.get_height() / 2,  # Center vertically
            f"{mean_val:.1f}±{std_val:.1f}ms",  # Format as mean±std in milliseconds
            ha="left",
            va="center",
            fontsize=16,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_smc_method_comparison(
    benchmark_results,
    true_poses,
    world,
    save_path=None,
    n_rays=8,
    n_particles=200,
    K=10,
):
    """Create comprehensive comparison plot for different SMC methods."""
    n_methods = len(benchmark_results)

    # Create 4-row layout: first timestep, final particles, rainclouds, timing
    fig = plt.figure(figsize=(6 * n_methods, 20))
    # Create main gridspec with 4 rows
    gs_main = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.4)
    # First row for initial particle distributions
    gs_first = gs_main[0].subgridspec(1, n_methods, hspace=0.05)
    # Second row for final particle distributions
    gs_second = gs_main[1].subgridspec(1, n_methods, hspace=0.05)
    # Third row for raincloud plots
    gs_third = gs_main[2].subgridspec(1, n_methods, hspace=0.05)
    # Bottom section for timing
    gs_bottom = gs_main[3]

    method_labels = {
        "smc_basic": f"Bootstrap filter (N={n_particles})",
        "smc_mh": f"SMC (N={n_particles}) + MH (K={K})",
        "smc_hmc": f"SMC (N={n_particles}) + HMC (K={K})",
        "smc_locally_optimal": f"SMC (N={n_particles}) + Locally Optimal (K={K})",
    }

    colors = {
        "smc_basic": "#1f77b4",
        "smc_mh": "#ff7f0e",
        "smc_hmc": "#2ca02c",
        "smc_locally_optimal": "#d62728",
    }

    # Grayscale colors for raincloud plots
    grayscale_colors = {
        "smc_basic": "#404040",
        "smc_mh": "#606060",
        "smc_hmc": "#808080",
        "smc_locally_optimal": "#a0a0a0",
    }

    for i, (method_name, result) in enumerate(benchmark_results.items()):
        particle_history = result["particle_history"]
        weight_history = result["weight_history"]
        diagnostic_weights = result["diagnostic_weights"]

        # First row: Initial particle distribution (timestep 1)
        ax_first = fig.add_subplot(gs_first[i])
        plot_world(world, ax_first)

        # Remove axis frames for clean visualization
        ax_first.grid(False)
        for spine in ax_first.spines.values():
            spine.set_visible(False)
        ax_first.set_xticks([])
        ax_first.set_yticks([])
        ax_first.set_xlabel("")
        ax_first.set_ylabel("")

        # Plot first timestep particle distribution
        first_particles = particle_history[0]
        first_weights = weight_history[0]
        plot_particles(first_particles, world, first_weights, ax_first, alpha=0.8)

        # Plot true first pose
        plot_pose(
            true_poses[0], ax_first, color="red", label="True Pose", arrow_length=0.4
        )

        ax_first.set_title("", fontsize=20, fontweight="bold")
        ax_first.legend().set_visible(False)

        # Color-code the method with frame
        for spine in ax_first.spines.values():
            spine.set_color(colors[method_name])
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Add "Start" label on the leftmost plot only
        if i == 0:
            ax_first.text(
                -0.1,
                0.5,
                "Start",
                transform=ax_first.transAxes,
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
            )

        # Second row: Final particle distribution
        ax_particles = fig.add_subplot(gs_second[i])
        plot_world(world, ax_particles)

        # Remove axis frames for clean visualization
        ax_particles.grid(False)
        for spine in ax_particles.spines.values():
            spine.set_visible(False)
        ax_particles.set_xticks([])
        ax_particles.set_yticks([])
        ax_particles.set_xlabel("")
        ax_particles.set_ylabel("")

        # Plot final particle distribution
        final_particles = particle_history[-1]
        final_weights = weight_history[-1]
        plot_particles(final_particles, world, final_weights, ax_particles, alpha=0.8)

        # Plot true final pose
        plot_pose(
            true_poses[-1],
            ax_particles,
            color="red",
            label="True Pose",
            arrow_length=0.4,
        )

        ax_particles.set_title("", fontsize=20, fontweight="bold")  # Remove title
        ax_particles.legend().set_visible(False)

        # Color-code the method with frame
        for spine in ax_particles.spines.values():
            spine.set_color(colors[method_name])
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Add "End" label on the leftmost plot only
        if i == 0:
            ax_particles.text(
                -0.1,
                0.5,
                "End",
                transform=ax_particles.transAxes,
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
            )

        # Third row: Raincloud plot with ESS (grayscale)
        ax_weights = fig.add_subplot(gs_third[i])

        # Remove frames except bottom x-axis
        ax_weights.grid(False)
        for spine_name, spine in ax_weights.spines.items():
            if spine_name == "bottom":
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1)
            else:
                spine.set_visible(False)

        if diagnostic_weights is not None and hasattr(diagnostic_weights, "shape"):
            from genjax.viz.raincloud import diagnostic_raincloud

            T, N = diagnostic_weights.shape
            ess_values = []
            ess_colors = []

            # Show selected timesteps with raincloud visualization
            selected_timesteps = [1, 6, 11, 16]
            for plot_pos, t in enumerate([t for t in selected_timesteps if t < T]):
                linear_weights = jnp.exp(diagnostic_weights[t])
                linear_weights = linear_weights / jnp.sum(linear_weights)

                # Use the modular raincloud function with particle count for ESS coloring
                ess, ess_color = diagnostic_raincloud(
                    ax_weights,
                    linear_weights,
                    position=plot_pos,  # Use plot position instead of timestep
                    color=grayscale_colors[method_name],
                    width=0.3,
                    n_particles=N,  # Pass particle count for ESS quality assessment
                )
                ess_values.append(ess)
                ess_colors.append(ess_color)

            # Add ESS values as color-coded text annotations on the right side
            # Also add timestep labels on the left
            selected_timesteps = [1, 6, 11, 16]
            valid_timesteps = [t for t in selected_timesteps if t < T]

            for plot_pos, (timestep, ess, ess_color) in enumerate(
                zip(valid_timesteps, ess_values, ess_colors)
            ):
                # ESS annotation on the right
                ax_weights.text(
                    0.98,
                    plot_pos,
                    f"ESS: {ess:.0f}",
                    fontsize=16,
                    va="center",
                    ha="right",
                    fontweight="bold",
                    color=ess_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                    transform=ax_weights.get_yaxis_transform(),
                )

            # Update y-axis to show the selected timesteps
            ax_weights.set_ylim(-0.5, len(valid_timesteps) - 0.5)
            ax_weights.set_yticks(range(len(valid_timesteps)))
            if i == 0:
                # Only show y-axis labels on the leftmost plot
                ax_weights.set_yticklabels(
                    [f"{t}" for t in valid_timesteps], fontsize=12
                )
            else:
                # Hide y-axis labels for columns 2 and 3
                ax_weights.set_yticklabels([])
                ax_weights.tick_params(left=False)  # Hide tick marks too

            # Only show x-axis label on the leftmost plot
            if i == 0:
                ax_weights.set_xlabel("Particle Weight", fontsize=18, fontweight="bold")
                ax_weights.set_ylabel("Timestep", fontsize=18, fontweight="bold")
            else:
                ax_weights.set_xlabel("")
                ax_weights.set_ylabel("")

            # Set x-axis limits and ticks
            ax_weights.set_xlim(0, 1)
            ax_weights.set_xticks([0, 1])
            ax_weights.set_xticklabels(["0", "1"], fontsize=14)
            ax_weights.tick_params(labelsize=14)
        ax_weights.set_title("")  # Remove title to save space

    # Bottom row: Timing comparison (spans all columns)
    ax_timing = fig.add_subplot(gs_bottom)

    # Extract timing data
    method_names = []
    timing_means = []
    timing_stds = []
    method_colors = []

    for method_name, result in benchmark_results.items():
        if method_name in colors:
            method_names.append(method_labels[method_name])
            timing_mean = (
                float(result["timing_stats"][0]) * 1000
            )  # Convert to milliseconds
            timing_std = (
                float(result["timing_stats"][1]) * 1000
            )  # Convert to milliseconds
            timing_means.append(timing_mean)
            timing_stds.append(timing_std)
            method_colors.append(colors[method_name])

    # Create horizontal bar chart
    y_pos = jnp.arange(len(method_names))
    bars = ax_timing.barh(
        y_pos, timing_means, xerr=timing_stds, color=method_colors, alpha=0.7, capsize=5
    )

    ax_timing.set_yticks(y_pos)
    ax_timing.set_yticklabels([])
    ax_timing.set_xlabel("Time (milliseconds)", fontsize=20, fontweight="bold")
    ax_timing.set_title("", fontsize=22, fontweight="bold", pad=20)  # Remove title
    ax_timing.tick_params(labelsize=16)
    ax_timing.grid(True, axis="x", alpha=0.3)

    # Add timing values as text on bars
    for i, (bar, mean, std) in enumerate(zip(bars, timing_means, timing_stds)):
        width = bar.get_width()
        mean_val = float(mean)
        std_val = float(std)
        ax_timing.text(
            width + std_val + max(timing_means) * 0.01,  # Position after error bar
            bar.get_y() + bar.get_height() / 2,  # Center vertically
            f"{mean_val:.1f}±{std_val:.1f}ms",  # Format as mean±std in milliseconds
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    # Add legend for SMC methods at the bottom
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[method], label=method_labels[method])
        for method in benchmark_results.keys()
        if method in colors
    ]
    legend = fig.legend(
        legend_elements,
        [
            method_labels[method]
            for method in benchmark_results.keys()
            if method in colors
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        fontsize=18,
    )
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight("bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
