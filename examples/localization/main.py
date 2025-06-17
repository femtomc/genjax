"""
Main execution script for the localization case study.

Demonstrates probabilistic robot localization using particle filtering.
"""

import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt

from .core import (
    Pose,
    create_multi_room_world,
    run_particle_filter,
    distance_to_wall_lidar,
)

from .data import (
    generate_ground_truth_data,
    generate_multiple_trajectories,
)

from .figs import (
    plot_world,
    plot_trajectory,
    plot_particle_filter_step,
    plot_particle_filter_evolution,
    plot_estimation_error,
    plot_sensor_observations,
    plot_ground_truth_trajectory,
    plot_multiple_trajectories,
)


def main():
    """Run the localization demo."""
    print("GenJAX Localization Case Study")
    print("=" * 40)

    # Set random seed for reproducibility
    key = jrand.key(42)

    # Create world
    print("Creating multi-room world...")
    world = create_multi_room_world()
    print(f"World dimensions: {world.width} x {world.height}")
    print(f"Internal walls: {world.num_walls}")

    # Generate ground truth data
    print("Generating ground truth trajectory...")
    key, subkey = jrand.split(key)
    initial_pose, controls, true_poses, observations = generate_ground_truth_data(
        world, subkey
    )

    print(f"Generated trajectory - poses type: {type(true_poses)}")
    # print(f"Observations type: {type(observations)}, shape: {getattr(observations, 'shape', 'no shape')}")
    if hasattr(true_poses, "__len__"):
        print(f"Generated trajectory with {len(true_poses)} steps")
    else:
        print(f"Generated single pose: {true_poses}")
    print(
        f"Initial pose: x={initial_pose.x:.2f}, y={initial_pose.y:.2f}, theta={initial_pose.theta:.2f}"
    )

    # Initialize particles using the initial_model with generate
    print("Initializing particles using initial_model...")
    n_particles = 1000  # Increased for complex cross-room navigation

    # Run particle filter
    print("Running particle filter...")
    key, subkey = jrand.split(key)

    particle_history, weight_history = run_particle_filter(
        n_particles, observations, world, subkey, initial_pose
    )
    print("Particle filter completed successfully!")

    # Compute estimated poses (weighted means)
    estimated_poses = []
    for particles, weights in zip(particle_history, weight_history):
        if jnp.sum(weights) > 0:
            weights_norm = weights / jnp.sum(weights)
            mean_x = jnp.sum(jnp.array([p.x for p in particles]) * weights_norm)
            mean_y = jnp.sum(jnp.array([p.y for p in particles]) * weights_norm)

            # Circular mean for angles
            sin_sum = jnp.sum(
                jnp.sin(jnp.array([p.theta for p in particles])) * weights_norm
            )
            cos_sum = jnp.sum(
                jnp.cos(jnp.array([p.theta for p in particles])) * weights_norm
            )
            mean_theta = jnp.arctan2(sin_sum, cos_sum)

            estimated_poses.append(Pose(mean_x, mean_y, mean_theta))
        else:
            # Fallback: uniform weights
            mean_x = jnp.mean(jnp.array([p.x for p in particles]))
            mean_y = jnp.mean(jnp.array([p.y for p in particles]))
            mean_theta = jnp.arctan2(
                jnp.mean(jnp.sin(jnp.array([p.theta for p in particles]))),
                jnp.mean(jnp.cos(jnp.array([p.theta for p in particles]))),
            )
            estimated_poses.append(Pose(mean_x, mean_y, mean_theta))

    # true_poses is now a list from sequential generation
    true_pose_list = [initial_pose] + true_poses  # Add initial pose to list

    # Display results
    print("\nResults Summary:")
    final_true_pose = true_pose_list[-1]
    print(
        f"Final true pose: x={final_true_pose.x:.2f}, y={final_true_pose.y:.2f}, theta={final_true_pose.theta:.2f}"
    )
    print(
        f"Final estimated pose: x={estimated_poses[-1].x:.2f}, y={estimated_poses[-1].y:.2f}, theta={estimated_poses[-1].theta:.2f}"
    )

    final_error = jnp.sqrt(
        (final_true_pose.x - estimated_poses[-1].x) ** 2
        + (final_true_pose.y - estimated_poses[-1].y) ** 2
    )
    print(f"Final position error: {final_error:.3f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # observations is now a list from sequential generation
    obs_list = observations

    # 1. Plot detailed ground truth trajectory
    fig1, axes1 = plot_ground_truth_trajectory(
        initial_pose, controls, true_poses, obs_list, world
    )
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/figs/ground_truth_detailed.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: figs/ground_truth_detailed.png")
    plt.close(fig1)

    # 1b. Plot simple ground truth trajectory (original style)
    fig1b, ax1b = plt.subplots(figsize=(10, 10))
    plot_world(world, ax1b)
    plot_trajectory(true_pose_list, world, ax1b, color="blue", label="True Trajectory")

    # Add observations as text (show minimum distance from LIDAR)
    for i, (pose, obs) in enumerate(
        zip(true_pose_list[::2], obs_list[::2])
    ):  # Every other step
        # obs is now a vector of 8 distances, show the minimum
        min_dist = jnp.min(obs) if hasattr(obs, "__len__") and len(obs) > 1 else obs
        ax1b.text(pose.x + 0.2, pose.y + 0.2, f"{min_dist:.1f}", fontsize=8, alpha=0.7)

    ax1b.legend()
    ax1b.set_title("Ground Truth Trajectory with Sensor Observations")
    plt.tight_layout()
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/figs/ground_truth.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: figs/ground_truth.png")
    plt.close(fig1b)

    # 2. Plot particle filter evolution (enhanced grid showing more steps)
    if len(particle_history) > 1:
        fig2, axes2 = plot_particle_filter_evolution(
            particle_history,  # Show all steps with intelligent subsampling
            weight_history,
            true_pose_list,
            world,
        )
        plt.savefig(
            "/Users/femtomc/Dev/genjax/examples/localization/figs/particle_evolution.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: figs/particle_evolution.png")
        plt.close(fig2)

    # 3. Plot final step comparison
    if len(particle_history) > 0 and len(estimated_poses) > 0:
        fig3, ax3 = plot_particle_filter_step(
            true_pose_list[-1],
            particle_history[-1],
            world,
            weight_history[-1],
            step_num=len(particle_history),
        )
        plt.savefig(
            "/Users/femtomc/Dev/genjax/examples/localization/figs/final_step.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: figs/final_step.png")
        plt.close(fig3)

    # 4. Plot estimation error
    if len(estimated_poses) > 1:
        fig4, axes4 = plot_estimation_error(
            true_pose_list[: len(estimated_poses)], estimated_poses
        )
        plt.savefig(
            "/Users/femtomc/Dev/genjax/examples/localization/figs/estimation_error.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: figs/estimation_error.png")
        plt.close(fig4)

    # 5. Plot sensor observations (LIDAR data)
    # Pass full LIDAR observations to show each ray individually
    # For true distances, generate LIDAR distances to match observation structure
    # Note: obs_list corresponds to poses[1:] (trajectory poses), so we need true_pose_list[1:]
    true_lidar_distances = [
        distance_to_wall_lidar(pose, world, n_angles=8) for pose in true_pose_list[1:]
    ]

    fig5, ax5 = plot_sensor_observations(obs_list, true_lidar_distances)
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/figs/sensor_observations.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: figs/sensor_observations.png")
    plt.close(fig5)

    # 6. Generate multiple trajectories for comparison (optional)
    print("\nGenerating multiple trajectory comparison...")
    key, subkey = jrand.split(key)
    multiple_trajectories = generate_multiple_trajectories(
        world,
        subkey,
        n_trajectories=3,
        trajectory_types=["room_navigation", "exploration", "wall_bouncing"],
    )

    fig6, ax6 = plot_multiple_trajectories(multiple_trajectories, world)
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/figs/multiple_trajectories.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: figs/multiple_trajectories.png")
    plt.close(fig6)

    print("\nLocalization demo completed!")
    print("Check the generated PNG files in the figs/ directory for visualizations.")
    print("\nGenerated visualizations:")
    print("  - ground_truth_detailed.png: Comprehensive trajectory analysis")
    print("  - ground_truth.png: Simple trajectory with observations")
    print("  - multiple_trajectories.png: Comparison of different trajectory types")
    print("  - particle_evolution.png: Particle filter over time")
    print("  - final_step.png: Final particle distribution")
    print("  - estimation_error.png: Localization accuracy")
    print("  - sensor_observations.png: Sensor data analysis")


if __name__ == "__main__":
    main()
