"""
Main execution script for the localization case study.

Demonstrates probabilistic robot localization using particle filtering.
"""

import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt

from core import (
    Pose,
    create_simple_world,
    generate_ground_truth_data,
    run_particle_filter,
    distance_to_wall,
)

from figs import (
    plot_world,
    plot_trajectory,
    plot_particle_filter_step,
    plot_particle_filter_evolution,
    plot_estimation_error,
    plot_sensor_observations,
)


def main():
    """Run the localization demo."""
    print("GenJAX Localization Case Study")
    print("=" * 40)

    # Set random seed for reproducibility
    key = jrand.key(42)

    # Create world
    print("Creating world...")
    world = create_simple_world()
    print(f"World dimensions: {world.width} x {world.height}")

    # Generate ground truth data
    print("Generating ground truth trajectory...")
    key, subkey = jrand.split(key)
    initial_pose, controls, true_poses, observations = generate_ground_truth_data(
        world, subkey
    )

    print(f"Generated trajectory - poses type: {type(true_poses)}")
    if hasattr(true_poses, "__len__"):
        print(f"Generated trajectory with {len(true_poses)} steps")
    else:
        print(f"Generated single pose: {true_poses}")
    print(
        f"Initial pose: x={initial_pose.x:.2f}, y={initial_pose.y:.2f}, theta={initial_pose.theta:.2f}"
    )

    # Initialize particles
    print("Initializing particles...")
    n_particles = 100
    key, subkey = jrand.split(key)

    # Initialize particles randomly around the true initial position
    particle_keys = jrand.split(subkey, n_particles)
    initial_particles = []

    for i in range(n_particles):
        # Add noise to initial position
        x_noise = jrand.normal(particle_keys[i], shape=()) * 0.5
        y_noise = jrand.normal(particle_keys[i], shape=()) * 0.5
        theta_noise = jrand.normal(particle_keys[i], shape=()) * 0.2

        pose = Pose(
            x=jnp.clip(initial_pose.x + x_noise, 0.0, world.width),
            y=jnp.clip(initial_pose.y + y_noise, 0.0, world.height),
            theta=initial_pose.theta + theta_noise,
        )
        initial_particles.append(pose)

    print(f"Initialized {n_particles} particles")

    # Run particle filter
    print("Running particle filter...")
    key, subkey = jrand.split(key)

    try:
        particle_history, weight_history = run_particle_filter(
            initial_particles, controls, observations, world, subkey
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

        # Convert vectorized poses to list for easier handling
        if hasattr(true_poses, "x") and hasattr(true_poses.x, "__len__"):
            # Vectorized pose from JAX Scan
            true_pose_list = [
                Pose(true_poses.x[i], true_poses.y[i], true_poses.theta[i])
                for i in range(len(true_poses.x))
            ]
        else:
            true_pose_list = true_poses

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

    except Exception as e:
        print(f"Error during particle filter execution: {e}")
        print("Proceeding with visualization of available data...")
        particle_history = [initial_particles]
        weight_history = [jnp.ones(n_particles) / n_particles]
        estimated_poses = [initial_pose]

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Handle vectorized observations
    if hasattr(observations, "__len__") and not isinstance(observations, (int, float)):
        obs_list = list(observations)
    else:
        obs_list = [observations]

    # 1. Plot ground truth trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    plot_world(world, ax1)
    plot_trajectory(true_poses, world, ax1, color="blue", label="True Trajectory")

    # Add observations as text
    for i, (pose, obs) in enumerate(
        zip(true_pose_list[::2], obs_list[::2])
    ):  # Every other step
        ax1.text(pose.x + 0.2, pose.y + 0.2, f"{obs:.1f}", fontsize=8, alpha=0.7)

    ax1.legend()
    ax1.set_title("Ground Truth Trajectory with Sensor Observations")
    plt.tight_layout()
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/ground_truth.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: ground_truth.png")

    # 2. Plot particle filter evolution (first few steps)
    if len(particle_history) > 1:
        fig2, axes2 = plot_particle_filter_evolution(
            particle_history[: min(4, len(particle_history))],
            weight_history[: min(4, len(weight_history))],
            true_pose_list[: min(4, len(true_pose_list))],
            world,
        )
        plt.savefig(
            "/examples/localization/figs/particle_evolution.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: particle_evolution.png")
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
            "/examples/localization/figs/final_step.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: final_step.png")
        plt.close(fig3)

    # 4. Plot estimation error
    if len(estimated_poses) > 1:
        fig4, axes4 = plot_estimation_error(
            true_pose_list[: len(estimated_poses)], estimated_poses
        )
        plt.savefig(
            "/Users/femtomc/Dev/genjax/examples/localization/estimation_error.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("Saved: estimation_error.png")
        plt.close(fig4)

    # 5. Plot sensor observations
    true_distances = [distance_to_wall(pose, world) for pose in true_pose_list]
    fig5, ax5 = plot_sensor_observations(obs_list, true_distances)
    plt.savefig(
        "/Users/femtomc/Dev/genjax/examples/localization/sensor_observations.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: sensor_observations.png")
    plt.close(fig5)

    print("\nLocalization demo completed!")
    print("Check the generated PNG files for visualizations.")

    # Keep the first plot open for display
    plt.show()


if __name__ == "__main__":
    main()
