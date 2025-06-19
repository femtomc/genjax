"""
Main execution script for the localization case study.

Demonstrates probabilistic robot localization using particle filtering.
"""

import argparse
import os
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
    plot_lidar_demo,
    plot_weight_evolution,
    plot_weight_flow,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GenJAX Localization Case Study - Probabilistic Robot Localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LIDAR configuration
    parser.add_argument(
        "--n-rays",
        type=int,
        default=128,
        help="Number of LIDAR rays for distance measurements",
    )

    # Particle filter configuration
    parser.add_argument(
        "--n-particles",
        type=int,
        default=5000,
        help="Number of particles for the particle filter",
    )

    # Trajectory configuration
    parser.add_argument(
        "--n-steps", type=int, default=16, help="Number of trajectory steps to generate"
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figs",
        help="Output directory for generated figures",
    )

    # Visualization options
    parser.add_argument(
        "--no-lidar-rays",
        action="store_true",
        help="Disable LIDAR ray visualization in plots",
    )

    # World configuration
    parser.add_argument(
        "--world-type",
        type=str,
        default="basic",
        choices=["basic", "complex"],
        help="World geometry type: 'basic' for simple rectangular walls, 'complex' for slanted walls",
    )

    return parser.parse_args()


def main():
    """Run the localization demo."""
    args = parse_args()

    print("GenJAX Localization Case Study")
    print("=" * 40)
    print("Configuration:")
    print(f"  LIDAR rays: {args.n_rays}")
    print(f"  Particles: {args.n_particles}")
    print(f"  Trajectory steps: {args.n_steps}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  LIDAR visualization: {'disabled' if args.no_lidar_rays else 'enabled'}")
    print(f"  World type: {args.world_type}")
    print("=" * 40)

    # Create output directory if it doesn't exist
    if os.path.isabs(args.output_dir):
        figs_dir = args.output_dir
    else:
        figs_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(figs_dir, exist_ok=True)

    # Set random seed for reproducibility
    key = jrand.key(args.seed)

    # Create world
    print(f"Creating {args.world_type} multi-room world...")
    world = create_multi_room_world(world_type=args.world_type)
    print(f"World dimensions: {world.width} x {world.height}")
    print(f"Internal walls: {world.num_walls}")
    print(f"World complexity: {args.world_type}")

    # Generate ground truth data
    print("Generating ground truth trajectory...")
    key, subkey = jrand.split(key)
    true_poses, controls, observations = generate_ground_truth_data(
        world, subkey, n_steps=args.n_steps, n_rays=args.n_rays
    )

    # Extract initial pose for particle filter initialization
    initial_pose = true_poses[0]

    print(f"Generated trajectory - poses type: {type(true_poses)}")
    # print(f"Observations type: {type(observations)}, shape: {getattr(observations, 'shape', 'no shape')}")
    if hasattr(true_poses, "__len__"):
        print(f"Generated trajectory with {len(true_poses)} steps")
    else:
        print(f"Generated single pose: {true_poses}")
    print(
        f"Initial pose: x={initial_pose.x:.2f}, y={initial_pose.y:.2f}, theta={initial_pose.theta:.2f}"
    )

    # Run particle filter
    print("Running particle filter...")
    key, subkey = jrand.split(key)

    particle_history, weight_history, diagnostic_weights = run_particle_filter(
        args.n_particles,
        observations,
        world,
        subkey,
        n_rays=args.n_rays,
        collect_diagnostics=True,
    )
    print("Particle filter completed successfully!")

    # true_poses already includes all poses (including initial pose)

    # Compute estimated poses (weighted means) and track error over time
    estimated_poses = []
    timestep_errors = []
    print("\nEstimation error over time (after incorporating observations):")
    print("Step | True Pose (x, y, θ) | Est Pose (x, y, θ) | Pos Error | ESS")
    print("-" * 75)

    for t, (particles, weights) in enumerate(zip(particle_history, weight_history)):
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

        # Compute position error for this timestep
        if t < len(true_poses):
            true_pose = true_poses[t]
            est_pose = estimated_poses[t]
            pos_error = jnp.sqrt(
                (true_pose.x - est_pose.x) ** 2 + (true_pose.y - est_pose.y) ** 2
            )
            timestep_errors.append(float(pos_error))

            # Compute effective sample size
            if jnp.sum(weights) > 0:
                weights_norm = weights / jnp.sum(weights)
                ess = 1.0 / jnp.sum(weights_norm**2)
            else:
                ess = len(particles)

            print(
                f"{t:4d} | ({true_pose.x:5.2f}, {true_pose.y:5.2f}, {true_pose.theta:5.2f}) | ({est_pose.x:5.2f}, {est_pose.y:5.2f}, {est_pose.theta:5.2f}) | {pos_error:8.3f} | {ess:6.1f}"
            )

    print("\nError statistics:")
    print(f"  Initial error: {timestep_errors[0]:.3f}")
    print(f"  Final error: {timestep_errors[-1]:.3f}")
    print(f"  Mean error: {jnp.mean(jnp.array(timestep_errors)):.3f}")
    print(f"  Max error: {jnp.max(jnp.array(timestep_errors)):.3f}")
    print(f"  Error std: {jnp.std(jnp.array(timestep_errors)):.3f}")

    # Display results
    print("\nResults Summary:")
    final_true_pose = true_poses[-1]
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

    # Create parametrized filename prefix
    param_prefix = f"r{args.n_rays}_p{args.n_particles}_{args.world_type}"

    # observations is now a list from sequential generation
    obs_list = observations

    # 1. Plot detailed ground truth trajectory
    fig1, axes1 = plot_ground_truth_trajectory(
        initial_pose, controls, true_poses[1:], obs_list, world
    )
    filename1 = f"{param_prefix}_ground_truth_detailed.png"
    plt.savefig(
        os.path.join(figs_dir, filename1),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename1}")
    plt.close(fig1)

    # 1b. Plot simple ground truth trajectory (original style)
    fig1b, ax1b = plt.subplots(figsize=(10, 10))
    plot_world(world, ax1b)
    plot_trajectory(true_poses, world, ax1b, color="blue", label="True Trajectory")

    # Add observations as text (show minimum distance from LIDAR)
    for i, (pose, obs) in enumerate(
        zip(true_poses[::2], obs_list[::2])
    ):  # Every other step
        # obs is now a vector of 8 distances, show the minimum
        min_dist = jnp.min(obs) if hasattr(obs, "__len__") and len(obs) > 1 else obs
        ax1b.text(pose.x + 0.2, pose.y + 0.2, f"{min_dist:.1f}", fontsize=8, alpha=0.7)

    ax1b.legend()
    ax1b.set_title("Ground Truth Trajectory with Sensor Observations")
    plt.tight_layout()
    filename1b = f"{param_prefix}_ground_truth.png"
    plt.savefig(
        os.path.join(figs_dir, filename1b),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename1b}")
    plt.close(fig1b)

    # 2. Plot particle filter evolution (enhanced grid showing more steps)
    if len(particle_history) > 1:
        show_lidar = not args.no_lidar_rays
        fig2, axes2 = plot_particle_filter_evolution(
            particle_history,  # Show all steps with intelligent subsampling
            weight_history,
            true_poses,
            world,
            show_lidar_rays=show_lidar,
            observations_history=obs_list
            if show_lidar
            else None,  # Pass observations for measurement points
            n_rays=args.n_rays,
        )
        filename2 = f"{param_prefix}_particle_evolution.png"
        plt.savefig(
            os.path.join(figs_dir, filename2),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved: figs/{filename2}")
        plt.close(fig2)

    # 3. Plot final step comparison
    if len(particle_history) > 0 and len(estimated_poses) > 0:
        show_lidar = not args.no_lidar_rays
        fig3, ax3 = plot_particle_filter_step(
            true_poses[-1],
            particle_history[-1],
            world,
            weight_history[-1],
            step_num=len(particle_history),
            show_lidar_rays=show_lidar,
            observations=obs_list[-1]
            if show_lidar and obs_list
            else None,  # Pass observations for measurement points
            n_rays=args.n_rays,
        )
        filename3 = f"{param_prefix}_final_step.png"
        plt.savefig(
            os.path.join(figs_dir, filename3),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved: figs/{filename3}")
        plt.close(fig3)

    # 4. Plot estimation error
    if len(estimated_poses) > 1:
        fig4, axes4 = plot_estimation_error(
            true_poses[: len(estimated_poses)], estimated_poses
        )
        filename4 = f"{param_prefix}_estimation_error.png"
        plt.savefig(
            os.path.join(figs_dir, filename4),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved: figs/{filename4}")
        plt.close(fig4)

    # 5. Plot sensor observations (LIDAR data)
    # Pass full LIDAR observations to show each ray individually
    # For true distances, generate LIDAR distances to match observation structure
    # Note: obs_list includes observations for ALL poses including initial pose
    true_lidar_distances = [
        distance_to_wall_lidar(pose, world, n_angles=args.n_rays) for pose in true_poses
    ]

    fig5, ax5 = plot_sensor_observations(obs_list, true_lidar_distances)
    filename5 = f"{param_prefix}_sensor_observations.png"
    plt.savefig(
        os.path.join(figs_dir, filename5),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename5}")
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
    filename6 = f"{param_prefix}_multiple_trajectories.png"
    plt.savefig(
        os.path.join(figs_dir, filename6),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename6}")
    plt.close(fig6)

    # 7. LIDAR Demo - Show measurement points
    print("\nGenerating LIDAR measurement demonstration...")
    # Use a pose from the middle of the trajectory for interesting ray patterns
    demo_pose = true_poses[len(true_poses) // 2]
    fig7, ax7 = plot_lidar_demo(demo_pose, world, n_rays=args.n_rays)
    filename7 = f"{param_prefix}_lidar_demo.png"
    plt.savefig(
        os.path.join(figs_dir, filename7),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename7}")
    plt.close(fig7)

    # 8. Weight Evolution - Show particle weight histograms over time
    print("\nGenerating particle weight evolution...")
    fig8, axes8 = plot_weight_evolution(weight_history)
    filename8 = f"{param_prefix}_weight_evolution.png"
    plt.savefig(
        os.path.join(figs_dir, filename8),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename8}")
    plt.close(fig8)

    # 9. Weight Flow - Show particle weight evolution using diagnostic weights from state API
    print("\nGenerating particle weight flow...")
    fig9, axes9 = plot_weight_flow(
        diagnostic_weights
    )  # Use diagnostic weights from state API
    filename9 = f"{param_prefix}_weight_flow.png"
    plt.savefig(
        os.path.join(figs_dir, filename9),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename9}")
    plt.close(fig9)

    print("\nLocalization demo completed!")
    print("Check the generated PNG files in the figs/ directory for visualizations.")
    print("\nGenerated visualizations:")
    print(f"  - {filename1}: Comprehensive trajectory analysis")
    print(f"  - {filename1b}: Simple trajectory with observations")
    print(f"  - {filename6}: Comparison of different trajectory types")
    print(f"  - {filename2}: Particle filter over time")
    print(f"  - {filename3}: Final particle distribution")
    print(f"  - {filename4}: Localization accuracy")
    print(f"  - {filename5}: Sensor data analysis")
    print(
        f"  - {filename7}: LIDAR measurement points demonstration ({args.n_rays} rays)"
    )
    print(f"  - {filename8}: Particle weight distribution evolution")
    print(f"  - {filename9}: Particle weight flow visualization")
    print("\nFilename format: r<rays>_p<particles>_<world_type>_<plottype>.png")
    print(
        f"Current parameters: {args.n_rays} rays, {args.n_particles} particles, {args.n_steps} steps, seed {args.seed}, world: {args.world_type}"
    )


if __name__ == "__main__":
    main()
