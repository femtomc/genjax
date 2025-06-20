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
    plot_weight_flow,
    plot_smc_timing_comparison,
    plot_smc_method_comparison,
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
        default=8,
        help="Number of LIDAR rays for distance measurements",
    )

    # Particle filter configuration
    parser.add_argument(
        "--n-particles",
        type=int,
        default=200,
        help="Number of particles for the particle filter",
    )

    parser.add_argument(
        "--k-rejuv",
        type=int,
        default=10,
        help="Number of rejuvenation steps (K) for MCMC methods",
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

    # Experimental mode
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run SMC method comparison experiment (basic, MH, HMC)",
    )

    # Timing experiment configuration
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=20,
        help="Number of timing repetitions for each method",
    )

    # Data export and plotting options
    parser.add_argument(
        "--export-data",
        action="store_true",
        help="Export experimental data to CSV files",
    )

    parser.add_argument(
        "--plot-from-data",
        type=str,
        help="Generate plots from existing data directory (path to data dir)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save/load experimental data (relative to localization dir)",
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
    print(f"  Data directory: {args.data_dir}")
    print(f"  LIDAR visualization: {'disabled' if args.no_lidar_rays else 'enabled'}")
    print(f"  World type: {args.world_type}")
    print(f"  Export data: {args.export_data}")
    print(f"  Plot from data: {args.plot_from_data or 'None'}")
    print("=" * 40)

    # Create output directory if it doesn't exist
    if os.path.isabs(args.output_dir):
        figs_dir = args.output_dir
    else:
        figs_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(figs_dir, exist_ok=True)

    # Create data directory if it doesn't exist
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Handle plot-from-data mode
    if args.plot_from_data:
        return plot_from_existing_data(args, figs_dir)

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

    # Check if we're running the experimental comparison
    if args.experiment:
        print("Running SMC method comparison experiment...")
        from .core import benchmark_smc_methods
        from .figs import plot_smc_timing_comparison, plot_smc_method_comparison

        key, subkey = jrand.split(key)
        benchmark_results = benchmark_smc_methods(
            args.n_particles,
            observations,
            world,
            subkey,
            n_rays=args.n_rays,
            repeats=args.timing_repeats,
            K=args.k_rejuv,
        )

        # Create parameter prefix for consistent naming
        param_prefix = f"r{args.n_rays}_p{args.n_particles}_{args.world_type}"

        # Export data if requested
        if args.export_data:
            print("\nExporting experimental data...")
            from .export import save_benchmark_results, save_ground_truth_data

            # Create timestamped data directory
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"smc_comparison_{param_prefix}_{timestamp}"
            experiment_data_dir = os.path.join(data_dir, experiment_name)
            os.makedirs(experiment_data_dir, exist_ok=True)

            # Experiment configuration
            config = {
                "n_rays": args.n_rays,
                "n_particles": args.n_particles,
                "n_steps": args.n_steps,
                "seed": args.seed,
                "world_type": args.world_type,
                "k_rejuv": args.k_rejuv,
                "timing_repeats": args.timing_repeats,
                "timestamp": timestamp,
            }

            # Save all experimental data
            save_ground_truth_data(
                experiment_data_dir, true_poses, observations, controls
            )
            save_benchmark_results(experiment_data_dir, benchmark_results, config)

            print(f"Data exported to: {experiment_data_dir}")

        # Generate timing comparison plot
        print("\nGenerating timing comparison plot...")
        timing_filename = f"{param_prefix}_smc_timing_comparison.pdf"
        timing_path = os.path.join(figs_dir, timing_filename)
        plot_smc_timing_comparison(
            benchmark_results,
            save_path=timing_path,
            n_particles=args.n_particles,
            K=args.k_rejuv,
        )
        print(f"Saved: figs/{timing_filename}")

        # Generate method comparison plot
        print("Generating method comparison plot...")
        comparison_filename = f"{param_prefix}_smc_method_comparison.pdf"
        comparison_path = os.path.join(figs_dir, comparison_filename)
        plot_smc_method_comparison(
            benchmark_results,
            true_poses,
            world,
            save_path=comparison_path,
            n_rays=args.n_rays,
            n_particles=args.n_particles,
            K=args.k_rejuv,
        )
        print(f"Saved: figs/{comparison_filename}")

        # Generate particle evolution plots for each method
        print("Generating particle evolution plots for each method...")
        evolution_filenames = []
        for method_name, result in benchmark_results.items():
            particle_history = result["particle_history"]
            weight_history = result["weight_history"]

            method_labels = {
                "smc_basic": f"Bootstrap filter (N={args.n_particles})",
                "smc_mh": f"SMC (N={args.n_particles}) + MH (K={args.k_rejuv})",
                "smc_hmc": f"SMC (N={args.n_particles}) + HMC (K={args.k_rejuv})",
                "smc_locally_optimal": f"SMC (N={args.n_particles}) + Locally Optimal (K={args.k_rejuv})",
            }

            evolution_filename = f"{param_prefix}_{method_name}_particle_evolution.pdf"
            evolution_path = os.path.join(figs_dir, evolution_filename)

            show_lidar = not args.no_lidar_rays

            # Generate observations history for LIDAR visualization
            obs_list = observations if show_lidar else None

            fig, axes = plot_particle_filter_evolution(
                particle_history,
                weight_history,
                true_poses,
                world,
                show_lidar_rays=show_lidar,
                observations_history=obs_list,
                n_rays=args.n_rays,
            )

            # Add method label to title
            fig.suptitle(
                f"{method_labels[method_name]} - Particle Evolution",
                fontsize=16,
                fontweight="bold",
                y=0.95,
            )

            plt.savefig(evolution_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            evolution_filenames.append(evolution_filename)
            print(f"Saved: figs/{evolution_filename}")

        print("\nExperimental comparison completed!")
        print("Generated plots:")
        print(f"  - {timing_filename}: SMC method timing comparison")
        print(f"  - {comparison_filename}: Final distributions and diagnostic weights")
        for i, filename in enumerate(evolution_filenames):
            method_name = list(benchmark_results.keys())[i]
            method_labels = {
                "smc_basic": f"Bootstrap filter (N={args.n_particles})",
                "smc_mh": f"SMC (N={args.n_particles}) + MH (K={args.k_rejuv})",
                "smc_hmc": f"SMC (N={args.n_particles}) + HMC (K={args.k_rejuv})",
                "smc_locally_optimal": f"SMC (N={args.n_particles}) + Locally Optimal (K={args.k_rejuv})",
            }
            print(f"  - {filename}: {method_labels[method_name]} particle evolution")
        return

    # Run standard particle filter
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
        trajectory_types=["room_navigation", "exploration"],
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

    # 8. Diagnostic Weights Flow - Raincloud plots showing weight distributions over time
    print("\nGenerating diagnostic weights flow (raincloud plots)...")
    # diagnostic_weights should always be available from ParticleCollection
    print(
        f"Diagnostic weights shape: {diagnostic_weights.shape if diagnostic_weights is not None else 'None'}"
    )
    print(f"Diagnostic weights type: {type(diagnostic_weights)}")
    if diagnostic_weights is not None:
        print(
            f"Diagnostic weights range: {jnp.min(diagnostic_weights):.6f} to {jnp.max(diagnostic_weights):.6f}"
        )
        print(
            f"Diagnostic weights unique values: {len(jnp.unique(diagnostic_weights))}"
        )
    fig8, ax8 = plot_weight_flow(diagnostic_weights)
    filename8 = f"{param_prefix}_diagnostic_weight_flow.png"
    plt.savefig(
        os.path.join(figs_dir, filename8),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: figs/{filename8}")
    plt.close(fig8)

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
    print(f"  - {filename8}: Diagnostic particle weight flow visualization")
    print("\nFilename format: r<rays>_p<particles>_<world_type>_<plottype>.png")
    print(
        f"Current parameters: {args.n_rays} rays, {args.n_particles} particles, {args.n_steps} steps, seed {args.seed}, world: {args.world_type}"
    )


def plot_from_existing_data(args, figs_dir):
    """Generate plots from existing experimental data."""
    from .export import (
        load_benchmark_results,
        load_ground_truth_data,
        load_experiment_metadata,
    )
    from .core import create_multi_room_world

    data_path = args.plot_from_data
    if not os.path.exists(data_path):
        print(f"Error: Data directory {data_path} does not exist")
        return

    print(f"Loading experimental data from: {data_path}")

    # Load metadata and create world
    config = load_experiment_metadata(data_path)
    world_type = config.get("world_type", "basic")
    if world_type == "basic":
        world = create_multi_room_world()
    else:
        from .core import create_complex_multi_room_world

        world = create_complex_multi_room_world()

    # Load ground truth and benchmark results
    true_poses, observations = load_ground_truth_data(data_path)
    benchmark_results = load_benchmark_results(data_path)

    print(f"Loaded data: {len(true_poses)} poses, {len(benchmark_results)} methods")

    # Generate plots using loaded data
    param_prefix = (
        f"r{config['n_rays']}_p{config['n_particles']}_{config['world_type']}"
    )

    # Timing comparison plot
    print("Generating timing comparison plot...")
    timing_filename = f"{param_prefix}_smc_timing_comparison_from_data.pdf"
    timing_path = os.path.join(figs_dir, timing_filename)
    plot_smc_timing_comparison(
        benchmark_results,
        save_path=timing_path,
        n_particles=config["n_particles"],
        K=config["k_rejuv"],
    )
    print(f"Saved: figs/{timing_filename}")

    # Method comparison plot
    print("Generating method comparison plot...")
    comparison_filename = f"{param_prefix}_smc_method_comparison_from_data.pdf"
    comparison_path = os.path.join(figs_dir, comparison_filename)
    plot_smc_method_comparison(
        benchmark_results,
        true_poses,
        world,
        save_path=comparison_path,
        n_rays=config["n_rays"],
        n_particles=config["n_particles"],
        K=config["k_rejuv"],
    )
    print(f"Saved: figs/{comparison_filename}")

    print("Plotting from existing data completed!")


if __name__ == "__main__":
    main()
