"""
Main execution script for the localization case study.

Demonstrates probabilistic robot localization using particle filtering.
Restructured to have two main commands:
1. generate-data: Generate all experimental data and save to data/
2. plot-figures: Generate all figures from saved data
"""

import argparse
import os
import datetime
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt

from .core import (
    Pose,
    create_multi_room_world,
    run_particle_filter,
    distance_to_wall_lidar,
    benchmark_smc_methods,
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
    plot_multiple_trajectories,
    plot_lidar_demo,
    plot_weight_flow,
    plot_smc_timing_comparison,
    plot_smc_method_comparison,
    plot_multi_method_estimation_error,
)

from .export import (
    save_experiment_metadata,
    save_ground_truth_data,
    save_benchmark_results,
    save_smc_results,
    load_experiment_metadata,
    load_ground_truth_data,
    load_benchmark_results,
    load_smc_results,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GenJAX Localization Case Study - Probabilistic Robot Localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main command
    parser.add_argument(
        "command",
        choices=["generate-data", "plot-figures"],
        help="Main command: generate-data or plot-figures",
    )

    # Data experiment name (for both commands)
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (for plot-figures, defaults to latest if not specified)",
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
        default=20,
        help="Number of rejuvenation steps (K) for MCMC methods",
    )

    parser.add_argument(
        "--n-particles-big-grid",
        type=int,
        default=5,
        help="Number of particles for the big grid locally optimal method",
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

    # Timing experiment configuration
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=20,
        help="Number of timing repetitions for each method",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save/load experimental data (relative to localization dir)",
    )

    # Run mode options
    parser.add_argument(
        "--include-smc-comparison",
        action="store_true",
        help="Include SMC method comparison in data generation (adds significant computation time)",
    )

    parser.add_argument(
        "--include-basic-demo",
        action="store_true",
        help="Include basic particle filter demo in data generation",
    )

    return parser.parse_args()


def generate_data(args):
    """Generate all experimental data and save to data directory."""
    print("GenJAX Localization Case Study - Data Generation")
    print("=" * 50)
    print("Configuration:")
    print(f"  LIDAR rays: {args.n_rays}")
    print(f"  Particles: {args.n_particles}")
    print(f"  Trajectory steps: {args.n_steps}")
    print(f"  Random seed: {args.seed}")
    print(f"  World type: {args.world_type}")
    print(f"  K rejuvenation steps: {args.k_rejuv}")
    print(f"  Timing repeats: {args.timing_repeats}")
    print(f"  Include SMC comparison: {args.include_smc_comparison}")
    print(f"  Include basic demo: {args.include_basic_demo}")
    print("=" * 50)

    # Create data directory
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Create timestamped experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_prefix = f"localization_r{args.n_rays}_p{args.n_particles}_{args.world_type}"
    experiment_name = args.experiment_name or f"{param_prefix}_{timestamp}"
    experiment_data_dir = os.path.join(data_dir, experiment_name)
    os.makedirs(experiment_data_dir, exist_ok=True)

    print(f"\nSaving data to: {experiment_data_dir}")

    # Set random seed
    key = jrand.key(args.seed)

    # Create world
    print(f"\nCreating {args.world_type} multi-room world...")
    world = create_multi_room_world(world_type=args.world_type)
    print(f"World dimensions: {world.width} x {world.height}")
    print(f"Internal walls: {world.num_walls}")

    # Generate ground truth data
    print("\nGenerating ground truth trajectory...")
    key, subkey = jrand.split(key)
    true_poses, controls, observations = generate_ground_truth_data(
        world, subkey, n_steps=args.n_steps, n_rays=args.n_rays
    )
    initial_pose = true_poses[0]
    print(f"Generated trajectory with {len(true_poses)} poses")
    print(
        f"Initial pose: x={initial_pose.x:.2f}, y={initial_pose.y:.2f}, theta={initial_pose.theta:.2f}"
    )

    # Save configuration
    config = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "n_rays": args.n_rays,
        "n_particles": args.n_particles,
        "n_particles_big_grid": args.n_particles_big_grid,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "world_type": args.world_type,
        "k_rejuv": args.k_rejuv,
        "timing_repeats": args.timing_repeats,
        "include_smc_comparison": args.include_smc_comparison,
        "include_basic_demo": args.include_basic_demo,
    }
    save_experiment_metadata(experiment_data_dir, config)

    # Save ground truth data
    save_ground_truth_data(experiment_data_dir, true_poses, observations, controls)

    # Generate multiple trajectories for comparison
    print("\nGenerating multiple trajectory types...")
    key, subkey = jrand.split(key)
    multiple_trajectories = generate_multiple_trajectories(
        world,
        subkey,
        n_trajectories=3,
        trajectory_types=["room_navigation", "exploration"],
    )

    # Save multiple trajectories
    multi_traj_dir = os.path.join(experiment_data_dir, "multiple_trajectories")
    os.makedirs(multi_traj_dir, exist_ok=True)
    for i, (traj_type, (poses, controls, obs)) in enumerate(multiple_trajectories):
        traj_dir = os.path.join(multi_traj_dir, f"trajectory_{i}_{traj_type}")
        os.makedirs(traj_dir, exist_ok=True)
        save_ground_truth_data(traj_dir, poses, obs, controls)

    # Run basic particle filter demo if requested
    if args.include_basic_demo:
        print("\nRunning basic particle filter demo...")
        key, subkey = jrand.split(key)

        particle_history, weight_history, diagnostic_weights = run_particle_filter(
            args.n_particles,
            observations,
            world,
            subkey,
            n_rays=args.n_rays,
            collect_diagnostics=True,
        )

        # Save basic demo results
        basic_result = {
            "particle_history": particle_history,
            "weight_history": weight_history,
            "diagnostic_weights": diagnostic_weights,
            "timing_stats": (0.0, 0.0),  # No timing for basic demo
        }
        save_smc_results(experiment_data_dir, "basic_demo", basic_result)
        print("Basic particle filter demo completed!")

    # Run SMC method comparison if requested
    if args.include_smc_comparison:
        print("\nRunning SMC method comparison experiment...")
        print(
            "This will compare: Bootstrap filter, SMC+MH, SMC+HMC, SMC+Locally Optimal"
        )

        key, subkey = jrand.split(key)
        benchmark_results = benchmark_smc_methods(
            args.n_particles,
            observations,
            world,
            subkey,
            n_rays=args.n_rays,
            repeats=args.timing_repeats,
            K=args.k_rejuv,
            K_hmc=25,  # Special K value for HMC
            n_particles_big_grid=args.n_particles_big_grid,
        )

        # Save benchmark results
        save_benchmark_results(experiment_data_dir, benchmark_results, config)
        print("SMC comparison experiment completed!")

    print("\n✅ Data generation complete!")
    print(f"All data saved to: {experiment_data_dir}")
    print("\nTo generate figures from this data, run:")
    print(
        f"  python -m examples.localization.main plot-figures --experiment-name {experiment_name}"
    )


def plot_figures(args):
    """Generate all figures from saved experimental data."""
    # Determine data directory
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)

    # Find experiment directory
    if args.experiment_name:
        experiment_data_dir = os.path.join(data_dir, args.experiment_name)
    else:
        # Find the most recent experiment
        experiments = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not experiments:
            print("Error: No experiments found in data directory")
            return
        experiments.sort()
        experiment_data_dir = os.path.join(data_dir, experiments[-1])
        print(f"Using most recent experiment: {experiments[-1]}")

    if not os.path.exists(experiment_data_dir):
        print(f"Error: Experiment data directory not found: {experiment_data_dir}")
        return

    print("GenJAX Localization Case Study - Figure Generation")
    print("=" * 50)
    print(f"Loading data from: {experiment_data_dir}")

    # Load metadata and configuration
    config = load_experiment_metadata(experiment_data_dir)

    # Override visualization options if specified
    if args.no_lidar_rays is not None:
        config["no_lidar_rays"] = args.no_lidar_rays

    print("\nExperiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    # Create output directory
    if os.path.isabs(args.output_dir):
        figs_dir = args.output_dir
    else:
        figs_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(figs_dir, exist_ok=True)

    # Create world
    world = create_multi_room_world(world_type=config["world_type"])

    # Load ground truth data
    true_poses, observations = load_ground_truth_data(experiment_data_dir)
    initial_pose = true_poses[0]

    # Create parametrized filename prefix
    param_prefix = (
        f"localization_r{config['n_rays']}_p{config['n_particles']}_{config['world_type']}"
    )

    print("\nGenerating figures...")

    # 1. Ground truth trajectory plots
    print("  - Ground truth trajectory plots...")

    # Note: Detailed plot requires controls which are not saved in export
    # So we only generate the simple plot
    fig1b, ax1b = plt.subplots(figsize=(10, 10))
    plot_world(world, ax1b)
    plot_trajectory(true_poses, world, ax1b, color="blue", label="True Trajectory")

    # Add observations as text
    for i, (pose, obs) in enumerate(zip(true_poses[::2], observations[::2])):
        if hasattr(obs, "__len__") and len(obs) > 1:
            obs_array = jnp.array(obs)
            min_dist = jnp.min(obs_array)
        else:
            min_dist = obs
        ax1b.text(pose.x + 0.2, pose.y + 0.2, f"{min_dist:.1f}", fontsize=8, alpha=0.7)

    ax1b.legend()
    ax1b.set_title("Ground Truth Trajectory with Sensor Observations")
    plt.tight_layout()
    filename1b = f"{param_prefix}_true_robot_path_with_lidar_observations.pdf"
    plt.savefig(os.path.join(figs_dir, filename1b), dpi=150, bbox_inches="tight")
    plt.close(fig1b)
    print(f"    Saved: {filename1b}")

    # 2. Multiple trajectories comparison
    print("  - Multiple trajectories comparison...")
    multi_traj_dir = os.path.join(experiment_data_dir, "multiple_trajectories")
    if os.path.exists(multi_traj_dir):
        trajectory_data_list = []
        for traj_dir in sorted(os.listdir(multi_traj_dir)):
            if traj_dir.startswith("trajectory_"):
                traj_type = traj_dir.split("_", 2)[
                    2
                ]  # Extract type from directory name
                traj_path = os.path.join(multi_traj_dir, traj_dir)
                poses, obs = load_ground_truth_data(traj_path)
                controls = [None] * (len(poses) - 1)
                trajectory_data_list.append((traj_type, (poses, controls, obs)))

        fig6, ax6 = plot_multiple_trajectories(trajectory_data_list, world)
        filename6 = f"{param_prefix}_trajectory_types_exploration_vs_navigation.pdf"
        plt.savefig(os.path.join(figs_dir, filename6), dpi=150, bbox_inches="tight")
        plt.close(fig6)
        print(f"    Saved: {filename6}")

    # 3. LIDAR demonstration
    print("  - LIDAR sensor demonstration...")
    demo_pose = true_poses[len(true_poses) // 2]
    fig7, ax7 = plot_lidar_demo(demo_pose, world, n_rays=config["n_rays"])
    filename7 = f"{param_prefix}_lidar_8ray_wall_detection_visualization.pdf"
    plt.savefig(os.path.join(figs_dir, filename7), dpi=150, bbox_inches="tight")
    plt.close(fig7)
    print(f"    Saved: {filename7}")

    # 4. Basic demo plots (if available)
    basic_demo_dir = os.path.join(experiment_data_dir, "basic_demo")
    if os.path.exists(basic_demo_dir):
        print("  - Basic particle filter demo plots...")
        basic_result = load_smc_results(experiment_data_dir, "basic_demo")

        particle_history = basic_result["particle_history"]
        weight_history = basic_result["weight_history"]
        diagnostic_weights = basic_result["diagnostic_weights"]

        # Compute estimated poses
        estimated_poses = []
        for particles, weights in zip(particle_history, weight_history):
            if jnp.sum(weights) > 0:
                weights_norm = weights / jnp.sum(weights)
                mean_x = jnp.sum(jnp.array([p.x for p in particles]) * weights_norm)
                mean_y = jnp.sum(jnp.array([p.y for p in particles]) * weights_norm)
                sin_sum = jnp.sum(
                    jnp.sin(jnp.array([p.theta for p in particles])) * weights_norm
                )
                cos_sum = jnp.sum(
                    jnp.cos(jnp.array([p.theta for p in particles])) * weights_norm
                )
                mean_theta = jnp.arctan2(sin_sum, cos_sum)
                estimated_poses.append(Pose(mean_x, mean_y, mean_theta))

        # Particle evolution
        show_lidar = not config.get("no_lidar_rays", False)
        fig2, axes2 = plot_particle_filter_evolution(
            particle_history,
            weight_history,
            true_poses,
            world,
            show_lidar_rays=show_lidar,
            observations_history=observations if show_lidar else None,
            n_rays=config["n_rays"],
        )
        filename2 = f"{param_prefix}_particle_filter_temporal_evolution_16steps.pdf"
        plt.savefig(os.path.join(figs_dir, filename2), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"    Saved: {filename2}")

        # Final step
        fig3, ax3 = plot_particle_filter_step(
            true_poses[-1],
            particle_history[-1],
            world,
            weight_history[-1],
            step_num=len(particle_history),
            show_lidar_rays=show_lidar,
            observations=observations[-1] if show_lidar else None,
            n_rays=config["n_rays"],
        )
        filename3 = f"{param_prefix}_final_particle_distribution_at_convergence.pdf"
        plt.savefig(os.path.join(figs_dir, filename3), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"    Saved: {filename3}")

        # Estimation error
        if len(estimated_poses) > 1:
            fig4, axes4 = plot_estimation_error(
                true_poses[: len(estimated_poses)], estimated_poses
            )
            filename4 = f"{param_prefix}_position_and_heading_error_over_time.pdf"
            plt.savefig(os.path.join(figs_dir, filename4), dpi=150, bbox_inches="tight")
            plt.close(fig4)
            print(f"    Saved: {filename4}")

        # Diagnostic weights flow
        fig8, ax8 = plot_weight_flow(diagnostic_weights)
        filename8 = f"{param_prefix}_particle_weights_and_ess_percentage_timeline.pdf"
        plt.savefig(os.path.join(figs_dir, filename8), dpi=300, bbox_inches="tight")
        plt.close(fig8)
        print(f"    Saved: {filename8}")

    # 5. Sensor observations
    print("  - Sensor observations plot...")
    true_lidar_distances = [
        distance_to_wall_lidar(pose, world, n_angles=config["n_rays"])
        for pose in true_poses
    ]
    fig5, ax5 = plot_sensor_observations(observations, true_lidar_distances)
    filename5 = f"{param_prefix}_lidar_distance_readings_along_trajectory.pdf"
    plt.savefig(os.path.join(figs_dir, filename5), dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"    Saved: {filename5}")

    # 6. SMC comparison plots (if available)
    if config.get("include_smc_comparison", False):
        print("  - SMC method comparison plots...")
        benchmark_results = load_benchmark_results(experiment_data_dir)

        # Timing comparison
        timing_filename = f"{param_prefix}_inference_runtime_performance_comparison.pdf"
        timing_path = os.path.join(figs_dir, timing_filename)
        plot_smc_timing_comparison(
            benchmark_results,
            save_path=timing_path,
            n_particles=config["n_particles"],
            K=config["k_rejuv"],
            n_particles_big_grid=config.get("n_particles_big_grid", 5),
        )
        print(f"    Saved: {timing_filename}")

        # Method comparison
        comparison_filename = f"{param_prefix}_comprehensive_4panel_smc_methods_analysis.pdf"
        comparison_path = os.path.join(figs_dir, comparison_filename)
        plot_smc_method_comparison(
            benchmark_results,
            true_poses,
            world,
            save_path=comparison_path,
            n_rays=config["n_rays"],
            n_particles=config["n_particles"],
            K=config["k_rejuv"],
            n_particles_big_grid=config.get("n_particles_big_grid", 5),
        )
        print(f"    Saved: {comparison_filename}")

        # Particle evolution plots for each method
        print("  - Particle evolution plots for each SMC method...")
        method_names = {
            "smc_basic": "Bootstrap Filter",
            "smc_hmc": "SMC + HMC",
            "smc_locally_optimal": "SMC + Locally Optimal",
            "smc_locally_optimal_big_grid": "SMC (N=5) + Locally Optimal (L=25)",
        }

        for method_key, method_display_name in method_names.items():
            if method_key in benchmark_results:
                result = benchmark_results[method_key]
                particle_history = result["particle_history"]
                weight_history = result["weight_history"]

                # Generate evolution plot for this method
                fig_evolution, _ = plot_particle_filter_evolution(
                    particle_history,
                    weight_history,
                    true_poses,
                    world,
                    show_lidar_rays=False,
                    observations_history=observations,
                    n_rays=config["n_rays"],
                    method_name=method_display_name,
                )

                # Save with method-specific filename
                method_slug = method_key.replace("_", "-")
                evolution_filename = f"{param_prefix}_{method_slug}_particle_evolution_timeline.pdf"
                evolution_path = os.path.join(figs_dir, evolution_filename)
                plt.savefig(evolution_path, dpi=150, bbox_inches="tight")
                plt.close(fig_evolution)
                print(f"    Saved: {evolution_filename}")

        # Multi-method estimation error plot
        print("  - Multi-method estimation error analysis...")
        error_filename = f"{param_prefix}_all_methods_tracking_accuracy_comparison.pdf"
        error_path = os.path.join(figs_dir, error_filename)
        plot_multi_method_estimation_error(
            benchmark_results, true_poses, save_path=error_path
        )
        print(f"    Saved: {error_filename}")

    print("\n✅ Figure generation complete!")
    print(f"All figures saved to: {figs_dir}")
    print("\nGenerated figures:")
    print("  - Ground truth trajectory (detailed and simple)")
    print("  - Multiple trajectory types comparison")
    print("  - LIDAR sensor demonstration")
    print("  - Sensor observations analysis")
    if os.path.exists(basic_demo_dir):
        print("  - Particle filter evolution")
        print("  - Final particle distribution")
        print("  - Estimation error over time")
        print("  - Diagnostic weight flow")
    if config.get("include_smc_comparison", False):
        print("  - SMC method timing comparison")
        print("  - SMC method comprehensive comparison")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "generate-data":
        generate_data(args)
    elif args.command == "plot-figures":
        plot_figures(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
