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
    plot_localization_problem_explanation,
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
        choices=["paper"],
        nargs="?",
        default="paper",
        help="Main command: paper (generate only paper figures)",
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


# Old generate_data and plot_figures functions removed - keeping only paper mode which combines both


def paper_mode(args):
    """Generate only the paper figures (1x4 explanation + 4panel SMC comparison)."""
    # Generate data and get the experiment directory
    experiment_data_dir = generate_data(args)

    # Load config
    config = load_experiment_metadata(experiment_data_dir)

    # Setup output directory
    if os.path.isabs(args.output_dir):
        figs_dir = args.output_dir
    else:
        genjax_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        figs_dir = os.path.join(genjax_root, args.output_dir)
    os.makedirs(figs_dir, exist_ok=True)

    # Load ground truth
    world = create_multi_room_world(world_type=config["world_type"])
    true_poses, observations = load_ground_truth_data(experiment_data_dir)

    param_prefix = f"localization_r{config['n_rays']}_p{config['n_particles']}_{config['world_type']}"

    print("\n=== Paper Mode: Generating only paper figures ===")

    # 1. Localization problem explanation (1x4)
    print("  Generating localization problem explanation (1x4)...")
    fig_explain, axes_explain = plot_localization_problem_explanation(
        true_poses, observations, world, n_rays=config["n_rays"]
    )
    filename_explain = f"{param_prefix}_localization_problem_1x4_explanation.pdf"
    plt.savefig(os.path.join(figs_dir, filename_explain), dpi=300, bbox_inches="tight")
    plt.close(fig_explain)
    print(f"    Saved: {filename_explain}")

    # 2. SMC method comparison (4panel)
    if config.get("include_smc_comparison", False):
        print("  Generating SMC method comparison (4panel)...")
        benchmark_results = load_benchmark_results(experiment_data_dir)
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

    print("\n=== Paper mode complete! ===")
    print(f"Generated 2 paper figures in: {figs_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    print("GenJAX Localization Case Study - Paper Mode")
    paper_mode(args)


if __name__ == "__main__":
    main()
