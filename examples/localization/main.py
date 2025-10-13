"""
Main execution script for the localization case study.

Demonstrates probabilistic robot localization using particle filtering.
Restructured to have two main commands:
1. generate-data: Generate all experimental data and save to data/
2. plot-figures: Generate all figures from saved data
"""

import argparse
import os
import jax.random as jrand
import matplotlib.pyplot as plt

from .core import (
    create_multi_room_world,
    benchmark_smc_methods,
)

from .data import (
    generate_ground_truth_data,
)

from .figs import (
    plot_smc_method_comparison,
    plot_localization_problem_explanation,
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
    print("GenJAX Localization Case Study - Paper Mode")
    print("=" * 50)
    print("Configuration:")
    print(f"  LIDAR rays: {args.n_rays}")
    print(f"  Particles: {args.n_particles}")
    print(f"  Trajectory steps: {args.n_steps}")
    print(f"  Random seed: {args.seed}")
    print(f"  World type: {args.world_type}")
    print(f"  K rejuvenation steps: {args.k_rejuv}")
    print(f"  Include SMC comparison: {args.include_smc_comparison}")
    print("=" * 50)

    # Setup output directory
    if os.path.isabs(args.output_dir):
        figs_dir = args.output_dir
    else:
        genjax_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        figs_dir = os.path.join(genjax_root, args.output_dir)
    os.makedirs(figs_dir, exist_ok=True)

    # Set random seed
    key = jrand.key(args.seed)

    # Create world
    print(f"\nCreating {args.world_type} multi-room world...")
    world = create_multi_room_world(world_type=args.world_type)
    print(f"World dimensions: {world.width} x {world.height}")

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

    param_prefix = f"localization_r{args.n_rays}_p{args.n_particles}_{args.world_type}"

    print("\n=== Paper Mode: Generating only paper figures ===")

    # 1. Localization problem explanation (1x4)
    print("  Generating localization problem explanation (1x4)...")
    fig_explain, axes_explain = plot_localization_problem_explanation(
        true_poses, observations, world, n_rays=args.n_rays
    )
    filename_explain = f"{param_prefix}_localization_problem_1x4_explanation.pdf"
    plt.savefig(os.path.join(figs_dir, filename_explain), dpi=300, bbox_inches="tight")
    plt.close(fig_explain)
    print(f"    Saved: {filename_explain}")

    # 2. SMC method comparison (4panel)
    if args.include_smc_comparison:
        print("  Generating SMC method comparison (4panel)...")
        print(
            "    Running benchmarks (Bootstrap filter, SMC+MH, SMC+HMC, SMC+Locally Optimal)..."
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
        comparison_filename = (
            f"{param_prefix}_comprehensive_4panel_smc_methods_analysis.pdf"
        )
        comparison_path = os.path.join(figs_dir, comparison_filename)
        plot_smc_method_comparison(
            benchmark_results,
            true_poses,
            world,
            save_path=comparison_path,
            n_rays=args.n_rays,
            n_particles=args.n_particles,
            K=args.k_rejuv,
            n_particles_big_grid=args.n_particles_big_grid,
        )
        print(f"    Saved: {comparison_filename}")
    else:
        print("  Skipping SMC comparison (use --include-smc-comparison to enable)")

    print("\n=== Paper mode complete! ===")
    figures_generated = 2 if args.include_smc_comparison else 1
    print(f"Generated {figures_generated} paper figure(s) in: {figs_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    print("GenJAX Localization Case Study - Paper Mode")
    paper_mode(args)


if __name__ == "__main__":
    main()
