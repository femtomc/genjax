"""Command-line interface for intuitive physics case study."""

import argparse
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import sys
import os

sys.path.append(os.path.dirname(__file__))
from figs import *


def main():
    parser = argparse.ArgumentParser(description="Intuitive Physics Case Study")
    parser.add_argument(
        "--environment", action="store_true", help="Generate environment visualization"
    )
    parser.add_argument(
        "--trajectory", action="store_true", help="Generate trajectory visualization"
    )
    parser.add_argument(
        "--action-space", action="store_true", help="Generate action space heatmaps"
    )
    parser.add_argument(
        "--samples", action="store_true", help="Generate sample actions visualization"
    )
    parser.add_argument(
        "--timing", action="store_true", help="Generate timing comparison"
    )
    parser.add_argument(
        "--inference", action="store_true", help="Generate inference demonstration"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all visualizations"
    )

    # Trajectory parameters
    parser.add_argument(
        "--theta", type=float, default=0.3, help="Trajectory angle (radians)"
    )
    parser.add_argument(
        "--impulse", type=float, default=2.0, help="Trajectory impulse magnitude"
    )

    args = parser.parse_args()

    if args.all:
        args.environment = True
        args.trajectory = True
        args.action_space = True
        args.samples = True
        args.timing = True
        args.inference = True

    # Default to environment if no specific option given
    if not any(
        [
            args.environment,
            args.trajectory,
            args.action_space,
            args.samples,
            args.timing,
            args.inference,
        ]
    ):
        args.environment = True

    print("🎯 Intuitive Physics Case Study")
    print("=" * 50)

    if args.environment:
        print("\n📊 Generating environment visualization...")
        fig = create_physics_visualization()
        save_figure(fig, "environment_setup")
        plt.close(fig)

    if args.trajectory:
        print(
            f"\n🎯 Generating trajectory visualization (θ={args.theta:.3f}, ι={args.impulse:.3f})..."
        )
        fig = visualize_trajectory(args.theta, args.impulse)
        save_figure(fig, f"trajectory_theta{args.theta:.3f}_impulse{args.impulse:.3f}")
        plt.close(fig)

    if args.action_space:
        print("\n🗺️ Generating action space analysis...")
        fig = action_space_heatmap()
        save_figure(fig, "action_space_analysis")
        plt.close(fig)

    if args.samples:
        print("\n🎲 Generating sample actions visualization...")
        fig = sample_actions_visualization(num_samples=3)
        save_figure(fig, "sample_actions")
        plt.close(fig)

    if args.timing:
        print("\n⏱️ Running timing benchmarks...")
        fig = timing_comparison_fig()
        save_figure(fig, "timing_comparison")
        plt.close(fig)

    if args.inference:
        print("\n🔍 Generating inference demonstration...")
        fig = inference_demonstration_fig()
        save_figure(fig, "inference_demonstration")
        plt.close(fig)

    print("\n✅ Case study complete!")
    print("\nGenerated figures:")
    print("- Environment setup: examples/intuitive_physics/figs/environment_setup.pdf")
    if args.trajectory:
        print(
            f"- Trajectory: examples/intuitive_physics/figs/trajectory_theta{args.theta:.3f}_impulse{args.impulse:.3f}.pdf"
        )
    if args.action_space:
        print(
            "- Action space: examples/intuitive_physics/figs/action_space_analysis.pdf"
        )
    if args.samples:
        print("- Sample actions: examples/intuitive_physics/figs/sample_actions.pdf")
    if args.timing:
        print(
            "- Timing comparison: examples/intuitive_physics/figs/timing_comparison.pdf"
        )
    if args.inference:
        print(
            "- Inference demonstration: examples/intuitive_physics/figs/inference_demonstration.pdf"
        )


if __name__ == "__main__":
    main()
