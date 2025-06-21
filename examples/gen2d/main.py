"""Main entry point for gen2d case study."""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path

from examples.gen2d.core import run_gen2d_inference
from examples.gen2d.data import generate_tracking_data
from examples.gen2d.figs import (
    plot_tracking_results,
    plot_trajectories,
    plot_diagnostics,
    plot_data_visualization,
)


def main():
    parser = argparse.ArgumentParser(
        description="Gen2D: Tracking objects in Game of Life with SMC"
    )

    # Figure generation modes
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument(
        "--tracking", action="store_true", help="Generate tracking visualization"
    )
    parser.add_argument(
        "--trajectories", action="store_true", help="Generate trajectory plot"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Generate SMC diagnostics"
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Generate data visualization (Game of Life patterns)",
    )

    # Model parameters
    parser.add_argument(
        "--pattern",
        type=str,
        default="oscillators",
        choices=["glider", "oscillators", "achims_p4", "random"],
        help="Initial Game of Life pattern",
    )
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument(
        "--n-frames", type=int, default=20, help="Number of frames to simulate"
    )
    parser.add_argument(
        "--n-components", type=int, default=5, help="Number of Gaussian components"
    )

    # Inference parameters
    parser.add_argument(
        "--n-particles", type=int, default=100, help="Number of SMC particles"
    )
    parser.add_argument(
        "--n-rejuvenation", type=int, default=2, help="Number of rejuvenation moves"
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Time step size")
    parser.add_argument(
        "--process-noise", type=float, default=0.5, help="Process noise std"
    )
    parser.add_argument("--obs-std", type=float, default=2.0, help="Observation std")

    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-pixels", type=int, default=500, help="Maximum pixels per frame"
    )

    args = parser.parse_args()

    # Default to tracking visualization if nothing specified
    if not any(
        [args.all, args.tracking, args.trajectories, args.diagnostics, args.data]
    ):
        args.tracking = True

    # Create output directory
    output_dir = Path("figs")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Handle data visualization mode (no inference needed)
    if args.data:
        print("Generating Game of Life data visualizations...")
        plot_data_visualization(save_dir=output_dir)
        print("Done!")
        return

    # Generate data
    print(f"Generating {args.pattern} pattern data...")
    key = jax.random.PRNGKey(args.seed)
    data_key, inference_key = jax.random.split(key)

    grids, observations, counts = generate_tracking_data(
        pattern=args.pattern,
        grid_size=args.grid_size,
        n_steps=args.n_frames,
        max_pixels=args.max_pixels,
        key=data_key,
    )

    print(
        f"Generated {args.n_frames + 1} frames with {jnp.sum(counts)} total active pixels"
    )

    # Run inference
    print(f"Running SMC inference with {args.n_particles} particles...")
    particles = run_gen2d_inference(
        observations=observations,
        observation_counts=counts,
        n_components=args.n_components,
        n_particles=args.n_particles,
        dt=args.dt,
        process_noise=args.process_noise,
        obs_std=args.obs_std,
        n_rejuvenation_moves=args.n_rejuvenation,
        key=inference_key,
    )

    print("Inference complete!")

    # Generate figures
    param_str = (
        f"{args.pattern}_K{args.n_components}_P{args.n_particles}_T{args.n_frames}"
    )

    if args.all or args.tracking:
        print("Generating tracking visualization...")
        # Show frames at different time points
        frame_indices = [0, args.n_frames // 3, 2 * args.n_frames // 3, args.n_frames]
        plot_tracking_results(
            grids,
            particles,
            frame_indices=frame_indices,
            obs_std=args.obs_std,
            save_path=output_dir / f"tracking_{param_str}.pdf",
        )
        print(f"Saved tracking_{param_str}.pdf")

    if args.all or args.trajectories:
        print("Generating trajectory plot...")
        plot_trajectories(
            particles,
            grid_size=args.grid_size,
            save_path=output_dir / f"trajectories_{param_str}.pdf",
        )
        print(f"Saved trajectories_{param_str}.pdf")

    if args.all or args.diagnostics:
        print("Generating diagnostics...")
        plot_diagnostics(
            particles, save_path=output_dir / f"diagnostics_{param_str}.pdf"
        )
        print(f"Saved diagnostics_{param_str}.pdf")

    print("Done!")


if __name__ == "__main__":
    main()
