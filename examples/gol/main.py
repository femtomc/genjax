import argparse
from .figs import (
    save_blinker_gibbs_figure,
    save_logo_gibbs_figure,
    save_timing_scaling_figure,
)


def main():
    """Main CLI for Game of Life case study."""
    parser = argparse.ArgumentParser(
        description="GenJAX Game of Life Case Study - Probabilistic Conway's Game of Life Inference"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["all", "blinker", "logo", "timing"],
        default="all",
        help="Which figures to generate (default: all)",
    )

    # Inference parameters
    parser.add_argument(
        "--chain-length",
        type=int,
        default=250,
        help="Number of Gibbs sampling steps (default: 250)",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.03,
        help="Probability of rule violations (default: 0.03)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)",
    )

    # Timing parameters
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 150, 200],
        help="Grid sizes for timing analysis (default: [10, 50, 100, 150, 200])",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=5,
        help="Number of timing repetitions (default: 5)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "both"],
        default="cpu",
        help="Device for computation (default: cpu)",
    )

    args = parser.parse_args()

    print("=== GenJAX Game of Life Case Study ===")
    print(f"Mode: {args.mode}")
    print(
        f"Parameters: chain_length={args.chain_length}, flip_prob={args.flip_prob}, seed={args.seed}"
    )

    if args.mode in ["all", "blinker"]:
        print("\nGenerating blinker pattern reconstruction...")
        save_blinker_gibbs_figure(
            chain_length=args.chain_length, flip_prob=args.flip_prob, seed=args.seed
        )

    if args.mode in ["all", "logo"]:
        print("\nGenerating logo pattern reconstruction...")
        save_logo_gibbs_figure(
            chain_length=args.chain_length,
            flip_prob=args.flip_prob,
            seed=args.seed,
            small=True,  # Use small version for reasonable computation time
            size=128,  # 128x128 logo for excellent logo preservation
        )

    if args.mode in ["all", "timing"]:
        print("\nGenerating timing scaling analysis...")
        print(f"Grid sizes: {args.grid_sizes}")
        print(f"Timing repeats: {args.timing_repeats}")
        print(f"Device: {args.device}")

        save_timing_scaling_figure(
            grid_sizes=args.grid_sizes,
            repeats=args.timing_repeats,
            device=args.device,
            chain_length=args.chain_length,
            flip_prob=args.flip_prob,
            seed=args.seed,
        )

    print("\n=== Game of Life case study complete! ===")


if __name__ == "__main__":
    main()
