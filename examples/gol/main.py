import argparse
from .figs import save_all_showcase_figures


def main():
    """Main CLI for Game of Life case study."""
    parser = argparse.ArgumentParser(
        description="GenJAX Game of Life Case Study - Probabilistic Conway's Game of Life Inference"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["showcase"],
        default="showcase",
        help="Which figures to generate (default: showcase - paper figures only)",
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
    print("Mode: showcase (paper artifact)")
    print(
        f"Parameters: chain_length={args.chain_length}, flip_prob={args.flip_prob}, seed={args.seed}"
    )

    print("\nGenerating showcase figures...")
    save_all_showcase_figures()

    print("\n=== Game of Life case study complete! ===")


if __name__ == "__main__":
    main()
