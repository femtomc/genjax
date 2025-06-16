"""Command-line interface for fair coin timing comparison."""

import argparse
from examples.faircoin.figs import timing_comparison_fig


def main():
    """Main CLI entry point for fair coin timing comparison."""
    parser = argparse.ArgumentParser(
        description="Fair coin (Beta-Bernoulli) timing comparison"
    )
    parser.add_argument(
        "--comparison", action="store_true", help="Include Pyro in timing comparison"
    )
    parser.add_argument(
        "--num-obs", type=int, default=50, help="Number of observations (default: 50)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=200,
        help="Number of timing repeats (default: 200)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of importance samples (default: 1000)",
    )

    args = parser.parse_args()

    print("Fair Coin (Beta-Bernoulli) Timing Comparison")
    print("=" * 50)
    print("Configuration:")
    print(f"  - Observations: {args.num_obs}")
    print(f"  - Timing repeats: {args.repeats}")
    print(f"  - Importance samples: {args.num_samples}")
    print(f"  - Include Pyro: {args.comparison}")
    print()

    timing_comparison_fig(
        num_obs=args.num_obs,
        repeats=args.repeats,
        num_samples=args.num_samples,
        include_pyro=args.comparison,
    )


if __name__ == "__main__":
    main()
