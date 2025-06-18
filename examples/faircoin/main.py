"""Command-line interface for fair coin timing comparison."""

import argparse
from examples.faircoin.figs import (
    timing_comparison_fig,
    posterior_comparison_fig,
    combined_comparison_fig,
    save_all_figures,
)


def main():
    """Main CLI entry point for fair coin timing comparison."""
    parser = argparse.ArgumentParser(
        description="Fair coin (Beta-Bernoulli) timing comparison"
    )
    parser.add_argument(
        "--posterior", action="store_true", help="Generate posterior comparison figure"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all figures (timing + posterior)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate combined figure (posterior + timing)",
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

    print("Fair Coin (Beta-Bernoulli) Analysis")
    print("=" * 40)
    print("Configuration:")
    print(f"  - Observations: {args.num_obs}")
    print(f"  - Timing repeats: {args.repeats}")
    print(f"  - Importance samples: {args.num_samples}")
    print(f"  - Generate posterior: {args.posterior}")
    print(f"  - Generate all: {args.all}")
    print(f"  - Generate combined: {args.combined}")
    print()

    if args.all:
        save_all_figures(
            num_obs=args.num_obs,
            num_samples=args.num_samples,
        )
    elif args.posterior:
        posterior_comparison_fig(
            num_obs=args.num_obs,
            num_samples=args.num_samples,
        )
    elif args.combined:
        combined_comparison_fig(
            num_obs=args.num_obs,
            num_samples=args.num_samples,
        )
    else:
        # Default: timing comparison
        timing_comparison_fig(
            num_obs=args.num_obs,
            repeats=args.repeats,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
