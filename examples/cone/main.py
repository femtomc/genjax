from __future__ import annotations

import argparse

from .core import (
    make_expressive_objective,
    make_naive_iwae_objective,
    naive_elbo_objective,
    optimize_objective,
    run_table4_suite,
)
from .figs import save_figure_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cone case study (ported from PLDI'24 artifact).")

    subparsers = parser.add_subparsers(dest="command", required=False)

    table_parser = subparsers.add_parser(
        "table4",
        help="Train objectives and print Table-4-style objective summaries.",
    )
    table_parser.add_argument("--n-steps", type=int, default=1000)
    table_parser.add_argument("--batch-size", type=int, default=64)
    table_parser.add_argument("--learning-rate", type=float, default=1e-3)
    table_parser.add_argument("--eval-samples", type=int, default=5000)

    fig_parser = subparsers.add_parser(
        "fig2",
        help="Train a subset of objectives and export posterior/prior figures.",
    )
    fig_parser.add_argument("--n-steps", type=int, default=1000)
    fig_parser.add_argument("--batch-size", type=int, default=64)
    fig_parser.add_argument("--learning-rate", type=float, default=1e-3)
    fig_parser.add_argument("--output-dir", type=str, default="figs")

    parser.set_defaults(command="table4")
    return parser.parse_args()


def run_table4(args: argparse.Namespace) -> None:
    results = run_table4_suite(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_samples=args.eval_samples,
    )

    for result in results:
        print(f"{result.name}:")
        print((result.mean, result.variance))


def run_fig2(args: argparse.Namespace) -> None:
    naive_elbo = optimize_objective(
        naive_elbo_objective,
        (0.0, 0.0, 1.0, 1.0),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed_value=0,
    )

    naive_iwae5 = optimize_objective(
        make_naive_iwae_objective(5),
        (3.0, 0.0, 1.0, 1.0),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed_value=1,
    )

    expressive_diwhvi = optimize_objective(
        make_expressive_objective(5, 5),
        (0.0, 0.0),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed_value=2,
    )

    save_figure_suite(
        naive_elbo_params=naive_elbo.params,
        naive_iwae5_params=naive_iwae5.params,
        expressive_diwhvi_params=expressive_diwhvi.params,
        output_dir=args.output_dir,
    )

    print("Saved figures:")
    print("  - cone_prior_samples.pdf")
    print("  - cone_naive_elbo_posterior.pdf")
    print("  - cone_naive_iwae5_posterior.pdf")
    print("  - cone_expressive_diwhvi_posterior.pdf")


def main() -> None:
    args = parse_args()

    if args.command == "table4":
        run_table4(args)
    elif args.command == "fig2":
        run_fig2(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
