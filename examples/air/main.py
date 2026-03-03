from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

from .core import (
    AIRSuiteResult,
    DEFAULT_CONFIG,
    SMALL_CONFIG,
    VALID_ESTIMATORS,
    estimate_count_accuracy,
    estimate_objective_statistics,
    init_air_params,
    make_air_objective,
    prepare_air_dataset,
    run_estimator_suite,
    save_suite_results_csv,
    save_training_history_csv,
    train_air,
)


def _parse_estimators(raw: str) -> tuple[str, ...]:
    estimators = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not estimators:
        raise ValueError("At least one estimator must be provided")

    invalid = [name for name in estimators if name not in VALID_ESTIMATORS]
    if invalid:
        raise ValueError(
            f"Unknown estimator(s): {', '.join(invalid)}. "
            f"Valid options: {', '.join(VALID_ESTIMATORS)}"
        )

    return estimators


def _resolve_config(use_small_config: bool):
    return SMALL_CONFIG if use_small_config else DEFAULT_CONFIG


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "multi-mnist"],
        default="synthetic",
        help="Dataset source.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to multi_mnist_uint8.npz when --dataset multi-mnist.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=512,
        help="Number of examples to load/sample.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIR case study for GenJAX (ported from the PLDI'24 programmable-VI "
            "artifact, GenJAX path only)."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser(
        "train",
        help="Train one estimator/objective configuration.",
    )
    _add_dataset_args(train_parser)
    train_parser.add_argument("--estimator", choices=VALID_ESTIMATORS, default="enum")
    train_parser.add_argument("--particles", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=6)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size for accuracy/objective metrics.",
    )
    train_parser.add_argument(
        "--eval-samples",
        type=int,
        default=6,
        help="MC samples for objective mean/variance estimation.",
    )
    train_parser.add_argument(
        "--history-csv",
        type=str,
        default=None,
        help="Optional output path for per-epoch training history CSV.",
    )
    train_parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional output path for one-row summary CSV.",
    )
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--small-config",
        action="store_true",
        help="Use a smaller architecture for smoke tests and quick iteration.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Train and compare multiple estimators under one configuration.",
    )
    _add_dataset_args(compare_parser)
    compare_parser.add_argument(
        "--estimators",
        type=str,
        default="enum,reinforce,mvd,hybrid",
        help="Comma-separated estimator list.",
    )
    compare_parser.add_argument("--particles", type=int, default=1)
    compare_parser.add_argument("--epochs", type=int, default=4)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--learning-rate", type=float, default=1e-4)
    compare_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size for accuracy/objective metrics.",
    )
    compare_parser.add_argument("--eval-samples", type=int, default=6)
    compare_parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional output path for suite summary CSV.",
    )
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument(
        "--small-config",
        action="store_true",
        help="Use a smaller architecture for smoke tests and quick iteration.",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-data",
        help="Download/generate multi-MNIST data and write an AIR NPZ file.",
    )
    fetch_parser.add_argument(
        "--output",
        type=str,
        default="examples/air/data/multi_mnist_uint8.npz",
        help="Output NPZ path containing x (uint8 images) and y (object labels).",
    )
    fetch_parser.add_argument(
        "--cache-root",
        type=str,
        default="/tmp/air-data",
        help="Cache directory used by pyro.contrib.examples.multi_mnist.",
    )
    fetch_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional prefix length to keep from the generated dataset.",
    )

    parser.set_defaults(command="compare")
    return parser.parse_args()


def _print_suite_table(results: Sequence[AIRSuiteResult]) -> None:
    print("\nEstimator results")
    print("=" * 88)
    print(
        f"{'estimator':<12} {'K':>3} {'final_loss':>14} "
        f"{'final_acc':>12} {'obj_mean':>12} {'obj_var':>12}"
    )
    print("-" * 88)
    for result in results:
        print(
            f"{result.estimator:<12} {result.num_particles:>3d} "
            f"{result.final_loss:>14.4f} {result.final_accuracy:>12.4f} "
            f"{result.objective_mean:>12.4f} {result.objective_variance:>12.4f}"
        )


def run_fetch_data(args: argparse.Namespace) -> None:
    try:
        from pyro.contrib.examples import multi_mnist
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "fetch-data requires pyro + torchvision. Run it via: "
            "`pixi run -e perfbench-pyro python -m examples.air.main fetch-data ...`"
        ) from exc

    cache_root = Path(args.cache_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading/generating Multi-MNIST under {cache_root} ...")
    x, y = multi_mnist.load(str(cache_root))

    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be >= 1 when provided")
        x = x[: args.max_examples]
        y = y[: args.max_examples]

    np.savez_compressed(out_path, x=x, y=y)
    print("saved AIR dataset")
    print(f"path:      {out_path}")
    print(f"examples:  {x.shape[0]}")
    print(f"shape:     {x.shape[1:]} (uint8)")


def run_train(args: argparse.Namespace) -> None:
    config = _resolve_config(args.small_config)

    init_params = init_air_params(jax.random.key(args.seed), config=config)
    dataset = prepare_air_dataset(
        dataset=args.dataset,
        config=config,
        n_samples=args.num_examples,
        seed_value=args.seed + 1,
        data_path=args.data_path,
        decoder_params=init_params.decoder,
    )

    training = train_air(
        dataset.observations,
        dataset.true_counts,
        estimator=args.estimator,
        config=config,
        num_particles=args.particles,
        init_params=init_params,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluate_accuracy_every=1,
        eval_batch_size=args.eval_batch_size,
        seed_value=args.seed + 2,
    )

    objective, _, guide = make_air_objective(
        args.estimator,
        config=config,
        num_particles=args.particles,
    )
    objective_mean, objective_var = estimate_objective_statistics(
        objective,
        training.params,
        dataset.observations,
        n_mc_samples=args.eval_samples,
        seed_value=args.seed + 3,
        batch_size=args.eval_batch_size,
    )
    final_accuracy, confusion = estimate_count_accuracy(
        guide,
        training.params,
        dataset.observations,
        dataset.true_counts,
        config=config,
        seed_value=args.seed + 4,
        batch_size=args.eval_batch_size,
    )

    result = AIRSuiteResult(
        estimator=args.estimator,
        num_particles=args.particles,
        final_loss=float(training.loss_history[-1]),
        final_accuracy=final_accuracy,
        objective_mean=objective_mean,
        objective_variance=objective_var,
        params=training.params,
    )

    print("AIR train summary")
    print("=" * 80)
    print(f"dataset:      {args.dataset}")
    print(f"examples:     {dataset.observations.shape[0]}")
    print(f"estimator:    {args.estimator}")
    print(f"particles:    {args.particles}")
    print(f"epochs:       {args.epochs}")
    print(f"batch size:   {args.batch_size}")
    print(f"final loss:   {result.final_loss:.6f}")
    print(f"final acc:    {result.final_accuracy:.6f}")
    print(f"obj mean:     {result.objective_mean:.6f}")
    print(f"obj variance: {result.objective_variance:.6f}")
    print("count confusion matrix:")
    print(confusion)

    if args.history_csv:
        save_training_history_csv(training, args.history_csv)
        print(f"saved history CSV to {args.history_csv}")

    if args.summary_csv:
        save_suite_results_csv([result], args.summary_csv)
        print(f"saved summary CSV to {args.summary_csv}")


def run_compare(args: argparse.Namespace) -> None:
    config = _resolve_config(args.small_config)
    estimators = _parse_estimators(args.estimators)

    base_params = init_air_params(jax.random.key(args.seed), config=config)
    dataset = prepare_air_dataset(
        dataset=args.dataset,
        config=config,
        n_samples=args.num_examples,
        seed_value=args.seed + 1,
        data_path=args.data_path,
        decoder_params=base_params.decoder,
    )

    results = run_estimator_suite(
        dataset.observations,
        dataset.true_counts,
        estimators=estimators,
        config=config,
        num_particles=args.particles,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_objective_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        seed_value=args.seed + 2,
    )

    print("AIR estimator comparison")
    print("=" * 88)
    print(f"dataset:      {args.dataset}")
    print(f"examples:     {dataset.observations.shape[0]}")
    print(f"particles:    {args.particles}")
    print(f"epochs:       {args.epochs}")
    print(f"batch size:   {args.batch_size}")
    _print_suite_table(results)

    if args.summary_csv:
        save_suite_results_csv(results, args.summary_csv)
        print(f"\nsaved summary CSV to {args.summary_csv}")


def main() -> None:
    args = parse_args()

    if args.command == "fetch-data":
        run_fetch_data(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
