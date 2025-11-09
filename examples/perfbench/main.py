"""
Helper CLI that wraps the legacy timing-benchmarks scripts so they can be invoked
from either the case-study directory or the repository root with the correct
PYTHONPATH.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = CASE_ROOT / "benchmarks"
DATA_DIR = CASE_ROOT / "data"
GENJL_SENTINEL = CASE_ROOT / ".genjl_instantiated"
EXPORT_DEFAULTS = [
    "benchmark_timings_is_all_frameworks.pdf",
    "benchmark_timings_hmc_all_frameworks.pdf",
]


def _ensure_bench_module(module: str):
    """Import a timing_benchmarks module, adding benchmarks/src to sys.path if needed."""
    bench_path = str(BENCH_ROOT / "src")
    if bench_path not in sys.path:
        sys.path.insert(0, bench_path)
    return importlib.import_module(module)


def _bench_env() -> dict[str, str]:
    env = os.environ.copy()
    bench_path = str(BENCH_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = bench_path if not current else f"{bench_path}{os.pathsep}{current}"
    depot_path = CASE_ROOT / ".julia_depot"
    depot_path.mkdir(exist_ok=True)
    env["JULIA_DEPOT_PATH"] = str(depot_path)
    home_path = CASE_ROOT / ".julia_home"
    home_path.mkdir(exist_ok=True)
    env["HOME"] = str(home_path)
    return env


def _run_python(relative_script: Path, *extra: str) -> None:
    script_path = CASE_ROOT / relative_script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    cmd = [sys.executable, str(script_path), *extra]
    subprocess.run(cmd, check=True, cwd=str(CASE_ROOT), env=_bench_env())


def _run_module(module: str, *extra: str) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    subprocess.run(cmd, check=True, cwd=str(CASE_ROOT), env=_bench_env())


def _ensure_genjl_ready() -> None:
    if GENJL_SENTINEL.exists():
        return
    cmd = [
        "julia",
        "--project=benchmarks/julia",
        "-e",
        "using Pkg; Pkg.instantiate(); Pkg.precompile()",
    ]
    subprocess.run(cmd, check=True, cwd=str(CASE_ROOT), env=_bench_env())
    GENJL_SENTINEL.touch()


def command_generate(args: argparse.Namespace) -> None:
    generation = _ensure_bench_module("timing_benchmarks.data.generation")
    dataset = generation.generate_polynomial_data(
        n_points=args.n_points, seed=args.seed
    )
    output_path = (CASE_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        xs=np.asarray(dataset.xs),
        ys=np.asarray(dataset.ys),
        true_a=dataset.true_a,
        true_b=dataset.true_b,
        true_c=dataset.true_c,
        noise_std=dataset.noise_std,
        n_points=dataset.n_points,
    )
    print(f"Saved polynomial dataset to {output_path}")


def command_run(args: argparse.Namespace) -> None:
    module_map = {
        "genjax": "timing_benchmarks.curvefit_benchmarks.genjax",
        "numpyro": "timing_benchmarks.curvefit_benchmarks.numpyro",
        "genjl": "timing_benchmarks.curvefit_benchmarks.genjl",
        "handcoded-jax": "timing_benchmarks.curvefit_benchmarks.handcoded_jax",
        "pyro": "timing_benchmarks.curvefit_benchmarks.pyro",
        "torch": "timing_benchmarks.curvefit_benchmarks.handcoded_torch",
    }
    module = module_map[args.framework]
    if args.framework == "genjl":
        _ensure_genjl_ready()
    extra = [
        "--output-dir",
        str(args.output_dir),
        "--repeats",
        str(args.repeats),
    ]
    if args.device:
        extra.extend(["--device", args.device])
    if args.framework_args:
        forwarded = [arg for arg in args.framework_args if arg != "--"]
        extra.extend(forwarded)
    if args.framework == "genjl":
        _ensure_genjl_ready()
    _run_module(module, *extra)


def command_combine(args: argparse.Namespace) -> None:
    extras: list[str] = [
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.frameworks:
        extras.extend(["--frameworks", *args.frameworks])
    _run_python(Path("benchmarks/combine_results.py"), *extras)


def command_clean(_: argparse.Namespace) -> None:
    for rel in ("data", "figs"):
        target = CASE_ROOT / rel
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed {target}")


def command_genjl_hmc(args: argparse.Namespace) -> None:
    _ensure_genjl_ready()
    extra = [
        "--chain-lengths",
        *[str(v) for v in args.chain_lengths],
        "--n-warmup",
        str(args.n_warmup),
        "--repeats",
        str(args.repeats),
        "--step-size",
        str(args.step_size),
        "--n-leapfrog",
        str(args.n_leapfrog),
        "--n-points",
        str(args.n_points),
        "--seed",
        str(args.seed),
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(args.output_dir),
    ]
    _run_python(Path("benchmarks/run_genjl_hmc.py"), *extra)


def command_export(args: argparse.Namespace) -> None:
    src = (CASE_ROOT / args.source_dir).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source directory {src} not found")
    dest = (CASE_ROOT / args.dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    patterns = args.files or EXPORT_DEFAULTS

    copied_any = False
    for pattern in patterns:
        for file_path in src.glob(pattern):
            target_name = file_path.name
            if args.prefix:
                target_name = f"{args.prefix}_{target_name}"
            target_path = dest / target_name
            shutil.copy2(file_path, target_path)
            print(f"Copied {file_path} -> {target_path}")
            copied_any = True

    if not copied_any:
        print(f"No matching artifacts found in {src} for patterns: {patterns}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perfbench helper CLI (wraps timing-benchmarks)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Generate the shared polynomial dataset")
    gen.add_argument("--n-points", type=int, default=50)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--output",
        type=Path,
        default=Path("data/curvefit/polynomial_data.npz"),
    )
    gen.set_defaults(func=command_generate)

    run = sub.add_parser("run", help="Run a single framework benchmark")
    run.add_argument(
        "--framework",
        required=True,
        choices=["genjax", "numpyro", "genjl", "handcoded-jax", "pyro", "torch"],
    )
    run.add_argument("--repeats", type=int, default=100)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", choices=["cpu", "cuda"], default=None)
    run.add_argument(
        "framework_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the framework script (prefix with --).",
    )
    run.set_defaults(func=command_run)

    comb = sub.add_parser("combine", help="Combine timing JSON into plots/tables")
    comb.add_argument("--data-dir", type=Path, default=Path("data"))
    comb.add_argument("--output-dir", type=Path, default=Path("figs"))
    comb.add_argument("--frameworks", nargs="*")
    comb.set_defaults(func=command_combine)

    clean = sub.add_parser("clean", help="Remove generated data/figures")
    clean.set_defaults(func=command_clean)

    genjl_hmc = sub.add_parser("genjl-hmc", help="Run Gen.jl HMC benchmarks")
    genjl_hmc.add_argument("--chain-lengths", nargs="+", type=int, default=[100, 500, 1000])
    genjl_hmc.add_argument("--n-warmup", type=int, default=500)
    genjl_hmc.add_argument("--repeats", type=int, default=10)
    genjl_hmc.add_argument("--step-size", type=float, default=0.01)
    genjl_hmc.add_argument("--n-leapfrog", type=int, default=20)
    genjl_hmc.add_argument("--n-points", type=int, default=50)
    genjl_hmc.add_argument("--seed", type=int, default=42)
    genjl_hmc.add_argument("--dataset", type=Path, default=Path("data/curvefit/polynomial_data.npz"))
    genjl_hmc.add_argument("--output-dir", type=Path, default=Path("data/curvefit/genjl"))
    genjl_hmc.set_defaults(func=command_genjl_hmc)

    export = sub.add_parser("export", help="Copy figures/tables to another directory")
    export.add_argument("--source-dir", type=Path, default=Path("figs"))
    export.add_argument("--dest-dir", type=Path, default=Path("../../figs"))
    export.add_argument("--prefix", type=str, default="perfbench")
    export.add_argument(
        "--files",
        nargs="*",
        help="Specific filenames or glob patterns to copy (defaults to key artifacts).",
    )
    export.set_defaults(func=command_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
