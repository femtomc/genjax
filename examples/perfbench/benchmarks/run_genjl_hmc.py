#!/usr/bin/env python
"""Run Gen.jl HMC benchmarks for the polynomial regression case study."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JULIA_DIR = PROJECT_ROOT / "julia"
JULIA_DATA_DIR = JULIA_DIR / "data"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from timing_benchmarks.data.generation import generate_polynomial_data  # noqa: E402


def load_dataset(dataset_path: Path, n_points: int, seed: int):
    if dataset_path.exists():
        data_npz = np.load(dataset_path)
        xs = data_npz["xs"]
        ys = data_npz["ys"]
    else:
        dataset = generate_polynomial_data(n_points=n_points, seed=seed)
        xs = np.asarray(dataset.xs)
        ys = np.asarray(dataset.ys)
    return xs, ys


def run_julia_script(args: argparse.Namespace, csv_path: Path) -> None:
    chain_lengths = ",".join(str(v) for v in args.chain_lengths)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_path = (PROJECT_ROOT / "benchmarks/julia").as_posix()
    timing_module = (PROJECT_ROOT / "benchmarks/julia/src/TimingBenchmarks.jl").as_posix()

    julia_code = dedent(
        f"""
        using Pkg
        Pkg.activate("{project_path}")
        using CSV, DataFrames, JSON

        include("{timing_module}")
        using .TimingBenchmarks

        df = CSV.read("{csv_path.as_posix()}", DataFrame)
        xs = Float64.(df.x)
        ys = Float64.(df.y)

        data = PolynomialData(xs, ys, Dict("a" => 0.0, "b" => 0.0, "c" => 0.0), 0.05, length(xs))
        chain_lengths = [{chain_lengths}]

        for n_samples in chain_lengths
            println("Running Gen.jl HMC with ", n_samples, " samples...")
            result = run_polynomial_hmc_benchmark(
                data,
                n_samples;
                n_warmup = {args.n_warmup},
                repeats = {args.repeats},
                step_size = {args.step_size},
                n_leapfrog = {args.n_leapfrog}
            )

            output_file = "{output_dir.as_posix()}/hmc_n$(n_samples).json"
            open(output_file, "w") do f
                JSON.print(f, result, 2)
            end

            println("✓ Gen.jl HMC (n=$(n_samples)): ", result["mean_time"], "s ± ", result["std_time"], "s")
            println("  Saved to: ", output_file)
        end
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jl", delete=False) as tmp_script:
        tmp_script.write(julia_code)
        tmp_script_path = Path(tmp_script.name)

    try:
        env = os.environ.copy()
        home = PROJECT_ROOT / ".julia_home"
        home.mkdir(exist_ok=True)
        depot = PROJECT_ROOT / ".julia_depot"
        depot.mkdir(exist_ok=True)
        env["HOME"] = str(home)
        env["JULIA_DEPOT_PATH"] = str(depot)
        result = subprocess.run(
            ["julia", "--project=julia", str(tmp_script_path)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Julia command failed with exit code {result.returncode}:\n{result.stderr}"
            )
    finally:
        tmp_script_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gen.jl HMC benchmarks.")
    parser.add_argument(
        "--chain-lengths",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Chain lengths (number of samples) to benchmark.",
    )
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--n-leapfrog", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/curvefit/polynomial_data.npz"),
        help="Path to existing polynomial_data.npz (regenerated if missing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curvefit/genjl"),
        help="Directory to store HMC timing JSON outputs.",
    )
    args = parser.parse_args()

    xs, ys = load_dataset(args.dataset, n_points=args.n_points, seed=args.seed)

    JULIA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = JULIA_DATA_DIR / "temp_dataset.csv"
    np.savetxt(
        csv_path,
        np.column_stack([xs, ys]),
        fmt="%.10f",
        delimiter=",",
        header="x,y",
        comments="",
    )

    try:
        run_julia_script(args, csv_path)
    finally:
        csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
