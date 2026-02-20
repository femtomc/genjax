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
import json

CASE_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = CASE_ROOT / "benchmarks"
DATA_DIR = CASE_ROOT / "data"
GENJL_SENTINEL = CASE_ROOT / ".genjl_instantiated"
EXPORT_DEFAULTS = [
    "benchmark_timings_is_all_frameworks.pdf",
    "benchmark_timings_hmc_all_frameworks.pdf",
]
DEFAULT_IS_FRAMEWORKS = [
    "genjax",
    "numpyro",
    "genjl",
    "handcoded-jax",
    "pyro",
    "torch",
]
DEFAULT_HMC_FRAMEWORKS = [
    "genjax",
    "numpyro",
    "handcoded_jax",
    "pyro",
    "handcoded_torch",
    "genjl",
]
IS_OUTPUT_SUBDIRS = {
    "genjax": "genjax",
    "numpyro": "numpyro",
    "genjl": "genjl_dynamic",
    "handcoded-jax": "handcoded_jax",
    "pyro": "pyro",
    "torch": "handcoded_torch",
}
DEFAULT_IS_REPEATS = 50
DEFAULT_IS_INNER_REPEATS = 50
HIGH_IS_FRAMEWORKS = {"genjax", "numpyro", "handcoded-jax"}
HIGH_IS_REPEATS = 100
HIGH_IS_INNER_REPEATS = 100
PYRO_IS_REPEATS = 5
PYRO_IS_INNER_REPEATS = 5


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


def _run_example_script(
    env_name: str | None,
    script: str,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> None:
    pixi_bin = shutil.which("pixi")
    if pixi_bin is None:
        raise RuntimeError("Pixi executable not found in PATH.")

    cmd = [pixi_bin, "run"]
    if env_name:
        cmd.extend(["-e", env_name])
    cmd.extend(["python", script, *args])

    env = _bench_env()
    if env_overrides:
        env.update(env_overrides)

    subprocess.run(cmd, check=True, cwd=str(CASE_ROOT), env=env)


def _run_main_subcommand(
    env_name: str | None, *cli_args: str, env_overrides: dict[str, str] | None = None
) -> None:
    _run_example_script(env_name, "main.py", *cli_args, env_overrides=env_overrides)


def _is_env_for_framework(framework: str, mode: str) -> str:
    norm = framework.replace("-", "_")
    if norm in {"genjax", "numpyro", "handcoded_jax", "genjl"}:
        return "perfbench-cuda" if mode == "cuda" else "perfbench"
    if norm == "pyro":
        return "perfbench-pyro"
    if norm in {"torch", "handcoded_torch"}:
        return "perfbench-torch"
    raise ValueError(f"Unknown framework '{framework}'")


def _normalize_hmc_framework(framework: str) -> str:
    return framework.replace("-", "_")


def _framework_dir(base: Path, framework: str) -> Path | None:
    base = Path(base)
    candidates = [base / "curvefit" / framework, base / framework]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _has_is_results(base: Path, frameworks: list[str]) -> bool:
    for fw in frameworks:
        fw_dir = _framework_dir(base, fw)
        if not fw_dir:
            continue
        for n in [100, 1000, 5000, 10000]:
            if (fw_dir / f"is_n{n}.json").exists():
                return True
    return False


def _has_hmc_results(base: Path, frameworks: list[str]) -> bool:
    for fw in frameworks:
        fw_dir = _framework_dir(base, fw)
        if not fw_dir:
            continue
        for n in [100, 500, 1000, 5000, 10000]:
            if (fw_dir / f"hmc_n{n}.json").exists():
                return True
    return False


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
    framework_args = [arg for arg in args.framework_args if arg != "--"]
    if args.particles:
        framework_args.extend(["--n-particles", *[str(p) for p in args.particles]])

    extra = [
        "--output-dir",
        str(args.output_dir),
        "--repeats",
        str(args.repeats),
        "--inner-repeats",
        str(args.inner_repeats),
    ]
    if args.device:
        extra.extend(["--device", args.device])
    if framework_args:
        extra.extend(framework_args)
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
    for rel in ("data", "data_cpu", "figs", "figs_cpu"):
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


def _resolve_is_output_dir(base_dir: Path, framework: str) -> Path:
    if framework not in IS_OUTPUT_SUBDIRS:
        raise ValueError(f"Unknown IS framework '{framework}'")
    return base_dir / "curvefit" / IS_OUTPUT_SUBDIRS[framework]


def _print_is_timings(framework: str, output_dir: Path, particles: list[int]) -> None:
    for n in particles:
        path = output_dir / f"is_n{n}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            mean = float(data.get("mean_time", float("nan")))
            std = float(data.get("std_time", float("nan")))
            print(f"    ↳ IS {framework} @ n={n}: {mean:.6f}s ± {std:.6f}s")
        except Exception as err:
            print(f"    ↳ IS {framework} @ n={n}: failed to read timings ({err})")


def _print_hmc_timings(framework: str, output_dir: Path, chain_lengths: list[int]) -> None:
    fw_dir = output_dir / framework
    for n in chain_lengths:
        path = fw_dir / f"hmc_n{n}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            mean = float(data.get("mean_time", float("nan")))
            std = float(data.get("std_time", float("nan")))
            print(f"    ↳ HMC {framework} @ n={n}: {mean:.6f}s ± {std:.6f}s")
        except Exception as err:
            print(f"    ↳ HMC {framework} @ n={n}: failed to read timings ({err})")


def command_pipeline(args: argparse.Namespace) -> None:
    # Interpret inference selection
    if args.inference == "is":
        args.skip_hmc = True
        args.skip_is = False
    elif args.inference == "hmc":
        args.skip_is = True
        args.skip_hmc = False

    mode = args.mode
    particles = args.particles or [1000, 5000, 10000]
    shared_fw = args.frameworks or []
    is_frameworks = args.is_frameworks or (shared_fw if shared_fw else DEFAULT_IS_FRAMEWORKS)
    hmc_frameworks = args.hmc_frameworks or (shared_fw if shared_fw else DEFAULT_HMC_FRAMEWORKS)
    norm_hmc = [_normalize_hmc_framework(fw) for fw in hmc_frameworks]
    device = "cuda" if mode == "cuda" else "cpu"

    data_root = (CASE_ROOT / ("data" if mode == "cuda" else "data_cpu")).resolve()
    figs_root = (CASE_ROOT / ("figs" if mode == "cuda" else "figs_cpu")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    figs_root.mkdir(parents=True, exist_ok=True)
    curvefit_root = (data_root / "curvefit").resolve()
    curvefit_root.mkdir(parents=True, exist_ok=True)
    dataset_output = (args.data_output or curvefit_root / "polynomial_data.npz").resolve()
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    export_prefix = args.fig_prefix or ("perfbench" if mode == "cuda" else "perfbench_cpu")
    export_dest = (CASE_ROOT / args.export_dest).resolve()
    export_dest.mkdir(parents=True, exist_ok=True)
    hmc_output_dir = (data_root if mode == "cuda" else curvefit_root).resolve()

    if not args.skip_generate:
        print("→ Generating dataset")
        gen_env = "perfbench-cuda" if mode == "cuda" else "perfbench"
        _run_main_subcommand(
            gen_env,
            "generate-data",
            "--n-points",
            str(args.data_n_points),
            "--seed",
            str(args.data_seed),
            "--output",
            str(dataset_output),
        )

    ran_is = False
    ran_hmc = False

    if not args.skip_is:
        for framework in is_frameworks:
            output_dir = _resolve_is_output_dir(data_root, framework).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            env_name = _is_env_for_framework(framework, mode)
            framework_repeats = args.is_repeats
            framework_inner_repeats = args.is_inner_repeats
            if framework == "pyro" and args.is_repeats == DEFAULT_IS_REPEATS and args.is_inner_repeats == DEFAULT_IS_INNER_REPEATS:
                framework_repeats = PYRO_IS_REPEATS
                framework_inner_repeats = PYRO_IS_INNER_REPEATS
            elif (
                framework in HIGH_IS_FRAMEWORKS
                and args.is_repeats == DEFAULT_IS_REPEATS
                and args.is_inner_repeats == DEFAULT_IS_INNER_REPEATS
            ):
                framework_repeats = HIGH_IS_REPEATS
                framework_inner_repeats = HIGH_IS_INNER_REPEATS

            cmd = [
                "run",
                "--framework",
                framework,
                "--repeats",
                str(framework_repeats),
                "--inner-repeats",
                str(framework_inner_repeats),
                "--output-dir",
                str(output_dir),
            ]
            if particles:
                cmd.append("--particles")
                cmd.extend(str(p) for p in particles)
            if framework in {"pyro", "torch"}:
                cmd.extend(["--device", device])
            extra_env = None
            if framework in {"pyro", "torch"}:
                extra_env = {"JAX_PLATFORMS": "cpu"}
            elif env_name == "perfbench-cuda":
                extra_env = {"JAX_PLATFORMS": "cuda"}
            print(f"→ IS {framework} (env={env_name})")
            _run_main_subcommand(env_name, *cmd, env_overrides=extra_env)
            _print_is_timings(framework, output_dir, particles)
            ran_is = True

    # Defer plotting until after inference stages; we'll combine selectively later.

    if not args.skip_hmc:
        def run_hmc_group(frameworks: list[str], env_name: str | None, device_arg: str | None = None, env_overrides: dict[str, str] | None = None):
            if not frameworks:
                return
            cli = [
                "benchmarks/run_hmc_benchmarks.py",
                "--frameworks",
                *frameworks,
                "--chain-lengths",
                *[str(v) for v in args.hmc_chain_lengths],
                "--repeats",
                str(args.hmc_repeats),
                "--n-warmup",
                str(args.hmc_warmup),
                "--step-size",
                str(args.hmc_step_size),
                "--n-leapfrog",
                str(args.hmc_n_leapfrog),
                "--output-dir",
                str(hmc_output_dir),
                "--n-points",
                str(args.data_n_points),
            ]
            if device_arg:
                cli.extend(["--device", device_arg])
            print(f"→ HMC {', '.join(frameworks)} (env={env_name or 'default'})")
            _run_example_script(env_name, *cli, env_overrides=env_overrides)

        jax_group = [fw for fw in norm_hmc if fw in {"genjax", "numpyro", "handcoded_jax"}]
        pyro_group = [fw for fw in norm_hmc if fw == "pyro"]
        torch_group = [fw for fw in norm_hmc if fw == "handcoded_torch"]
        genjl_requested = "genjl" in norm_hmc

        jax_env_name = "perfbench-cuda" if mode == "cuda" else "perfbench"
        jax_env_overrides = {"JAX_PLATFORMS": "cuda"} if mode == "cuda" else None
        run_hmc_group(jax_group, jax_env_name, device, env_overrides=jax_env_overrides)
        for fw in jax_group:
            _print_hmc_timings(fw, hmc_output_dir, args.hmc_chain_lengths)
            ran_hmc = True

        run_hmc_group(
            pyro_group,
            "perfbench-pyro",
            device,
            env_overrides={"JAX_PLATFORMS": "cpu"},
        )
        for fw in pyro_group:
            _print_hmc_timings(fw, hmc_output_dir, args.hmc_chain_lengths)
            ran_hmc = True

        run_hmc_group(
            torch_group,
            "perfbench-torch",
            device,
            env_overrides={"JAX_PLATFORMS": "cpu"},
        )
        for fw in torch_group:
            _print_hmc_timings(fw, hmc_output_dir, args.hmc_chain_lengths)
            ran_hmc = True

        if genjl_requested:
            genjl_dir = (curvefit_root / "genjl").resolve()
            genjl_dir.mkdir(parents=True, exist_ok=True)
            genjl_cmd = [
                "genjl-hmc",
                "--chain-lengths",
                *[str(v) for v in args.hmc_chain_lengths],
                "--n-warmup",
                str(args.hmc_warmup),
                "--repeats",
                str(args.hmc_repeats),
                "--step-size",
                str(args.hmc_step_size),
                "--n-leapfrog",
                str(args.hmc_n_leapfrog),
                "--dataset",
                str(dataset_output),
                "--output-dir",
                str(genjl_dir),
                "--n-points",
                str(args.data_n_points),
            ]
            print("→ HMC genjl")
            _run_main_subcommand(jax_env_name, *genjl_cmd)
            _print_hmc_timings("genjl", genjl_dir, args.hmc_chain_lengths)
            ran_hmc = True

    if not ran_is and _has_is_results(data_root, is_frameworks):
        ran_is = True
    if not ran_hmc and _has_hmc_results(hmc_output_dir, norm_hmc):
        ran_hmc = True

    combine_env = "perfbench-cuda" if mode == "cuda" else "perfbench"

    if not args.skip_plots and (ran_is or ran_hmc):
        if ran_is:
            print("→ Combining IS results")
            _run_main_subcommand(
                combine_env,
                "combine",
                "--data-dir",
                str(data_root),
                "--output-dir",
                str(figs_root),
            )
        if ran_hmc:
            plot_args = [
                "benchmarks/combine_results.py",
                "--data-dir",
                str(hmc_output_dir),
                "--output-dir",
                str(figs_root),
                "--frameworks",
                *norm_hmc,
            ]
            print("→ Combining HMC results")
            _run_example_script(combine_env, *plot_args)

    if not args.skip_export and (ran_is or ran_hmc):
        print("→ Exporting figures")
        _run_main_subcommand(
            combine_env,
            "export",
            "--source-dir",
            str(figs_root),
            "--dest-dir",
            str(export_dest),
            "--prefix",
            export_prefix,
        )


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
    run.add_argument("--inner-repeats", type=int, default=10)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", choices=["cpu", "cuda"], default=None)
    run.add_argument(
        "--particles",
        type=int,
        nargs="+",
        help="Importance sampling particle counts forwarded as --n-particles.",
    )
    run.add_argument(
        "framework_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the framework script (prefix with --).",
    )
    run.set_defaults(func=command_run)

    pipeline = sub.add_parser("pipeline", help="Run the full perfbench case study")
    pipeline.add_argument("--mode", choices=["cpu", "cuda"], default="cpu")
    pipeline.add_argument("--inference", choices=["all", "is", "hmc"], default="all",
                          help="Select which inference stages to run (default: all).")
    pipeline.add_argument("--particles", type=int, nargs="+", help="Particle counts for IS sweeps.")
    pipeline.add_argument("--is-frameworks", nargs="+", help="Frameworks to include in the IS sweep (overrides --frameworks).")
    pipeline.add_argument("--is-repeats", type=int, default=DEFAULT_IS_REPEATS, help="Timing repeats for IS.")
    pipeline.add_argument("--is-inner-repeats", type=int, default=DEFAULT_IS_INNER_REPEATS, help="Inner timing repeats for IS.")
    pipeline.add_argument("--hmc-frameworks", nargs="+", help="Frameworks to include in the HMC sweep (overrides --frameworks).")
    pipeline.add_argument("--frameworks", nargs="+", help="Convenience list applied to --is-frameworks/--hmc-frameworks when those flags are omitted.")
    pipeline.add_argument("--hmc-chain-lengths", type=int, nargs="+", default=[100, 500, 1000])
    pipeline.add_argument("--hmc-repeats", type=int, default=100)
    pipeline.add_argument("--hmc-warmup", type=int, default=50)
    pipeline.add_argument("--hmc-step-size", type=float, default=0.01)
    pipeline.add_argument("--hmc-n-leapfrog", type=int, default=20)
    pipeline.add_argument("--data-output", type=Path, help="Override dataset output path.")
    pipeline.add_argument("--data-n-points", type=int, default=50)
    pipeline.add_argument("--data-seed", type=int, default=42)
    pipeline.add_argument("--fig-prefix", help="Prefix for exported figures.")
    pipeline.add_argument(
        "--export-dest",
        type=Path,
        default=Path("../../figs"),
        help="Destination directory for exported figures.",
    )
    pipeline.add_argument("--skip-generate", action="store_true", help="Reuse existing dataset.")
    pipeline.add_argument("--skip-is", action="store_true", help="Skip IS sweep.")
    pipeline.add_argument("--skip-hmc", action="store_true", help="Skip HMC sweep.")
    pipeline.add_argument("--skip-plots", action="store_true", help="Skip plotting steps.")
    pipeline.add_argument("--skip-export", action="store_true", help="Skip exporting figures.")
    pipeline.set_defaults(func=command_pipeline)

    comb = sub.add_parser("combine", help="Combine timing JSON into plots/tables")
    comb.add_argument("--data-dir", type=Path, default=Path("data"))
    comb.add_argument("--output-dir", type=Path, default=Path("figs"))
    comb.add_argument("--frameworks", nargs="*")
    comb.set_defaults(func=command_combine)

    clean = sub.add_parser("clean", help="Remove generated data/figures")
    clean.set_defaults(func=command_clean)

    genjl_hmc = sub.add_parser("genjl-hmc", help="Run Gen.jl HMC benchmarks")
    genjl_hmc.add_argument("--chain-lengths", nargs="+", type=int, default=[100, 500, 1000])
    genjl_hmc.add_argument("--n-warmup", type=int, default=50)
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
