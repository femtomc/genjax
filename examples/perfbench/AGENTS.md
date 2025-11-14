# Performance Benchmark Case Study Guide

This directory is the **timing-benchmarks** project from commit `d4433b0`. It reproduces the POPL Figure 16(b) survey that compares GenJAX against hand-written JAX/PyTorch code, NumPyro, Pyro, TensorFlow Probability, and Gen.jl on the shared polynomial regression workload.

## Layout

- `pyproject.toml` / `pixi.lock` – local Pixi project; tasks are scoped here (not to the repo root).  
- `benchmarks/src/timing_benchmarks/` – benchmark implementations for each framework, shared data generation utilities, and plotting helpers.  
- `benchmarks/run_hmc_benchmarks.py`, `benchmarks/run_genjl_hmc.py`, `benchmarks/combine_results.py` – the only preserved orchestration scripts (everything else now flows through `main.py`).  
- `julia/` – Gen.jl project used whenever `main.py run --framework genjl` (or the pipeline’s Gen.jl stage) executes; the helper auto-runs `Pkg.instantiate()/Pkg.precompile()` on first use.  

All other historical `run_*.py`/`.sh` helpers were removed; everything funnels through Pixi tasks and `main.py`. Outputs are written to `data_{,cpu}/` and `figs_{,cpu}/` (both gitignored), and `main.py` hides the legacy orchestration scripts so you rarely have to call them directly.

## Setup

```bash
cd examples/perfbench
pixi install            # creates the local environments defined in benchmarks/pyproject.toml
```

> **Platform note:** Only the `linux-64` solve exposes CUDA-enabled JAX/Pyro/PyTorch. macOS installs are CPU-only but otherwise follow the same commands. In both cases the entire IS+HMC sweep typically finishes within ~5–10 minutes; reduce particle/chain grids if you need a quicker smoke test. Install Julia ≥1.10 ahead of time (we recommend [juliaup](https://github.com/JuliaLang/juliaup); `curl -fsSL https://install.julialang.org | sh` will put `julia` on your `PATH`).

The local project defines four environments:

| Environment | Purpose |
| --- | --- |
| `default` | GenJAX + NumPyro (CPU) |
| `cuda` | Adds CUDA-enabled JAX + TensorFlow Probability |
| `pyro` | PyTorch + Pyro (prefers CUDA, falls back to CPU) |
| `torch` | Hand-coded PyTorch HMC kernels |

Most commands are launched for you; when you need to run something manually, use `pixi run -e <env> …`.

## High-level workflow

Everything now flows through **one** command:

```bash
python main.py pipeline [flags]
```

- Repo-level shortcut: `pixi run paper-perfbench …`
- Case-study shortcuts: `pixi run perfbench` (CUDA) and `pixi run perfbench-cpu`
- Cleanup: `pixi run paper-perfbench-clean` (root) or `pixi run clean` (here)

What the pipeline does:

1. Generate (or reuse) the polynomial dataset.
2. Run IS and/or HMC for the requested frameworks (respecting per-stage framework lists, repeats, particle counts, etc.). Each framework is executed inside the correct Pixi environment with explicit `JAX_PLATFORMS`/`--device` settings.
3. Combine whatever results exist into `figs_{,cpu}/benchmark_timings_{is,hmc}_all_frameworks.{pdf,png}` plus `benchmark_summary_*.csv` and `benchmark_table.tex`.
4. Copy PDFs into `../../figs/` unless `--skip-export` is set.

### Key pipeline flags

- `--mode {cpu,cuda}` controls output roots (`data_cpu`/`figs_cpu` vs `data`/`figs`) **and** whether JAX frameworks set `JAX_PLATFORMS=cuda`.
- `--inference {all,is,hmc}` toggles stages; the helper also honours `--skip-generate`, `--skip-is`, `--skip-hmc`, `--skip-plots`, `--skip-export`.
- `--frameworks …` feeds both stages; use `--is-frameworks` / `--hmc-frameworks` when you need different sets.
- `--particles …`, `--is-repeats`, `--is-inner-repeats` customise the IS sweep (defaults: 1k/5k/10k, 50 repeats, 50 inner repeats; GenJAX/NumPyro/handcoded JAX auto-bump to 100×100 unless overridden).
- `--hmc-chain-lengths`, `--hmc-repeats`, `--hmc-warmup`, `--hmc-step-size`, `--hmc-n-leapfrog` feed both the shared HMC runner and the Gen.jl helper.
- Plotting/export runs only if at least one corresponding stage finished or if existing JSON is detected on disk.
- Pyro, hand-coded PyTorch, and Gen.jl HMC runs are clamped to 5 outer × 5 inner repeats inside `benchmarks/run_hmc_benchmarks.py` so the complete sweep stays within the ~5–10 minute target window; edit that script or execute it manually if you need larger grids.

Common snippets:

```bash
# CUDA, IS + HMC, restricted frameworks
python main.py pipeline --mode cuda --frameworks genjax numpyro handcoded_jax

# CPU IS-only rerun with custom particle counts
python main.py pipeline --mode cpu --inference is --particles 1000 2000 4000

# HMC-only run for Gen.jl + Pyro on CUDA, skip plots/export
python main.py pipeline \
  --mode cuda \
  --hmc-frameworks genjl pyro \
  --inference hmc \
  --skip-plots --skip-export
```

### Other subcommands

You generally shouldn’t need these outside of debugging, but they remain available:

- `python main.py generate-data …`
- `python main.py run --framework {genjax,numpyro,handcoded-jax,pyro,torch,genjl} …`
- `python main.py combine …` (IS plots/table)
- `python benchmarks/combine_results.py …` (HMC plots/table; invoked automatically by the pipeline)
- `python main.py genjl-hmc …` (Julia-only helper when you want to rerun Gen.jl without touching other frameworks)
- `python main.py export --source-dir figs_cpu --dest-dir ../../figs --prefix perfbench_cpu`

Raw timing JSON resides in `data{,_cpu}/curvefit/<framework>/`. Figures + tables live under `figs{,_cpu}/`. The pipeline prints per-framework summaries to stdout after each stage to make regression hunting easier.

Gen.jl instantiates automatically the first time you request it (`julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`). Install Julia via juliaup beforehand, and delete `julia/.julia` if you want to reset the environment.

## Cleanup

- Remove benchmark artifacts: `pixi run clean-data`
- Remove generated figures: `pixi run clean-figs`

Both `data/` and `figs/` are ignored by git, so you can safely delete them between runs.
