# Performance Benchmark Case Study (POPL Figure 16b)

This directory packages the “timing-benchmarks” project (commit `d4433b0`) that produced the multi-framework performance survey in the paper. The legacy files now live under `benchmarks/` so the case-study root stays tidy; the new `main.py` provides a lightweight wrapper for the most common steps.

## Layout

```
examples/perfbench/
├── AGENTS.md                  # case-study guide
├── README.md                  # this file
├── main.py                    # light CLI that wraps the preserved scripts
├── pixi.lock / pyproject.toml # local Pixi project (Linux + macOS CPU support)
└── benchmarks/                # original timing-benchmarks repo (curated)
    ├── src/timing_benchmarks/ # Python package with frameworks + helpers
    ├── julia/                 # Gen.jl implementations
    ├── run_hmc_benchmarks.py  # HMC driver for GenJAX/NumPyro/handcoded JAX
    ├── run_genjl_hmc.py       # HMC driver for Gen.jl
    ├── combine_results.py     # plotting/table script for curvefit
    └── README_LEGACY.md       # untouched legacy instructions
```

The `benchmarks/` tree now keeps only the code paths exercised by this case study (HMC driver + plot combiners + the `timing_benchmarks` package). The pile of ad-hoc `run_*.py` / `.sh` helpers from the original repository have been dropped—use the Pixi tasks or `python main.py …` for every workflow, and consult `benchmarks/README_LEGACY.md` when you need the original level of detail.

## Prerequisites & Platform Note

The Pixi manifest includes `linux-64` (with CUDA) and `osx-arm64` (CPU). On Linux, install CUDA if you plan to run the Pyro/TensorFlow Probability baselines; on macOS you can still run the CPU-sized variant, but GPU-only tasks will fail to solve.

CUDA is required for the Pyro, hand-coded PyTorch, and TensorFlow Probability benchmarks in the paper. You can still run the commands with `--device cpu` when needed, but expect very long runtimes and different figures.

## Common Workflow

From `examples/perfbench/`:

```bash
pixi install                      # once per machine

# Full Figure 16b sweep (all frameworks, GPU-capable envs where needed). Importance sampling uses 1k/5k/10k particles; HMC uses 100/500/1k chain lengths.
pixi run curvefit-all

# Or call the wrapped runners directly
pixi run python main.py generate-data
pixi run python main.py run --framework genjax --output data/curvefit/genjax
pixi run python main.py run --framework numpyro --output data/curvefit/numpyro
pixi run python main.py run --framework genjl  --output data/curvefit/genjl    # Julia deps installed automatically
pixi run -e cuda python main.py run --framework handcoded-jax  --output data/curvefit/handcoded_jax
pixi run -e pyro python main.py run --framework pyro  --output data/curvefit/pyro --device cuda
pixi run -e pyro python main.py run --framework torch --output data/curvefit/handcoded_torch --device cuda
pixi run python main.py combine --data-dir data --output-dir figs
pixi run python main.py export --source-dir figs --dest-dir ../../figs --prefix perfbench
```

The `combine` step writes `figs/benchmark_timings_is_all_frameworks.{pdf,png}` and `figs/benchmark_timings_hmc_all_frameworks.{pdf,png}` (plus `benchmark_summary_*.csv` and `benchmark_table.tex` left in-place). Run the `export` step (or `pixi run curvefit-export`) to copy only the PDFs into the repository-level `figs/` directory with names like `perfbench_benchmark_timings_is_all_frameworks.pdf`. Raw JSON timings stay under `data/curvefit/<framework>/`.

### HMC Panel

The lower panel of Figure 16b uses the legacy runners directly (HMC requires per-framework options). These commands benchmark chain lengths 100/500/1 000. Run from the case-study root:

```bash
pixi run python benchmarks/run_hmc_benchmarks.py --frameworks genjax numpyro handcoded_jax --device cuda
pixi run -e pyro python benchmarks/run_hmc_benchmarks.py --frameworks pyro handcoded_torch --device cuda
pixi run python main.py genjl-hmc
pixi run python benchmarks/combine_results.py --frameworks genjax numpyro handcoded_jax handcoded_torch genjl --data-dir data --output-dir figs
```

### CPU-only workflow

Run every framework on CPU with tiny particle grids/repeats (1k particles) plus a reduced HMC sweep for the JAX-based frameworks (chain lengths 100/500/1 000):

```bash
pixi run curvefit-cpu
pixi run curvefit-export-cpu
```

This command (and the matching root-level `pixi run paper-perfbench-cpu`) generates a 20-point dataset, runs GenJAX, NumPyro, TensorFlow Probability, Pyro, hand-coded PyTorch, and Gen.jl with 1 000 particles / 5 repeats each, triggers a CPU HMC job for GenJAX/NumPyro/hand-coded JAX (chain lengths 100/500/1 000) plus Gen.jl, and writes results to `data_cpu/curvefit/<framework>/`. Plots live under `figs_cpu/`, and `curvefit-export-cpu` copies the PDFs (IS/HMC) to `../../figs/perfbench_cpu_*.pdf`. Use this when you only need the CPU repro path. (The first Gen.jl invocation automatically runs `julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`; rerun it manually if you want to pre-seed the environment.)

## Legacy Scripts & Tests

Only the pieces that still feed this case study (the `timing_benchmarks` package, `run_hmc_benchmarks.py`, and the `combine_*results.py` plotters) are kept in-tree. Historical notes remain in `benchmarks/README_LEGACY.md` for reference, but the ad-hoc shell/Python wrappers they described have been removed—drive everything via the Pixi tasks or `python main.py …`.
