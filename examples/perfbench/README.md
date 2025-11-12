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

The Pixi manifest exposes `default`, `cuda`, `pyro`, and `torch` environments. CUDA support (Linux + NVIDIA) is required if you want Pyro, hand-coded PyTorch, or TensorFlow Probability to run on GPUs; macOS installs stay CPU-only. Regardless of device, plan for roughly 5–10 minutes to complete the full sweep (IS + HMC) unless you trim the grids or repeats.

Install the local environments once:

```bash
cd examples/perfbench
pixi install
```

## Quick Start

- **Repo root**: `pixi run paper-perfbench [flags]` (CPU by default) or `pixi run paper-perfbench --mode cuda`.
- **Case-study root**: `pixi run perfbench` ≡ `python main.py pipeline --mode cuda`, `pixi run perfbench-cpu` ≡ `python main.py pipeline --mode cpu`.
- **Cleanup**: `pixi run paper-perfbench-clean` (root) or `pixi run clean` (case-study) removes `data*/` and `figs*/`.

The pipeline regenerates the shared polynomial dataset (unless `--skip-generate`), runs IS and/or HMC for the requested frameworks, then combines whatever results exist into `figs_{,cpu}/benchmark_timings_{is,hmc}_all_frameworks.{pdf,png}` and `benchmark_table.tex`. PDFs are copied to `../../figs/` unless `--skip-export` is set.

## Pipeline CLI

`python main.py pipeline [options]` (the root-level `pixi run paper-perfbench …` forwards its flags here).

Core options:

| Flag | Meaning |
| --- | --- |
| `--mode {cpu,cuda}` | Chooses data/fig roots (`data_cpu`/`figs_cpu` vs `data`/`figs`) and whether JAX stages set `JAX_PLATFORMS=cuda`. Pyro/hand-coded Torch receive `--device` accordingly (and fall back to CPU if CUDA is unavailable). |
| `--inference {all,is,hmc}` + `--skip-*` | Decide which stages run. Plotting/export only trigger when their inputs exist. |
| `--frameworks …` | Convenience list that feeds both IS and HMC. Use `--is-frameworks` / `--hmc-frameworks` for independent control. Framework names match `genjax`, `numpyro`, `handcoded_jax`, `pyro`, `handcoded_torch`, `genjl`. |
| `--particles …` | Particle counts for IS (default 1 000 / 5 000 / 10 000). |
| `--is-repeats`, `--is-inner-repeats` | Forwarded to every IS timing helper (defaults: 50 / 50, but GenJAX/NumPyro/handcoded JAX automatically bump to 100 / 100 unless you override). |
| `--hmc-chain-lengths`, `--hmc-repeats`, `--hmc-warmup`, `--hmc-step-size`, `--hmc-n-leapfrog` | Shared HMC knobs (defaults: 100 / 500 / 1 000 chains, 100 repeats, 50 warmup, step size 0.01, 20 leapfrog). Gen.jl’s dedicated runner receives the same values. |
| `--fig-prefix`, `--export-dest` | Control the filenames copied into `../../figs/`. |

Example invocations:

```bash
# CPU IS-only sweep for GenJAX + NumPyro with custom repeats
python main.py pipeline \
  --mode cpu \
  --inference is \
  --frameworks genjax numpyro \
  --is-repeats 20 \
  --is-inner-repeats 20

# CUDA HMC sweep, custom frameworks and chain lengths, skip plotting/export
python main.py pipeline \
  --mode cuda \
  --frameworks genjax numpyro pyro \
  --inference hmc \
  --hmc-chain-lengths 1000 5000 \
  --hmc-repeats 25 \
  --skip-plots --skip-export
```

Implementation notes:

- IS runs execute sequentially per framework. Each call shells into the proper Pixi environment (default, `cuda`, `pyro`, or `torch`) and forwards `--repeats`/`--inner-repeats`/`--device` as needed.
- HMC timings depend on `benchmarks/run_hmc_benchmarks.py`. Pyro, hand-coded PyTorch, and Gen.jl are intentionally capped at 5 outer × 5 inner repeats inside that helper so the entire sweep stays within the ~5–10 minute budget; customize those by invoking the helper directly if you need denser statistics.
- The pipeline defers plotting until after the requested inference stages finish. When you re-run with `--skip-is`/`--skip-hmc`, it detects existing JSON and will still combine them if present.

### Other subcommands

- `python main.py generate-data --n-points 50 --seed 42 --output data_cpu/curvefit/polynomial_data.npz`
- `python main.py run --framework <name> ...` (single-framework IS/likelihood timing; usually called by the pipeline).
- `python main.py combine --data-dir … --output-dir …` consolidates IS timings; `python benchmarks/combine_results.py …` handles HMC summaries (the pipeline runs both automatically).
- `python main.py genjl-hmc …` shells into the Julia project for the Gen.jl HMC sweep.
- `python main.py export --source-dir figs_cpu --dest-dir ../../figs --prefix perfbench_cpu` copies PDFs into the repo-level `figs/`.

## Data & Figures

- CPU mode → `data_cpu/curvefit/<framework>/` JSON + `figs_cpu/…`.
- CUDA mode → `data/curvefit/<framework>/` JSON + `figs/…`.
- Summaries (`benchmark_summary_*.csv`, `benchmark_table.tex`) live next to the generated plots.
- The pipeline prints timing summaries per stage to stdout; `combine_results.py` also emits `benchmark_summary_*.csv` and warns when baseline data is missing.

Gen.jl automatically instantiates `benchmarks/julia/Project.toml` the first time you request it. Delete `julia/.julia` inside this directory if you need a clean slate.

## Legacy Scripts & Tests

Only the pieces that still feed this case study (the `timing_benchmarks` package, `run_hmc_benchmarks.py`, and the `combine_*results.py` plotters) are kept in-tree. Historical notes remain in `benchmarks/README_LEGACY.md` for reference, but the ad-hoc shell/Python wrappers they described have been removed—drive everything via the Pixi tasks or `python main.py …`.
