# Performance Benchmark Case Study Guide

This directory is the **timing-benchmarks** project from commit `d4433b0`. It reproduces the POPL Figure 16(b) survey that compares GenJAX against hand-written JAX/PyTorch code, NumPyro, Pyro, TensorFlow Probability, and Gen.jl on the shared polynomial regression workload.

## Layout

- `pyproject.toml` / `pixi.lock` – local Pixi project; tasks are scoped here (not to the repo root).  
- `benchmarks/src/timing_benchmarks/` – benchmark implementations for each framework, shared data generation utilities, and plotting helpers.  
- `benchmarks/run_hmc_benchmarks.py`, `benchmarks/run_genjl_hmc.py`, `benchmarks/combine_results.py` – the only preserved orchestration scripts (everything else now flows through `main.py`).  
- `julia/` – Gen.jl project used by the `curvefit-genjl` Pixi task via `main.py run --framework genjl` (the helper CLI runs `Pkg.instantiate()/Pkg.precompile()` the first time automatically).  

All other historical `run_*.py`/`.sh` helpers were removed; everything funnels through the Pixi tasks and `main.py`.
- Outputs are written to `data/` and `figs/` (both gitignored).
- `main.py` – small helper that shells out to the preserved scripts so you can drive the workflow without touching the legacy surface area.

## Setup

```bash
cd examples/perfbench
pixi install            # creates the local environments defined in benchmarks/pyproject.toml
```

> **Platform note:** `pyproject.toml` lists both `linux-64` and `osx-arm64`. Only the Linux environments expose CUDA-enabled dependencies (Pyro/PyTorch + TensorFlow Probability); macOS installs are limited to the CPU-sized workflow.

The local project defines four environments:

- `default` (GenJAX + NumPyro, CPU)
- `cuda` (adds TensorFlow Probability + CUDA-enabled JAX)
- `pyro` (PyTorch + Pyro, defaults to CUDA but can fall back to CPU)
- `torch` (hand-coded PyTorch HMC tasks)

Use `pixi run …` inside this directory. When a command requires another environment, prefix it with `pixi run -e <env>`, for example `pixi run -e pyro pyro-curvefit`.

## Reproducing Figure 16(b)

1. **Generate the shared dataset**
   ```bash
   pixi run generate-data
   ```
2. **Run importance-sampling sweeps per framework**
   ```bash
   pixi run curvefit-genjax
   pixi run curvefit-numpyro
   pixi run curvefit-genjl            # requires Julia ≥1.9 (installs deps automatically on first run)
   pixi run -e cuda cuda-curvefit-handcoded # TensorFlow Probability baseline (GPU-backed JAX)
   pixi run -e pyro pyro-curvefit     # Pyro baseline (default `--device cuda`; append `--device cpu` for CPU-only runs)
   pixi run -e pyro pyro-torch        # hand-coded PyTorch baseline
   ```
   > These commands produce the 1k/5k/10k particle sweeps used in the paper figures.
3. **(Optional) Hand-coded Gen.jl variants**
   ```bash
   pixi run curvefit-genjl --method is --repeats 50          # dynamic Gen.jl
   pixi run curvefit-genjl --method is --optimized --repeats 50
   ```
4. **Run the HMC benchmarks (produces the lower panel)**
   ```bash
   pixi run -e cuda hmc-genjax
   pixi run -e cuda hmc-numpyro
   pixi run -e cuda hmc-handcoded-jax     # TensorFlow Probability baseline
   # Optional extras:
   pixi run -e pyro hmc-pyro -- --device cpu
   pixi run -e pyro hmc-torch -- --device cpu
   pixi run curvefit-genjl --method hmc --repeats 20
   pixi run genjl-hmc                 # Gen.jl HMC benchmarks
   ```
5. **Combine everything into plots**
   ```bash
   pixi run curvefit-plot             # calls combine_results.py, writes figs/benchmark_timings_{is,hmc}_all_frameworks.{pdf,png}
   pixi run hmc-plot                  # optional HMC-only summary
   ```
6. **Export figures to the repo-level `figs/` directory**
   ```bash
   pixi run curvefit-export           # copies PDFs to ../../figs/perfbench_*
   ```

From the repository root you can invoke the same sequences via:

```bash
pixi run paper-perfbench          # runs curvefit-all in examples/perfbench
pixi run paper-perfbench-hmc      # runs hmc-all then hmc-plot
pixi run paper-perfbench-clean    # deletes data/ and figs/ under examples/perfbench
```

All raw timing JSON lives in `data/curvefit/<framework>/`. The final Figure 16(b) artifacts live at `figs/benchmark_timings_{is,hmc}_all_frameworks.{pdf,png}`, and `figs/benchmark_table.tex` contains the table that goes into the paper.

### CPU-only workflow

To validate the full pipeline (all frameworks, plus a reduced HMC sweep for the JAX-based baselines) without provisioning GPUs, run:

```bash
pixi run curvefit-cpu
pixi run curvefit-export-cpu
```

This reuses the wrapper CLI to: generate a 20-point dataset, run GenJAX/NumPyro/TFP/Gen.jl plus the Pyro + hand-coded PyTorch baselines on CPU with 1 000 particles and 5 repeats, execute `run_hmc_benchmarks.py` with GenJAX/NumPyro/hand-coded JAX on CPU (chain lengths 100/500/1 000), and combine the outputs into `figs_cpu/benchmark_timings_{is,hmc}_all_frameworks.pdf`. The Pyro/PyTorch steps automatically invoke the `pyro` environment with `--device cpu`, and the Gen.jl stage auto-runs `Pkg.instantiate()/Pkg.precompile()` the first time it’s used.

## Cleanup

- Remove benchmark artifacts: `pixi run clean-data`
- Remove generated figures: `pixi run clean-figs`

Both `data/` and `figs/` are ignored by git, so you can safely delete them between runs.
