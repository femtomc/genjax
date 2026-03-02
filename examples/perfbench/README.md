# Performance Benchmark Case Study (POPL Figure 16b)

This case study runs the multi-framework timing benchmark used for Figure 16b.

## Scope

The pipeline compares GenJAX with:
- NumPyro
- Pyro
- hand-coded JAX
- hand-coded PyTorch
- Gen.jl (when Julia is installed)

for both importance sampling (IS) and HMC workflows.

## Layout

```text
examples/perfbench/
├── AGENTS.md
├── README.md
├── main.py
└── benchmarks/
    ├── src/timing_benchmarks/
    ├── julia/
    ├── run_hmc_benchmarks.py
    ├── run_genjl_hmc.py
    └── combine_results.py
```

## Running from Repository Root

Use the root Pixi tasks:

```bash
# CPU pipeline
pixi run paper-perfbench

# CUDA pipeline
pixi run paper-perfbench-cuda

# Clean generated data/figures for perfbench
pixi run paper-perfbench-clean
```

You can pass pipeline flags directly:

```bash
pixi run paper-perfbench --inference is --frameworks genjax numpyro
pixi run paper-perfbench-cuda --inference hmc --hmc-chain-lengths 1000 5000
```

## Direct CLI Usage

```bash
pixi run -e perfbench python examples/perfbench/main.py pipeline --mode cpu
pixi run -e perfbench-cuda python examples/perfbench/main.py pipeline --mode cuda
```

Key flags:
- `--inference {all,is,hmc}`
- `--frameworks ...` (or `--is-frameworks` / `--hmc-frameworks`)
- `--particles ...`
- `--is-repeats`, `--is-inner-repeats`
- `--hmc-chain-lengths`, `--hmc-repeats`, `--hmc-warmup`, `--hmc-step-size`, `--hmc-n-leapfrog`
- `--skip-generate`, `--skip-is`, `--skip-hmc`, `--skip-plots`, `--skip-export`

## Outputs

- CUDA mode:
  - data: `examples/perfbench/data/`
  - figures/tables: `examples/perfbench/figs/`
- CPU mode:
  - data: `examples/perfbench/data_cpu/`
  - figures/tables: `examples/perfbench/figs_cpu/`

When export is enabled, PDFs are copied to the repository `figs/` directory.

## Julia / Gen.jl

Gen.jl benchmarks require Julia (>= 1.10) on PATH.
The pipeline auto-instantiates the Julia project on first use.
