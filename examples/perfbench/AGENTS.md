# Performance Benchmark Case Study Guide

This directory contains the `timing-benchmarks` integration used for POPL Figure 16(b):
GenJAX vs NumPyro, Pyro, hand-coded JAX/PyTorch, and Gen.jl on the shared
polynomial regression benchmark.

## Layout

- `benchmarks/src/timing_benchmarks/` – framework implementations and plotting utilities
- `benchmarks/run_hmc_benchmarks.py`, `benchmarks/run_genjl_hmc.py`, `benchmarks/combine_results.py` – orchestration helpers
- `main.py` – single CLI entry point used by root Pixi tasks
- `julia/` – Gen.jl project used for `genjl` baselines

Outputs are written under:
- `examples/perfbench/data{,_cpu}/`
- `examples/perfbench/figs{,_cpu}/`
- exported PDFs copied to repo-level `figs/` (unless `--skip-export`)

## Environments (top-level Pixi)

Perfbench now uses root-level environments (not a nested Pixi project):

- `perfbench` – CPU JAX/NumPyro/plotting stack
- `perfbench-cuda` – same + CUDA JAX
- `perfbench-pyro` – PyTorch + Pyro runners
- `perfbench-torch` – hand-coded PyTorch runners

`main.py pipeline` dispatches each framework into the right env automatically.

## Typical commands

```bash
# CPU full pipeline
pixi run paper-perfbench

# CUDA full pipeline
pixi run paper-perfbench-cuda

# CPU IS-only quick rerun
pixi run paper-perfbench --inference is --particles 1000 2000 4000
```

## Notes

- Install Julia (>=1.10) if you want Gen.jl baselines. If unavailable, skip `genjl` via `--frameworks` / `--hmc-frameworks`.
- Pyro / hand-coded torch runs are intentionally clamped in the HMC helper for manageable total runtime.
