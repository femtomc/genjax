# Perfbench Case Study Guide

Perfbench wraps the multi-framework timing benchmark used for paper Figure 16(b).

## Purpose

Benchmark GenJAX against NumPyro, Pyro, hand-coded JAX/PyTorch, and optional Gen.jl on a shared curvefit task.

## Key Files

- `main.py`: orchestration CLI (`pipeline`, `run`, `combine`, `export`, etc.)
- `benchmarks/`: framework runners + result combiner scripts
- `README.md`: broader user-facing benchmark notes

## Main Commands

```bash
# Full CPU pipeline
pixi run paper-perfbench

# Full CUDA pipeline
pixi run paper-perfbench-cuda

# Manual entrypoint
pixi run -e perfbench python examples/perfbench/main.py pipeline --mode cpu
```

Useful pipeline controls:
- `--inference {all,is,hmc}`
- `--frameworks ...` (or `--is-frameworks` / `--hmc-frameworks`)
- `--particles ...`
- `--skip-generate`, `--skip-is`, `--skip-hmc`, `--skip-plots`, `--skip-export`

## Environment Routing

`main.py pipeline` dispatches each framework into the appropriate Pixi environment (`perfbench`, `perfbench-cuda`, `perfbench-pyro`, `perfbench-torch`).

## Data / Output Locations

- Data: `examples/perfbench/data*/`
- Figures/tables: `examples/perfbench/figs*/`
- Exported PDFs copied to repo-level `figs/` unless `--skip-export`

## Practical Notes

- Gen.jl benchmarks require Julia and first-run package instantiation.
- Some frameworks are intentionally runtime-clamped in helper scripts to keep end-to-end runtime manageable.
- Keep command behavior aligned with top-level Pixi tasks when editing CLI flags.
