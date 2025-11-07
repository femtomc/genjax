# Curve Fitting Case Study Guide

This case study compares GenJAX importance sampling and HMC on a polynomial regression model, with optional NumPyro and Pyro baselines for parity checks.

## Key Files
- `core.py`: GenJAX models (`polynomial`, `npoint_curve`, outlier extensions) and inference helpers
- `data.py`: reproducible polynomial datasets shared across frameworks
- `figs.py`: figure builders that rely on `genjax.viz.standard`
- `main.py`: CLI orchestrating quick demos, full figure suites, scaling runs, and outlier experiments

## Typical Commands
```bash
pixi run python -m examples.curvefit.main --quick          # lightweight smoke test
pixi run python -m examples.curvefit.main --full           # full paper figures
pixi run python -m examples.curvefit.main --scaling        # scaling diagnostics
pixi run python -m examples.curvefit.main --outlier        # mixture-model workflow
pixi run -e curvefit-cuda python -m examples.curvefit.main --scaling  # GPU timing
```

Figures are written to the repository-level `figs/` directory. Create it (`mkdir -p figs`) before running the CLI or pass `--output-dir`.

### Resource-Constrained Benchmarks
- `--scaling-max-samples <N>` limits the largest particle count used in the GPU scaling figure (default grid tops out at 1M particles).
- `--scaling-particle-counts <comma separated list>` replaces the grid entirely (e.g., `--scaling-particle-counts 10,100,1000`).
- `--scaling-trials` and `--scaling-max-large-trials` control how many repetitions are run per particle count.

To reproduce a minimal GPU-friendly variant of the paper figures:
```bash
pixi run -e curvefit python -m examples.curvefit.main paper \
  --scaling-max-samples 20000 --scaling-trials 2
```
When using Pixi tasks, pass custom flags after `--`, e.g.
`pixi run -e curvefit -- curvefit --scaling-max-samples 20000 --scaling-trials 2`.

## Modeling Notes
- All static sizes (sample counts, chain lengths) should use the `Const[...]` wrapper.
- When building mixture models, define branch functions at module scope and reuse the `Cond` combinator pattern shown in `core.py`.
- Keep inference utilities seedable: `seeded = genjax.pjax.seed(fn)` and call `seeded(key, ...)` before `jax.jit` or batching.

## Visualization Notes
- Import fonts, colours, and layout helpers from `genjax.viz.standard` rather than setting `matplotlib` globals.
- Reuse the figure helpers in `figs.py` when adding or modifying plots so that timing, ESS, and posterior visualisations stay stylistically aligned.

## Extension Guidelines
- Add new CLI modes by updating `main.py` and wiring command-line flags to the relevant helpers in `core.py` or `figs.py`.
- Place shared utilities in `core.py`/`data.py`; avoid duplicating inference code inside `main.py`.
- If integrating new baselines, mirror the structure used for NumPyro/Pyro to keep comparisons symmetric.
