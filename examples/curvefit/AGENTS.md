# Curvefit Case Study Guide

This case study drives the polynomial-regression artifact figures (including scaling and outlier analyses).

## Current CLI Shape

`examples/curvefit/main.py` currently exposes **paper mode** only.

```bash
pixi run -e curvefit python -m examples.curvefit.main paper
pixi run -e curvefit python -m examples.curvefit.main paper --scaling-max-samples 20000 --scaling-trials 2
pixi run -e curvefit-cuda python -m examples.curvefit.main paper
```

Outputs are written to the repo-level `figs/` directory.

## Key Files

- `core.py`: models (`polynomial`, point/npoint variants), inference helpers, framework baselines
- `data.py`: synthetic dataset helpers
- `figs.py`: paper figure generation and scaling visualizations
- `main.py`: CLI argument parsing + paper workflow orchestration

## Important Flags

- `--scaling-trials`
- `--scaling-max-large-trials`
- `--scaling-max-samples`
- `--scaling-particle-counts`
- `--scaling-extended-timing`

## Modeling / Inference Idioms

- Keep static counts wrapped with `Const[...]` in inference helpers.
- Use seeded call sites before vectorized/JIT-heavy runs.
- For outlier branches, follow the existing `Cond`-based pattern in `core.py`.
- Keep framework comparison paths symmetric (GenJAX vs NumPyro/Pyro/etc.).

## When Modifying

1. Add logic to `core.py` / `figs.py`, not directly in CLI branches.
2. Keep generated artifact filenames stable unless intentionally changing outputs.
3. Keep CLI behavior aligned with Pixi tasks (`paper-curvefit-gen`, GPU variants).

## Tests

- `tests/test_benchmarks.py`
- `tests/test_simple_benchmark.py`
- relevant regressions for vectorized pathways
