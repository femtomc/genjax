# Cone Case Study Guide

Cone reproduces the PLDI'24 cone objectives with current GenJAX APIs.

## Purpose

Compare objective families:
- naive ELBO
- naive IWAE (`K` particles)
- expressive auxiliary-variable objectives (HVI/IWHVI/DIWHVI-style settings)

## Key Files

- `core.py`: model/guide definitions, objective factories, optimization loops
- `figs.py`: posterior/prior figure generation
- `main.py`: CLI (`table4`, `fig2`)

## CLI Commands

```bash
pixi run python -m examples.cone.main table4
pixi run python -m examples.cone.main fig2 --output-dir figs
```

Useful knobs:
- `--n-steps`
- `--batch-size`
- `--learning-rate`
- `--eval-samples` (table mode)

## Idioms in this Case

- Objectives are built with `@expectation` and optimized by stochastic gradient ascent.
- Sampling-heavy batched objective evaluation uses seeded/vectorized call paths.
- Keep particle counts and objective structure static during traced execution.

## When Modifying

1. Keep objective factories pure/composable (`make_*_objective`).
2. Keep CLI in `main.py` thin; move logic into `core.py` / `figs.py`.
3. Ensure naming and output artifacts remain stable for paper workflows.

## Tests

- Primary: `tests/test_cone_example.py`
