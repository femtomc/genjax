# Cone Case Study Guide

This case study ports the PLDI'24 cone objectives into the current GenJAX API.

## Scope
- Naive variational family with ELBO and IWAE objectives.
- Expressive auxiliary-variable family with nested importance objectives:
  - HVI-style (`N=1, K=1`)
  - IWHVI-style (`N=5, K=1`)
  - DIWHVI-style (`N=5, K=5`)

## Key Files
- `core.py`: model/guide definitions, objective builders, optimization helpers
- `figs.py`: figure generation for prior and posterior samples
- `main.py`: CLI entry point for table-style summaries and figure export

## Typical Commands
```bash
pixi run python -m examples.cone.main table4
pixi run python -m examples.cone.main fig2 --output-dir figs
```

## Implementation Notes
- ADEV objectives are written with `@expectation` and optimized via Monte Carlo gradient ascent.
- Seed any objective evaluation that is `vmap`/`jit` transformed via `genjax.pjax.seed`.
- Keep probabilistic control flow static (fixed particle counts in objective factories).

## Testing
Use the focused test:
```bash
pixi run test -m "not benchmark" tests/test_cone_example.py
```
