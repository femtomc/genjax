# Examples Directory Guide

`examples/` contains case studies used for artifact reproduction and API demonstration.
Each case has its own `AGENTS.md`—read it before editing that case.

## Case Study Map

- `air/`: AIR estimator comparison (GenJAX-only port from PLDI'24 artifact)
- `cone/`: ADEV objective variants (ELBO / IWAE / HVI-style families)
- `curvefit/`: polynomial regression + scalable inference + outlier modeling
- `faircoin/`: Beta–Bernoulli baseline benchmarking
- `gol/`: inverse Game of Life via Gibbs updates
- `localization/`: particle-filter localization with optional rejuvenation
- `perfbench/`: multi-framework timing pipeline (Figure 16b)

## Shared Conventions

- Keep CLI orchestration in `main.py`; keep model/inference logic in `core.py`.
- Keep plotting logic in `figs.py` and prefer `genjax.viz.standard` helpers.
- Use seeded probabilistic call sites for staged/vectorized code.
- Keep static counts/configuration explicit (`Const[...]` where relevant).

## External-Agent Workflow

1. Read `examples/<case>/AGENTS.md`.
2. Inspect `<case>/main.py` parser to confirm current CLI flags.
3. Reuse existing helper functions; avoid duplicating logic in CLI branches.
4. Add or update matching tests in `tests/`.

## Validation

Run case-specific commands in the appropriate Pixi environment (see each case guide), then run focused tests under `tests/`.
