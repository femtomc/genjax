# Examples Directory Guide

The `examples/` tree houses publication-quality case studies that exercise GenJAX across application domains. Use this guide to maintain consistency when modifying or adding case studies.

## Standard Layout
```
examples/<case>/
├── AGENTS.md        # case-specific guidance
├── core.py          # models and inference routines
├── data.py          # dataset generation (optional for trivial cases)
├── figs.py          # figure builders
├── main.py          # CLI entry point
└── export.py / data/ (optional)  # result management
```

Follow the same structure for new case studies unless a user specifies otherwise.

## Development Workflow
1. Start by reading the case-specific `AGENTS.md` and associated tests under `tests/`.
2. Treat `main.py` as the orchestrator: add new CLI flags there and delegate logic to helpers in `core.py`, `data.py`, or `figs.py`.
3. Keep stochastic helpers seedable: call `seeded_fn = genjax.pjax.seed(fn)` and invoke `seeded_fn(key, ...)`; wrap static parameters with `Const[...]`.
4. Place shared utilities in `core.py` or `data.py`; avoid duplicating inference code inside scripts or notebooks.

## Visualization Standards
- Import typography, colour palettes, and layout helpers from `genjax.viz.standard`.
- Let `figs.py` own figure aesthetics; avoid ad-hoc `matplotlib` configuration in CLI code.
- Emit figures to the repository-level `figs/` directory by default and expose an `--output-dir` escape hatch.

## Exported Artefacts
- Use `export.py` when experiments produce reusable artefacts (CSV, JSON, etc.). Provide matching load helpers so that plotting routines can consume stored results.
- Store experiment outputs under `examples/<case>/data/` with timestamped directories or user-specified names.
- Document export formats in `AGENTS.md` when non-obvious.

## Testing Expectations
- Mirror new functionality with targeted tests. For example, Gibbs or SMC updates should be exercised via regression tests in `tests/test_vmap_*` or dedicated files.
- Prefer small problem sizes for unit tests while preserving the logic found in the case study.
- Validate CLI changes with smoke-test command sequences in the relevant `AGENTS.md`.

## Adding a New Case Study
1. Scaffold the directory using the structure above.
2. Populate `AGENTS.md` with a concise summary of purpose, key files, environment requirements, and workflow.
3. Register supporting tests under `tests/` to cover inference logic or data handling.
4. Integrate visualization helpers through `genjax.viz.standard`.
5. Coordinate with paper artefact scripts if results feed into documentation or figures.
