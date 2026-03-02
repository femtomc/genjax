# Localization Case Study Guide

This case study runs particle-filter localization (with optional rejuvenation comparisons) and generates paper figures.

## Key Files

- `core.py`: world model, localization dynamics, SMC benchmark routines
- `data.py`: trajectory/observation generation
- `figs.py`: localization + comparison plots
- `main.py`: paper-mode orchestration
- `export.py`: data serialization helpers (when used by supporting workflows)

## Current CLI Usage

`main.py` currently exposes paper-oriented execution.

```bash
pixi run -e localization python -m examples.localization.main paper --include-smc-comparison
pixi run -e localization-cuda python -m examples.localization.main paper --include-smc-comparison
```

Common flags:
- `--n-rays`
- `--n-particles`
- `--n-steps`
- `--k-rejuv`
- `--timing-repeats`
- `--world-type`
- `--output-dir`

## Output Artifacts

Paper mode writes figure PDFs to `figs/` (or the provided `--output-dir`).

## Notes for External Agents

- `--include-smc-comparison` controls whether the expensive benchmark panel is generated.
- `--include-basic-demo` exists in parser for compatibility but is not central to current paper-mode flow.
- Keep static counts/configuration explicit for traced performance-sensitive code.

## When Modifying

1. Keep model/benchmark logic in `core.py`.
2. Keep plotting/layout logic in `figs.py`.
3. Keep CLI as orchestration only.
4. Update tests and this AGENTS file if workflow flags or outputs change.

## Tests

- `tests/test_vmap_rejuvenation_smc.py`
- `tests/test_smc.py` (integration behavior)
