# Game of Life Case Study Guide

The GoL case study demonstrates inverse cellular-automata inference with Gibbs sampling plus timing/showcase figures.

## Key Files

- `core.py`: probabilistic GoL dynamics + Gibbs sampler utilities
- `data.py`: pattern generators / asset preprocessing helpers
- `figs.py`: showcase and timing figure generation
- `main.py`: CLI entrypoint (currently showcase-oriented)

## Current CLI Usage

```bash
pixi run -e gol python -m examples.gol.main
pixi run -e gol python -m examples.gol.main --mode showcase
pixi run -e gol-cuda python -m examples.gol.main --mode showcase
```

Primary output artifacts:
- `figs/gol_integrated_showcase_*.pdf`
- `figs/gol_gibbs_timing_bar_plot.pdf`

## Important Implementation Note

`main.py` currently calls `save_all_showcase_figures()` with internal defaults.
Some parsed CLI parameters are informational today and not fully threaded through.
If you expand CLI behavior, update both parser wiring and this AGENTS file.

## Idioms

- Keep lattice/update dimensions static in JIT-sensitive paths.
- Reuse sampler utilities from `core.py`; do not duplicate update loops in CLI code.
- Keep figure style routed through shared viz helpers where already used.

## Tests

- Regression and vectorization checks in `tests/test_vmap_generate_bug.py` and related suites.
