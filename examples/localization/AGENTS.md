# Localization Case Study Guide

GenJAX localization demonstrates particle filtering with optional MCMC rejuvenation for drift-only robot dynamics in a simple indoor map.

## Key Files
- `core.py`: model definition, particle filter variants, utility dataclasses
- `data.py`: reproducible trajectory generation
- `figs.py`: plotting utilities backed by `genjax.viz.standard`
- `export.py`: experiment export/import helpers
- `main.py`: CLI entry point (generate data, plot figures, full pipeline)

## Environment
Run localization tasks inside the CUDA environment so that JAX sees GPU-capable dependencies:

```bash
pixi run -e cuda python -m examples.localization.main generate-data
pixi run -e cuda python -m examples.localization.main plot-figures --experiment-name <name>
pixi run cuda-localization  # end-to-end shortcut
```

Generated artefacts live under `examples/localization/data/` and figures default to the repository `figs/` directory.

## Workflow Summary
1. `generate-data` produces ground-truth trajectories, observations, and serialized particle dumps via `export.py`.
2. `plot-figures` consumes a saved experiment and renders comparison figures (ESS diagnostics, timing, trajectory overlays).
3. `run` combines both steps for quick smoke tests.

The CLI exposes knobs for particle count, rejuvenation iterations, and which figure suites to emit; prefer adjusting these via command-line flags rather than modifying source.

## Modeling Notes
- State variables cover position and heading only. Velocity terms are intentionally absent to keep rejuvenation effective.
- Observation constraints rely on vectorised LIDAR beams; always wrap static configuration (particle counts, ray counts) with `Const[...]`.
- Rejuvenation kernels are optional: pass `None` to rely on the model’s proposal or supply an `mcmc_kernel` callable constructed in `core.py`.

## Visualization Notes
- Import typography and colour utilities from `genjax.viz.standard` (`setup_publication_fonts`, `get_method_color`, `save_publication_figure`, etc.).
- Keep plots free of ad-hoc styling; reuse the helpers in `figs.py` when adding new figures so that ESS diagnostics and timing charts stay comparable.

## Implementation Checklist
- Before `jax.jit`, `jax.vmap`, or `jax.scan`, derive a seeded callable (`seeded_fn = genjax.pjax.seed(fn)`) and invoke it as `seeded_fn(key, ...)`.
- Use `Const[int]` for static loop bounds (e.g., particle counts, rejuvenation steps).
- When extending exports, update both `save_*` and `load_*` helpers to maintain backward compatibility.
- Validate changes against `tests/test_vmap_rejuvenation_smc.py` when modifying rejuvenation logic.
