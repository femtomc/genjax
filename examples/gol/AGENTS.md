# Game of Life Case Study Guide

The Game of Life (GoL) example showcases GenJAX Gibbs sampling for inverse cellular automata problems, including animation and performance benchmarking.

## Key Files
- `core.py`: probabilistic GoL update rules, Gibbs sampler utilities, animation helpers
- `data.py`: pattern generators and optional image-to-grid loaders
- `figs.py`: figure builders using `genjax.viz.standard`
- `main.py`: CLI with presets for blinker demos, logo reconstructions, and timing studies
- `assets/`: optional input imagery (not tracked)

## Running the Case Study
```bash
pixi run python -m examples.gol.main --demo          # quick blinker walkthrough
pixi run python -m examples.gol.main --logo          # logo reconstruction figures
pixi run python -m examples.gol.main --timing        # scaling benchmarks
pixi run -e gol-cuda python -m examples.gol.main --timing --device gpu
```

Figures are emitted to the repository `figs/` directory; ensure it exists or pass `--output-dir`.

## Modeling Notes
- Cells are updated with a vectorised single-cell Gibbs move; `Const[...]` keeps lattice dimensions static for JIT.
- The sampler maintains both the latent board and the deterministic forward step. Use the provided helpers in `core.py` instead of reimplementing scan loops.
- When loading custom assets, normalise to binary grids before passing them into the sampler utilities.

## Visualization Notes
- `figs.py` centralises layout and colour choices; import typography and palette helpers from `genjax.viz.standard`.
- Animation and monitoring plots are separated—extend them by following the existing helper signatures rather than adding inline plotting logic.

## Extension Guidelines
- Add new experiments by introducing CLI flags in `main.py` that call composable routines from `core.py` and `figs.py`.
- Update tests in `tests/test_vmap_generate_bug.py` and related harnesses if modifying the vectorised update pathways.
