# Fair Coin Case Study Guide

The fair coin example benchmarks GenJAX against NumPyro and a hand-coded JAX baseline on a Beta–Bernoulli model. Outputs include timing comparisons and posterior accuracy checks.

## Key Files
- `core.py`: model definition (`beta_ber_multi`), timing harnesses, posterior sampling utilities.
- `figs.py`: figure builders (timing bars, posterior histograms, combined layouts). Uses local Matplotlib styling; migrate to `genjax.viz.standard` when refactoring.
- `main.py`: CLI entry point exposing `--timing`, `--posterior`, `--combined`, and `--all` modes.
- `README.md`: user-facing description (optional for agents).

## Typical Commands
```bash
pixi run -e faircoin python -m examples.faircoin.main --combined
pixi run -e faircoin python -m examples.faircoin.main --timing --num-samples 5000
pixi run -e faircoin-cuda python -m examples.faircoin.main --all     # GPU variant
```

Figures are saved to the repository `figs/` directory; ensure it exists or pass `--output-dir`.

## Modeling Notes
- Prior: `Beta(10, 10)` with Bernoulli observations stored under `"obs"`.
- Data generator: 80% heads, 20% tails (`num_heads = int(0.8 * num_obs)`).
- Importance sampling uses the prior as proposal; ensure particle counts and repeat counts are wrapped in `Const[...]` when wiring automated benchmarks.

## Visualization Notes
- `figs.py` currently sets Matplotlib globals directly; replicate patterns exactly to keep styling consistent until GRVS migration occurs.
- `timing_comparison_fig` expects timing summaries from `core.py.timing`.
- Posterior figures overlay analytic Beta posteriors computed via `exact_beta_posterior_stats`.

## Testing / Validation
- Smoke tests for timing harnesses live in `tests/test_simple_benchmark.py` and `tests/test_benchmarks.py`.
- When modifying the model or data generation logic, add coverage under `tests/test_faircoin.py` (create if absent) to assert posterior mean/variance against analytic values.

## Extension Guidelines
- Add new frameworks by mirroring the pattern in `core.py` (separate timing + posterior helpers) and extending CLI switches in `main.py`.
- Keep analytic references up to date in `exact_beta_posterior_stats` when altering priors or data composition.
