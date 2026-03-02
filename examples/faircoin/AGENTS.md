# Fair Coin Case Study Guide

Faircoin benchmarks GenJAX against NumPyro and hand-coded baselines on a Beta–Bernoulli model with analytic posterior checks.

## Key Files

- `core.py`: model, timing harnesses, posterior sampling helpers
- `figs.py`: timing/posterior plotting helpers
- `main.py`: CLI dispatcher (`--combined`, `--posterior`, `--all`, default timing)

## CLI Commands

```bash
pixi run -e faircoin python -m examples.faircoin.main --combined
pixi run -e faircoin python -m examples.faircoin.main --posterior --num-samples 5000
pixi run -e faircoin python -m examples.faircoin.main --all --num-obs 50 --num-samples 2000
pixi run -e faircoin-cuda python -m examples.faircoin.main --combined
```

## Important Flags

- `--num-obs`
- `--num-samples`
- `--repeats`

Figures are written under `figs/`.

## Modeling Notes

- Prior is Beta–Bernoulli; analytic posterior reference lives in `exact_beta_posterior_stats`.
- Importance-sampling paths should keep sample counts explicit/static when staged.

## Styling Note

`figs.py` currently uses case-local Matplotlib/Seaborn styling rather than full GRVS helpers.
Preserve existing output style unless you are intentionally migrating styling.

## Tests / Validation

- benchmark smoke: `tests/test_simple_benchmark.py`, `tests/test_benchmarks.py`
- add dedicated posterior-accuracy tests if model assumptions change
