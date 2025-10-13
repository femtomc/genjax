# Extras Module Guide

The `genjax.extras` package bundles exact state-space models that serve as baselines for probabilistic inference algorithms.

## Layout
- `state_space.py`: Discrete Hidden Markov Model (HMM) and linear Gaussian state-space utilities, plus helper functions for synthetic datasets and exact log marginal likelihoods.
- `__init__.py`: public re-exports.

## Key APIs (`state_space.py`)
- `discrete_hmm(...)` / `linear_gaussian_ssm(...)`: generative functions constructed with the standard addressing scheme (`state_0`, `obs_0`, `scan_steps/...`).
- Dataset helpers: `discrete_hmm_test_dataset()`, `linear_gaussian_test_dataset()`.
- Baselines: `discrete_hmm_exact_log_marginal(...)`, `linear_gaussian_exact_log_marginal(...)`.
- Convenience bundles: `discrete_hmm_inference_problem(...)`, `linear_gaussian_inference_problem(...)` returning `(dataset, exact_log_marginal)`.

## Usage Patterns
1. Generate a dataset with the relevant helper.
2. Call the exact log marginal function for the baseline.
3. Feed both into approximate inference tests (SMC, MCMC, VI) to validate convergence.

The dataset helpers always return dictionaries with keys `"z"` (latent states) and `"obs"` (observations).

## Implementation Notes
- All routines are JAX-compatible; wrap them with `genjax.pjax.seed` before tracing (`jit`, `vmap`, etc.).
- Static dimensions (time steps, state sizes) should be passed as Python integers or `Const[...]`.
- Ensure covariance matrices stay positive definite when customising linear Gaussian parameters.

## Testing Checklist
- Cross-check approximate methods against the exact log marginal values.
- Include short sequences (small `T`) to keep tests fast.
- Add regression tests in `tests/test_extras.py` or algorithm-specific suites when extending functionality.
