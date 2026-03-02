# Extras Module Guide

`genjax.extras` provides exact state-space baselines used to validate approximate inference code.

## What Lives Here

Main file: `state_space.py`

- Discrete HMM utilities:
  - `discrete_hmm`
  - `forward_filter`, `backward_sample`, `forward_filtering_backward_sampling`
  - `sample_hmm_dataset`
- Linear Gaussian state-space utilities:
  - `linear_gaussian`
  - `kalman_filter`, `kalman_smoother`
  - `sample_linear_gaussian_dataset`
- Testing/inference helpers:
  - `discrete_hmm_test_dataset`, `discrete_hmm_exact_log_marginal`
  - `linear_gaussian_test_dataset`, `linear_gaussian_exact_log_marginal`
  - `discrete_hmm_inference_problem`, `linear_gaussian_inference_problem`

## Canonical Workflow

1. Generate a dataset (`*_test_dataset` or `*_inference_problem`).
2. Compute exact log marginal baseline (`*_exact_log_marginal`).
3. Compare approximate inference outputs (SMC/MCMC/VI) against this baseline in tests.

Most dataset helpers return:
- `"z"`: latent states
- `"obs"`: observations

## Idioms

- Keep sequence lengths/static dimensions explicit (often via `Const[int]`).
- Use seeded call sites before staging/vectorization (`seed(fn)(key, ...)`).
- Prefer these exact utilities in regression tests instead of ad-hoc hand-derived checks.

## Common Mistakes

- Confusing `linear_gaussian` with names from older docs (`linear_gaussian_ssm`).
- Ignoring matrix shape compatibility in LGSSM parameters.
- Using long trajectories in unit tests (keep tests fast; push heavy sweeps to scripts).

## Tests to Consult

- `tests/test_discrete_hmm.py`
- `tests/test_linear_gaussian.py`
- `tests/test_smc.py` (integration with approximate inference)
