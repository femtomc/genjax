# Test Suite Guide

The test tree mirrors `src/genjax/` and case-study behavior.
Use tests as executable documentation for API and idiom expectations.

## Running Tests

- Full suite: `pixi run test`
- Single file: `pixi run test -- tests/test_core.py`
- Single test: `pixi run test -- tests/test_core.py::test_fn_simulate_assess_consistency`
- Keyword filter: `pixi run test -- tests/test_smc.py -k rejuvenation -vv`

## Coverage Map (High Level)

- Core runtime: `test_core.py`, `test_pjax.py`, `test_distributions.py`
- Inference: `test_mcmc.py`, `test_smc.py`, `test_vi.py`
- ADEV: `test_adev.py`, `test_mvnormal_estimators.py`
- Baselines/extras: `test_discrete_hmm.py`, `test_linear_gaussian.py`
- Regressions: `test_vmap_generate_bug.py`, `test_vmap_rejuvenation_smc.py`
- Case-study checks: `test_cone_example.py`, benchmark smoke tests

## Testing Idioms

- Use deterministic randomness (`jax.random.key(...)`) and explicit seeding.
- Keep unit tests small (few chains/particles/steps) unless specifically testing convergence.
- Use `Const[...]` for static loop/shape controls where model APIs expect it.
- Prefer analytic checks where available (exact posteriors/log marginals).

## Numeric Tolerance Guidance

- Deterministic identities: very tight (`~1e-10` to `1e-8`).
- Log-density/probability checks: moderate (`~1e-6`).
- Monte Carlo tests: looser, justified by sample count and variance.

Document tolerance reasoning in-test when non-obvious.

## Before Submitting

1. Run the most local affected tests.
2. Run a broader smoke subset (or full suite when feasible).
3. Ensure new tests fail without your change and pass with it.
4. Update AGENTS docs if public behavior changed.
