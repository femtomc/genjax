# Test Suite Guide

The test suite mirrors the `src/genjax/` package. Each `tests/test_*.py` file targets the module with the matching name (for example `tests/test_core.py` validates `src/genjax/core.py` APIs). Regression utilities such as `tests/test_vmap_generate_bug.py` and `tests/test_vmap_rejuvenation_smc.py` guard previously fixed issues.

## Running Tests
- `pixi run test` – full suite (pytest + coverage profile)
- `pixi run test -m tests/test_smc.py` – single file
- `pixi run test -m tests/test_smc.py::test_case` – individual test
- `pixi run test -m tests/test_smc.py -k keyword -vv` – focused debugging

## Writing Tests
- Use deterministic randomness: `jax.random.PRNGKey(seed)` and pass keys explicitly.
- Keep sizes small but representative (few chains, low particle counts) to minimise runtime.
- Name tests `test_<feature>_<expected_behaviour>` and group related ones with module-level comments instead of nested classes.
- Reuse shared helpers or fixtures by adding them to `conftest.py` / `conftest_jit.py`.

## Algorithm-Specific Expectations
- **SMC** (`tests/test_smc.py`): include effective-sample-size checks and log marginal likelihood comparisons across multiple particle counts.
- **MCMC** (`tests/test_mcmc.py`): verify acceptance ratios, convergence trends, and trace invariants (e.g., scores after `update` / `regenerate`).
- **VI** (`tests/test_vi.py`): track ELBO progression or posterior moments against analytic baselines where available.
- **ADEV / gradient estimators**: compare analytical gradients or finite-difference references against estimator outputs.
- **Vectorisation**: maintain dedicated regression tests (e.g., `test_vmap_generate_bug.py`) whenever adding new `vmap` pathways.

## Numerical Tolerances
Apply the smallest tolerance that still passes reliably:
- Deterministic identities: `atol <= 1e-10`
- Log-density comparisons: `atol <= 1e-6`
- Monte Carlo estimates: choose tolerances based on sample size (document rationale inside the test).

## Performance Hygiene
- Cache compiled functions with fixtures (`jit_compiler`, `jitted_distributions`, etc.) when tests exceed ~0.5 s.
- Prefer `jax.vmap` over Python loops in tests that exercise many inputs.
- For new benchmarks, add smoke tests under `tests/test_benchmarks.py` or `tests/test_simple_benchmark.py` but keep heavy workloads in notebooks or scripts outside the suite.

## Before Submitting Changes
1. Run the relevant subset of tests plus a quick `pixi run test` smoke pass.
2. Confirm new tests fail without the code change and pass afterwards.
3. Keep fixtures and helper utilities in-sync with module APIs (update imports when refactoring).
