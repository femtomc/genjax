# Inference Module Guide

`genjax.inference` hosts reusable drivers for Markov chain Monte Carlo (MCMC), sequential Monte Carlo (SMC), and variational inference (VI). All algorithms operate on GenJAX generative functions and respect the addressing interface defined in `src/genjax/core.py`.

## Module Map
- `mcmc.py`: Metropolis–Hastings (`mh`), Metropolis-adjusted Langevin (`mala`), Hamiltonian Monte Carlo (`hmc`), and the `chain` driver for running multiple chains with diagnostics.
- `smc.py`: Particle filters (`init`, `extend`, `resample`), rejuvenation helpers (`rejuvenation_smc`, `rejuvenate`), ESS utilities, and particle collection dataclasses.
- `vi.py`: ELBO optimisation utilities (`variational_inference`, `elbo_estimator`, mean-field helpers).
- `__init__.py`: curated exports for the API surface.

## Common Usage Patterns

### MCMC
```python
from genjax import sel
from genjax.inference import chain, mh

def kernel(trace):
    return mh(trace, sel("theta"))

run = chain(kernel)
result = run(key, initial_trace, n_steps=const(1000), n_chains=const(4))
```
- Pass selections to restrict which addresses are updated.
- `chain` returns diagnostics (`rhat`, `ess_bulk`, `ess_tail`, acceptance rates) aligned with the trace structure.

### SMC
```python
from genjax.inference import init, extend, rejuvenation_smc

particles = init(model, args, n_particles=const(256), constraints=data)
particles = extend(particles, next_model, next_args, constraints=next_obs)
particles = rejuvenation_smc(
    particles,
    transition_proposal=None,   # default to model proposal
    mcmc_kernel=None            # optional rejuvenation
)
```
- Particle collections expose `.log_marginal_likelihood()` and `.diagnostics` (ESS, log weights).
- All static counts use `Const[...]` to avoid recompilation across JIT invocations.

### Variational Inference
```python
from genjax.inference import variational_inference

result = variational_inference(
    target_model,
    target_args,
    guide_family,
    n_steps=const(2000),
    optimizer=optax.adam(1e-3)
)
```
- `result` provides ELBO histories and final variational parameters.
- Combine with ADEV estimators (see `genjax.adev`) for unbiased gradients.

## Implementation Notes
- All drivers assume you call `seeded = genjax.pjax.seed(fn)` and then invoke `seeded(key, ...)` before applying `jax.jit`, `jax.vmap`, or `jax.scan`.
- Selection objects (`genjax.sel`) must remain static across traced calls.
- `rejuvenation_smc` accepts optional `transition_proposal` and `mcmc_kernel`; default behaviour relies on the model’s internal proposal.
- Particle collections retain log-normalised weights for diagnostics; resampling utilities reset them as needed.

## Testing Expectations
- See `tests/test_mcmc.py`, `tests/test_smc.py`, and `tests/test_vi.py` for regression coverage.
- When adding new kernels, supply targeted tests that check convergence trends (acceptance rates, log marginal likelihood, ELBO monotonicity) using small synthetic models.
- Update documentation in `AGENTS.md` files for any new public helpers.
