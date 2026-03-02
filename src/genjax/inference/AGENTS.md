# Inference Module Guide

`genjax.inference` provides reusable programmable inference components over GFI traces.

## Module Map

- `mcmc.py`
  - kernels: `mh`, `mala`, `hmc`
  - driver: `chain`
  - diagnostics container: `MCMCResult`
- `smc.py`
  - particle lifecycle: `init`, `change`, `extend`, `rejuvenate`, `resample`
  - higher-level loop: `rejuvenation_smc`
  - diagnostics container: `ParticleCollection`
- `vi.py`
  - objective builder: `elbo_factory`
  - families: `mean_field_normal_family`, `full_covariance_normal_family`
  - pipeline: `elbo_vi`
  - result container: `VariationalApproximation`

## Core Idioms

### MCMC

```python
from genjax import sel
from genjax.inference import mh, chain

def kernel(trace):
    return mh(trace, sel("theta"))

runner = chain(kernel)
result = seed(runner)(key, initial_trace, n_steps=Const(1000), n_chains=Const(4))
```

- Pass **static selections** (`sel(...)`) for targeted updates.
- Use `chain` for diagnostics and multi-chain orchestration.

### SMC / importance-style workflows

Low-level particle lifecycle:

```python
particles = init(model, target_args, Const(256), first_obs)
particles = extend(particles, model, particle_args, constraints=next_obs)
particles = resample(particles)
particles = rejuvenate(particles, mcmc_kernel)
```

High-level convenience loop:

```python
particles = rejuvenation_smc(
    model,
    transition_proposal=None,
    mcmc_kernel=Const(kernel),
    observations=obs_seq,
    initial_model_args=init_args,
    n_particles=Const(256),
)
```

- `ParticleCollection` exposes ESS and log marginal likelihood utilities.
- `resample(...)` resets particle ancestry/weights as needed.

### Variational inference

```python
elbo = elbo_factory(target_gf, variational_family, constraint, target_args)
result = seed(lambda: elbo_vi(
    target_gf,
    variational_family,
    init_params,
    constraint,
    target_args=target_args,
))(key)
```

- `elbo_vi` wraps ELBO construction + optimization.
- Families in `vi.py` are reference implementations; extend carefully.

## JAX / PJAX Rules

- Seed probabilistic callables before `jit`, `vmap`, `scan`.
- Use `modular_vmap` for vectorized probabilistic code paths.
- Keep static counts/config as `Const[...]` where required.

## Common Failure Modes

- Selections changing shape/structure across traced calls.
- Mixing keyed and keyless APIs inside library internals.
- Using plain `jax.vmap` where probabilistic primitives are present.
- Ignoring incremental weights from `update`/`regenerate` when composing kernels.

## Testing Expectations

- `tests/test_mcmc.py`: acceptance behavior + convergence diagnostics
- `tests/test_smc.py`: ESS / log-marginal sanity + particle evolution
- `tests/test_vi.py`: ELBO/parameter progression + API integrity

For new algorithmic helpers:
1. add minimal unit coverage,
2. add at least one end-to-end regression test,
3. document public additions here.
