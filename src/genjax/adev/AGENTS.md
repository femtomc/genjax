# ADEV Module Guide

`genjax.adev` implements **Automatic Differentiation of Expected Values (ADEV)**:
unbiased gradient estimators for stochastic programs.

## What to Use First

- `@expectation`: wraps a stochastic objective.
- `Expectation.estimate(...)`: Monte Carlo estimate of the objective.
- `Expectation.grad_estimate(...)`: unbiased gradient estimate.
- `Expectation.jvp_estimate(...)`: low-level forward-mode estimate.

## Estimator Primitives (Current Public Set)

- Reparameterized: `normal_reparam`, `uniform_reparam`, `multivariate_normal_reparam`
- Score-function / REINFORCE: `flip_reinforce`, `geometric_reinforce`, `normal_reinforce`, `uniform_reinforce`, `multivariate_normal_reinforce`
- Discrete estimators: `flip_enum`, `flip_enum_parallel`, `flip_mvd`, `categorical_enum_parallel`

Choose low-variance estimators when possible (reparameterization for continuous sites, exact/enum when tractable for discrete sites).

## Canonical Pattern

```python
from genjax.adev import expectation, normal_reparam

@expectation
def objective(theta):
    x = normal_reparam(theta, 1.0)
    return x**2

value = objective.estimate(theta)
grad = objective.grad_estimate(theta)
```

## Key Idioms

- Inside `@expectation`, use **ADEV primitives**, not plain `normal(...)` etc.
- Seed call sites before staging/vectorization:
  - `seeded = genjax.pjax.seed(fn)`
  - `seeded(key, ...)`
- If probabilistic vectorization is needed, prefer `modular_vmap`.

## Extending ADEV (Custom Primitives)

Subclass `ADEVPrimitive` and implement:

1. `sample(self, *args)`
2. `sample_with_key(self, key, *args, sample_shape=())`
3. `prim_jvp_estimate(self, dual_tree, konts)`

Guidelines:
- Keep output structures pytree-compatible.
- Make `sample_with_key` obey PJAX keyful sampler conventions.
- Ensure `prim_jvp_estimate` returns a `Dual`-compatible value/tangent pair shape.

## Common Mistakes

- Mixing plain distribution sites with ADEV expectation sites.
- Missing keyful sampler support in custom primitives.
- Using `jax.vmap` directly over probabilistic ADEV sites when `modular_vmap` is needed.
- Assuming low variance from REINFORCE in difficult objectives.

## Testing

Primary coverage lives in:
- `tests/test_adev.py`
- `tests/test_mvnormal_estimators.py`
- relevant VI tests (`tests/test_vi.py`) for end-to-end objective optimization

When adding estimators:
1. test finite outputs,
2. test gradient sanity vs analytic/finite-difference references where possible,
3. add seeded/vectorized regression tests.
