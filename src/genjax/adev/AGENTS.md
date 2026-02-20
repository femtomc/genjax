# ADEV Module Guide

`genjax.adev` implements Automatic Differentiation of Expected Values (ADEV), providing unbiased gradient estimators for probabilistic programs.

## Layout
- `__init__.py`: public API (decorators, primitives, helper dataclasses)
- `REFERENCES.md`: bibliography for the underlying theory

## Core APIs
- `@expectation`: wraps a probabilistic program; exposes `.estimate(...)`, `.grad_estimate(...)`, and `.jvp_estimate(...)`.
- `Dual`: tree container for primal/tangent pairs. Use `Dual.tree_pure` and `Dual.dual_tree` to build inputs for low-level estimators.
- Primitive families (imported from `__init__.py`):
  - Reparameterised gradients: `normal_reparam`, `beta_reparam`, etc.
  - Score-function gradients: `flip_reinforce`, `categorical_reinforce`, etc.
  - Enumeration / measure-valued estimators: `flip_enum`, `flip_mvd`, …
- `ADEVPrimitive`: base class for custom estimators (override `sample`, `sample_with_key`, and `prim_jvp_estimate`).

## Usage Pattern
```python
from genjax.adev import expectation, normal_reparam

@expectation
def objective(theta):
    x = normal_reparam(theta, 1.0)
    return x**2

value = objective.estimate(theta)
grad = objective.grad_estimate(theta)
```

- Choose estimator variants per distribution according to variance/availability (e.g., reparameterisation for continuous, score-function for discrete).
- Wrap ADEV programs with `genjax.pjax.seed` before applying `jax.jit`.
- When vectorization crosses probabilistic sampling sites, prefer `genjax.modular_vmap` over plain `jax.vmap`.

## Implementation Notes
- All primitives are pytrees and support JAX transformations; keep static metadata (e.g., estimator selection) outside traced values.
- When subclassing `ADEVPrimitive`, ensure `prim_jvp_estimate` returns a pair `(dual_value, trace)` compatible with downstream continuations.
- Use helper functions in `genjax.adev.continuations` (imported via `__all__`) rather than reimplementing CPS plumbing.

## Testing Checklist
- Compare estimator outputs against analytic gradients or finite-difference approximations (see `tests/test_adev.py`).
- Validate unbiasedness empirically by averaging multiple gradient estimates and checking against known targets.
- Keep variance-mitigation techniques (control variates, baselines) encapsulated so tests exercise both raw estimators and tuned variants.
