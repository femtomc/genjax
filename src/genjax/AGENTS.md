# GenJAX Core Runtime Guide

This guide covers the core runtime in `src/genjax/`:

- `core.py`: generative functions, traces, selections, combinators
- `distributions.py`: built-in distributions
- `pjax.py`: probabilistic JAX transforms (`seed`, `modular_vmap`)
- `state.py`: state-tagging interpreter (`@state`, `save`, `namespace`)

For inference algorithms, see `inference/AGENTS.md`.
For ADEV estimators, see `adev/AGENTS.md`.

## Core Mental Model

A GenJAX program is a **generative function** (GF) with the GFI contract:

- `simulate(*args, **kwargs) -> Trace`
- `assess(choices, *args, **kwargs) -> (log_density, retval)`
- `generate(constraints_or_none, *args, **kwargs) -> (Trace, weight)`
- `update(trace, constraints_or_none, *new_args, **new_kwargs) -> (Trace, weight, discard)`
- `regenerate(trace, selection, *args, **kwargs) -> (Trace, weight, discard)`

A `Trace` carries:
- random choices,
- args,
- return value,
- score (`-log p(choices | args)`).

## Canonical Idioms

### 1) Modeling with `@gen`

```python
@gen
def model(x):
    z = normal(0.0, 1.0) @ "z"
    y = normal(z + x, 0.1) @ "y"
    return y
```

- Always address stochastic sites with `@ "address"`.
- Keep addresses stable across executions.

### 2) Constraints and importance weights

```python
trace, w = model.generate({"y": observed_y}, x)
```

- `generate` is the basic entry point for likelihood weighting / SMC-style initialization.

### 3) Randomness + JAX transforms

Library functions should remain keyless; seed at the call site:

```python
seeded_sim = seed(model.simulate)
trace = seeded_sim(key, x)
```

Apply `seed(...)` before `jit`, `scan`, `vmap`, or other staged/vectorized use.

### 4) Probabilistic vectorization

Prefer `modular_vmap` when random sampling is in the mapped function:

```python
batched = modular_vmap(fn, in_axes=(0, None), axis_size=N)
out = batched(xs, shared_arg)
```

### 5) Static config via `Const`

Use `Const[...]` for static values (loop lengths, fixed configuration) that must not be traced dynamically.

## Selections and Addressing

- Build selections with `sel(...)`.
- Compose selections with `|`, `&`, and `~`.
- Use selections in `regenerate` and inference kernels (`mh`, `mala`, `hmc`) to target subsets of choices.

Keep address hierarchies predictable so selections remain robust.

## State Interpreter Idiom

Use `@state` + `save(...)` for diagnostics without breaking JAX transforms:

```python
@state
def kernel_step(trace):
    new_trace = mh(trace, sel("theta"))
    save(theta=new_trace.get_choices()["theta"])
    return new_trace
```

Use `namespace(fn, "name")` to organize saved values hierarchically.

## Combinators

- `Scan(callee, length=Const[int])`: sequential probabilistic loops
- `Vmap` via `.vmap(...)`: structure-preserving vectorization of GFs
- `Cond(branch_true, branch_false)`: probabilistic branching compatible with shared addresses

Prefer these over Python control flow inside traced probabilistic code.

## Common Pitfalls

- Missing `@ "addr"` on stochastic sites (choice not tracked).
- Python `if/for/while` in traced probabilistic paths.
- Calling raw `jax.vmap` around probabilistic code when `modular_vmap` is needed.
- Forgetting to seed before `jit`/`scan`/`vmap` usage.
- Dynamic address construction (hard to select/debug).

## When Editing Core Runtime

1. Preserve GFI semantics first; optimize second.
2. Keep pytrees and shape behavior transform-safe.
3. Add tests in matching files (`tests/test_core.py`, `tests/test_pjax.py`, etc.).
4. Validate edge cases (empty selections, full selections, constrained/unconstrained paths).
