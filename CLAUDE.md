# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Policy

- **NEVER create documentation files** unless explicitly requested
- Focus on implementation tasks and working code

## Overview

GenJAX is a probabilistic programming language embedded in Python centered on programmable inference.

## Core Concepts

### Generative Functions & Traces

- **Generative Function**: Probabilistic program implementing the Generative Function Interface (GFI)
- **Trace**: Execution record containing random choices, arguments, return value, and score (`log 1/P(choices)`)

### Generative Function Interface (GFI)

**Mathematical Foundation**: Generative functions bundle:

- Measure kernel $P(dx; a \in A)$ over measurable space $X$ (the model distribution)
- Function $f(x, a \in A) \rightarrow R$ (return value function)
- Proposal family $Q(dx; a \in A, x' \in X')$ (internal proposals)

**GFI Methods**:

```python
# Core interface (defined in src/core.py)
GFI[X, R].simulate(args: tuple) -> Trace[X, R]                    # Sample execution
GFI[X, R].assess(args: tuple, x: X) -> tuple[Density, R]          # Evaluate density
GFI[X, R].generate(args: tuple, x: X_) -> tuple[Trace[X, R], Weight]
GFI[X, R].update(args: tuple, trace: Trace[X, R], x_: X_) -> tuple[Trace[X, R], Weight, X_]
GFI[X, R].regenerate(args: tuple, trace: Trace[X, R], s: Sel) -> tuple[Trace[X, R], Weight, X_]

# Trace methods
Trace[X, R].get_retval() -> R         # Return value
Trace[X, R].get_choices() -> X        # Random choices
Trace[X, R].get_score() -> Score      # log(1/P(choices))
Trace[X, R].get_args() -> tuple       # Arguments
Trace[X, R].get_gen_fn() -> GFI[X, R] # Source function
```

## Generative Function Types

### Distributions

Built-in distributions implement GFI and wrap TensorFlow Probability:

```python
from genjax import normal, beta, exponential, categorical, flip

# Usage: parameters as args, not constructor arguments
normal(mu, sigma)  # ✅ CORRECT
exponential(rate)  # ✅ CORRECT
```

### `@gen` Functions (`Fn` type)

Transform JAX-compatible Python functions into generative functions:

```python
@gen
def beta_ber():
    f = beta(10.0, 10.0) @ "fairness"  # @ operator for addressing
    return flip(f) @ "obs"

# Creates hierarchical addressing through composition
@gen
def nested_model():
    result = simple_model() @ "sub"  # choices["sub"]["inner_address"]
```

### Combinators

Higher-order generative functions that compose other generative functions:

**Scan** - Sequential iteration (like `jax.lax.scan`):

```python
@gen
def step_fn(carry, _):
    x = normal(carry, 1.0) @ "x"  # ✅ Static addressing only
    return x, x

scan_gf = Scan(step_fn, length=10)
result = scan_gf((init_carry, None)) @ "scan"
```

**Vmap** - Vectorization:

```python
# Vectorize over parameters
vectorized_normal = normal.vmap(in_axes=(0, None))
traces = vectorized_normal.simulate((mus_array, sigma_scalar))

# Independent sampling
batch_sampler = single_sample.repeat(n=10)  # axis_size=10
```

**Cond** - Conditional branching:

```python
@gen
def branch_a(): return exponential(1.0) @ "value"
@gen
def branch_b(): return exponential(2.0) @ "value"

cond_gf = Cond(branch_a, branch_b)
result = cond_gf((condition,)) @ "conditional"
```

## Critical API Patterns

**Generative Function Usage**:

```python
# ✅ CORRECT patterns
x = normal(mu, sigma) @ "x"                    # In @gen functions
log_density, retval = normal.assess((mu, sigma), sample)  # GFI calls with tuple args

# ❌ WRONG patterns
x = normal(mu, sigma)                          # Not traced
x = normal(mu=mu, sigma=sigma) @ "x"          # No kwargs
normal(mu, sigma).assess((), sample)          # Wrong arg structure
```

## JAX Integration & Constraints

### CRITICAL JAX Python Restrictions

**NEVER use Python control flow in `@gen` functions**:

```python
# ❌ WRONG - These break JAX compilation
@gen
def bad_model():
    if condition:        # Python if
        x = normal(0, 1) @ "x"
    for i in range(n):   # Python for loop
        y = normal(0, 1) @ f"y_{i}"  # Dynamic addressing

# ✅ CORRECT - Use JAX-compatible patterns
@gen
def good_model():
    # Use Cond combinator for conditionals
    cond_gf = Cond(branch_a, branch_b)
    x = cond_gf((condition,)) @ "x"

    # Use Scan combinator for iteration
    scan_gf = Scan(step_fn, length=n)
    results = scan_gf((init, None)) @ "scan"
```

### PJAX: Probabilistic JAX

PJAX extends JAX with probabilistic primitives (`assume_p`, `log_density_p`).

**Key Transformations**:

- **`seed`**: Eliminates PJAX primitives → enables standard JAX transformations → requires explicit keys
- **`modular_vmap`**: Preserves PJAX primitives → specialized vectorization → automatic key management

```python
# Use seed for JAX transformations
seeded_model = seed(model.simulate)
result = seeded_model(key, args)
jit_model = jax.jit(seeded_model)

# Use modular_vmap for probabilistic vectorization
vmap_model = modular_vmap(model.simulate, in_axes=(0,))
```

### Static vs Dynamic Arguments

JAX transformations make all arguments dynamic, but some GenJAX operations need static values:

```python
# ❌ PROBLEMATIC - T becomes a tracer
def inference_fn(T, args):
    model = model_factory(T)  # T must be static!

# ✅ CORRECT - Use closures for static values
T = 5  # Static
def inference_closure(args):
    model = model_factory(T)  # Captured as static
    return inference_logic(model, args)

seeded_inference = seed(inference_closure)
```

### Pytree Usage

**CRITICAL**: All GenJAX datatypes inherit from `Pytree` for automatic JAX vectorization:

- **DO NOT use Python lists** for multiple Pytree instances
- **DO use JAX transformations** - they automatically vectorize Pytree leaves
- **Pattern**: Use single vectorized `Trace`, not `[trace1, trace2, ...]`

## Common Error Patterns

### `LoweringSamplePrimitiveToMLIRException`

**Cause**: PJAX primitives inside JAX control flow or JIT compilation.

**Solution**: Apply `seed` transformation:

```python
# ❌ Problematic
trace = model_with_scan.simulate(())

# ✅ Fixed
seeded_model = seed(model_with_scan.simulate)
trace = seeded_model(key, ())
```

## Testing Patterns

**Density Validation**:

```python
def test_model():
    trace = model.simulate(())
    choices = trace.get_choices()

    # Use Distribution.assess for validation
    expected_density = sum(dist.assess(params, choice)[0] for dist, params, choice in distributions)
    actual_density, _ = model.assess((), choices)
    assert jnp.allclose(actual_density, expected_density)
    assert jnp.allclose(trace.get_score(), -actual_density)
```

## Development Commands

```bash
pixi install              # Setup
pixi run format          # Format code
pixi run python examples/simple.py  # Run examples
```

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: Probabilistic extension to JAX with primitives `assume_p`, `log_density_p`
- **Trace**: Execution record with choices, args, return value, score
- **Score**: `log(1/P(choices))` - negative log probability
