# GenJAX Concepts Guide

This file provides detailed guidance on GenJAX concepts and usage patterns for Claude Code.

## Core Concepts

### Generative Functions & Traces

- **Generative Function**: Probabilistic program implementing the Generative Function Interface (GFI)
- **Trace**: Execution record containing random choices, arguments, return value, and score (`log 1/P(choices)`)

### Generative Function Interface (GFI)

**Mathematical Foundation**: Generative functions bundle:

- Measure kernel $P(dx; args)$ over measurable space $X$ (the model distribution)
- Return value function $f(x, args) \rightarrow R$ (deterministic computation from choices)
- Internal proposal family $Q(dx; args, context)$ (for efficient inference)

**Core GFI Methods** (all densities in log space):

```python
# Forward sampling
trace = model.simulate(args)                    # Sample (choices, retval) ~ P(·; args)
# trace.get_score() = log(1/P(choices; args))   # Negative log probability

# Density evaluation
log_density, retval = model.assess(args, choices)  # Compute log P(choices; args)

# Constrained generation (importance sampling)
trace, weight = model.generate(args, constraints)
# weight = log[P(all_choices; args) / Q(unconstrained; constrained, args)]

# Incremental updates (MCMC, SMC)
new_trace, weight, discarded = model.update(new_args, trace, constraints)
# weight = log[P(new_choices; new_args)/Q(new; old, constraints)] - log[P(old_choices; old_args)/Q(old)]

# Selective regeneration
new_trace, weight, discarded = model.regenerate(args, trace, selection)
# weight = log P(new_selected | non_selected; args) - log P(old_selected | non_selected; args)
```

**Mathematical Properties**:
- **Importance weights** enable unbiased Monte Carlo estimation
- **Weight ratios** from update/regenerate enable MCMC acceptance probabilities
- **Incremental computation** allows efficient inference on large models
- **Selection interface** enables fine-grained control over which choices to modify

**Trace Interface**:
```python
trace.get_retval()     # Return value: R
trace.get_choices()    # Random choices: X
trace.get_score()      # Negative log probability: log(1/P(choices))
trace.get_args()       # Function arguments: tuple
trace.get_gen_fn()     # Source generative function: GFI[X, R]
```

### Selection Interface

**Selections** specify which addresses to target for regeneration/update operations:

```python
from genjax import sel

# Basic selections
sel("x")                    # Select address "x"
sel()                       # Select nothing (empty selection)
Selection(AllSel())         # Select everything

# Combinators
sel("x") | sel("y")         # OR: select "x" or "y"
sel("x") & sel("y")         # AND: select "x" and "y" (intersection)
~sel("x")                   # NOT: select everything except "x" (complement)

# Usage in regenerate
selection = sel("mu") | sel("sigma")  # Select parameters to resample
new_trace, weight, discarded = model.regenerate(args, trace, selection)
```

**Selection Semantics**:
- `match(addr) -> (bool, subselection)` determines if address is selected
- Supports hierarchical addressing for nested generative functions
- Empty selection → no regeneration, weight = 0
- Full selection → equivalent to `simulate()` from scratch

## Generative Function Types

### Distributions

Built-in distributions implement the GFI and wrap TensorFlow Probability distributions:

```python
from genjax import normal, beta, exponential, categorical, flip

# Usage: parameters as args, not constructor arguments
x = normal.sample(mu, sigma)        # ✅ CORRECT
exponential.sample(rate)            # ✅ CORRECT
logp = normal.logpdf(x, mu, sigma)  # ✅ CORRECT

# Usage: GFI methods, same idea.
normal.simulate((mu, sigma)) # ✅ CORRECT
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
    result = beta_ber() @ "sub"
    # choices["sub"]["fairness"]
    # OR choices["sub"]["obs"]
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

**Vmap** - Vectorization (like `jax.vmap` for generative functions):

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
x = normal(mu, sigma)                         # Not traced
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

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: Probabilistic extension to JAX with primitives `assume_p`, `log_density_p`
- **Trace**: Execution record with choices, args, return value, score
- **Score**: `log(1/P(choices))` - negative log probability
