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
trace = model.simulate(*args)                   # Sample (choices, retval) ~ P(·; args)
# trace.get_score() = log(1/P(choices; args))   # Negative log probability

# Density evaluation
log_density, retval = model.assess(choices, *args)  # Compute log P(choices; args)

# Constrained generation (importance sampling)
trace, weight = model.generate(constraints, *args)
# weight = log[P(all_choices; args) / Q(unconstrained; constrained, args)]

# Edit moves (MCMC, SMC)
new_trace, weight, discarded = model.update(trace, constraints, *new_args)
# weight = log[P(new_choices; new_args)/Q(new; old, constraints)] - log[P(old_choices; old_args)/Q(old)]

# Selective regeneration (edit move)
new_trace, weight, discarded = model.regenerate(trace, selection, *args, **kwargs)
# weight = log P(new_selected | non_selected; args) - log P(old_selected | non_selected; args)
```

**Mathematical Properties**:

- **Importance weights** enable unbiased Monte Carlo estimation
- **Incremental importance weights** from update/regenerate enable MCMC acceptance probabilities
- **Edit moves** (update, regenerate) provide efficient inference transitions
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
new_trace, weight, discarded = model.regenerate(trace, selection, *args, **kwargs)
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
normal.simulate(mu, sigma) # ✅ CORRECT
```

**Creating Custom Distributions with `distribution()` helper**:

When working with static parameters that need to survive JAX transformations, use the `distribution()` helper function:

```python
from genjax import distribution, const

# ✅ CORRECT - Using distribution() helper for custom distributions
@gen
def model_with_custom_dist():
    # Create distribution with static parameters using Const pattern
    custom_normal = distribution(
        _sample=lambda key, mu, sigma: ...,     # Sampling function
        _logpdf=lambda x, mu, sigma: ...,       # Log density function
        mu=const(0.0),                          # Static mean parameter
        sigma=const(1.0)                        # Static std parameter
    )
    return custom_normal.simulate() @ "x"

# The distribution() helper automatically handles Const fields properly
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
result = scan_gf(init_carry, None) @ "scan"
```

**Vmap** - Vectorization (like `jax.vmap` for generative functions):

```python
# Vectorize over parameters
vectorized_normal = normal.vmap(in_axes=(0, None))
traces = vectorized_normal.simulate(mus_array, sigma_scalar)

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
result = cond_gf(condition) @ "conditional"
```

## Critical API Patterns

**Generative Function Usage**:

```python
# ✅ CORRECT patterns
x = normal(mu, sigma) @ "x"                    # In @gen functions
log_density, retval = normal.assess(sample, mu, sigma)  # GFI calls with unpacked args

# ❌ WRONG patterns
x = normal(mu, sigma)                         # Not traced
x = normal(mu=mu, sigma=sigma) @ "x"          # No kwargs
normal.assess((mu, sigma), sample)           # Wrong arg structure (old format)
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
    results = scan_gf(init, None) @ "scan"
```

### PJAX: Probabilistic JAX

PJAX extends JAX with probabilistic primitives (`sample_p`, `log_density_p`).

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

### State Interpreter: Tagged Value Inspection

The state interpreter allows you to inspect intermediate values within JAX computations using tagged values:

```python
from genjax.state import state, tag_state, save

# Tag single values for inspection
@state
def computation(x):
    y = x + 1
    tagged_y = tag_state(y, name="intermediate")
    return tagged_y * 2

result, state_dict = computation(5)
# result = 12, state_dict = {"intermediate": 6}

# Tag multiple values at once
@state  
def multi_value_computation(x):
    y = x + 1
    z = x * 2
    tagged_y, tagged_z = tag_state(y, z, name="pair")
    return tagged_y + tagged_z

result, state_dict = multi_value_computation(5)
# result = 16, state_dict = {"pair": [6, 10]}

# Convenience function for multiple named values
@state
def convenient_computation(x):
    y = x + 1
    z = x * 2
    values = save(first=y, second=z)  # Returns {"first": y, "second": z}
    return values["first"] + values["second"]

result, state_dict = convenient_computation(5)
# result = 16, state_dict = {"first": 6, "second": 10}
```

**JAX Compatibility**: The state interpreter works with all JAX transformations (`jit`, `vmap`, `grad`) by using `initial_style_bind` for proper JAX primitive handling.

**Key Features**:
- **Multiple value tagging**: `tag_state(a, b, c, name="multi")` 
- **Required naming**: All tags must have a `name` parameter
- **JAX transformation compatibility**: Works with `jit`, `vmap`, `grad`
- **Convenience functions**: `save(x=val1, y=val2)` for multiple named values

### Static vs Dynamic Arguments

JAX transformations make all arguments dynamic, but some GenJAX operations need static values:

```python
# ❌ PROBLEMATIC - length becomes a tracer
def bad_scan_model(length, init_carry, xs):
    scan_gf = Scan(step_fn, length=length)  # length must be static!
    return scan_gf(init_carry, xs)

# ✅ CORRECT - Use Const[...] for static values
from genjax import Const, const

@gen
def scan_model(length: Const[int], init_carry, xs):
    scan_gf = Scan(step_fn, length=length.value)  # length.value is static
    return scan_gf(init_carry, xs) @ "scan"

# Usage with static values
args = (const(10), init_carry, xs)  # Wrap static value with const()
trace = seed(scan_model.simulate)(key, *args)

# Works with any static configuration
@gen
def configurable_model(config: Const[dict], data):
    if config.value["use_hierarchical"]:
        prior = normal(config.value["prior_mean"], config.value["prior_std"]) @ "prior"
    else:
        prior = exponential(config.value["rate"]) @ "prior"
    return normal(prior, 1.0) @ "obs"

# Pass static configuration
config = const({"use_hierarchical": True, "prior_mean": 0.0, "prior_std": 1.0})
trace = configurable_model.simulate(config, data)
```

**Const[...] Pattern Benefits**:

- Preserves static values across JAX transformations like `seed(fn)(...)`
- Enables type-safe static parameters with proper type hints
- Works seamlessly with scan lengths, model configurations, conditionals
- Cleaner and more explicit than closure-based alternatives
- Integrates naturally with GenJAX's type system

### Pytree Usage

**CRITICAL**: All GenJAX datatypes inherit from `Pytree` for automatic JAX vectorization:

- **DO NOT use Python lists** for multiple Pytree instances
- **DO use JAX transformations** - they automatically vectorize Pytree leaves
- **Pattern**: Use single vectorized `Trace`, not `[trace1, trace2, ...]`

## Common Error Patterns

### Address Collision Detection

**GenJAX now automatically detects duplicate addresses** at the same level in `@gen` functions:

```python
# ❌ ERROR - Address collision detected
@gen
def problematic_model():
    x = normal(0.0, 1.0) @ "duplicate"  # First use
    y = normal(2.0, 3.0) @ "duplicate"  # ❌ Same address!
    return x + y

# Error message includes location information:
# ValueError: Address collision detected: 'duplicate' is used multiple times at the same level.
# Each address in a generative function must be unique.
# Function: function 'problematic_model' at /path/to/file.py:42
# Location: file.py:44
```

**Address collision detection runs in all GFI methods**:

- `simulate()`, `assess()`, `generate()`, `update()`, `regenerate()`
- Provides clear error messages with function and line information
- Helps debug complex generative functions with many addresses

**Valid patterns that DON'T trigger collisions**:

```python
# ✅ CORRECT - Same address in different scopes
@gen
def inner():
    return normal(0, 1) @ "x"

@gen
def outer():
    a = inner() @ "call1"  # inner has address "x"
    b = inner() @ "call2"  # inner has address "x" - this is fine!
    return a + b
    # Final structure: choices["call1"]["x"], choices["call2"]["x"]

# ✅ CORRECT - Unique addresses at same level
@gen
def valid_model():
    x = normal(0.0, 1.0) @ "first"
    y = normal(2.0, 3.0) @ "second"  # Different address
    return x + y
```

### Enhanced Error Reporting

**GenJAX provides enhanced error messages** with source location information for debugging:

```python
# Error messages now include:
# 1. Function name and file location where error occurs
# 2. Specific line number of the problematic code
# 3. Clear description of the issue and how to fix it

# Example error output:
# ValueError: Address collision detected: 'x' is used multiple times at the same level.
# Each address in a generative function must be unique.
# Function: function 'my_model' at /path/to/model.py:15
# Location: model.py:18

# Benefits:
# - Quickly locate problematic code in large generative functions
# - Stack trace filtering removes internal GenJAX frames
# - Filters out beartype wrapper noise for cleaner error messages
```

**Error reporting improvements apply to**:

- Address collision detection
- Invalid trace operations
- Type checking violations
- GFI method constraint violations

### `LoweringSamplePrimitiveToMLIRException`

**Cause**: PJAX primitives inside JAX control flow or JIT compilation.

**Solution**: Apply `seed` transformation:

```python
# ❌ Problematic
trace = model_with_scan.simulate()

# ✅ Fixed
seeded_model = seed(model_with_scan.simulate)
trace = seeded_model(key)
```

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: Probabilistic extension to JAX with primitives `sample_p`, `log_density_p`
- **State Interpreter**: JAX interpreter for inspecting tagged intermediate values
- **Trace**: Execution record with choices, args, return value, score
- **Score**: `log(1/P(choices))` - negative log probability

## References

### Theoretical Foundation

- **SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals (SMCP3)**: Lew, A. K., Matheos, G., Zhi-Xuan, T., Ghavamizadeh, M., Russell, N., Cusumano-Towner, M., & Mansinghka, V. K. (2023). Sequential Monte Carlo with programmable proposals. In Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023). [Paper](https://proceedings.mlr.press/v206/lew23a/lew23a.pdf)

### Gen Julia Implementation

- **Gen Julia Documentation**: Comprehensive documentation for the original Gen probabilistic programming language. [https://www.gen.dev/docs/stable/](https://www.gen.dev/docs/stable/)
- **Gen Julia GitHub Repository**: Source code and examples for the Julia implementation. [https://github.com/probcomp/Gen.jl](https://github.com/probcomp/Gen.jl)
- **Generative Function Interface**: Mathematical specification and API reference. [https://www.gen.dev/docs/stable/api/model/gfi/](https://www.gen.dev/docs/stable/api/model/gfi/)

### Notes

GenJAX implements the same mathematical foundations as Gen Julia, with the GFI methods (`simulate`, `assess`, `generate`, `update`, `regenerate`) following identical mathematical specifications. The `update` and `regenerate` methods are examples of SMCP3 edit moves that enable efficient probabilistic inference.
