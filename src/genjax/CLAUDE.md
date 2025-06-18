# GenJAX Core Concepts Guide

This guide covers the core GenJAX concepts implemented in:
- `core.py`: Generative functions, traces, Fixed infrastructure
- `distributions.py`: Probability distributions  
- `pjax.py`: Probabilistic JAX (PJAX) primitives and interpreters
- `state.py`: State inspection interpreter

**For inference algorithms**, see `inference/CLAUDE.md`  
**For gradient estimation**, see `adev/CLAUDE.md`  
**For testing utilities**, see `extras/CLAUDE.md`

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

### Fixed Values & Model Structure Debugging

**The `Fixed` Infrastructure** tracks whether random choices were externally constrained (provided by user/constraints) versus internally proposed (sampled by the model). This enables debugging model structure issues during inference.

#### Core Components

**`Fixed[A]` Wrapper**:
```python
from genjax import Fixed, fixed

# Create Fixed wrapper for constrained values
constrained_value = fixed(1.5)
print(constrained_value)  # Fixed(1.5)

# Access the underlying value
actual_value = constrained_value.value  # 1.5
```

**`trace.verify()` Method**:
```python
# Check if all choices were properly constrained
try:
    trace.verify()
    print("✅ All values properly constrained")
except NotFixedException as e:
    print("❌ Model structure issue detected:")
    print(e)  # Shows detailed choice map with Fixed/NOT_FIXED status
```

#### How Fixed Works

**Distribution Methods Automatically Use Fixed**:
- `generate(constrained_value, ...)` → stores `Fixed(constrained_value)` in choices
- `update(trace, constrained_value, ...)` → stores `Fixed(constrained_value)` in choices
- `regenerate(trace, selection, ...)` → keeps unselected values as `Fixed`

**`get_choices()` Strips Fixed Wrappers**:
```python
# Internal storage (with Fixed wrappers)
trace._choices = {"x": Fixed(1.5), "y": Fixed(2.0)}

# User-facing interface (Fixed wrappers stripped)
trace.get_choices()  # {"x": 1.5, "y": 2.0}
```

**Return values are NEVER Fixed** - only internal choice storage uses Fixed wrappers.

#### Debugging Model Structure Issues

**Common Patterns**:

```python
# ✅ CORRECT: Mixed Fixed/Unfixed in sequential models
{
  "t0": {"state": Fixed, "obs": Fixed},      # Initial state constrained by prior+data
  "t1": {"state": NOT_FIXED, "obs": Fixed}, # State proposed from t0, obs constrained
  "t2": {"state": NOT_FIXED, "obs": Fixed}  # State proposed from t1, obs constrained
}

# ❌ PROBLEM: All states unfixed in sequential models
{
  "t0": {"state": NOT_FIXED, "obs": Fixed}, # Missing temporal constraint
  "t1": {"state": NOT_FIXED, "obs": Fixed}, # Missing temporal constraint
  "t2": {"state": NOT_FIXED, "obs": Fixed}  # Missing temporal constraint
}
```

**Interpreting verify() Results**:

- **All Fixed**: Perfect constraint (may indicate over-constrained model)
- **Mixed Fixed/NOT_FIXED**: Normal for sequential models with temporal dependencies
- **All NOT_FIXED**: Missing constraints (indicates model structure problems)

#### Debugging Sequential Models

**Example: SMC vs Kalman Filter Issues**
```python
# Run SMC inference
particles = init(model, args, n_particles, constraints)

# Check if model structure is correct
try:
    particles.traces.verify()
    print("✅ Model has proper temporal structure")
except NotFixedException as e:
    print("❌ Model structure issue - likely missing temporal dependencies:")
    print(e)
    # Output shows which states are NOT_FIXED when they should be Fixed
```

**What Fixed Status Indicates**:
- **Observations Fixed**: ✅ Properly constrained by data
- **States Fixed**: ✅ Properly constrained by previous timestep or prior
- **States NOT_FIXED**: Either correctly proposed OR missing constraints

**Use Cases**:
1. **Sequential Models**: Verify temporal dependencies are properly captured
2. **Constraint Debugging**: Ensure all required values are constrained
3. **SMC Troubleshooting**: Diagnose particle filter model structure issues
4. **MCMC Validation**: Check that regeneration properly maintains constraints

#### Technical Implementation

**Tree Operations with Fixed**:
```python
# Fixed detection using is_leaf
def check_instance_fixed(x):
    return isinstance(x, Fixed)

# Flatten to check all leaves
leaf_values, tree_def = jtu.tree_flatten(choices, is_leaf=check_instance_fixed)
all_fixed = all(isinstance(leaf, Fixed) for leaf in leaf_values)

# Create boolean status map
choice_map_status = jtu.tree_map(
    lambda x: isinstance(x, Fixed),
    choices,
    is_leaf=check_instance_fixed
)
```

**JAX Compatibility**: Fixed wrappers work seamlessly with all JAX transformations and tree operations.

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
x = normal(mu, sigma) @ "x"                           # Positional args in @gen functions
x = normal(mu=mu, sigma=sigma) @ "x"                  # Keyword args in @gen functions
log_density, retval = normal.assess(sample, mu, sigma)        # GFI calls with positional args
log_density, retval = normal.assess(sample, mu=mu, sigma=sigma)  # GFI calls with keyword args

# ❌ WRONG patterns
x = normal(mu, sigma)                                 # Not traced (missing @ "address")
normal.assess((mu, sigma), sample)                   # Wrong arg structure (old format)
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

### State Interpreter: Tagged Value Inspection & Organization

The state interpreter allows you to inspect and organize intermediate values within JAX computations using a powerful API for hierarchical state collection.

#### Core API

```python
from genjax.state import state, save, namespace

# Basic pattern: @state decorator + save() function (named mode)
@state
def computation(x):
    y = x + 1
    z = x * 2
    # Save intermediate values for inspection
    save(intermediate=y, doubled=z)
    return y + z

result, state_dict = computation(5)
# result = 16, state_dict = {"intermediate": 6, "doubled": 10}
```

#### Leaf Mode for Direct Storage

The `save()` function supports two modes:

**Named Mode** (`save(**kwargs)`): Save values with explicit names (original behavior):
```python
save(first=value1, second=value2)  # → {"first": value1, "second": value2}
```

**Leaf Mode** (`save(*args)`): Save values directly at current namespace leaf:
```python
# Single value
namespace(lambda: save(42), "coords")()  # → {"coords": 42}

# Multiple values (stored as tuple)
namespace(lambda: save(1, 2, 3), "coords")()  # → {"coords": (1, 2, 3)}
```

#### Namespace Organization

**Hierarchical State Collection**: Use `namespace(fn, ns)` to organize state into nested structures:

```python
@state
def complex_computation(x):
    # Root level state (named mode)
    save(input=x, stage="preprocessing")

    # Named mode in namespace
    processing_fn = namespace(
        lambda y: save(step1=y*2, step2=y+10),
        "processing"
    )
    processing_fn(x)

    # Leaf mode in namespace - store values directly
    coords_fn = namespace(
        lambda z: save(z, z*2, z*3),  # Leaf mode: saves tuple at "coords"
        "coords"
    )
    coords_fn(x)

    # Nested namespaces with mixed modes
    stats_fn = namespace(
        namespace(lambda z: save(mean=z, variance=z**2), "statistics"),  # Named mode
        "analysis"
    )
    stats_fn(x)

    return x * 3

result, state_dict = complex_computation(5)
# state_dict = {
#     "input": 5,
#     "stage": "preprocessing",
#     "processing": {"step1": 10, "step2": 15},  # Named mode
#     "coords": (5, 10, 15),                     # Leaf mode: tuple directly
#     "analysis": {"statistics": {"mean": 5, "variance": 25}}  # Named mode
# }
```

**Namespace Composition**: Namespaces can be nested to arbitrary depth:

```python
# Create deeply nested organization
deep_analysis = namespace(
    namespace(
        namespace(lambda data: save(metric=data), "metrics"),
        "detailed"
    ),
    "analysis"
)
# Results in: {"analysis": {"detailed": {"metrics": {...}}}}
```

#### MCMC Integration

**Acceptance Tracking**: Used internally for MCMC diagnostics:

```python
@state
def mcmc_step(trace):
    new_trace = some_mcmc_move(trace)
    accept = compute_acceptance(new_trace, trace)
    save(accept=accept)  # Save acceptance for diagnostics
    return new_trace

new_trace, diagnostics = mcmc_step(trace)
# diagnostics = {"accept": True/False}

# Organized MCMC diagnostics
@state
def organized_mcmc_step(trace):
    # Proposals under "proposal" namespace
    proposal_fn = namespace(
        lambda: save(type="mh", step_size=0.1),
        "proposal"
    )
    proposal_fn()

    # Acceptance under "diagnostics" namespace
    diag_fn = namespace(
        lambda: save(accepted=True, log_ratio=-0.5),
        "diagnostics"
    )
    diag_fn()

    return new_trace
# Result: {"proposal": {"type": "mh", "step_size": 0.1},
#          "diagnostics": {"accepted": True, "log_ratio": -0.5}}
```

#### JAX Compatibility

**Full JAX Integration**: Works with all JAX transformations using JAX primitives:

```python
# Works with jit
jitted_computation = jax.jit(computation)
result, state_dict = jitted_computation(5)

# Works with vmap
vmapped_computation = jax.vmap(computation)
results, state_dicts = vmapped_computation(jnp.array([1, 2, 3]))

# Works with grad (on the result, not the state)
grad_fn = jax.grad(lambda x: computation(x)[0])  # Differentiate result
gradient = grad_fn(5.0)
```

**Implementation Details**:
- Uses `initial_style_bind` for proper JAX primitive handling
- Namespace stack managed via JAX primitives (`namespace_push_p`, `namespace_pop_p`)
- Error-safe: namespace stack cleaned up even when functions raise exceptions
- Compatible with scan, vmap, and other JAX combinators

#### Key Features

- **Two Storage Modes**: Named mode (`save(**kwargs)`) and leaf mode (`save(*args)`)
- **Hierarchical Organization**: `namespace(fn, ns)` for structured state collection
- **Composable Namespaces**: Unlimited nesting depth through function composition
- **JAX Transformation Safety**: Full compatibility with `jit`, `vmap`, `grad`, `scan`
- **Error Handling**: Automatic namespace stack cleanup on exceptions
- **MCMC Integration**: Used internally for acceptance tracking and diagnostics
- **Mixed State**: Combine root-level and namespaced state in same computation
- **Leaf Mode Benefits**: Store values directly at namespace paths without additional keys
- **Performance**: Zero overhead when not using `@state` decorator

**When to Use Each Mode**:
- **Named Mode**: When you want explicit keys for multiple values in same namespace
- **Leaf Mode**: When you want to store coordinates, vectors, or single values directly at namespace path

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

**GenJAX detects duplicate addresses** at the same level in `@gen` functions:

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

### Error Reporting

**GenJAX provides error messages** with source location information for debugging:

```python
# Error messages include:
# 1. Function name and file location where error occurs
# 2. Specific line number of the problematic code
# 3. Description of the issue and how to fix it

# Example error output:
# ValueError: Address collision detected: 'x' is used multiple times at the same level.
# Each address in a generative function must be unique.
# Function: function 'my_model' at /path/to/model.py:15
# Location: model.py:18

# Benefits:
# - Locate problematic code in large generative functions
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

## Performance and Optimization

### Memory Management

- **Trace Size**: Be mindful of trace memory with large choice maps
- **Fixed Infrastructure**: Use `Fixed` wrapper for static values to avoid recompilation
- **Batching**: Use `vmap` for parallel operations on multiple traces

### Numerical Stability

- **Log Space**: Always work in log space for probabilities
- **Score Accumulation**: Scores are accumulated as negative log probabilities
- **Small Probabilities**: Use `logsumexp` for stable probability aggregation

### JAX Compilation

- **JIT Compilation**: Use `@jax.jit` for hot loops
- **Static Arguments**: Mark static arguments with `Const[T]` type hints
- **Avoid Recompilation**: Use consistent shapes and types

## References

### Theoretical Foundation

### Gen Julia Implementation

- **Gen Julia Documentation**: Comprehensive documentation for the original Gen probabilistic programming language. [https://www.gen.dev/docs/stable/](https://www.gen.dev/docs/stable/)
- **Gen Julia GitHub Repository**: Source code and examples for the Julia implementation. [https://github.com/probcomp/Gen.jl](https://github.com/probcomp/Gen.jl)
- **Generative Function Interface**: Mathematical specification and API reference. [https://www.gen.dev/docs/stable/api/model/gfi/](https://www.gen.dev/docs/stable/api/model/gfi/)

### Notes

GenJAX implements the same mathematical foundations as Gen Julia, with the GFI methods (`simulate`, `assess`, `generate`, `update`, `regenerate`) following identical mathematical specifications. The `update` and `regenerate` methods are edit moves that enable probabilistic inference. The MCMC and SMC implementations provide JAX-native vectorization and diagnostics.

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: Probabilistic extension to JAX with primitives `sample_p`, `log_density_p`
- **State Interpreter**: JAX interpreter for inspecting tagged intermediate values
- **Trace**: Execution record with choices, args, return value, score
- **Score**: `log(1/P(choices))` - negative log probability
