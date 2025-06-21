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
- Internal proposal family $Q(dx'; args, x)$

**Core GFI Methods** (all densities in log space):

#### simulate
**Method**: `simulate(*args, **kwargs) -> Trace[X, R]`
**Location**: `core.py:1032-1060`
**Purpose**: Forward sampling from the generative function
- Samples `(choices, retval) ~ P(·; args)`
- Returns trace with `score = log(1/P(choices; args))`

#### assess
**Method**: `assess(x: X, *args, **kwargs) -> tuple[Density, R]`
**Location**: `core.py:1098-1127`
**Purpose**: Evaluate density at given choices
- Returns `(log P(choices; args), retval)`

#### generate
**Method**: `generate(x: X | None, *args, **kwargs) -> tuple[Trace[X, R], Weight]`
**Location**: `core.py:1062-1096`
**Purpose**: Constrained generation via importance sampling
- Returns trace and weight: `log[P(all_choices) / Q(unconstrained | constrained)]`

#### update
**Method**: `update(tr: Trace[X, R], x_: X | None, *args, **kwargs) -> tuple[Tr[X, R], Weight, X | None]`
**Location**: `core.py:1129-1166`
**Purpose**: Edit move for MCMC/SMC
- Returns new trace, incremental weight, discarded choices

#### regenerate
**Method**: `regenerate(tr: Trace[X, R], sel: Selection, *args, **kwargs) -> tuple[Tr[X, R], Weight, X | None]`
**Location**: `core.py:1168-1208`
**Purpose**: Selective regeneration of addresses
- Weight: `log P(new_selected | non_selected) - log P(old_selected | non_selected)`

#### merge
**Method**: `merge(x: X, x_: X, check: jnp.ndarray | None = None) -> tuple[X, X | None]`
**Location**: `core.py:1210-1227`
**Purpose**: Merge two choice maps, with optional conditional selection
- Returns `(merged_choices, discarded_values)`
- If `check` is provided, uses `jnp.where(check, x, x_)` for conditional selection at leaf level
- Used internally for compositional generative functions and Cond combinator
- **Enhanced API (June 2025)**: Added `check` parameter and tuple return for conditional merge support

#### filter
**Method**: `filter(x: X, selection: Selection) -> tuple[X | None, X | None]`
**Location**: `core.py:1227-1252`
**Purpose**: Filter choice map into selected and unselected parts
- Returns `(selected_choices, unselected_choices)`

#### log_density
**Method**: `log_density(x: X, *args, **kwargs) -> Score`
**Location**: `core.py:1278-1285`
**Purpose**: Convenience method for assess that sums log densities
- Returns total log density as scalar

**Mathematical Properties**:

- **Importance weights** enable unbiased Monte Carlo estimation
- **Incremental importance weights** from update/regenerate enable MCMC acceptance probabilities and SMC weight updates
- **Selection interface** enables fine-grained control over which choices to modify

**Trace Interface**:

**Type**: `Trace[X, R]`
**Location**: `core.py:537-630`
**Methods**:
- `get_retval() -> R`: Return value
- `get_choices() -> X`: Random choices
- `get_score() -> Score`: Negative log probability
- `get_args() -> Any`: Function arguments
- `get_gen_fn() -> GFI[X, R]`: Source generative function
- `get_fixed_choices() -> X`: Choices preserving Fixed wrappers

### Selection Interface

**Selections** specify which addresses to target for regeneration and choice filter operations.

**Function**: `sel(*v: tuple[()] | str | dict[str, Any] | None) -> Selection`
**Location**: `core.py:930-975`
**Purpose**: Create selection objects for targeting specific addresses

**Selection Combinators**:
- `sel("x")`: Select address "x"
- `sel()`: Empty selection (selects nothing)
- `Selection(AllSel())`: Select everything
- `sel("x") | sel("y")`: OR combinator
- `sel("x") & sel("y")`: AND combinator (intersection)
- `~sel("x")`: NOT combinator (complement)

**Selection Semantics**:
- `match(addr) -> (bool, subselection)` determines if address is selected
- Supports hierarchical addressing for nested generative functions
- Used in `regenerate` and other GFI methods for selective operations

## Generative Function Types

### Distributions

**Location**: `distributions.py`
**Purpose**: Built-in probability distributions implementing the GFI

**Available Distributions**:
- `normal`, `beta`, `exponential`, `gamma`, `poisson`
- `categorical`, `flip`, `uniform`, `dirichlet`
- `multivariate_normal`, `bernoulli`

**API Pattern**:
- Parameters passed as arguments, not constructor: `normal(mu, sigma)`
- All distributions have: `sample()`, `logpdf()`, and full GFI methods
- Wrap TensorFlow Probability distributions internally

### `@gen` Functions

**Decorator**: `@gen`
**Function**: `gen(fn: Callable[..., R]) -> Fn[R]`
**Location**: `core.py:2212-2248`
**Purpose**: Transform Python functions into generative functions

**Key Features**:
- Use `@` operator for addressing random choices
- Supports hierarchical composition
- Full GFI implementation
- JAX-compatible (with restrictions)

**Addressing Pattern**:
- Single level: `normal(0, 1) @ "x"`
- Hierarchical: `sub_model() @ "sub"` creates nested addresses

### Combinators

Higher-order generative functions for composition:

#### Scan
**Class**: `Scan[X, R](callee: GFI[X, R], length: Const[int])`
**Location**: `core.py:2284-2492`
**Purpose**: Sequential iteration like `jax.lax.scan`
- `length` must be static (use `Const[int]`)
- Addresses indexed automatically: `x_0`, `x_1`, ...
- Returns `(final_carry, stacked_outputs)`

#### Vmap
**Method**: `vmap(in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0, axis_size=None, axis_name: str | None = None, spmd_axis_name: str | None = None) -> Vmap[X, R]`
**Location**: `core.py:1254-1267`
**Purpose**: Vectorization over generative functions
- Works like `jax.vmap` but for probabilistic programs
- `repeat(n)` method for independent sampling

#### Cond
**Class**: `Cond[X, R](callee: GFI[X, R], callee_: GFI[X, R])`
**Location**: `core.py:2545-2688`
**Purpose**: Conditional execution with full support for same-address branches
- First argument to resulting GF must be boolean condition
- Both branches must have same return type
- **Enhanced (June 2025)**: Now supports branches with same addresses via conditional merge API
- Uses `merge(x, x_, check=condition)` for efficient conditional selection
- `CondTr.get_choices()` automatically applies conditional selection using enhanced merge
- Enables natural mixture models without address conflicts
- No NaN masking needed when structures match - direct `jnp.where` selection instead

## Critical API Patterns

### Generative Function Usage

**Correct Patterns**:
- In `@gen` functions: `x = normal(mu, sigma) @ "x"`
- GFI method calls: `normal.assess(sample, mu, sigma)`
- Both positional and keyword args supported

**Common Mistakes**:
- Missing `@` operator: Random choices not traced
- Wrong argument order in `assess`: choices come first
- Old tuple-based API no longer supported

## JAX Integration & Constraints

### CRITICAL JAX Python Restrictions

**NEVER use Python control flow in `@gen` functions**:

**Forbidden Patterns**:
- Python `if`/`else` statements
- Python `for`/`while` loops
- Dynamic string formatting for addresses
- Any Python construct that creates dynamic control flow

**Required Patterns**:
- Use `Cond` combinator for conditionals
- Use `Scan` combinator for iteration
- Use static addressing only
- See combinators section for proper usage

### PJAX: Probabilistic JAX

**Location**: `pjax.py`
**Purpose**: Extends JAX with probabilistic primitives

**Core Primitives**:
- `sample_p`: Probabilistic sampling primitive
- `log_density_p`: Density evaluation primitive

**Key Transformations**:

#### seed
**Function**: `seed(fn) -> seeded_fn`
**Location**: `pjax.py:seed`
- Eliminates PJAX primitives for JAX compatibility
- Requires explicit PRNG keys
- **CRITICAL**: Only use external to `src/`, never inside library code

#### modular_vmap
**Function**: `modular_vmap(fn, in_axes, axis_size) -> vmapped_fn`
**Location**: `pjax.py:modular_vmap`
- Preserves PJAX primitives during vectorization
- Automatic PRNG key management
- Use for probabilistic operations inside library

### State Interpreter: Tagged Value Inspection & Organization

**Location**: `state.py`
**Purpose**: Inspect and organize intermediate values in JAX computations

#### Core API

**Decorator**: `@state`
**Location**: `state.py:state`
- Transforms function to return `(result, state_dict)` tuple
- Collects all `save()` calls during execution

**Function**: `save(*args, **kwargs)`
**Location**: `state.py:save`
- Named mode: `save(key=value)` stores with explicit keys
- Leaf mode: `save(value)` stores directly at namespace

**Function**: `namespace(fn, name)`
**Location**: `state.py:namespace`
- Creates hierarchical state organization
- Returns wrapped function that executes in namespace

#### Namespace Organization

**Hierarchical State Collection**:
- Use `namespace(fn, name)` to create nested state structures
- Supports arbitrary nesting depth
- Combines with both named and leaf save modes

**Common Patterns**:
- Root level: Direct `save()` calls
- Single namespace: `namespace(fn, "section")`
- Nested: `namespace(namespace(fn, "inner"), "outer")`
- See `state.py` for implementation details

### Enhanced Cond Combinator (June 2025)

**Mixture Model Pattern with Same Addresses**:
```python
@gen
def mixture_observation(condition, value):
    """Example: Mixture model with same observation address in both branches."""
    @gen
    def normal_branch():
        return normal(value, 0.1) @ "obs"  # Same address!

    @gen
    def heavy_tail_branch():
        return normal(value, 1.0) @ "obs"  # Same address!

    # This now works correctly thanks to enhanced merge API
    cond_model = Cond(heavy_tail_branch, normal_branch)
    observation = cond_model(condition) @ "mixture"
    return observation
```

**How It Works**:
- `CondTr` stores choices from both branches
- When `get_choices()` is called, uses `merge(branch1_choices, branch2_choices, check=condition)`
- Distribution.merge implements `jnp.where(condition, branch1_val, branch2_val)` at leaves
- Result: Efficient conditional selection without address conflicts

#### MCMC Integration

**Purpose**: Track acceptance and diagnostics in MCMC algorithms
**Usage**: See `mcmc.py` for actual implementation

**Common Pattern**:
- Wrap MCMC kernel with `@state`
- Save acceptance decisions with `save(accept=...)`
- Use namespaces for organized diagnostics
- Enables acceptance rate computation

#### JAX Compatibility

**Full JAX Integration**:
- Compatible with all JAX transformations: `jit`, `vmap`, `grad`, `scan`
- Uses JAX primitives for proper transformation behavior
- Namespace stack managed safely across transformations
- Zero overhead when `@state` decorator not used

**Implementation Details**: See `state.py` for primitive definitions

#### Key Features Summary

- **Two Storage Modes**: Named (`save(**kwargs)`) and leaf (`save(*args)`)
- **Hierarchical Organization**: Via `namespace(fn, name)`
- **JAX Compatible**: Works with all transformations
- **MCMC Integration**: Tracks acceptance diagnostics
- **Error Safe**: Automatic cleanup on exceptions

### The `Const[...]` Pattern

**Type**: `Const[A]`
**Function**: `const(a: A) -> Const[A]`
**Location**: `core.py:194-437`
**Purpose**: Preserve static values across JAX transformations

**When to Use**:
- Scan lengths: `Scan(gf, length=const_value)`
- Configuration dictionaries
- Any value that must remain static during JAX transformations
- Type annotations: `param: Const[int]`

**API Pattern**:
- Wrap values: `const(10)`
- Access values: `const_param.value`
- Type safe with proper annotations

**Benefits**:
- Prevents JAX tracer errors
- Enables static configuration
- Cleaner than closure workarounds

### Pytree Usage

**CRITICAL**: All GenJAX datatypes inherit from `Pytree` for automatic JAX vectorization:

- **DO NOT use Python lists** for multiple Pytree instances
- **DO use JAX transformations** - they automatically vectorize Pytree leaves
- **Pattern**: Use single vectorized `Trace`, not `[trace1, trace2, ...]`

## Common Error Patterns

### Address Collision Detection

**Feature**: Automatic duplicate address detection
**Location**: Implemented in `core.py:Fn` class
**Purpose**: Prevent accidentally reusing addresses at same level

**Detection Scope**:
- Runs in all GFI methods
- Checks addresses at same hierarchical level
- Provides file location and line numbers in errors

**Error Pattern**:
- Using same address twice at same level triggers error
- Error message includes function name and exact location

**Valid Patterns**:
- Same address in different scopes (via composition) is allowed
- Creates hierarchical structure: `choices["outer"]["inner"]`

### Error Reporting

**Enhanced Error Messages** include:
- Function name and file location
- Specific line number of error
- Clear description and fix suggestions
- Filtered stack traces (removes internal frames)

**Error Types Covered**:
- Address collision detection
- Invalid trace operations
- Type checking violations
- GFI method constraint violations

**Implementation**: See `core.py` for error formatting logic

### `LoweringSamplePrimitiveToMLIRException`

**Cause**: PJAX primitives inside JAX control flow or JIT compilation

**Solution**: Apply `seed` transformation before JAX operations
- Transform: `seed(model.simulate)`
- Then apply: `jax.jit`, `jax.vmap`, etc.
- See `pjax.py:seed` for implementation

## Performance and Optimization

### Numerical Stability

- **Log Space**: Always work in log space for probabilities
- **Score Accumulation**: Scores are accumulated as negative log probabilities
- **Small Probabilities**: Use `logsumexp` for stable probability aggregation

### JAX Compilation

- **JIT Compilation**: Use `@jax.jit` as high as possible in the computation graph
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
