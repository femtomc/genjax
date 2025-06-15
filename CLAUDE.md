# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Commands

The codebase is managed through [`pixi`](https://pixi.sh/latest/). All toplevel codebase commands should go through `pixi`.

### Environment Setup

```bash
# Install dependencies using pixi (a package manager)
pixi install
```

### Running Examples

```bash
# Run examples (via pixi)
pixi run python examples/simple.py
pixi run python examples/regression.py
pixi run python examples/marginal.py
pixi run python examples/vi.py
```

### Generating Figures

```bash
# Generate figures for the beta-bernoulli benchmark
pixi run betaber-timing

# Generate figures for Game of Life simulations
pixi run gol-timing
pixi run gol-figs

# Generate figures for curve fitting
pixi run curvefit-figs
```

### Development

```bash
# Format code
pixi run format

# Check for unused code
pixi run vulture
```

### Documentation

```bash
# Preview documentation
pixi run preview

# Deploy documentation to GitHub Pages
pixi run deploy
```

## Overview

GenJAX is a probabilistic programming language embedded in Python.  

A probabilistic program language (PPL) is a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation of expected values, and more.

The design of GenJAX is centered on programmable inference: automation which allows users of GenJAX to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions, answering questions like "`x` probabilistically affects `y`, and I observe `y`: what are my new beliefs about `x`?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

## Key concepts

### Datatypes: Generative Functions & Traces

- **Generative Function**: A probabilistic program that implements the Generative Function Interface (GFI)
- **Trace**: A recording of execution containing random choices sampled during the execution, the arguments of the execution, the return value of the execution, and the score of the execution (the score is `log 1 / P(random choices)`, the reciprocal of the density in logspace).

### Generative Function Interface

Generative functions are probabilistic programs which bundle together a set of probabilistic ingredients (measures and deterministic functions), and expose a computational interface that provides automation for doing computations with those ingredients.

In the following, we use tags:

- (**MATH**) indicates a description in terms of the abstract mathematical ingredients.
- (**COMPUTATIONAL**) indicates a description in terms of Python types.

The (**MATH**) ingredients of generative functions are as follows:

- A measure kernel $P(dx; a \in A)$ over a measurable space $X$ given arguments $a \in A$ (informally called: the `P` distribution).
- A measurable function $f(x, a \in A) \rightarrow R$ (informally called: the return value function).
- An indexed family of measure kernels $Q(dX; a \in A, x' \in X')$ given arguments $A$ and sample $x' \in X' \subset X$ (informally called _the internal proposal distribution family_)

**IMPORTANT**:

- (**COMPUTATIONAL**) The **Computational Interface** provides access to computations using these ingredients, and is the primary set of methods you will be working with when using GenJAX.

#### Computational Interface

The generative function interface, or GFI, is the set of interface methods for working with generative functions and traces. The definition of `GFI` and `Trace` is given in `src/core.py`.

Below, we enumerate the list of interfaces, with their signatures. We use GenJAX Python types: `X` is the Python type of "random samples" (**MATH**: the measurable space $X$) and `R` is the type of "return values" (**MATH**: the measurable space $R$).

**Computational Interface on the type `GFI`**

- **simulate**

   ```python
   GFI[X, R].simulate(args: tuple) -> Trace[X, R]
   ```

  (**COMPUTATIONAL**) Given `args: A`, sample a sample `x : X`, evaluate `log (1 / P(x; args))`, and the return value function `f(x, args)`. Store these values in a `Trace[X, R]` and return it.
- **assess**

   ```python
   GFI[X, R].assess(args: tuple, x: X) -> tuple[Density, R]
   ```

  (**COMPUTATIONAL**): Given `args: A` and sample `x : X`, evaluate the log density `log P(x; args)` and the return value function. Returns the log density and the value of the return value function.
- **generate**

   ```python
   GFI[X, R].generate(args: tuple, x: X_) -> tuple[Trace[X, R], Weight]
   ```

  (**COMPUTATIONAL**): Given `args: A` and sample `x_ : X_`, sample a complete sample `x : X` using `Q(dx; x_, args)` (the internal proposal distribution family) and evaluate the density for the `P` distribution at `x` and the density ratio `P(x; args) / Q(x; x_, args)`.
- **update**

   ```python
   GFI[X, R].update(args: tuple, trace: Trace[X, R], x_: X_) -> tuple[Trace[X, R], Weight, X_]
   ```

- **regenerate**

   ```python
   GFI[X, R].regenerate(args: tuple, trace: Trace[X, R], s: Sel) -> tuple[Trace[X, R], Weight, X_]
   ```

**Methods on the type `Trace`**

- **get_retval**

   ```python
   Trace[X, R].get_retval() -> R
   ```

   (**COMPUTATIONAL**) Get the return value for the execution that produced the trace.
- **get_gen_fn**

   ```python
   Trace[X, R].get_gen_fn() -> GFI[X, R]
   ```

  (**COMPUTATIONAL**) Get the GFI which the trace was created from.
- **get_args**

   ```python
   Trace[X, R].get_args() -> tuple
   ```

  (**COMPUTATIONAL**) Get the arguments for the execution which created the trace.
- **get_choices**

   ```python
   Trace[X, R].get_choices() -> X
   ```

  (**COMPUTATIONAL**) Get the traced random choices of the execution that produced the trace.
- **get_score**

   ```python
   Trace[X, R].get_score() -> Score
   ```

  (**COMPUTATIONAL**) Get the score of the execution.

### Generative Function Languages

GenJAX supports syntactic abstractions to make constructing generative functions convenient and concise.

#### The `@gen` Decorator: transforms JAX-compatible Python functions into generative functions

```python
import jax.numpy as jnp
from genjax import beta, flip, gen

# Example: beta-bernoulli model
@gen
def beta_ber():
    # define the hyperparameters that control the Beta prior
    alpha0 = jnp.array(10.0)
    beta0 = jnp.array(10.0)
    # sample f from the Beta prior
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip(f) @ "obs"
```

**`@gen` Language String Addressing System**

Within the `@gen` decorator, invocation of other generative functions are named using the `@` operator. The below example illustrates how this addressing works.

```python
from genjax import gen, normal

# Example: 1-level addressing
@gen
def simple_model():
   # Read: normal at "x"
   x = normal(0.0, 1.0) @ "x"  # "x" is the address

# Example: nested addressing
@gen
def nested_model():
   # Read: simple_model at "s"
   v = simple_model() @ "s" # "s" is the address

# Get a trace by using `nested_model.simulate`
trace = nested_model.simulate()

# The nesting stores the value at ("x", "s")
# (this will be the normal sample from within `simple_model`)
get_choices(trace)["x"]["s"]
```

Nesting can be applied to arbitrary depths.

#### Generative Function Combinators

**IMPORTANT**: Generative function combinators are higher-order generative functions that compose other generative functions (callees) to implement complex probabilistic computations. They implement the GFI by orchestrating calls to their callees' GFI methods.

**Key Combinator Design Principles**:

- **Composition over Implementation**: Combinators don't implement probabilistic primitives directly - they compose existing generative functions
- **GFI Preservation**: Combinators maintain the GFI contract while transforming execution semantics
- **JAX Transformation Integration**: Combinators in GenJAX leverage JAX transformations (scan, vmap, cond) while preserving probabilistic semantics

**Examples in GenJAX**:

- `Scan(callee)`: Sequential iteration using `jax.lax.scan`
- `Vmap(callee)`: Vectorization using `modular_vmap`
- `Cond(callee1, callee2)`: Conditional branching using `jax.lax.cond`

**Pattern**: `Combinator(callees...).method(args) -> orchestrates callee.method() calls`

### Programmable inference algorithms

## Codebase Architecture

1. **Core primitives** (`src/genjax/core.py`):
   - `GFI` (Generative Function Interface) - The base class for all generative functions. Contains the `abstractmethod` definitions of the GFI and their Python signatures.
   - `Trace` - Records samples, scores, and return values from generative functions.
   - `Pytree` - Base class for JAX `Pytree` compatibility. `Pytree` is a class whose implementors inform JAX how to break down its structure into lists of arrays, and zip back up into instances of the class. You can read more about this here: [Pytree](https://docs.jax.dev/en/latest/pytrees.html).

2. **Distributions** (`src/genjax/distributions.py`):
   - Implements the `GFI` (**distributions are generative functions too**).
   - Includes common distributions like bernoulli, beta, categorical, normal, etc, which are exposed by wrapping TensorFlow Probability distributions.

3. **ADEV** (`src/genjax/adev.py`):
   - JAX implementation of [ADEV: Sound Automatic Differentiation for Expected Values of Probabilistic Programs](https://dl.acm.org/doi/10.1145/3571198).
   - Provides gradient estimation strategies for expected value probabilistic programs. An expected value probabilistic program is a JAX-compatible Python function decorated with `@genjax.adev.expectation`.
   - Implements various samplers equipped with gradient estimators (REINFORCE, enumeration, etc).

4. **Programmable variational inference** (`src/genjax/vi.py`):
   - JAX implementation of [Probabilistic Programming with Programmable Variational Inference](https://dl.acm.org/doi/10.1145/3656463).

The codebase uses JAX extensively for automatic differentiation, vectorization, and JIT compilation, with extensions to support generative functions through custom primitive operations.

## Development Overview

### Code Development Constraints

**IMPORTANT - Documentation Policy**:

- **NEVER create documentation files (.md, README, etc.) unless explicitly requested by the user**
- Focus on implementation tasks and working code
- Existing documentation should only be modified when explicitly asked
- Prefer code comments and docstrings over separate documentation files

### Key workflows

1. (**core-dev**) Developing or making changes to core functionality.
   - Make changes to source code in `src/genjax/`
   - Run relevant tests to verify functionality
   - Update examples if adding new features
   - Ensure JAX transformations (`jit`, `vmap`) still work

2. (**example-dev**) Developing new examples using `GenJAX`.
   - Create a new folder under `examples`, for instance: `examples/foo`.
   - Create a new "core" Python file (for instance: `examples/foo/core.py`) which will contain the definitions of generative functions, and any inference functionality.
   - Create a new "figures" Python file (for instance: `examples/foo/figures.py`) which will contain plotting code.

### High-level Compiler Reference

- users author generative functions
- users apply inference algorithms to their generative functions, written using the GFI
- the algorithms lower to PJAX
- which eventually lowers to JAX.

```raw
GenJAX
├── Inference algorithms implemented using GFI: Monte Carlo, Variational, MCMC
├── Generative functions: @gen decorator + Python DSL syntax
├── PJAX: Probabilistic intermediate representation
└── JAX Backend: Vectorization, JIT compilation
```

### Directory Structure

```
genjax/
├── src/genjax/
│   ├── core.py           # PJAX, definition of GFI, Trace, Distribution
│   ├── distributions.py  # Various standard probability distributions, 
│   │                     # as generative functions.
│   ├── vi.py             # Programmable variational inference 
│   └── adev.py           # Automatic differentiation of expected values
├── examples/             # Tutorial notebooks and examples
└── quarto/               # Quarto website source
```

## Key Usage Workflows

### 1. Defining a generative function

```python
# Imports
from genjax import gen

@gen
def model():
    # Define prior distributions
    # Specify likelihood
    # Return observables
```

### 2. Basic Usage of the GFI

#### Methods on `GFI`

```python
# Sample a trace.
trace = model.simulate(args)

# Evaluate the density and return value given fixed
# random choices.
density, retval = model.assess(args, choices)

trace, weight = model.generate(args, choices)

# Updating a trace to be consistent with new random choices.
# The `discard` is any choices replaced.
new_trace, weight, discard = trace.update(new_args, new_choices)

# Updating a trace by asking a generative function to re-propose
# some of the random choices.
# The `discard` is any choices replaced.
new_trace, weight, discard = trace.update(new_args, new_choices)
```

#### Methods on `Trace`

```python
# Get the arguments of the trace.
args = trace.get_args()
```

### 3. Variational Inference

```python
from genjax import gen, beta, bernoulli, expectation
from genjax.vi import beta_implicit

# Define a model.
@gen
def beta_ber(alpha, beta):
   p = beta(1.0, 1.0) @ "p"
   _ = flip(p) @ "f"

# Define a variational guide.
@gen
def guide(alpha, beta):
   _ = beta_implicit(alpha, beta) @ "p"

# NOTE: model and guide should accept the same arguments.

# Either define your own objective:
@expectation
def objective(alpha, beta):
    # Define ELBO or other VI objective
    # using the GFI.

# Or use a standard library objective:
objective = ELBO(beta_ber, guide)

# Optimize with gradient estimation
init_params = (2.0, 2.0)
params_grad = objective.grad_estimate(*params)
```

## Design Patterns

### Programmable Monte Carlo Inference

#### High-level workflow

1. Define generative functions using distributions, the `@gen` language, and generative function combinators.
2. Use GFI methods for basic inference.
3. Implement custom algorithms using trace manipulation.

#### Generative Function Composition

- Within `@gen`, a generative function can call other generative functions by using the addressing syntax e.g. `other_gen_fn(*args) @ "addr"`
- Given a generative function, `.vmap()` can be used to create a `Vmap` generative function.
- Address namespacing prevents conflicts

#### Trace Manipulation

- Traces are the data _lingua franca_ of generative functions: they store samples of probabilistic choices, probabilistic data like scores (reciprocals of densities), the return value from executing the generative function.
- Traces are treated as immutable records of execution, including random choices and probabilistic data like scores.
- To mutate a trace, the interfaces `update()` and `regenerate()` are used: these interfaces create new traces with modifications.
- These interfaces are useful for expressing MCMC proposals and importance sampling algorithms, including sequential Monte Carlo.

#### Debugging

- Examine traces to understand model behavior
- Use `assess()` to check density computations
- Leverage JAX debugging tools (`jax.debug`) if you encounter NaN values or JAX runtime errors.

### Programmable variational inference

### Optimization of Expected Values

- Different samplers support different gradient estimators
- `normal_reparam`: Reparameterization gradients
- `normal_reinforce`: REINFORCE gradients
- `flip_enum`: Exact enumeration for discrete variables

## JAX Integration

### Transformations

- All GenJAX code is JAX-compatible. In general, you should only be writing code within "JAX Python", the subset of Python which is JAX compatible.
- Can use `jax.jit`, `jax.vmap`, `jax.grad` _on GFI interface invocations_, but not on generative functions (as instances of `GFI`) themselves.
- (**IMPORTANT**) GenJAX provides a custom version of `vmap` called `modular_vmap`. You should use this version when working with any code which uses generative function interface methods.

### GenJAX Constraints

**CRITICAL**: When writing code with GenJAX (inside `@gen` functions, combinators, etc.), you must follow JAX Python restrictions:

**Control Flow**:

- **NEVER use Python `if...else` statements** - these are not JAX-compatible
- **Use JAX control flow**: `jax.lax.cond`, `jax.lax.switch` for deterministic conditional logic, `jax.lax.scan` for deterministic iterative logic
- **Use GenJAX combinators**: `Cond(branch1, branch2)` for conditional generative functions, `Scan(callee)` for iterative generative functions
- **Example Pattern for Conditional Generative Functions**: Extract branches into separate `@gen` functions, then use combinators

**Example - WRONG**:

```python
@gen
def bad_conditional():
    x = normal(0.0, 1.0) @ "x"
    if x > 0:  # ❌ Python if statement
        y = exponential(1.0) @ "y"
    else:
        y = exponential(2.0) @ "y"
    return y
```

**Example - CORRECT**:

```python
@gen
def high_branch():
    return exponential(1.0) @ "y"

@gen  
def low_branch():
    return exponential(2.0) @ "y"

@gen
def good_conditional():
    x = normal(0.0, 1.0) @ "x"
    condition = x > 0
    cond_gf = Cond(high_branch, low_branch)
    result = cond_gf((condition,)) @ "cond"  # ✅ Use Cond combinator
    return result
```

### GenJAX API Patterns

**CRITICAL**: GenJAX generative functionshave specific API patterns that must be followed exactly:

**Generative Function Usage in `@gen` Functions**:

```python
# ✅ CORRECT: Use @ "address" syntax
x = normal(mu, sigma) @ "x"
y = exponential(rate) @ "y" 

# ❌ WRONG: Don't call methods directly
x = normal(mu, sigma)  # Won't be traced
```

**Generative Function Density Computations**:

```python
# ✅ CORRECT: Parameters in first argument as tuple
log_density, retval = normal.assess((mu, sigma), sample_value)
log_density, retval = exponential.assess((rate,), sample_value)

# ❌ WRONG: Parameters as constructor arguments  
log_density, retval = normal(mu, sigma).assess((), sample_value)  # Invalid
log_density, retval = exponential(rate).assess((), sample_value)  # Invalid
```

**Testing Pattern**:

```python
# ✅ CORRECT: For validating densities in tests
manual_log_density, _ = normal.assess((0.0, 1.0), sample)
fn_density, _ = my_gen_fn.assess(args, choices)
assert jnp.allclose(fn_density, manual_log_density)

# ❌ WRONG: Manual density formulas
manual_log_density = -(sample**2)/2.0 - 0.5*jnp.log(2*jnp.pi)  # Error-prone
```

### Vectorization and Pytree Usage

**CRITICAL**: All GenJAX datatypes inherit from `Pytree`, enabling automatic JAX vectorization. When implementing combinators or working with collections of traces/choices:

- **DO NOT use Python lists** for storing multiple instances of Pytree-inheriting types
- **DO use JAX's automatic vectorization** - JAX will vectorize the "leaves" of any Pytree instance via transformations like `jax.vmap` (**IMPORTANT**: prefer `genjax.modular_vmap`) and primitives like `jax.lax.scan`.
- **Example**: `jax.lax.scan` automatically handles vectorized `Trace` objects when passed in the `xs` argument
- **Pattern**: Instead of `[trace1, trace2, ...]`, use a single vectorized `Trace` that JAX creates automatically

**Key Principles**:

- Leverage JAX's built-in Pytree vectorization rather than manual list management
- Trust JAX transformations to handle vectorized probabilistic data structures
- Avoid overcomplicating implementations that should use JAX's automatic capabilities

### Vectorization Benefits

- Native support for batched operations
- Automatic vectorization of probabilistic programs
- Efficient vectorized sampling and inference

## Performance Considerations

### JIT Compilation

- Most operations can be JIT compiled
- Compilation happens lazily on first call
- Significant speedups for complex models

### Memory Management

- Traces contain full execution history
- Consider memory usage for large models
- Use JAX memory profiling tools

## Testing Strategy

### Unit Tests

- Test individual distributions and combinators
- Verify GFI method implementations

### Testing Generative Functions

**CRITICAL Testing Patterns**: Always validate probabilistic computations using these specific approaches:

**1. Density Validation Pattern**:

```python
@gen
def my_model():
    x = normal(0.0, 1.0) @ "x"
    y = exponential(x + 1.0) @ "y"
    return x + y

def test_my_model():
    trace = my_model.simulate(())
    choices = trace.get_choices()
    
   # ✅ CORRECT: Use Distribution.assess for access to distribution densities 
    # Validate using distribution densities
    x_density, _ = normal.assess((0.0, 1.0), choices["x"])
    y_density, _ = exponential.assess((choices["x"] + 1.0,), choices["y"])
    expected_total_density = x_density + y_density
    
    # Test assess
    actual_density, _ = my_model.assess((), choices)
    assert jnp.allclose(actual_density, expected_total_density)
    
    # Test simulate/assess consistency
    assert jnp.allclose(trace.get_score(), -actual_density)
```

**2. Nested Function Testing**:

```python
# Test nested @gen functions with proper choice extraction
def test_nested_functions():
    trace = outer_fn.simulate(())
    choices = trace.get_choices()
    
    # Access nested choices correctly
    x_choice = choices["x"]
    inner_result = choices["inner_fn_address"]  # Result from inner function
    
    # Validate each level separately
    inner_density, _ = inner_fn.assess(inner_args, inner_choices)
    outer_density, _ = outer_fn.assess(args, choices)
```

### Testing Generative Function Combinators

**Key Testing Approach**: Compare combinator implementations against manual JAX implementations to validate correctness.

**Combinator Testing Pattern**:

```python
# Test Scan against manual jax.lax.scan implementation
def test_scan_vs_manual():
    scan_gf = Scan(my_step_function)
    trace = scan_gf.simulate(args)
    
    # Manual implementation using jax.lax.scan
    def manual_scan_fn(carry, input_and_choice):
        input_val, choice = input_and_choice
        # Replicate same logic as my_step_function
        density, _ = distribution.assess(params, choice)
        return new_carry, (output, density)
    
    manual_carry, (manual_outputs, manual_densities) = jax.lax.scan(
        manual_scan_fn, init_carry, (inputs, choices)
    )
    
    # Compare results
    assert jnp.allclose(trace.get_score(), -jnp.sum(manual_densities))
```

**For density computations**:

- Test `simulate()`: Compare `trace.get_score()` (reciprocal density) against density computations using the `Distribution.logpdf` method
- Test `assess()`: Compare combinator density against using the same probabilistic logic
- Test consistency: Verify `simulate_score = -assess_density` for same choices

**Key Principles**:

- **Use `Distribution.assess()`** instead of manual density formulas
- **Test consistency** between `simulate` and `assess` methods  
- **Validate combinators** against equivalent JAX implementations
- **Handle nested addressing** properly in choice extraction

### Integration Tests

- End-to-end model fitting
- Inference algorithm correctness
- JAX transformation compatibility

### Performance Tests

- Benchmark against other PPLs
- Memory usage profiling
- JIT compilation overhead

## Sharp Edges & Limitations of GenJAX

### Known Issues

- `vmap` within ADEV `@expectation` programs has undefined semantics
- Not all gradient estimation strategies support batching
- Some JAX transformations may not compose cleanly

### Debugging Tips

- Start with simple models and build complexity gradually
- Use `jax.debug.print()` for debugging inside JIT-compiled code
- Check trace consistency with `assess()` calls

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: stands for _Probabilistic_ JAX, an extension of JAX with custom primitives (`sample_p`, `log_density_p`) which GenJAX uses as an intermediate representation of its probabilistic computations.
- **ADEV**: Automatic Differentiation of Expected Values
- **Trace**: Recording of the execution of a generative function.
- **Address**: String identifier for random variables (e.g., `"x"`, `"alpha"`, etc)
- **Score**: Negative log probability (log of the reciprocal `1/P(choices)`)
- **Programmable Inference**: Ability to customize inference algorithms

## Further Reading

- [GenJAX Documentation](https://probcomp.github.io/genjax/)
- [ADEV Paper](https://dl.acm.org/doi/10.1145/3571198) - Automatic differentiation of expected values
- [Programmable VI Paper](https://dl.acm.org/doi/10.1145/3656463) - Programmable variational inference
- [Gen Paper](https://dl.acm.org/doi/10.1145/3314221.3314642) - Original Gen system (Julia version)

---

_This codebase implements cutting-edge research in vectorized probabilistic programming. The design prioritizes flexibility and performance while maintaining mathematical correctness._
