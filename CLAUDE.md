# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Commands

The codebase is managed through [`pixi`](https://pixi.sh/latest/).

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

GenJAX is a probabilistic programming language embedded in Python.  A probabilistic program language (PPL) is a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation for expected values, and more.

The design of GenJAX is centered on programmable inference: automation which allows users of GenJAX to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions, answering questions like "`x` probabilistically affects `y`, and I observe `y`: what are my new beliefs about `x`?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

## Key concepts

### 1. Datatypes: Generative Functions & Traces

- **Generative Function**: A probabilistic program that implements the Generative Function Interface (GFI)
- **Trace**: A recording of execution containing random choices sampled durng the execution, the arguments of the execution, the return value of the execution, and the score of the execution (the score is $\log \frac{1}{P(\text{random choices})}$, the reciprocal of the density in logspace).

### 2. Generative Function Interface

The generative function interface, or GFI consists of a set of interface methods for working with generative functions and traces.

Below, we enumerate the list of interfaces, with their signatures. We use GenJAX Python types: `X` denotes the type of "random samples" and `R` denotes the type of "return values".

**Methods on the type `GFI`**

- `GFI[X, R].simulate(args: tuple) -> Trace[X, R]` (sample a trace from the prior)
- `GFI[X, R].assess(args: tuple, x: X) -> tuple[Density, R]` (evaluate the density at the sample `x: X` and)
- `GFI[X, R].generate(args: tuple, x: X) -> tuple[Trace[X, R], Weight]`
- `GFI[X, R].update(args: tuple, trace: Trace[X, R], x: X))`
- `GFI[X, R].regenerate(args: tuple, trace: Trace[X, R], s: Sel)`

**Methods on the type `Trace`**

- `Trace[X, R].get_retval() -> R` (Get the return value for the execution that produced the trace)
- `Trace[X, R].get_gen_fn() -> GFI[X, R]` (Get the GFI which the trace was created from)
- `Trace[X, R].get_args() -> tuple` (Get the arguments for the execution which created the trace)
- `Trace[X, R].get_score() -> Score` (Get the score of the execution)
- `Trace[X, R].get_choices() -> X` (Get the traced random choices of the execution that produced the trace)

### 3. Generative Function Languages

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

#### Generative function combinators

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

1. Make changes to source code in `src/genjax/`
2. Run relevant tests to verify functionality
3. Update examples if adding new features
4. Ensure JAX transformations (`jit`, `vmap`) still work

### Compiler Overview

This is a high-level sketch of how GenJAX (thought of as a compiler) works:

- users author generative functions
- users apply inference algorithms to their generative functions, written using the GFI
- the algorithms lower to PJAX
- which eventually lowers to JAX.

```raw
GenJAX
├── Generative functions: @gen decorator + Python DSL syntax
├── Inference algorithms implemented using GFI: Monte Carlo, Variational, MCMC
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

## Key Workflows

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

### 2. GFI Basic Usage

```python
# Sampling a trace.
trace = model.simulate(args)

# Evaluating the density and return value given fixed
# random choices.
density, retval = model.assess(args, choices)

# Updating a trace to be consistent with new random choices.
new_trace, weight, _ = trace.update(new_args, new_choices)
```

### 3. Variational Inference

```python
@expectation
def objective(params):
    # Define ELBO or other VI objective
    
# Optimize with gradient estimation
grad = objective.grad_estimate(params)
```

## Design Patterns

### Generative Function Composition

- Within `@gen`, a generative function can call other generative functions by using the addressing syntax e.g. `other_gen_fn(*args) @ "addr"`
- Given a generative function, `.vmap()` can be used to create a `Vmap` generative function.
- Address namespacing prevents conflicts

### Trace Manipulation

- Traces are the data _lingua franca_ of generative functions: they store samples of probabilistic choices, probabilistic data like scores (reciprocals of densities), the return value from executing the generative function.
- Traces are treated as immutable records of execution, including random choices and probabilistic data like scores.
- To mutate a trace, the interfaces `update()` and `regenerate()` are used: these interfaces create new traces with modifications.
- These interfaces are useful for expressing MCMC proposals and importance sampling algorithms, including sequential Monte Carlo.

### Model + Inference

1. Define generative model with `@gen`
2. Use GFI methods for basic inference
3. Implement custom algorithms using trace manipulation
4. Optimize with ADEV for variational approaches

### Debugging

- Examine traces to understand model behavior
- Use `assess()` to check density computations
- Leverage JAX debugging tools (jax.debug)

### Gradient Estimation Strategies

- Different samplers support different gradient estimators
- `normal_reparam`: Reparameterization gradients
- `normal_reinforce`: REINFORCE gradients
- `flip_enum`: Exact enumeration for discrete variables

## JAX Integration

### Transformations

- All GenJAX code is JAX-compatible. In general, you should only be writing code within "JAX Python", the subset of Python which is JAX compatible.
- Can use `jax.jit`, `jax.vmap`, `jax.grad` _on GFI interface invocations_, but not on generative functions (as instances of `GFI`) themselves.
- GenJAX provides a custom version of `vmap` called `modular_vmap` In general, you should use this version when working with probabilistic code.

### Vectorization

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

## Sharp Edges & Limitations

### Known Issues

- `vmap` within ADEV `@expectation` programs has undefined semantics
- Not all gradient estimation strategies support batching
- Some JAX transformations may not compose cleanly

### Debugging Tips

- Start with simple models and build complexity gradually
- Use `jax.debug.print()` for debugging inside JIT-compiled code
- Check trace consistency with `assess()` calls

## Testing Strategy

### Unit Tests

- Test individual distributions and combinators
- Verify GFI method implementations
- Check gradient estimation accuracy

### Integration Tests

- End-to-end model fitting
- Inference algorithm correctness
- JAX transformation compatibility

### Performance Tests

- Benchmark against other PPLs
- Memory usage profiling
- JIT compilation overhead

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: stands for _Probabilistic_ JAX, an extension of JAX with custom primitives (`sample_p`, `log_density_p`) which GenJAX uses as an intermediate representation of its probabilistic computations.
- **ADEV**: Automatic Differentiation of Expected Values
- **Trace**: Recording of the execution of a generative function.
- **Address**: String identifier for random variables (e.g., `"x"`, `"alpha"`)
- **Score**: Negative log probability (log of the reciprocal `1/P(choices)`)
- **Programmable Inference**: Ability to customize inference algorithms

## Further Reading

- [GenJAX Documentation](https://probcomp.github.io/genjax/)
- [ADEV Paper](https://dl.acm.org/doi/10.1145/3571198) - Automatic differentiation of expected values
- [Programmable VI Paper](https://dl.acm.org/doi/10.1145/3656463) - Programmable variational inference
- [Gen Paper](https://dl.acm.org/doi/10.1145/3314221.3314642) - Original Gen system (Julia version)

---

_This codebase implements cutting-edge research in vectorized probabilistic programming. The design prioritizes flexibility and performance while maintaining mathematical correctness._
