# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GenJAX is a probabilistic programming language embedded in Python.  

A probabilistic program language (PPL) is a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation of expected values, and more.

The design of GenJAX is centered on programmable inference: automation which allows users of GenJAX to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions, answering questions like "`x` probabilistically affects `y`, and I observe `y`: what are my new beliefs about `x`?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

## Development Constraints

### Documentation Policy

**CRITICAL - Documentation Policy**:

- **NEVER create documentation files (.md, README, etc.) unless explicitly requested by the user**
- **NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User**
- Focus on implementation tasks and working code
- Existing documentation should only be modified when explicitly asked
- Prefer code comments and docstrings over separate documentation files

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

#### Distributions: Probabilistic Building Blocks

**IMPORTANT**: Distributions are generative functions that implement a `sample` and `logpdf` interface (the interface of probability distributions). They are the fundamental building blocks of probabilistic programs in GenJAX.

**Key Concepts**:

- **Distributions are Generative Functions**: Every distribution implements the GFI (simulate, assess, generate, update, regenerate)
- **Wrapper for TensorFlow Probability distributions**: Most distributions wrap TensorFlow Probability distributions for robustness and reliability
- **Parameter Convention**: Distribution parameters are passed as arguments to the GFI methods, not as constructor arguments

**Common Distributions**:

```python
from genjax import normal, beta, categorical, exponential, uniform, bernoulli

# Continuous distributions
x = normal(mu, sigma) @ "x"              # Normal(mu, sigma)
y = beta(alpha, beta_param) @ "y"        # Beta(alpha, beta)
z = exponential(rate) @ "z"              # Exponential(rate)
w = uniform(low, high) @ "w"             # Uniform(low, high)

# Discrete distributions
coin = bernoulli(p) @ "coin"             # Bernoulli(p)
choice = categorical(logits) @ "choice"  # Categorical(logits)
```

**Distribution Usage Patterns**:

```python
# ✅ CORRECT: Use distributions within @gen functions with addressing
@gen
def my_model():
    mu = normal(0.0, 1.0) @ "mu"
    sigma = exponential(1.0) @ "sigma"
    obs = normal(mu, sigma) @ "obs"
    return obs

# ✅ CORRECT: Direct simulation and assessment
trace = normal.simulate((0.0, 1.0))
sample = trace.get_retval()
log_density, _ = normal.assess((0.0, 1.0), sample)

# ❌ WRONG: Don't instantiate distributions as objects
dist = normal(0.0, 1.0)  # This doesn't create a usable distribution
```

**Parameter Passing Convention**:

```python
# ✅ CORRECT: Parameters as tuple in GFI methods
log_density, value = normal.assess((mu, sigma), sample)
trace = exponential.simulate((rate,))

# ❌ WRONG: Parameters as constructor arguments
log_density, value = normal(mu, sigma).assess((), sample)  # Invalid API
```

**Vectorization with Distributions**:

```python
# Vectorize over parameters
mus = jnp.array([0.0, 1.0, 2.0])
vectorized_normal = normal.vmap(in_axes=(0, None))
traces = vectorized_normal.simulate((mus, 1.0))

# Vectorize over samples (using repeat)
batch_normal = normal.repeat(10)  # 10 independent samples
traces = batch_normal.simulate((0.0, 1.0))
```

**Custom Distributions**:

```python
from genjax import distribution, tfp_distribution
import tensorflow_probability.substrates.jax as tfp

# Method 1: Using tfp_distribution wrapper
student_t = tfp_distribution(
    lambda df, loc, scale: tfp.distributions.StudentT(df, loc, scale),
    name="student_t"
)

# Method 2: Custom distribution from scratch
def custom_sampler(key, a, b):
    # Custom sampling logic
    return a + b * jax.random.normal(key)

def custom_logpdf(x, a, b):
    # Custom log probability density
    return -0.5 * ((x - a) / b)**2 - jnp.log(b)

custom_dist = distribution(custom_sampler, custom_logpdf, name="custom")

# Usage in @gen functions
@gen
def model_with_custom():
    x = student_t(3.0, 0.0, 1.0) @ "x"
    y = custom_dist(0.0, 1.0) @ "y"
    return x + y
```

#### `Fn` Generative Functions

**IMPORTANT**: The `@gen` decorator is GenJAX's primary mechanism for creating `Fn` generative functions from JAX-compatible Python programs. `Fn` is the core generative function type that implements complex probabilistic models through function composition.

**What `@gen` Does**:

- **Transforms JAX Python Functions**: Converts ordinary JAX-compatible Python functions into `Fn` generative functions that implement the full GFI
- **Enables Probabilistic Addressing**: Adds the `@` operator for addressing random choices made by distributions and other generative functions
- **GFI Methods Implemented Using Execution Handlers**: Implements the different GFI methods (simulate, assess, generate, update, regenerate) through handler stacks
- **Preserves JAX Semantics**: Maintains compatibility with JAX transformations while adding probabilistic capabilities

##### `@gen` Decorator: Transform JAX-Compatible Python Functions into `Fn`

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

# The @gen decorator creates an Fn generative function
print(type(beta_ber))  # <class 'genjax.core.Fn'>
```

**How `Fn` Works Internally**:

```python
# When you call a @gen function, it uses handler stacks to intercept @ calls
@gen
def hierarchical_model(N):
    # Handler intercepts these @ calls and manages the trace
    mu = normal(0.0, 1.0) @ "mu"
    sigma = exponential(1.0) @ "sigma"
    obs = normal.repeat(n=10)(mu, sigma) @ "obs"
    return obs

# Behind the scenes, when you call a GFI method on the Fn (e.g., hierarchical_model.simulate):
# 1. Pushes a handler onto the handler stack
# 2. Executes the function body
# 3. Intercepts @ calls to invoke the GFI method on the callee 
# 4. Records state into the handler
# 5. Pops the handler and returns state associated with the GFI method
```

**`@gen` Language: String Addressing System**

Within `@gen` functions, the `@` operator creates addressed random choices. This addressing system enables trace manipulation and inference algorithms.

```python
from genjax import gen, normal

# Example: 1-level addressing
@gen
def simple_model():
   # The @ operator creates an addressed random choice
   x = normal(0.0, 1.0) @ "x"  # "x" is the address
   return x

# Example: nested addressing through function composition
@gen
def nested_model():
   # Calling another @gen function creates nested addressing
   v = simple_model() @ "s" # "s" is the address for the sub-model

# Get a trace by using the Fn's simulate method
trace = nested_model.simulate(())

# The nesting creates hierarchical addressing
# Access: choices["address_in_outer"]["address_in_inner"]
choices = get_choices(trace)
inner_x = choices["s"]  # This gets the sample from simple_model, a choice map
# For the actual value: choices["s"]["x"] gives the value
```

**Advanced `Fn` Patterns**:

```python
# CRITICAL: Prioritize never using Python for loops in @gen functions
# Use `jax.vmap` for deterministic vectorization
# Use Vmap combinator for generative vectorization
# Use `jax.lax.scan` for deterministic iteration
# Use Scan combinator for any generative iteration

@gen
def step_function(carry, x):
    """Single step for time series model."""
    prev_state, transition_noise, observation_noise = carry
    
    # State transition
    state = normal(prev_state, transition_noise) @ "state"
    
    # Observation (x is unused input)
    obs = normal(state, observation_noise) @ "obs"
    
    # New carry
    new_carry = (state, transition_noise, observation_noise)
    
    return new_carry, (state, obs)

@gen
def time_series_model(T):
    # Priors
    initial_state = normal(0.0, 1.0) @ "initial_state"
    transition_noise = exponential(1.0) @ "transition_noise"
    observation_noise = exponential(1.0) @ "observation_noise"
    
    # Initial observation
    initial_obs = normal(initial_state, observation_noise) @ "obs_0"
    
    if T == 1:
        return jnp.array([initial_state]), jnp.array([initial_obs])
    
    # Use Scan for remaining steps
    scan_fn = Scan(step_function, length=T-1)
    init_carry = (initial_state, transition_noise, observation_noise)
    
    final_carry, (states, observations) = scan_fn((init_carry, None)) @ "time_steps"
    
    # Combine initial and remaining
    all_states = jnp.concatenate([jnp.array([initial_state]), states])
    all_obs = jnp.concatenate([jnp.array([initial_obs]), observations])
    
    return all_states, all_obs

# ❌ WRONG: Never use Python for loops
@gen
def bad_time_series(T):
    states = []
    for t in range(T):  # This will typically break JAX compilation!
        state = normal(0.0, 1.0) @ f"state_{t}"
        states.append(state)
    return states

# ✅ CORRECT: Always use Scan for iteration
# (See time_series_model above)
```

**`Fn` Trace Structure**:

```python
# Fn traces have choice_type X = dict[str, Any]
trace = time_series_model.simulate((5,))  # 5 time steps
choices = get_choices(trace)

# Choices is a dictionary with string keys
initial = choices["initial_state"]      # Float
transition = choices["transition_noise"] # Float
states = [choices["time_steps"]["state"][t] for t in range(1, 5)]  # List of floats
observations = [choices["time_steps"]["obs"][t] for t in range(1, 5)]  # List of floats
```

**JAX Compatibility Requirements for `@gen`**:

```python
# ✅ CORRECT: JAX-compatible patterns in @gen
@gen
def good_model(data):
    # Simple computations without iteration
    mu = normal(0.0, 1.0) @ "mu"
    sigma = jnp.exp(normal(0.0, 1.0)) @ "log_sigma"  # Ensure positivity
    
    # Single observations
    obs = normal(mu, sigma) @ "obs"
    
    return obs

# ✅ CORRECT: Use combinators for any iteration
@gen
def observation_step(carry, data_point):
    mu, sigma = carry
    obs = normal(mu + data_point, sigma) @ "obs"
    return (mu, sigma), obs

@gen
def good_model_with_iteration(data):
    mu = normal(0.0, 1.0) @ "mu"
    sigma = jnp.exp(normal(0.0, 1.0)) @ "log_sigma"
    
    # Use Scan for iteration over data
    scan_fn = Scan(observation_step, length=len(data))
    init_carry = (mu, sigma)
    final_carry, observations = scan_fn((init_carry, data)) @ "observations"
    
    return observations

# ❌ WRONG: Python control flow patterns
@gen
def bad_model(data):
    results = []  # Python list - wrong data structure
    
    for x in data:  # Python for loop - typically breaks JAX compilation!
        if x > 0:  # Python if statement - not JAX compatible
            y = normal(x, 1.0) @ f"obs_{x}"  # Dynamic addressing - wrong!
            results.append(y)
    
    return results  # Returns Python list, not JAX array
```

**Key Principles**:

- **Static Structure**: The addressing structure must be statically determinable
- **JAX Compatibility**: Use JAX operations, not Python operations for dynamic computation
- **Deterministic Execution**: The function should execute the same sequence of @ calls given the same arguments (before randomness)

Nesting can be applied to arbitrary depths, creating complex hierarchical models while maintaining the benefits of JAX transformations.

#### Generative Function Combinators

**IMPORTANT**: Generative function combinators are higher-order generative functions that compose other generative functions (callees) to implement complex probabilistic computations. They implement the GFI by orchestrating calls to their callees' GFI methods.

**Key Combinator Design Principles**:

- **Composition over Implementation**: Combinators don't implement probabilistic primitives directly - they compose existing generative functions
- **GFI Preservation**: Combinators maintain the GFI contract while transforming execution semantics
- **JAX Transformation Integration**: Combinators in GenJAX leverage JAX transformations (scan, vmap, cond) while preserving probabilistic semantics

**Examples in GenJAX**:

- `Scan`: Sequential iteration for generative functions akin to `jax.lax.scan`
- `Vmap`: Vectorization for generative functions using `genjax.modular_vmap`
- `Cond`: Conditional branching for generative functions akin to `jax.lax.cond`

**Pattern**: `Combinator(*callees, **kwargs).method(args) -> orchestrates callee.method() calls`

##### Scan Combinator Usage

**CRITICAL**: The `Scan` combinator has specific addressing and input requirements that must be followed exactly:

**Static Addressing Only**:

```python
# ❌ WRONG: Dynamic addressing in scan
@gen
def bad_scan_step(carry, t):
    x = normal(0.0, 1.0) @ f"x_{t}"  # Dynamic address - will break!
    return carry + x, x

# ✅ CORRECT: Static addressing in scan
@gen
def good_scan_step(carry, x_input):
    x = normal(0.0, 1.0) @ "x"  # Static address - will be vectorized
    return carry + x, x
```

**Scan Input Requirements**:

```python
# Use None for unused scan inputs
scan_fn = Scan(step_function, length=10)
result = scan_fn((init_carry, None)) @ "scan_result"

# Or use meaningful inputs if needed by the step function
scan_fn = Scan(step_function, length=len(data))
result = scan_fn((init_carry, data)) @ "scan_result"
```

**Trace Structure and Choice Extraction**:

```python
trace = scan_fn.simulate((init_carry, inputs))
choices = get_choices(trace)

# Vectorized choices from scan are nested
scan_choices = choices["scan_result"]
vectorized_x = scan_choices["x"]  # Array of all x values from scan steps

# For complex nested scans
nested_scan_choices = choices["outer_scan"]["inner_scan"]["variable"]
```

**Why These Constraints Exist**:

- Scan automatically vectorizes choices under static addresses
- Dynamic addressing breaks JAX's compilation and vectorization
- The scan combinator needs to know the trace structure statically

**Combinator Debugging Patterns**:

**Static vs Dynamic Addressing Issues**:

```python
# ❌ CRITICAL ERROR: This will break during vectorization
@gen
def broken_step(carry, t):
    # Problem: t is a static value that cannot be passed into scan
    state = normal(carry, 1.0) @ f"state_{t}"  # Dynamic address breaks vectorization
    return state, state

# ✅ CORRECT: Static addresses work with vectorization
@gen
def working_step(carry, x):
    # x can be None if unused, or actual scan input
    state = normal(carry, 1.0) @ "state"  # Static address - gets vectorized
    obs = normal(state, 0.1) @ "obs"     # Static address - gets vectorized
    return state, (state, obs)
```

**Trace Extraction from Vectorized Operations**:

```python
# Proper extraction of choices from Scan results
trace = hmm_model.simulate(args)
choices = get_choices(trace)

# Initial state/obs (not vectorized)
initial_state = choices["state_0"]
initial_obs = choices["obs_0"]

# Vectorized states/obs from scan
if "scan_steps" in choices:
    scan_choices = choices["scan_steps"]
    scan_states = scan_choices["state"]  # Array of states from scan steps
    scan_obs = scan_choices["obs"]       # Array of observations from scan steps
    
    # Combine initial + scan results
    all_states = jnp.concatenate([jnp.array([initial_state]), scan_states])
    all_obs = jnp.concatenate([jnp.array([initial_obs]), scan_obs])
```

**Common Scan Input Mistakes**:

```python
# ❌ WRONG: Using dummy values for unused inputs
scan_fn = Scan(step_function, length=T-1)
result = scan_fn((init_carry, jnp.zeros(T-1))) @ "steps"  # Unnecessary dummy array

# ✅ CORRECT: Use None for unused inputs
scan_fn = Scan(step_function, length=T-1)
result = scan_fn((init_carry, None)) @ "steps"  # Clean and efficient
```

##### Vmap Combinator Usage

**IMPORTANT**: The `Vmap` combinator vectorizes generative functions across batch dimensions while preserving probabilistic semantics.

**Basic Vectorization Patterns**:

```python
# Vectorize a distribution over multiple parameters
vectorized_normal = normal.vmap(in_axes=(0, None))  # vectorize over first parameter
mus = jnp.array([0.0, 1.0, 2.0])
traces = vectorized_normal.simulate((mus, 1.0))  # sigma=1.0 broadcast to all

# Vectorize a generative function
@gen
def single_observation(mu):
    return normal(mu, 1.0) @ "obs"

# Create vectorized version
batch_observations = Vmap(single_observation)
vectorized_traces = batch_observations.simulate((mus,))
```

**Advanced Vmap Patterns**:

```python
# Multiple parameter vectorization
@gen
def regression_point(x, slope, intercept):
    noise = normal(0.0, 0.1) @ "noise"
    y = slope * x + intercept + noise
    return y

# Vectorize over data points (x values)
x_data = jnp.array([1.0, 2.0, 3.0, 4.0])
vectorized_regression = Vmap(regression_point, in_axes=(0, None, None))
traces = vectorized_regression.simulate((x_data, 2.0, 1.0))
```

**Vmap In-Axes Specification**:

```python
# in_axes controls which arguments get vectorized
# 0 = vectorize over first dimension
# None = broadcast (don't vectorize)
# (0, None, 1) = vectorize arg1 over dim 0, broadcast arg2, vectorize arg3 over dim 1

vectorized_fn = Vmap(my_function, in_axes=(0, None, 0))
```

**Repeat and Axis Size**:

```python
# .repeat() - Convenient method for independent sampling
@gen
def single_sample():
    return normal(0.0, 1.0) @ "x"

# Generate 10 independent samples (no input vectorization)
batch_sampler = single_sample.repeat(10)
traces = batch_sampler.simulate(())  # Empty args since no inputs needed
choices = get_choices(traces)
samples = choices["x"]  # Array of 10 independent normal samples

# Equivalent to:
manual_vmap = Vmap(single_sample, in_axes=None, axis_size=10)
traces = manual_vmap.simulate(())

# axis_size parameter - specify batch size when not inferrable from inputs
@gen
def parameterized_model(mu, sigma):
    return normal(mu, sigma) @ "sample"

# When using axis_size, all arguments must be broadcast
fixed_params_vmap = Vmap(
    parameterized_model, 
    in_axes=(None, None),  # Broadcast both parameters
    axis_size=5            # Generate 5 samples with same parameters
)
traces = fixed_params_vmap.simulate((0.0, 1.0))
choices = get_choices(traces)
samples = choices["sample"]  # Array of 5 independent samples with mu=0.0, sigma=1.0

# axis_size with mixed vectorization
mixed_vmap = Vmap(
    parameterized_model,
    in_axes=(0, None),     # Vectorize mu, broadcast sigma
    axis_size=3            # Must match length of vectorized inputs
)
mus = jnp.array([0.0, 1.0, 2.0])  # Length must equal axis_size
traces = mixed_vmap.simulate((mus, 1.0))
```

**Key Points**:

- **`.repeat(n)`**: Shorthand for `Vmap(gen_fn, in_axes=None, axis_size=n)` - generates n independent samples
- **`axis_size`**: Explicitly specifies batch dimension size when not inferrable from inputs
- **Consistency requirement**: When using `axis_size` with vectorized inputs, array lengths must match `axis_size`
- **Broadcasting**: Use `in_axes=None` with `axis_size` to broadcast the same parameters across all samples

##### Cond Combinator Usage

**IMPORTANT**: The `Cond` combinator implements conditional branching in generative functions using JAX-compatible control flow.

**Basic Conditional Pattern**:

```python
@gen
def positive_branch():
    return exponential(1.0) @ "value"

@gen  
def negative_branch():
    return exponential(2.0) @ "value"

@gen
def conditional_model():
    x = normal(0.0, 1.0) @ "x"
    condition = x > 0
    cond_fn = Cond(positive_branch, negative_branch)
    result = cond_fn((condition,)) @ "conditional"
    return result
```

**Parameterized Branches**:

```python
@gen
def high_noise_branch(mu):
    return normal(mu, 2.0) @ "obs"

@gen
def low_noise_branch(mu):
    return normal(mu, 0.1) @ "obs"

@gen
def adaptive_noise_model(data_quality):
    mu = normal(0.0, 1.0) @ "mu"
    use_high_noise = data_quality < 0.5
    
    cond_fn = Cond(high_noise_branch, low_noise_branch)
    observation = cond_fn((use_high_noise, mu)) @ "obs"
    return observation
```

**Key Cond Constraints**:

- Both branches are always evaluated during simulation (JAX requirement)
- Branches must have compatible return types
- Use same addressing within branches for proper trace merging
- Condition must be a JAX-compatible boolean expression

##### Combinator Trace Structures

**Understanding Combinator Operations**: Combinators operate on their callees' trace structures in specific ways - they don't create new nested structures, but transform existing ones.

**Scan: Vectorizes Leaves, Preserves Addressing**:

```python
# Callee trace structure (single step):
# {"state": scalar, "obs": scalar}

# Scan vectorizes the leaves while keeping addresses unchanged
@gen
def step_fn(carry, x):
    state = normal(carry, 1.0) @ "state"
    obs = normal(state, 0.1) @ "obs"
    return state, obs

scan_fn = Scan(step_fn, length=5)
trace = scan_fn.simulate(args)
choices = get_choices(trace)

# Scan result: addresses stay the same, values become vectors
scan_results = choices["scan_steps"]
all_states = scan_results["state"]     # Array: [state_1, state_2, state_3, state_4, state_5]
all_observations = scan_results["obs"] # Array: [obs_1, obs_2, obs_3, obs_4, obs_5]
```

**Vmap: Vectorizes Leaves, Preserves Addressing**:

```python
# Callee trace structure (single instance):
# {"x": scalar, "y": scalar}

@gen
def single_model():
    x = normal(0.0, 1.0) @ "x"
    y = exponential(x + 1.0) @ "y"
    return y

vmap_fn = Vmap(single_model)
vectorized_trace = vmap_fn.simulate(args)
vectorized_choices = get_choices(vectorized_trace)

# Vmap result: addresses stay the same, values become vectors
vectorized_x = vectorized_choices["x"]  # Array: [x_1, x_2, x_3, ...]
vectorized_y = vectorized_choices["y"]  # Array: [y_1, y_2, y_3, ...]
```

**Cond: Combines Trace Structures**:

```python
# Branch A trace structure: {"value": scalar}
# Branch B trace structure: {"value": scalar}

@gen
def branch_a():
    return exponential(1.0) @ "value"

@gen  
def branch_b():
    return exponential(2.0) @ "value"

cond_fn = Cond(branch_a, branch_b)
cond_trace = cond_fn.simulate(args)
cond_choices = get_choices(cond_trace)

# Cond result: combines both branch structures, selects based on condition
# Addresses from both branches are present, but values reflect the selected branch
selected_value = cond_choices["value"]  # Value from whichever branch was selected
```

**Key Principles**:

- **Scan/Vmap**: Keep callee addressing unchanged, vectorize the leaf values
- **Cond**: Merge callee trace structures, use condition to select appropriate values
- **Nested Combinators**: Composition follows the same rules recursively

## Programmable inference

## GenJAX API Patterns

**CRITICAL**: GenJAX generative functions specific API patterns that must be followed exactly:

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

## JAX Integration

### JAX Basics

**IMPORTANT**: GenJAX is built on top of JAX, so understanding JAX fundamentals is essential for effective GenJAX usage. JAX is a Python library for high-performance machine learning research that provides NumPy-compatible operations with automatic differentiation and JIT compilation.

**Core JAX Concepts**:

**JAX Arrays and Operations**:

```python
import jax.numpy as jnp
import jax

# JAX arrays (similar to NumPy, but immutable)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([[1.0, 2.0], [3.0, 4.0]])

# JAX operations (similar to NumPy)
result = jnp.sum(x)
matrix_mult = jnp.dot(y, x[:2])
broadcasted = x + 10.0

# JAX arrays are immutable - operations return new arrays
x_modified = x.at[0].set(5.0)  # Functional update, x unchanged
```

**JAX Python Restrictions**:

```python
# ✅ CORRECT: JAX-compatible patterns
def jax_compatible_function(x):
    # Use JAX control flow
    result = jax.lax.cond(
        x > 0,
        lambda x: x * 2,      # true branch
        lambda x: x * -1      # false branch
    )
    
    # Use JAX operations
    return jnp.exp(result) + jnp.log(jnp.abs(x) + 1e-8)

# ❌ WRONG: Non-JAX patterns that break compilation
def non_jax_function(x):
    if x > 0:  # Python if - not typically JAX compatible
        result = x * 2
    else:
        result = x * -1
    
    # Python list operations - not JAX compatible
    results = []
    for i in range(len(x)):  # Python for loop with dynamic length
        results.append(x[i] * 2)
    
    return results  # Python list, not JAX array
```

**JAX Random Number Generation**:

```python
import jax.random as jrand

# JAX uses explicit PRNG keys (no global state)
key = jrand.key(42)

# Split keys for independent randomness
key, subkey = jrand.split(key)
sample = jrand.normal(subkey, shape=(10,))

# Multiple splits
key, *subkeys = jrand.split(key, 4)  # Creates 3 subkeys
samples = [jrand.normal(sk, shape=(5,)) for sk in subkeys]
```

**Note on GenJAX Distributions**: When using GenJAX distributions, you typically don't need to manage PRNG keys manually. GenJAX handles key management internally through the `seed` transformation and PJAX primitives:

```python
from genjax import normal, exponential

# ✅ PREFERRED: Use GenJAX distributions (key management handled internally)
trace = normal.simulate((0.0, 1.0))  # No key needed
sample = trace.get_retval()

# Direct sampling from distributions (for testing/debugging/low-level sampling algorithms)
sample = normal.sample(0.0, 1.0)  # GenJAX manages keys internally

# Only use JAX random directly for non-probabilistic operations
# or when implementing custom distributions
```

**JAX Transformations Overview**:

```python
# jit - Just-In-Time compilation for speed
@jax.jit
def fast_function(x):
    return jnp.sum(x**2) + jnp.exp(x).mean()

# grad - Automatic differentiation
def loss_function(params, data):
    return jnp.sum((params * data)**2)

grad_fn = jax.grad(loss_function)  # Gradient w.r.t. first argument
gradients = grad_fn(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

# vmap - Vectorization
def process_single(x):
    return x**2 + jnp.sin(x)

# Vectorize over batch dimension
vectorized_process = jax.vmap(process_single)
batch_data = jnp.array([1.0, 2.0, 3.0, 4.0])
batch_results = vectorized_process(batch_data)
```

**Key JAX Principles for GenJAX**:

- **Immutability**: JAX arrays are immutable; operations return new arrays
- **Functional style**: No side effects, pure functions only
- **JAX-compatible control flow**: Use `jax.lax.cond`, `jax.lax.scan`, not Python `if`/`for`
- **Static shapes**: Array shapes should be known at compile time when possible

### Transformations

- All GenJAX code is JAX-compatible. In general, you should only be writing code within "JAX Python", the subset of Python which is JAX compatible.
- Can use `jax.jit`, `jax.vmap`, `jax.grad` _on GFI interface invocations_, but not on generative functions (as instances of `GFI`) themselves.
- (**IMPORTANT**) GenJAX provides a custom version of `vmap` called `modular_vmap`. You should prioritize using this version when working with any code which uses generative functions.

### PJAX: Probabilistic JAX

**IMPORTANT**: PJAX (Probabilistic JAX) is GenJAX's intermediate representation that extends JAX with custom primitives for probabilistic programming. Understanding PJAX helps explain how GenJAX works under the hood.

**Core PJAX Primitives**:

PJAX adds two fundamental primitives to JAX:

- **`assume_p`**: Represents probabilistic sampling operations
- **`log_density_p`**: Represents log probability density computations

**How PJAX Works**:

```python
# When you write GenJAX code:
@gen
def my_model():
    x = normal(0.0, 1.0) @ "x"
    return x

trace = my_model.simulate(())

# GenJAX compiles this to a program with PJAX primitives that 
# looks conceptually like:
def my_model_pjax():
    x = assume_p(normal_sampler, 0.0, 1.0)  # Sampling primitive
    log_density = log_density_p(normal_logpdf, x, 0.0, 1.0)  # Density primitive
    return Trace(my_model, (), x, x, log_density)
```

**PJAX Transformations**:

PJAX primitives are foreign to JAX, so we must be careful when using JAX transformations with PJAX programs. PJAX introduces two transformations: `modular_vmap` and `seed`.

**1. `seed` Transformation**:

The `seed` transformation eliminates `assume_p` primitives by providing explicit PRNG keys, making the resulting function compatible with standard JAX transformations.

```python
from genjax import seed

@gen
def my_model():
    x = normal(0.0, 1.0) @ "x"
    y = normal(x, 0.5) @ "y"
    return y

# Original function uses assume_p primitives 
# When executing this code, a global key is evolved
# in the background
trace = my_model.simulate(())

# seed transformation eliminates assume_p by threading PRNG keys
# giving users control over keys
seeded_model = seed(my_model.simulate)

# Now we can use JAX transformations
import jax
import jax.random as jrand

key = jrand.key(42)
result = seeded_model(key, ())  # Explicit key required

# JAX transformations work on seeded functions
jit_model = jax.jit(seeded_model)
vmap_model = jax.vmap(seeded_model, in_axes=(0, None))

# Vectorize over multiple keys
keys = jrand.split(key, 10)
batch_results = vmap_model(keys, ())
```

**2. `modular_vmap` Transformation**:

The `modular_vmap` transformation extends `jax.vmap` to work directly with PJAX primitives, enabling vectorization of probabilistic programs without requiring `seed`.

```python
from genjax import modular_vmap

@gen
def single_observation(mu):
    return normal(mu, 1.0) @ "obs"

# Standard jax.vmap doesn't understand PJAX primitives
# vmap_fn = jax.vmap(single_observation.simulate)  # Would fail or give non-intuitive results

# modular_vmap handles PJAX primitives correctly
vmap_fn = modular_vmap(single_observation.simulate, in_axes=(0,))

# Works with vectorized arguments
mus = jnp.array([0.0, 1.0, 2.0])
vectorized_traces = vmap_fn((mus,))

# modular_vmap preserves probabilistic semantics
choices = get_choices(vectorized_traces)
samples = choices["obs"]  # Array of 3 samples

# Can compose modular_vmap with seed
vectorized_traces = seed(vmap_fn)(key, (mus, ))
```

**Key Differences**:

- **`seed`**: Eliminates PJAX primitives → enables standard JAX transformations → requires explicit key management
- **`modular_vmap`**: Preserves PJAX primitives → specialized for vectorization → maintains GenJAX's automatic key management

**When to Use Each**:

```python
# Use seed when:
# 1. You need to use JAX transformations other than vmap, such as jit
# 2. You're implementing low-level inference algorithms
# 3. You need explicit control over randomness

seeded_fn = seed(model.simulate)
grad_fn = jax.grad(lambda key, args: seeded_fn(key, args).get_score())

# Use modular_vmap when:
# 1. You need vectorization of probabilistic programs
# 2. You want to maintain GenJAX's probabilistic semantics
# 3. You're working within GenJAX's high-level APIs

batch_model = modular_vmap(model.simulate, in_axes=(0,))
```

**Key PJAX Concepts**:

- **Staged Execution**: PJAX separates the "what" (probabilistic operations) from the "how" (execution strategy)
- **Primitive Binding**: GFI methods (simulate, assess, generate) bind primitives as part of their implementations
- **JAX Integration**: PJAX primitives are proper JAX primitives, so they work with all JAX transformations
- **Vmap Transformation**: The `modular_vmap` function handles PJAX primitives correctly during vectorization, unlike standard `jax.vmap` which doesn't understand probabilistic operations
- **Seed Transformation**: The `seed` function eliminates `assume_p` primitives by providing explicit PRNG keys

### GenJAX Constraints

**CRITICAL**: When writing code with GenJAX (inside `@gen` functions, combinators, etc.), you must follow JAX Python restrictions. Violating these constraints will typically cause compilation errors or incorrect behavior.

**Control Flow - NEVER USE PYTHON CONTROL FLOW**:

- **ABSOLUTELY NEVER use Python `if...else` statements** - these are fundamentally incompatible with JAX compilation
- **ABSOLUTELY NEVER use Python `for` loops** - these break JAX vectorization and compilation
- **ABSOLUTELY NEVER use Python `while` loops** - these are not supported in JAX
- **ALWAYS use JAX control flow**: `jax.lax.cond`, `jax.lax.switch` for deterministic conditional logic, `jax.lax.scan` for deterministic iterative logic
- **ALWAYS use GenJAX combinators**: `Cond(branch1, branch2)` for conditional generative functions, `Scan(callee)` for iterative generative functions
- **Pattern for Conditional Generative Functions**: Extract branches into separate `@gen` functions, then use combinators

**WRONG - These patterns will break**:

```python
@gen
def bad_conditional():
    x = normal(0.0, 1.0) @ "x"
    if x > 0:  # ❌ CRITICAL ERROR: Python if statement
        y = exponential(1.0) @ "y"
    else:
        y = exponential(2.0) @ "y"
    return y

@gen
def bad_iteration(T):
    results = []
    for t in range(T):  # ❌ CRITICAL ERROR: Python for loop
        x = normal(0.0, 1.0) @ f"x_{t}"  # ❌ CRITICAL ERROR: Dynamic addressing
        results.append(x)
    return results
```

**CORRECT - Always use these patterns**:

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

@gen
def step_function(carry, x):
    state = normal(carry, 1.0) @ "state"  # ✅ Static addressing
    return state, state

@gen
def good_iteration(T):
    scan_fn = Scan(step_function, length=T)
    final_carry, results = scan_fn((0.0, None)) @ "scan"  # ✅ Use Scan combinator
    return results
```

**Why These Constraints Exist**:

- Python control flow prevents JAX from compiling and vectorizing code
- Dynamic addressing breaks GenJAX's trace structure requirements
- GenJAX combinators provide JAX-compatible alternatives for all control flow needs
- Violating these constraints typically results in cryptic JAX compilation errors

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

### JIT Compilation

- Most operations can be JIT compiled
- Compilation happens lazily on first call
- Significant speedups for complex models

### Memory Management

- Traces contain full execution history
- Consider memory usage for large models
- Use JAX memory profiling tools

## Common Error Patterns and Solutions

**CRITICAL**: Understanding these common error patterns will prevent most GenJAX development issues:

### Dynamic Addressing in Combinators

**Error Pattern**:

```python
# ❌ This breaks vectorization
@gen
def bad_step(carry, t):
    x = normal(0.0, 1.0) @ f"state_{t}"  # Dynamic addressing
    return x, x
```

**Solution**:

```python
# ✅ Use static addressing
@gen
def good_step(carry, x):
    state = normal(carry, 1.0) @ "state"  # Static addressing
    return state, state
```

### Python Control Flow in @gen Functions

**Error Pattern**:

```python
# ❌ Python control flow breaks JAX compilation
@gen
def bad_model(condition):
    if condition:  # Python if statement
        x = normal(0.0, 1.0) @ "x"
    else:
        x = normal(1.0, 1.0) @ "x"
    return x
```

**Solution**:

```python
# ✅ Use Cond combinator
@gen
def branch_a():
    return normal(0.0, 1.0) @ "x"

@gen
def branch_b():
    return normal(1.0, 1.0) @ "x"

@gen
def good_model(condition):
    cond_fn = Cond(branch_a, branch_b)
    result = cond_fn((condition,)) @ "conditional"
    return result
```

### Incorrect Distribution.assess() Usage

**Error Pattern**:

```python
# ❌ Wrong parameter passing
log_density, _ = normal(mu, sigma).assess((), sample)  # Invalid API
# ❌ Manual density calculations
manual_density = -(sample**2)/2.0 - 0.5*jnp.log(2*jnp.pi)  # Error-prone
```

**Solution**:

```python
# ✅ Correct parameter passing
log_density, _ = normal.assess((mu, sigma), sample)
# ✅ Use Distribution.assess() for reliable densities
```

### Dummy Values in Scan Inputs

**Error Pattern**:

```python
# ❌ Unnecessary dummy values
scan_fn = Scan(step_function, length=T)
result = scan_fn((init_carry, jnp.zeros(T))) @ "steps"
```

**Solution**:

```python
# ✅ Use None for unused inputs
scan_fn = Scan(step_function, length=T)
result = scan_fn((init_carry, None)) @ "steps"
```

### General Debugging Tips

- Start with simple models and build complexity gradually
- Use `jax.debug.print()` for debugging inside JIT-compiled code
- Check trace consistency with `assess()` calls
- When encountering compilation errors, check for Python control flow violations
- When vectorization fails, verify static addressing in combinators
- Always test density computations using `Distribution.assess()` methods

## Development Overview

### Codebase Commands

The codebase is managed through [`pixi`](https://pixi.sh/latest/). All toplevel codebase commands should go through `pixi`.

#### Environment Setup

```bash
# Install dependencies using pixi (a package manager)
pixi install
```

#### Running Examples

```bash
# Run examples (via pixi)
pixi run python examples/simple.py
pixi run python examples/regression.py
pixi run python examples/marginal.py
pixi run python examples/vi.py
```

#### Generating Figures

```bash
# Generate figures for the beta-bernoulli benchmark
pixi run betaber-timing

# Generate figures for Game of Life simulations
pixi run gol-timing
pixi run gol-figs

# Generate figures for curve fitting
pixi run curvefit-figs
```

#### Development

```bash
# Format code
pixi run format

# Check for unused code
pixi run vulture
```

#### Documentation

```bash
# Preview documentation
pixi run preview

# Deploy documentation to GitHub Pages
pixi run deploy
```

### Codebase Architecture

```
genjax/
├── src/genjax/                   # Core GenJAX library
│   ├── __init__.py                # Package exports and main API
│   ├── core.py                    # PJAX, GFI, Trace, Distribution, Fn, combinators
│   ├── distributions.py           # Standard probability distributions as generative functions
│   ├── adev.py                    # Automatic differentiation of expected values (ADEV)
│   └── stdlib.py                  # Standard inference library functions and utilities
├── examples/                     # Examples, tutorials, and performance benchmarks
├── tests/                        # Test suite
│   ├── test_core.py               # Core functionality tests
│   ├── test_stdlib.py             # Standard library tests
│   └── discrete_hmm.py            # Hidden Markov Model test cases
├── quarto/                       # Documentation source files for website generation
├── docs/                         # Generated documentation (GitHub Pages)
├── pixi.lock                     # Pixi lock file for reproducible environments
├── pyproject.toml                # Python project configuration
├── README.md                     # Project overview and quick start
├── LICENSE.md                    # License information
├── CLAUDE.md                     # Claude Code instructions (this file)
└── logo.png                      # Project logo
```

### Development Workflow

**IMPORTANT**: When working on GenJAX development tasks, follow this structured approach to ensure consistency and quality:

1. **Understand First**: Use Read and search tools (Grep, Glob) to understand existing code patterns and conventions before making changes
2. **Research Context**: Use search tools to find similar implementations and understand codebase patterns
3. **Follow Conventions**: Mimic existing code style, library usage, and architectural patterns
4. **Implement Changes**: Make changes that follow established patterns and JAX/GenJAX constraints
5. **Test and Verify**: Run appropriate pixi commands (`pixi run format`, `pixi run test`) to verify functionality
6. **Validate Results**: Ensure changes work as expected and follow GenJAX semantics before concluding

**Key Workflow Principles**:

- Never assume patterns - always research existing code first
- Prioritize consistency with existing codebase over personal preferences
- Always run formatter and linting tools before concluding work
- Test both basic functionality and edge cases
- Verify JAX compilation compatibility

### Testing Strategy

#### Unit Tests

- Test individual distributions and combinators
- Verify GFI method implementations

#### Testing Generative Functions

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

## Sharp Edges & Limitations of GenJAX

### Known Issues

- `vmap` within ADEV `@expectation` programs has undefined semantics
- Not all gradient estimation strategies support batching
- Some JAX transformations may not compose cleanly

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
