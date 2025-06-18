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

### State Interpreter: Tagged Value Inspection

The state interpreter allows you to inspect intermediate values within JAX computations using two core APIs:

```python
from genjax.state import state, save

# Core pattern: @state decorator + save() function
@state
def computation(x):
    y = x + 1
    z = x * 2
    # Save intermediate values for inspection
    save(intermediate=y, doubled=z)
    return y + z

result, state_dict = computation(5)
# result = 16, state_dict = {"intermediate": 6, "doubled": 10}

# Use in MCMC for acceptance tracking
@state
def mcmc_step(trace):
    new_trace = some_mcmc_move(trace)
    accept = compute_acceptance(new_trace, trace)
    save(accept=accept)  # Save acceptance for diagnostics
    return new_trace

new_trace, diagnostics = mcmc_step(trace)
# diagnostics = {"accept": True/False}
```

**JAX Compatibility**: The state interpreter works with all JAX transformations (`jit`, `vmap`, `grad`) by using `initial_style_bind` for proper JAX primitive handling.

**Key Features**:
- **Simple API**: Only `@state` decorator and `save()` function
- **Named value collection**: `save(name1=val1, name2=val2)`
- **JAX transformation compatibility**: Works with `jit`, `vmap`, `grad`
- **MCMC integration**: Used internally for acceptance tracking

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

## Inference Algorithms

GenJAX provides implementations of standard inference algorithms with vectorization and diagnostics.

### MCMC (Markov Chain Monte Carlo)

**Core MCMC Components**:

```python
from genjax import mh, mala, chain, seed, MCMCResult

# Basic Metropolis-Hastings step
def mh_kernel(trace):
    selection = sel("param")  # Select which addresses to resample
    return mh(trace, selection)

# MALA (Metropolis-Adjusted Langevin Algorithm) step with gradient information
def mala_kernel(trace):
    selection = sel("param")  # Select which addresses to resample
    step_size = 0.01  # Step size parameter (smaller = more conservative)
    return mala(trace, selection, step_size)

# Create MCMC chain algorithm
mcmc_chain = chain(mh_kernel)  # or chain(mala_kernel)
seeded_chain = seed(mcmc_chain)

# Run single chain
result = seeded_chain(key, initial_trace, n_steps=const(1000))
```

**Multi-Chain MCMC with Diagnostics**:

```python
# Run multiple parallel chains with diagnostics
result = seeded_chain(
    key,
    initial_trace,
    n_steps=const(1000),
    n_chains=const(4),              # Number of parallel chains
    burn_in=const(200),             # Burn-in samples to discard
    autocorrelation_resampling=const(2)  # Thinning (keep every N-th sample)
)

# MCMCResult contains diagnostics
assert isinstance(result, MCMCResult)
assert result.n_chains.value == 4
assert result.rhat is not None          # R-hat convergence diagnostic
assert result.ess_bulk is not None      # Bulk effective sample size
assert result.ess_tail is not None      # Tail effective sample size

# Access diagnostics (same structure as choices)
choices = result.traces.get_choices()
print(f"R-hat for param: {result.rhat['param']}")
print(f"Bulk ESS: {result.ess_bulk['param']}")
print(f"Tail ESS: {result.ess_tail['param']}")
```

**MCMC Diagnostics**:

- **R-hat (Potential Scale Reduction Factor)**: Convergence assessment comparing between-chain and within-chain variance. Values close to 1.0 indicate convergence.
- **Effective Sample Size (ESS)**:
  - **Bulk ESS**: Efficiency for bulk of the distribution
  - **Tail ESS**: Efficiency for distribution tails (5th and 95th percentiles)
- **Acceptance Rate**: Proportion of proposed moves that were accepted

**Chain Function Architecture**:

The `chain` higher-order function transforms simple MCMC kernels into full algorithms:

```python
# Transform into full MCMC algorithm
mcmc_algorithm = chain(mh)
```

**MALA (Metropolis-Adjusted Langevin Algorithm)**:

MALA uses gradient information to make more efficient proposals than standard Metropolis-Hastings:

```python
# MALA proposal: x_new = x + step_size^2/2 * ∇log(p(x)) + step_size * noise
def mala_kernel(trace):
    selection = sel("mu") | sel("sigma")  # Select parameters to update
    step_size = 0.01  # Controls proposal variance and drift strength
    return mala(trace, selection, step_size)

# MALA works well for continuous parameters with smooth log densities
# Step size tuning: smaller = higher acceptance but slower mixing
mala_chain = chain(mala_kernel)
result = seed(mala_chain)(key, trace, n_steps=const(1000))
```

**Key Features**:
- Parallel chain execution using `modular_vmap`
- Burn-in and thinning support
- Pytree-structured diagnostics matching choice structure
- State collection for acceptance tracking
- JAX-compatible design for JIT compilation
- MALA: Gradient-guided proposals for improved efficiency on smooth densities

### SMC (Sequential Monte Carlo)

**Particle Collection System**:

```python
from genjax import init, change, extend, rejuvenate, resample, ParticleCollection

# Initialize particle collection with importance sampling
particles = init(
    target_gf=model,
    target_args=args,
    n_samples=const(1000),
    constraints={"obs": observed_data},
    proposal_gf=custom_proposal  # Optional custom proposal
)

# Access particle statistics
ess = particles.effective_sample_size()
log_marginal = particles.log_marginal_likelihood()
```

**SMC Move Types**:

**Change Move** - Translate particles between models:
```python
# choice_fn must be a bijection on address space only
# Valid: remap keys while preserving all values exactly
def choice_fn(old_choices):
    return {"new_param": old_choices["old_param"], "obs": old_choices["obs"]}

particles = change(
    particles,
    new_target_gf=new_model,
    new_target_args=new_args,
    choice_fn=choice_fn  # Bijective mapping of address space
)

# Identity mapping (simplest valid choice_fn)
particles = change(particles, new_model, new_args, lambda x: x)
```

**Extension Move** - Add new random choices:
```python
# Use extended target's internal proposal (default)
particles = extend(
    particles,
    extended_target_gf=extended_model,
    extended_target_args=extended_args,
    constraints={"new_obs": observed_value}  # Constraints on new variables
)

# Or use custom extension proposal
@gen
def custom_proposal():
    # Proposal for new variables only
    return normal(0.5, 0.2) @ "new_param"

particles = extend(
    particles,
    extended_target_gf=extended_model,
    extended_target_args=extended_args,
    constraints={},  # No hard constraints
    extension_proposal=custom_proposal
)
```

**Rejuvenation Move** - Apply MCMC to combat degeneracy:
```python
# Use MCMC kernel from mcmc.py
def mcmc_kernel(trace):
    return mh(trace, sel("param"))

particles = rejuvenate(particles, mcmc_kernel)
# Weights remain unchanged due to detailed balance
# Model density ratio cancels with proposal density ratio
```

**Resampling** - Combat particle degeneracy:
```python
# Resample when effective sample size is low
if particles.effective_sample_size() < threshold:
    particles = resample(particles, method="systematic")  # or "categorical"
```

**Choice Function Specification**:

For the `change` move, the `choice_fn` parameter has strict requirements:

```python
# CRITICAL: choice_fn must be a bijection on address space only

# Valid examples (preserve all values exactly):
lambda x: x                                    # Identity mapping
lambda d: {"mu": d["param"], "obs": d["obs"]}  # Key remapping only
lambda d: {"new_key": d["old_key"]}            # Single key remap

# Invalid examples (break probability density):
lambda x: x + 1                               # Modifies scalar values
lambda d: {"key": d["key"] * 2}               # Modifies dict values
lambda d: {"key": d["key1"] + d["key2"]}      # Combines values

# Mathematical requirement:
# If choice_fn(x) = y, then probability density p(x) must equal p(y)
# This is only possible if choice_fn preserves values exactly
```

**SMC Algorithm Composition**:

```python
# Complete SMC algorithm with rejuvenation
final_particles = rejuvenation_smc(
    initial_model=model_t0,
    extended_model=model_extended,
    transition_proposal=transition_proposal,
    mcmc_kernel=lambda trace: mh(trace, sel("latent")),
    observations=time_series_data,
    choice_fn=lambda x: x,  # Identity mapping for same address space
    n_particles=const(1000)
)

# The algorithm automatically handles:
# 1. Initialization with first observation
# 2. Sequential extension with remaining observations
# 3. Adaptive resampling based on effective sample size
# 4. MCMC rejuvenation to maintain particle diversity
```

For implementation details, see `rejuvenation_smc` in `src/genjax/smc.py`.

**Key SMC Features**:
- **Vectorized Operations**: All moves use `modular_vmap` for parallel processing
- **Weight Tracking**: Importance weights maintained across all moves
- **Marginal Likelihood Estimation**: Computation via importance sampling
- **Flexible Proposals**: Support for both default and custom proposal distributions
- **MCMC Integration**: Kernels from `mcmc.py` work directly with `rejuvenate`
- **Mathematical Correctness**: Choice functions enforce bijection constraints
- **Detailed Balance**: Rejuvenation preserves weights via MCMC detailed balance
- **JAX Compatibility**: Full integration with JAX transformations via `jax.lax.scan`

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
