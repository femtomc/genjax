# inference/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX inference algorithms module.

## Overview

The `inference` module provides implementations of standard probabilistic inference algorithms including Markov Chain Monte Carlo (MCMC), Sequential Monte Carlo (SMC), and Variational Inference (VI). These algorithms enable approximate posterior inference in complex probabilistic models.

## Module Structure

```
src/genjax/inference/
├── __init__.py          # Module exports
├── mcmc.py             # Markov Chain Monte Carlo algorithms
├── smc.py              # Sequential Monte Carlo algorithms
├── vi.py               # Variational Inference algorithms
└── CLAUDE.md           # This file
```

## MCMC Algorithms (`mcmc.py`)

### Helper Functions

#### Log Density for Selected Choices
**Function**: `_create_log_density_wrt_selected(target_gf, args, unselected_choices) -> Callable`
**Location**: `mcmc.py:59-88`
**Purpose**: Create log density function that only depends on selected addresses

**Usage**: Internal helper shared by MALA and HMC for gradient computation. Creates a closure that:
- Reconstructs full choice map by merging selected with unselected choices
- Computes log density via `target_gf.assess`
- Enables `jax.grad` to compute gradients w.r.t. selected addresses only

### Core Components

#### Metropolis-Hastings (`mh`)
Standard Metropolis-Hastings algorithm for discrete and continuous parameters:

```python
from genjax.inference import mh

def mh_kernel(trace):
    selection = sel("param")  # Select which addresses to resample
    return mh(trace, selection)
```

#### MALA - Metropolis-Adjusted Langevin Algorithm (`mala`)
Gradient-informed MCMC for continuous parameters:

```python
from genjax.inference import mala

def mala_kernel(trace):
    selection = sel("continuous_param")
    step_size = 0.01  # Tune based on acceptance rate
    return mala(trace, selection, step_size)
```

#### HMC - Hamiltonian Monte Carlo (`hmc`)
Hamiltonian dynamics for efficient exploration of continuous parameter spaces:

```python
from genjax.inference import hmc

def hmc_kernel(trace):
    selection = sel("continuous_param")
    step_size = 0.1    # Leapfrog step size (eps)
    n_steps = 10       # Number of leapfrog steps (L)
    return hmc(trace, selection, step_size, n_steps)
```

#### Chain Function (`chain`)
Higher-order function to create full MCMC algorithms:

```python
from genjax.inference import chain

# Create MCMC algorithm from kernel
mcmc_algorithm = chain(mh_kernel)

# Run with diagnostics
result = seed(mcmc_algorithm)(
    key,
    initial_trace,
    n_steps=const(1000),
    n_chains=const(4),           # Multiple parallel chains
    burn_in=const(200),          # Burn-in samples to discard
    autocorrelation_resampling=const(2)  # Thinning factor
)
```

### MCMC Results and Diagnostics

The `MCMCResult` dataclass provides comprehensive diagnostics:

```python
# Access results
traces = result.traces          # Final traces (post burn-in, thinned)
choices = result.traces.get_choices()

# Convergence diagnostics
r_hat = result.rhat            # R-hat convergence diagnostic
ess_bulk = result.ess_bulk     # Bulk effective sample size
ess_tail = result.ess_tail     # Tail effective sample size
n_chains = result.n_chains     # Number of chains

# Diagnostics structure matches choice structure
print(f"R-hat for param: {result.rhat['param']}")
print(f"Bulk ESS: {result.ess_bulk['param']}")
```

### MCMC Best Practices

#### Selection Strategy
```python
# Select all continuous parameters
continuous_selection = sel("mu") | sel("sigma") | sel("beta")

# Select subset for partial updates
partial_selection = sel("mu") | sel("sigma")  # Leave beta unchanged

# Select hierarchical parameters
hierarchical_selection = sel("global_params") | sel("group_params")
```

#### Step Size Tuning for MALA
```python
# Start with small step sizes and tune based on acceptance rate
step_sizes = [0.001, 0.01, 0.1]
target_acceptance = 0.6  # Optimal for MALA

for step_size in step_sizes:
    result = run_mala_chain(step_size)
    acceptance_rate = compute_acceptance_rate(result)

    if abs(acceptance_rate - target_acceptance) < 0.1:
        optimal_step_size = step_size
        break
```

#### Parameter Tuning for HMC
```python
# HMC requires tuning both step size and number of steps
# Target acceptance rate: 0.6-0.9 (higher than MALA due to volume preservation)
def tune_hmc_parameters(initial_trace, selection):
    step_sizes = [0.01, 0.05, 0.1, 0.2]
    n_steps_options = [5, 10, 20, 50]

    best_params = None
    best_ess_per_step = 0

    for step_size in step_sizes:
        for n_steps in n_steps_options:
            def hmc_kernel(trace):
                return hmc(trace, selection, step_size, n_steps)

            result = run_hmc_chain(hmc_kernel, initial_trace)
            acceptance_rate = result.acceptance_rate

            # Good HMC should have acceptance rate 0.6-0.9
            if 0.6 <= acceptance_rate <= 0.9:
                # Compute effective sample size per leapfrog step
                ess_per_step = compute_ess(result) / n_steps
                if ess_per_step > best_ess_per_step:
                    best_ess_per_step = ess_per_step
                    best_params = (step_size, n_steps)

    return best_params

# HMC step size guidelines:
# - Simple models (normal distributions): 0.1 - 0.5
# - Complex models (banana, hierarchical): 0.001 - 0.05
# - High-dimensional models: start with 0.01 and adjust
```

#### Multi-Chain Diagnostics
```python
# Use multiple chains to assess convergence
result = mcmc_algorithm(key, trace, n_steps=const(1000), n_chains=const(4))

# Check R-hat < 1.1 for convergence
converged = all(r_hat < 1.1 for r_hat in jax.tree.leaves(result.rhat))

# Check effective sample size > 100 for reliable estimates
adequate_ess = all(ess > 100 for ess in jax.tree.leaves(result.ess_bulk))
```

## SMC Algorithms (`smc.py`)

### Core Data Structure

#### ParticleCollection
Container for SMC particles with importance weights and diagnostics:

```python
from genjax.inference import ParticleCollection

# ParticleCollection attributes:
# - traces: Vectorized trace containing all particles
# - log_weights: Log importance weights for each particle
# - n_samples: Number of particles (static)
# - log_marginal_estimate: Accumulated log marginal likelihood

# Methods:
ess = particles.effective_sample_size()  # Effective sample size
log_ml = particles.log_marginal_likelihood()  # Marginal likelihood estimate

# Weighted estimation of functions:
posterior_mean = particles.estimate(lambda choices: choices["param"])
posterior_variance = particles.estimate(lambda choices: choices["param"]**2) - posterior_mean**2
custom_expectation = particles.estimate(lambda choices: jnp.sin(choices["x"]) + choices["y"])
```


### Core Components

#### Particle Initialization (`init`)
Initialize particle collection with importance sampling:

```python
from genjax.inference import init

# Basic initialization with target's internal proposal
particles = init(
    target_gf=model,
    target_args=(),  # Args for the model
    n_samples=const(1000),
    constraints={"obs": observed_data},
    proposal_gf=None  # Uses target's internal proposal
)

# Custom proposal initialization
# Proposal signature: (constraints, *target_args) -> trace
particles = init(
    target_gf=model,
    target_args=(param1, param2),
    n_samples=const(1000),
    constraints={"obs": observed_data},
    proposal_gf=custom_proposal  # Custom importance sampling proposal
)
```

#### SMC Move Types

**Change Move** - Translate particles between models:
```python
from genjax.inference import change

# CRITICAL: choice_fn must be bijection on address space only
def identity_choice_fn(choices):
    return choices  # Identity mapping (most common)

def remap_choice_fn(choices):
    # Only remap keys, preserve all values exactly
    return {"new_param": choices["old_param"], "obs": choices["obs"]}

particles = change(
    particles,
    new_target_gf=new_model,
    new_target_args=new_args,
    choice_fn=identity_choice_fn  # Bijective address mapping
)
```

**Extension Move** - Add new random choices:
```python
from genjax.inference import extend

# Basic extension with target's internal proposal
particles = extend(
    particles,
    extended_target_gf=extended_model,
    extended_target_args=extended_args,  # Can be tuple or vectorized
    constraints={"new_obs": observed_value},
    extension_proposal=None  # Uses extended target's internal proposal
)

# Extension with custom proposal
particles = extend(
    particles,
    extended_target_gf=extended_model,
    extended_target_args=extended_args,
    constraints={"new_obs": observed_value},
    extension_proposal=transition_proposal  # Custom extension proposal
)
```

**Rejuvenation Move** - Apply MCMC to combat degeneracy:
```python
from genjax.inference import rejuvenate

def mcmc_kernel(trace):
    return mh(trace, sel("latent_state"))

particles = rejuvenate(particles, mcmc_kernel)
# Weights remain unchanged due to detailed balance
# Mathematical foundation: log incremental weight = 0
```

**Resampling** - Combat particle degeneracy:
```python
from genjax.inference import resample

# Categorical resampling (default)
particles = resample(particles, method="categorical")

# Systematic resampling (lower variance)
particles = resample(particles, method="systematic")

# After resampling:
# - Weights reset to uniform (log 0)
# - Marginal likelihood estimate updated
```

### Complete SMC Algorithm

#### Rejuvenation SMC
**Function**: `rejuvenation_smc(model, transition_proposal=None, mcmc_kernel=None, observations, initial_model_args, n_particles, return_all_particles, n_rejuvenation_moves) -> ParticleCollection`
**Location**: `smc.py:540-643`
**Purpose**: Full SMC algorithm with automatic resampling and rejuvenation

**API Contract**:
- Model signature: `(*args) -> return_value` where return feeds next timestep
- Uses feedback loop: return value from t becomes args for t+1
- `transition_proposal` is optional (default: None) - uses model's internal proposal if not provided
- `mcmc_kernel` is optional (default: None) - no rejuvenation if not provided, must be wrapped in `Const[]` if used
- Observations can be any Pytree structure
- `return_all_particles` must be wrapped in `Const[bool]` (default: `const(False)`)
- `n_rejuvenation_moves` must be wrapped in `Const[int]` (default: `const(1)`)
- Automatic resampling when ESS < n_particles/2

**Return Behavior**:
- `return_all_particles=const(False)`: Returns final `ParticleCollection` only
- `return_all_particles=const(True)`: Returns `ParticleCollection` with leading time dimension (T, ...)

**Rejuvenation Moves**:
- Performs `n_rejuvenation_moves` MCMC steps at each timestep
- Uses nested `jax.lax.scan` for efficient implementation
- Each move applies the same `mcmc_kernel` to all particles


### Getting All Timesteps from SMC

**Built-in Support**: Use `return_all_particles=const(True)` parameter:
```python
# Get all timesteps with model's internal proposal (no custom proposal)
all_particles = rejuvenation_smc(
    model=model,
    # transition_proposal=None,  # Optional - uses model's internal proposal
    # mcmc_kernel=None,         # Optional - no rejuvenation moves
    observations=observations,
    initial_model_args=initial_args,
    n_particles=const(1000),
    return_all_particles=const(True)  # Returns all timesteps
)

# With custom proposal and rejuvenation
all_particles = rejuvenation_smc(
    model=model,
    transition_proposal=transition_proposal,  # Optional custom proposal
    mcmc_kernel=const(mcmc_kernel),          # Optional rejuvenation
    observations=observations,
    initial_model_args=initial_args,
    n_particles=const(1000),
    return_all_particles=const(True),
    n_rejuvenation_moves=const(5)  # 5 MCMC moves per timestep
)

# Model-only SMC (no custom proposal, no rejuvenation)
final_particles = rejuvenation_smc(
    model=model,
    observations=observations,
    initial_model_args=initial_args,
    n_particles=const(1000)
    # All other parameters use defaults
)
```

**Alternative**: Implement custom SMC loop using building blocks (`init`, `extend`, `resample`, `rejuvenate`)

### SMC Best Practices

#### Particle Count Guidelines
- **Prototyping**: 100 particles (fast iteration)
- **Standard inference**: 1000 particles (good accuracy/speed tradeoff)
- **High precision**: 5000+ particles (publication quality)
- Always use `const()` wrapper for static values

#### Effective Sample Size Monitoring
- ESS indicates particle diversity (higher is better)
- Resample when ESS/n_particles < 0.5 (common threshold)
- `rejuvenation_smc` automatically resamples at ESS < n_particles/2
- Check ESS with `particles.effective_sample_size()`

#### Choice Function Constraints
- **Valid**: Identity (`lambda x: x`), key remapping
- **Invalid**: Value modification, arithmetic operations
- Must be bijective on address space
- See `change` docstring in `smc.py:290-310` for detailed specs

### Locally Optimal Proposals

**Concept**: A locally optimal proposal uses the model's `assess` method to evaluate multiple candidate proposals and selects the most promising one, providing more informed transitions than simple prior sampling.

**Mathematical Foundation**:
For a transition proposal at timestep t, instead of sampling directly from the prior:
1. **Grid Evaluation**: Create a grid of candidate values for latent variables
2. **Model Assessment**: Use `model.assess(candidates, observations, *args)` to evaluate log probability at each grid point
3. **Optimal Selection**: Choose the candidate with maximum log probability: `argmax_i log p(candidate_i | observations)`
4. **Noise Injection**: Add Gaussian noise around the selected point for smooth proposals

**API Contract**:
```python
@gen
def locally_optimal_proposal(obs, prev_choices, *args):
    """Locally optimal proposal using grid evaluation.

    Args:
        obs: Current observation constraints
        prev_choices: Previous particle's choices
        *args: Model arguments for this timestep
    """
    # Create grid of candidate values
    candidates = create_grid(variable_bounds)

    # Vectorized assessment over all candidates
    def assess_single_candidate(candidate):
        proposed_choices = update_choices(prev_choices, candidate)
        return model.assess(merge(obs, proposed_choices), *args).get_score()

    vectorized_assess = jax.vmap(assess_single_candidate)
    log_probs = vectorized_assess(candidates)

    # Select optimal candidate
    best_idx = jnp.argmax(log_probs)
    best_candidate = candidates[best_idx]

    # Add noise for smooth proposals
    x_prop = normal(best_candidate[0], noise_std) @ "x"
    y_prop = normal(best_candidate[1], noise_std) @ "y"

    return x_prop, y_prop
```

**Performance Characteristics**:
- **Computational Cost**: O(grid_size^d × model_complexity) where d is dimension
- **Accuracy**: Higher than prior sampling, especially with informative observations
- **JAX Compatibility**: Fully vectorized using `jax.vmap` for efficiency
- **Grid Size Trade-off**: Larger grids provide better proposals but increase computation

**Use Cases**:
- **High-dimensional spaces** where prior sampling is inefficient
- **Informative observations** that significantly constrain the posterior
- **Complex models** where likelihood evaluation is cheap relative to MCMC steps
- **Real-time applications** requiring efficient particle transitions

**Implementation Example**:
See `examples/localization/core.py` for a complete implementation using 15×15×15 grid evaluation over (x, y, θ) space for robot localization.

## Variational Inference (`vi.py`)

### Core Components

#### Variational Families

**Mean Field Normal Family**
**Function**: `mean_field_normal_family(parameter_names) -> VariationalApproximation`
**Location**: `vi.py:115-156`
**Purpose**: Creates independent normal distributions for each parameter

**API Contract**:
- Each parameter gets its own mean and log standard deviation
- Parameters are independent (no covariance)
- Supports reparameterization gradient estimation
- Most efficient for high-dimensional problems

**Full Covariance Normal Family**
**Function**: `full_covariance_normal_family(parameter_names) -> VariationalApproximation`
**Location**: `vi.py:159-213`
**Purpose**: Creates multivariate normal with full covariance matrix

**API Contract**:
- Parameters can be correlated
- Uses Cholesky parameterization for positive definite covariance
- More expressive but scales O(d²) in memory
- Better for capturing parameter correlations

#### ELBO Factory
**Function**: `elbo_factory(target_gf, variational_gf, estimator_mapping) -> Callable`
**Location**: `vi.py:34-71`
**Purpose**: Creates ELBO computation function with gradient estimation

**API Contract**:
- Returns function: `(variational_params, target_args, constraints, n_samples) -> float`
- `estimator_mapping` specifies gradient estimators (see `adev/CLAUDE.md`)
- Computes variational lower bound on log marginal likelihood
- Result is differentiable w.r.t. variational parameters

#### Complete VI Pipeline
**Function**: `elbo_vi(target_gf, target_args, constraints, variational_approximation, ...) -> VIResult`
**Location**: `vi.py:216-284`
**Purpose**: Full variational inference with optimization

**API Contract**:
- Uses optax for optimization (Adam by default)
- Returns `VIResult` with final parameters and loss history
- `n_samples` controls Monte Carlo approximation quality
- `n_steps` sets optimization iterations
- Supports custom optimizers and learning rate schedules

### VI Best Practices

#### Learning Rate Scheduling
- Use optax for advanced schedules (exponential decay, cosine annealing)
- Start with learning rate 0.01-0.1, decay over time
- Monitor loss plateaus to trigger decay
- See `optax` documentation for scheduler options

#### Convergence Monitoring
- Check loss change over sliding window (e.g., 100 steps)
- Convergence threshold: relative change < 1e-4
- Early stopping prevents overfitting
- Plot loss history to diagnose optimization issues

#### Sample Size Guidelines
- **Early exploration**: 100 samples (fast, approximate)
- **Refinement**: 500 samples (better gradient estimates)
- **Final optimization**: 1000+ samples (low variance)
- Trade-off: larger n_samples = better gradients but slower

## Algorithm Selection Guidelines

### When to Use Each Algorithm

#### MCMC
- **Best for**: Exact sampling from posterior (asymptotically)
- **Use when**:
  - High-precision posterior estimates needed
  - Model has complex dependencies
  - Computational time is not critical
- **Avoid when**: Real-time inference required

**MCMC Method Selection**:
- **MH**: General purpose, works with discrete and continuous parameters
- **MALA**: Continuous parameters, uses gradients for better proposals
- **HMC**: Continuous parameters, excellent for high-dimensional problems and models with complex posterior geometry

#### SMC
- **Best for**: Sequential/temporal models, particle filtering
- **Use when**:
  - Time series data
  - Online inference
  - Model evidence (marginal likelihood) needed
- **Avoid when**: Static models with no temporal structure

#### VI
- **Best for**: Fast approximate inference, large-scale problems
- **Use when**:
  - Speed is critical
  - Approximate posteriors acceptable
  - Gradient-based optimization possible
- **Avoid when**: High-precision posteriors essential

### Hybrid Approaches

Combine algorithms for better performance:

```python
# Use VI for initialization, MCMC for refinement
vi_result = variational_inference(...)
initial_trace = create_trace_from_vi_params(vi_result.params)

mcmc_result = mcmc_algorithm(key, initial_trace, n_steps=const(500))

# Use SMC for model comparison, MCMC for detailed posterior
smc_evidence = compute_marginal_likelihood_with_smc(model, data)
mcmc_posterior = detailed_posterior_with_mcmc(model, data)
```

## Performance Optimization

### JAX Integration

All inference algorithms are designed for JAX:

```python
# JIT compilation
jitted_mcmc = jax.jit(mcmc_algorithm)
jitted_smc = jax.jit(smc_algorithm)
jitted_vi = jax.jit(vi_step)

# Vectorization over multiple datasets
batched_inference = jax.vmap(inference_algorithm, in_axes=(0, None))
```

### Memory Management

```python
# For large-scale problems, use checkpointing
@jax.checkpoint
def memory_efficient_inference(...):
    return inference_algorithm(...)

# Monitor memory usage with particle counts
def adaptive_particle_count(model_complexity):
    if model_complexity < 10:
        return const(5000)
    elif model_complexity < 100:
        return const(1000)
    else:
        return const(500)
```

### Convergence Acceleration

```python
# Use warm starts
def warm_start_mcmc(previous_result, new_data):
    # Start from previous posterior
    initial_trace = previous_result.traces[-1]  # Last sample
    return mcmc_algorithm(key, initial_trace, n_steps=const(500))

# Progressive training for VI
def progressive_vi(target, constraints):
    # Start with simple approximation
    simple_family = MeanFieldNormalFamily(["param1"])
    result1 = variational_inference(..., variational_family=simple_family)

    # Expand to full approximation
    full_family = FullCovarianceNormalFamily(["param1", "param2"])
    result2 = variational_inference(..., variational_family=full_family,
                                   initial_params=expand_params(result1.params))

    return result2
```

## Testing and Validation

### Algorithm Correctness

```python
# Test against known analytical solutions
def test_inference_accuracy():
    # Use conjugate models with known posteriors
    true_posterior = analytical_solution(data)
    mcmc_posterior = mcmc_inference(model, data)

    # Compare moments
    assert jnp.allclose(true_posterior.mean, mcmc_posterior.mean, rtol=0.1)
    assert jnp.allclose(true_posterior.std, mcmc_posterior.std, rtol=0.2)

# Test convergence properties
def test_convergence():
    # Increasing computational resources should improve accuracy
    errors = []
    for n_samples in [100, 500, 1000, 2000]:
        result = inference_algorithm(..., n_samples=const(n_samples))
        error = compute_error(result, ground_truth)
        errors.append(error)

    # Errors should generally decrease
    assert errors[-1] < errors[0]
```

### Cross-Algorithm Validation

```python
def cross_validate_algorithms():
    # All algorithms should agree on simple problems
    mcmc_result = mcmc_inference(simple_model, data)
    smc_result = smc_inference(simple_model, data)
    vi_result = vi_inference(simple_model, data)

    # Compare posterior means (within tolerance)
    mcmc_mean = mcmc_result.posterior.mean
    smc_mean = smc_result.posterior.mean
    vi_mean = vi_result.posterior.mean

    assert jnp.allclose(mcmc_mean, smc_mean, rtol=0.2)
    assert jnp.allclose(mcmc_mean, vi_mean, rtol=0.3)  # VI less precise
```

## Common Patterns

### Sequential Data Processing

```python
def process_time_series(data_stream):
    particles = init(initial_model, initial_args, n_particles=const(1000), {})

    results = []
    for t, observation in enumerate(data_stream):
        # Extend model with new timestep
        particles = extend(particles, extended_model, new_args,
                         constraints={f"obs_{t}": observation})

        # Rejuvenate if needed
        if particles.effective_sample_size() < 500:
            particles = rejuvenate(particles, mcmc_kernel)
            particles = resample(particles)

        results.append(particles.log_marginal_likelihood())

    return results
```

### Hierarchical Model Inference

```python
def hierarchical_inference(grouped_data):
    # Use different algorithms at different levels

    # Global parameters with VI (fast)
    global_vi_result = variational_inference(global_model, global_data, ...)

    # Group-specific parameters with MCMC (precise)
    group_results = {}
    for group_id, group_data in grouped_data.items():
        # Initialize from global VI result
        initial_trace = create_trace_from_global_params(global_vi_result.params)
        group_results[group_id] = mcmc_inference(group_model, group_data,
                                                initial_trace=initial_trace)

    return global_vi_result, group_results
```

### Model Selection

```python
def model_selection(models, data):
    # Use SMC for marginal likelihood computation
    evidences = {}

    for model_name, model in models.items():
        particles = init(model, model_args, n_particles=const(2000),
                        constraints={"obs": data})
        evidences[model_name] = particles.log_marginal_likelihood()

    # Select model with highest evidence
    best_model = max(evidences, key=evidences.get)
    return best_model, evidences
```

## Integration with Other GenJAX Modules

### With State Space Models
```python
from genjax.extras.state_space import discrete_hmm, linear_gaussian

# SMC for state space models
def state_space_inference(model, observations):
    # Use exact inference for validation
    exact_result = forward_filter(observations, ...)

    # Use SMC for approximate inference
    smc_result = rejuvenation_smc(model, observations, ...)

    # Compare log marginal likelihoods
    error = abs(exact_result.log_marginal - smc_result.log_marginal)
    return smc_result, error
```

### With ADEV
```python
from genjax.adev import adev

# VI with automatic gradient estimation
@adev(normal="reparam", categorical="reinforce")
def variational_model(params):
    return complex_probabilistic_computation(params)

# Use with VI module
result = variational_inference(
    target_model, target_args, constraints,
    variational_family=CustomFamily(variational_model),
    ...
)
```

## References

### Theoretical Background
- **MCMC**: Robert & Casella, "Monte Carlo Statistical Methods"
- **SMC**: Doucet & Johansen, "A Tutorial on Particle Filtering and Smoothing"
- **VI**: Blei et al., "Variational Inference: A Review for Statisticians"

### Implementation Details
All algorithms are implemented using JAX primitives for performance and composability. The implementations follow mathematical specifications from the theoretical literature while being optimized for practical use in probabilistic programming.
