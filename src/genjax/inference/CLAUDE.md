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

### Core Components

#### Particle Initialization (`init`)
Initialize particle collection with importance sampling:

```python
from genjax.inference import init

particles = init(
    target_gf=model,
    target_args=args,
    n_samples=const(1000),
    constraints={"obs": observed_data},
    proposal_gf=custom_proposal  # Optional custom proposal
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

particles = extend(
    particles,
    extended_target_gf=extended_model,
    extended_target_args=extended_args,
    constraints={"new_obs": observed_value}
)
```

**Rejuvenation Move** - Apply MCMC to combat degeneracy:
```python
from genjax.inference import rejuvenate

def mcmc_kernel(trace):
    return mh(trace, sel("latent_state"))

particles = rejuvenate(particles, mcmc_kernel)
# Weights remain unchanged due to detailed balance
```

**Resampling** - Combat particle degeneracy:
```python
from genjax.inference import resample

# Resample when effective sample size drops
ess = particles.effective_sample_size()
if ess < threshold:
    particles = resample(particles, method="systematic")
```

### Complete SMC Algorithm

```python
from genjax.inference import rejuvenation_smc

# Full SMC with rejuvenation
final_particles = rejuvenation_smc(
    initial_model=model_t0,
    extended_model=model_extended,
    transition_proposal=transition_proposal,
    mcmc_kernel=lambda trace: mh(trace, sel("latent")),
    observations=time_series_data,
    choice_fn=lambda x: x,  # Identity mapping
    n_particles=const(1000)
)
```

### SMC Best Practices

#### Particle Count Guidelines
```python
# Start with moderate particle counts
n_particles_small = const(100)    # Prototyping
n_particles_medium = const(1000)  # Standard inference
n_particles_large = const(5000)   # High-precision applications
```

#### Effective Sample Size Monitoring
```python
def adaptive_resampling(particles, threshold=0.5):
    ess = particles.effective_sample_size()
    n_particles = particles.n_particles
    
    if ess / n_particles < threshold:
        particles = resample(particles)
    
    return particles
```

#### Choice Function Constraints
```python
# VALID choice functions (bijective on address space)
identity_fn = lambda x: x
remap_fn = lambda d: {"new_key": d["old_key"]}

# INVALID choice functions (modify values)
invalid_fn = lambda d: {"key": d["key"] + 1}     # Modifies values
invalid_fn2 = lambda d: {"key": d["k1"] + d["k2"]}  # Combines values
```

## Variational Inference (`vi.py`)

### Core Components

#### Variational Families

**Mean Field Normal Family**:
```python
from genjax.inference import MeanFieldNormalFamily

family = MeanFieldNormalFamily(
    parameter_names=["mu", "sigma"],
    estimator_mapping={"normal": "reparam"}  # Uses ADEV
)
```

**Full Covariance Normal Family**:
```python
from genjax.inference import FullCovarianceNormalFamily

family = FullCovarianceNormalFamily(
    parameter_names=["mu", "sigma"],
    estimator_mapping={"normal": "reparam"}
)
```

#### ELBO Factory
```python
from genjax.inference import elbo_factory

# Create ELBO function
elbo_fn = elbo_factory(
    target_gf=target_model,
    variational_gf=variational_model,
    estimator_mapping={"normal": "reparam", "categorical": "reinforce"}
)

# Compute ELBO
elbo_value = elbo_fn(
    variational_params=var_params,
    target_args=target_args,
    constraints=constraints,
    n_samples=const(100)
)
```

#### Complete VI Pipeline
```python
from genjax.inference import variational_inference

result = variational_inference(
    target_gf=target_model,
    target_args=(),
    constraints={"obs": observed_data},
    variational_family=MeanFieldNormalFamily(["param1", "param2"]),
    n_samples=const(100),
    n_steps=const(1000),
    learning_rate=0.01
)

# Access results
final_params = result.params
loss_history = result.losses
```

### VI Best Practices

#### Learning Rate Scheduling
```python
# Use adaptive learning rates
import optax

# Start high, decay over time
scheduler = optax.exponential_decay(
    init_value=0.1,
    transition_steps=100,
    decay_rate=0.95
)

optimizer = optax.adam(scheduler)
```

#### Convergence Monitoring
```python
def check_vi_convergence(losses, window=100, threshold=1e-4):
    if len(losses) < window:
        return False
    
    recent_losses = losses[-window:]
    loss_change = abs(recent_losses[-1] - recent_losses[0]) / window
    
    return loss_change < threshold
```

#### Sample Size Guidelines
```python
# Progressive sample size increase
sample_schedule = [
    (100, 100),    # (n_samples, n_steps) - early exploration
    (500, 500),    # medium precision
    (1000, 1000)   # final high precision
]

for n_samples, n_steps in sample_schedule:
    result = variational_inference(..., n_samples=const(n_samples), n_steps=const(n_steps))
    if check_convergence(result.losses):
        break
```

## Algorithm Selection Guidelines

### When to Use Each Algorithm

#### MCMC
- **Best for**: Exact sampling from posterior (asymptotically)
- **Use when**: 
  - High-precision posterior estimates needed
  - Model has complex dependencies
  - Computational time is not critical
- **Avoid when**: Real-time inference required

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