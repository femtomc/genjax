# extras/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX extras module.

## Overview

The `extras` module contains additional functionality that builds on GenJAX core capabilities but extends beyond the main inference algorithms. Currently, it focuses on state space models with exact inference for testing approximate methods.

## Module Structure

```
src/genjax/extras/
├── __init__.py          # Module exports
├── state_space.py       # State space models with exact inference
└── CLAUDE.md           # This file
```

## State Space Models (`state_space.py`)

### Purpose

Provides exact inference algorithms for discrete and continuous state space models to serve as baselines for testing approximate inference methods (MCMC, SMC, VI).

### Model Types

#### 1. Discrete Hidden Markov Models (HMMs)
- **Generative function**: `discrete_hmm`
- **Exact inference**: Forward filtering backward sampling (FFBS)
- **Use case**: Testing discrete latent variable inference

#### 2. Linear Gaussian State Space Models
- **Generative function**: `linear_gaussian_ssm`
- **Exact inference**: Kalman filtering and smoothing
- **Use case**: Testing continuous state inference

### Inference Testing API

**CRITICAL**: All testing should use the inference testing API for consistency and ease of use.

#### Core Testing Functions

```python
from genjax.extras.state_space import (
    # Inference testing API - use these for testing
    discrete_hmm_test_dataset,
    discrete_hmm_exact_log_marginal,
    discrete_hmm_inference_problem,
    linear_gaussian_test_dataset,
    linear_gaussian_exact_log_marginal,
    linear_gaussian_inference_problem,
)
```

#### Standardized Dataset Format

**All testing functions return datasets in this format**:
```python
dataset = {
    "z": latent_states,    # True latent sequence for validation
    "obs": observations    # Observed sequence for inference
}
```

#### Testing Patterns

**Pattern 1: Generate dataset and evaluate separately**:
```python
# Generate test dataset
dataset = discrete_hmm_test_dataset(initial_probs, transition_matrix, emission_matrix, T)

# Compute exact log marginal likelihood
exact_log_marginal = discrete_hmm_exact_log_marginal(
    dataset["obs"], initial_probs, transition_matrix, emission_matrix
)
```

**Pattern 2: One-call inference problem generation (RECOMMENDED)**:
```python
# Generate complete inference problem in one call
dataset, exact_log_marginal = discrete_hmm_inference_problem(
    initial_probs, transition_matrix, emission_matrix, T
)

# Now test your approximate algorithm
approximate_log_marginal = your_inference_algorithm(dataset["obs"], ...)
error = jnp.abs(approximate_log_marginal - exact_log_marginal)
```

### Example: Testing SMC vs Exact HMM Inference

```python
from genjax.pjax import seed
from genjax.smc import init
from genjax.extras.state_space import discrete_hmm_inference_problem

# Generate inference problem
seeded_problem = seed(lambda: discrete_hmm_inference_problem(
    initial_probs, transition_matrix, emission_matrix, T=20
))
dataset, exact_log_marginal = seeded_problem(key)

# Test SMC approximation
constraints = {"obs": dataset["obs"]}
smc_result = init(discrete_hmm, model_args, n_particles=1000, constraints=constraints)
smc_log_marginal = smc_result.log_marginal_likelihood()

# Validate accuracy
error = jnp.abs(smc_log_marginal - exact_log_marginal)
assert error < tolerance, f"SMC error {error} exceeds tolerance {tolerance}"
```

### Example: Testing MCMC vs Exact Kalman Filtering

```python
from genjax.mcmc import mh, chain
from genjax.extras.state_space import linear_gaussian_inference_problem

# Generate inference problem
seeded_problem = seed(lambda: linear_gaussian_inference_problem(
    initial_mean, initial_cov, A, Q, C, R, T=15
))
dataset, exact_log_marginal = seeded_problem(key)

# Test MCMC approximation (would need appropriate constraints setup)
# ... MCMC inference code ...

# Compare results
# ... validation code ...
```

### Critical Guidelines for Testing

1. **Always use the inference testing API** (`*_test_dataset`, `*_exact_log_marginal`, `*_inference_problem`)
2. **Use inference problem generators** (`*_inference_problem`) for new tests - they're more convenient
3. **Validate dataset format** - ensure `{"z": ..., "obs": ...}` structure
4. **Test convergence properties** - increasing computational resources should improve accuracy
5. **Use proper seeding** - wrap functions with `seed()` before calling with JAX keys

### Model Parameters

#### Discrete HMM Parameters
- `initial_probs`: Initial state distribution (K,)
- `transition_matrix`: State transition probabilities (K, K)
- `emission_matrix`: Observation emission probabilities (K, M)
- `T`: Number of time steps

#### Linear Gaussian Parameters
- `initial_mean`: Initial state mean (d_state,)
- `initial_cov`: Initial state covariance (d_state, d_state)
- `A`: State transition matrix (d_state, d_state)
- `Q`: Process noise covariance (d_state, d_state)
- `C`: Observation matrix (d_obs, d_state)
- `R`: Observation noise covariance (d_obs, d_obs)
- `T`: Number of time steps

### Implementation Details

#### Exact Inference Algorithms

**Discrete HMM**:
- Forward filtering: `forward_filter()` - computes p(x_t | y_{1:t}) in log space
- Backward sampling: `backward_sample()` - samples states given forward messages
- Log marginal: Sum over forward filter final messages

**Linear Gaussian**:
- Kalman filtering: `kalman_filter()` - computes p(x_t | y_{1:t}) with Gaussian messages
- Kalman smoothing: `kalman_smoother()` - computes p(x_t | y_{1:T}) using RTS smoother
- Log marginal: Sum of innovation log-likelihoods

#### JAX Integration

**PJAX Compatibility**:
- All generative functions use PJAX primitives
- Must use `seed()` transformation before JAX operations
- Compatible with `jit`, `vmap`, `grad` after seeding

**Addressing Structure**:
- Initial step: `"state_0"`, `"obs_0"`
- Remaining steps: `"scan_steps"` containing vectorized `"state"` and `"obs"`

### Error Handling

Common issues and solutions:

**LoweringSamplePrimitiveToMLIRException**:
```python
# ❌ Wrong
dataset = discrete_hmm_test_dataset(...)

# ✅ Correct
seeded_fn = seed(lambda: discrete_hmm_test_dataset(...))
dataset = seeded_fn(key)
```

**Shape Mismatches**:
- Ensure covariance matrices are positive definite
- Check observation matrix dimensions match state dimensions
- Verify time series length T > 1 for scan operations

### Testing Strategy

**For New Inference Algorithms**:

1. **Start with simple problems** - small T, well-conditioned parameters
2. **Test convergence** - increasing compute should improve accuracy
3. **Compare multiple model types** - validate on both discrete and continuous
4. **Validate edge cases** - T=1, degenerate parameters, extreme noise levels

**For Algorithm Comparison**:
1. **Use same inference problems** - generate once, test multiple algorithms
2. **Test multiple difficulties** - vary T, noise levels, model complexity
3. **Report convergence curves** - plot error vs computational cost
4. **Test robustness** - random seeds, initialization sensitivity

### Performance Notes

- **Kalman filtering**: O(T * d_state^3) complexity
- **HMM forward filtering**: O(T * K^2) complexity
- **Memory usage**: Linear in T for all algorithms
- **JAX compilation**: First call slower due to compilation, subsequent calls fast

### References

See the comprehensive reference list in `state_space.py` for theoretical background on:
- Hidden Markov Models (Rabiner, Bishop, Murphy)
- Kalman Filtering (Kalman, Sarkka, Murphy)
- Forward Filtering Backward Sampling (Carter & Kohn, Frühwirth-Schnatter)
- Rauch-Tung-Striebel Smoothing (Rauch et al., Anderson & Moore)
