# extras/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX extras module.

**For core GenJAX concepts**, see `../CLAUDE.md`
**For inference algorithms to test**, see `../inference/CLAUDE.md`

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

**Location**: `state_space.py`
**Purpose**: Generate test datasets and compute exact baselines

**Discrete HMM Functions**:
- `discrete_hmm_test_dataset()`: Generate synthetic data
- `discrete_hmm_exact_log_marginal()`: Compute exact log p(y)
- `discrete_hmm_inference_problem()`: One-call dataset + baseline

**Linear Gaussian Functions**:
- `linear_gaussian_test_dataset()`: Generate synthetic data
- `linear_gaussian_exact_log_marginal()`: Compute exact log p(y)
- `linear_gaussian_inference_problem()`: One-call dataset + baseline

#### Standardized Dataset Format

**All functions return**:
- `"z"`: True latent states (for validation)
- `"obs"`: Observations (for inference)

#### Testing Patterns

**Pattern 1: Generate dataset and evaluate separately**
- Call `*_test_dataset()` to generate data
- Call `*_exact_log_marginal()` for baseline
- Compare with approximate inference results

**Pattern 2: One-call inference problem (RECOMMENDED)**
- Call `*_inference_problem()` for dataset + baseline
- Returns tuple: `(dataset, exact_log_marginal)`
- More convenient for testing workflows

### Example Use Cases

#### Testing SMC vs Exact HMM Inference
1. Use `discrete_hmm_inference_problem()` to generate test case
2. Apply `seed()` transformation before calling with key
3. Run SMC with `init()` using observations as constraints
4. Compare `log_marginal_likelihood()` with exact baseline
5. See test files for complete examples

#### Testing MCMC vs Exact Kalman Filtering
1. Use `linear_gaussian_inference_problem()` for test case
2. Run MCMC chain on linear Gaussian model
3. Compare posterior samples with exact Kalman smoother
4. Validate log marginal likelihood estimates
5. See test files for implementation patterns

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
- **Forward Filter**: `forward_filter()` in `state_space.py`
  - Computes p(x_t | y_{1:t}) in log space
  - Returns log messages for each timestep
- **Backward Sampling**: `backward_sample()` in `state_space.py`
  - Samples states given forward messages
  - Used for FFBS algorithm
- **Log Marginal**: Sum of final forward messages

**Linear Gaussian**:
- **Kalman Filter**: `kalman_filter()` in `state_space.py`
  - Computes p(x_t | y_{1:t}) as Gaussian
  - Returns means, covariances, log likelihoods
- **Kalman Smoother**: `kalman_smoother()` in `state_space.py`
  - Computes p(x_t | y_{1:T}) via RTS algorithm
  - Returns smoothed means and covariances
- **Log Marginal**: Sum of innovation log-likelihoods

#### JAX Integration

**PJAX Compatibility**:
- All generative functions use PJAX primitives
- Must use `seed()` transformation before JAX operations
- Compatible with `jit`, `vmap`, `grad` after seeding

**Addressing Structure**:
- Initial step: `"state_0"`, `"obs_0"`
- Remaining steps: `"scan_steps"` containing vectorized `"state"` and `"obs"`

### Error Handling

#### Common Issues

**LoweringSamplePrimitiveToMLIRException**:
- **Cause**: Calling PJAX functions without `seed()` transformation
- **Solution**: Wrap function calls with `seed()` before using JAX key
- **Pattern**: `seeded_fn = seed(lambda: fn(...)); result = seeded_fn(key)`

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

**Location**: See docstring in `state_space.py` for comprehensive reference list

**Topics Covered**:
- Hidden Markov Models theory and algorithms
- Kalman filtering and smoothing
- Forward filtering backward sampling (FFBS)
- Rauch-Tung-Striebel (RTS) smoothing

**Related Files**:
- Test implementations: `tests/test_extras.py`
- Example usage: Various test files using exact baselines
