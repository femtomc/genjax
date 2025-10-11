# GenJAX Gaussian Process Module Guide

This guide covers the GenJAX-native Gaussian Process implementation that treats GPs as generative functions with exact inference as the internal proposal.

## Overview

The GP module implements Gaussian processes as first-class GenJAX generative functions, enabling:
- Composability with other probabilistic programs
- Exact GP inference as the proposal distribution
- Full support for the Generative Function Interface (GFI)
- Integration with GenJAX's inference algorithms

## Core Components

### GP Generative Function

**Class**: `GP(kernel, mean_fn=None, noise_variance=1e-6, jitter=1e-6)`
**Location**: `gp.py`
**Purpose**: Main GP implementation as a generative function

**Key Features**:
- Implements full GFI: `simulate`, `assess`, `generate`, `update`, `regenerate`
- Uses exact Gaussian conditioning for efficient inference
- Supports conditioning on training data via `x_train`, `y_train` arguments
- Numerical stability via jitter parameter

**Usage Pattern**:
```python
# Create GP
kernel = RBF(variance=1.0, lengthscale=0.5)
gp = GP(kernel, noise_variance=0.01)

# Use in @gen function
@gen
def model(x_train, y_train, x_test):
    y_test = gp(x_test, x_train=x_train, y_train=y_train) @ "gp"
    return y_test
```

### Kernel Functions

**Location**: `kernels.py`
**Purpose**: Covariance functions for GPs

**Available Kernels**:
- `RBF(variance, lengthscale)`: Radial Basis Function (squared exponential)
- `Matern12(variance, lengthscale)`: Matérn 1/2 (exponential)
- `Matern32(variance, lengthscale)`: Matérn 3/2
- `Matern52(variance, lengthscale)`: Matérn 5/2
- `Linear(variance, offset, input_dim)`: Linear kernel
- `Polynomial(variance, offset, degree)`: Polynomial kernel
- `White(variance)`: White noise kernel
- `Constant(variance)`: Constant kernel

**Kernel Combinators**:
- `Sum(k1, k2)`: Additive combination
- `Product(k1, k2)`: Multiplicative combination

**ARD Support**: Automatic Relevance Determination via per-dimension lengthscales:
```python
lengthscales = jnp.array([0.5, 1.0, 2.0])  # Different scale per dimension
kernel = RBF(variance=1.0, lengthscale=lengthscales)
```

### Mean Functions

**Location**: `mean.py`
**Purpose**: Prior mean functions for GPs

**Available Mean Functions**:
- `Zero()`: Zero mean (default)
- `Constant(value)`: Constant mean
- `Linear(weights, bias)`: Linear mean function

**Usage**:
```python
mean_fn = Linear(weights=jnp.array([0.5, -0.3]), bias=1.0)
gp = GP(kernel, mean_fn=mean_fn)
```

## Key Design Decisions

### Exact Inference as Proposal

The GP uses exact Gaussian conditioning as its internal proposal distribution. This means:
- When all outputs are constrained, the importance weight is 0 (log space)
- Partial constraints use conditional Gaussian distributions
- No MCMC or variational approximation needed for GP inference

### Integration with GFI

**Trace Structure**: `GPTrace` contains:
- Training data (`x_train`, `y_train`)
- Test locations (`x_test`)
- Generated/constrained values (`y_test`)
- Score (negative log probability)

**Choices**: The random choices are the function values at test points (`y_test`)

### Numerical Stability

- Cholesky decomposition for covariance matrix inversion
- Jitter added to diagonal for numerical stability
- Log-space computations for probabilities

## Common Patterns

### Basic GP Regression

```python
# Fixed hyperparameters
kernel = RBF(variance=1.0, lengthscale=0.5)
gp = GP(kernel, noise_variance=0.01)

# Sample from posterior
trace = gp.simulate(x_test, x_train=x_train, y_train=y_train, key=key)
y_pred = trace.get_retval()
```

### Hierarchical GP Model

```python
@gen
def hierarchical_gp(x_train, y_train, x_test):
    # Sample hyperparameters
    lengthscale = exponential(1.0) @ "lengthscale"
    variance = gamma(2.0, 2.0) @ "variance"

    # Create GP with sampled hyperparameters
    kernel = RBF(variance=variance, lengthscale=lengthscale)
    gp = GP(kernel, noise_variance=0.01)

    # Sample function values
    f_test = gp(x_test, x_train=x_train, y_train=y_train) @ "f_test"

    return f_test
```

### Additive GP Model

```python
@gen
def additive_model(x_train, y_train, x_test):
    # Long-term trend
    gp_trend = GP(RBF(variance=1.0, lengthscale=2.0))
    trend = gp_trend(x_test, x_train, y_train) @ "trend"

    # Short-term variations
    gp_local = GP(Matern32(variance=0.5, lengthscale=0.1))
    local = gp_local(x_test) @ "local"

    # Combine
    return trend + local
```

## Testing and Validation

The implementation is validated against TinyGP, a well-tested GP library:
- Kernel computations match exactly
- Log likelihood calculations are consistent
- Posterior statistics align with analytical solutions

See `tests/test_gp_tinygp_comparison.py` for comprehensive validation.

## Performance Considerations

- **Cubic Scaling**: Standard GP inference scales as O(n³) with training points
- **Memory**: O(n²) memory for covariance matrices
- **JAX Compilation**: All operations are JAX-compatible and JIT-compilable
- **Vectorization**: Kernel computations are fully vectorized

## Future Extensions

Potential enhancements could include:
- Sparse GP approximations (inducing points)
- Multi-output GPs
- Non-Gaussian likelihoods
- Specialized kernels (periodic, spectral mixture)
- Partial regeneration based on selection

## References

- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning
- TinyGP documentation: https://tinygp.readthedocs.io/
