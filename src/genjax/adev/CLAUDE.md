# adev/CLAUDE.md

This file provides guidance for Claude Code when working with GenJAX's Automatic Differentiation of Expected Values (ADEV) module.

**For core GenJAX concepts**, see `../CLAUDE.md`
**For inference algorithms using ADEV**, see `../inference/CLAUDE.md`
**For academic references**, see `REFERENCES.md`

## Overview

The `adev` module provides automatic differentiation capabilities specifically designed for unbiased gradient estimation of expected values. ADEV (Automatic Differentiation of Expectation Values) is a source-to-source transformation that extends forward-mode automatic differentiation to correctly handle probabilistic computations, based on the research presented in "ADEV: Sound Automatic Differentiation of Expected Values of Probabilistic Programs" (Lew et al., POPL 2023).

## Module Structure

```
src/genjax/adev/
├── __init__.py          # Main ADEV implementation (moved from adev.py)
├── CLAUDE.md           # This file
└── REFERENCES.md       # Academic references
```

## Core Concepts

### What is ADEV?

ADEV solves a fundamental problem in probabilistic machine learning: computing gradients of expectations ∇E[f(X)] where X is a random variable. Traditional automatic differentiation fails here because:
- The expectation operator E[·] doesn't have a simple chain rule
- Different distributions require different gradient estimation strategies
- Naive approaches produce biased estimates

ADEV provides a principled framework that:
1. Transforms probabilistic programs using continuation-passing style (CPS)
2. Allows modular selection of gradient strategies per distribution
3. Guarantees unbiased gradient estimates with provable soundness

### Key Components

#### 1. Dual Numbers
- **Type**: `Dual(primal, tangent)`
- **Purpose**: Carries value and derivative information through computations
- **Usage**: Internal representation for forward-mode AD

#### 2. ADEVPrimitive
- **Purpose**: Base class for stochastic operations with custom gradient strategies
- **Methods**:
  - `sample(*args)`: Forward sampling operation
  - `prim_jvp_estimate(dual_tree, konts)`: Custom gradient estimation strategy
- **Design**: Enables modular gradient estimation per distribution type

#### 3. Continuation-Passing Style (CPS)
- **Pure continuation (kpure)**: Evaluates on primal values only
- **Dual continuation (kdual)**: Applies ADEV transformation
- **Usage**: Enables different gradient strategies to compose correctly

#### 4. Expectation Decorator
- **Syntax**: `@expectation`
- **Creates**: Expectation object with gradient estimation methods
- **Methods**:
  - `grad_estimate(*args)`: Compute unbiased gradient estimate
  - `estimate(*args)`: Compute expectation value only
  - `jvp_estimate(*duals)`: Low-level JVP computation

### Gradient Estimators

ADEV provides several gradient estimators for different types of random variables:

#### Reparameterization Trick (`reparam`)
- **Use case**: Continuous variables with reparameterizable distributions
- **Distributions**: Normal, Beta (with appropriate transformations)
- **Advantages**: Low variance, exact gradients for simple cases
- **Implementation**: `normal_reparam`, `multivariate_normal_reparam`
- **Theory**: ∇E[f(g(ε; θ))] = E[∇f(g(ε; θ))] where g is differentiable

#### REINFORCE (Score Function)
- **Use case**: Discrete variables or non-reparameterizable continuous variables
- **Distributions**: Categorical, Bernoulli, Geometric, any with differentiable log-pdf
- **Advantages**: General applicability
- **Disadvantages**: High variance, requires variance reduction
- **Implementation**: `normal_reinforce`, `flip_reinforce`, `geometric_reinforce`
- **Theory**: ∇E[f(X)] = E[f(X) * ∇log p(X; θ)]

#### Enumeration
- **Use case**: Discrete variables with small support
- **Distributions**: Categorical with few categories, Bernoulli
- **Advantages**: Exact gradients, zero variance
- **Disadvantages**: Exponential complexity in number of variables
- **Implementation**: `flip_enum`, `flip_enum_parallel`, `categorical_enum_parallel`
- **Theory**: ∇E[f(X)] = Σ_x p(x; θ) * ∇f(x) (exact computation)

#### Measure-Valued Derivatives (MVD)
- **Use case**: Variance reduction for discrete variables
- **Advantages**: Lower variance than standard REINFORCE
- **Implementation**: `flip_mvd`
- **Theory**: Uses phantom estimation with discrete differences

## API Reference

### Core Classes

**Class**: `Dual(primal, tangent)`
**Location**: `__init__.py:214-349`
**Purpose**: Forward-mode AD data structure
**Key Methods**:
- `tree_pure(v)`: Convert tree to dual with zero tangents
- `dual_tree(primals, tangents)`: Combine into dual tree
- `tree_primal(v)`: Extract primal values
- `tree_tangent(v)`: Extract tangent values

**Class**: `ADEVPrimitive()`
**Location**: `__init__.py:100-171`
**Purpose**: Base class for custom gradient strategies
**Abstract Methods**:
- `sample(*args)`: Forward sampling
- `prim_jvp_estimate(dual_tree, konts)`: Gradient estimation

**Class**: `Expectation(prog)`
**Location**: `__init__.py:629-773`
**Purpose**: User-facing interface for gradient estimation
**Key Methods**:
- `grad_estimate(*primals)`: Compute gradient estimate
- `estimate(*args)`: Compute expectation value
- `jvp_estimate(*duals)`: Low-level JVP interface

### Gradient Estimator Primitives

**Continuous Distributions (Reparameterization)**:
- `normal_reparam(loc, scale)`: Normal distribution with pathwise gradients
- `multivariate_normal_reparam(loc, cov)`: Multivariate normal with Cholesky

**Continuous Distributions (REINFORCE)**:
- `normal_reinforce(loc, scale)`: Normal with score function gradients
- `multivariate_normal_reinforce(loc, cov)`: Multivariate normal REINFORCE

**Discrete Distributions**:
- `flip_enum(p)`: Exact Bernoulli gradients via enumeration
- `flip_reinforce(p)`: Bernoulli with REINFORCE
- `flip_mvd(p)`: Bernoulli with measure-valued derivatives
- `geometric_reinforce(p)`: Geometric distribution REINFORCE
- `categorical_enum_parallel(probs)`: Categorical enumeration

### Factory Functions

**Function**: `expectation(source) -> Expectation`
**Location**: `__init__.py:775-833`
**Purpose**: Decorator to create gradient-enabled expectations
**Usage**:
```python
@expectation
def objective(theta):
    x = normal_reparam(theta, 1.0)
    return x**2
```

**Function**: `reinforce(sample_func, logpdf_func) -> REINFORCE`
**Location**: `__init__.py:998-1012`
**Purpose**: Create REINFORCE estimator for any distribution
**Usage**: Internal factory for creating score function estimators

## Usage Patterns

### Basic Gradient Estimation

```python
from genjax.adev import expectation, normal_reparam

@expectation
def quadratic_loss(mu, sigma):
    x = normal_reparam(mu, sigma)
    return x**2

# Single parameter gradient
grad = quadratic_loss.grad_estimate(0.5, 1.0)

# Just compute expectation
value = quadratic_loss.estimate(0.5, 1.0)
```

### Mixed Gradient Strategies

```python
@expectation
def mixed_objective(theta, p):
    # Use reparameterization for continuous
    x = normal_reparam(theta, 1.0)
    # Use enumeration for discrete
    b = flip_enum(p)
    return x * jnp.float32(b)

grad_theta, grad_p = mixed_objective.grad_estimate(0.5, 0.3)
```

### Integration with Variational Inference

```python
from genjax.inference import elbo_factory
from genjax.adev import expectation, normal_reparam

@gen
def variational_family(data, theta):
    normal_reparam(theta, 1.0) @ "x"

# ELBO automatically uses ADEV for gradient estimation
elbo = elbo_factory(target_model, variational_family, data)
gradient = elbo.grad_estimate(initial_theta)
```

### Working with GenJAX Models

```python
@gen
def model_with_adev():
    # ADEV estimators work seamlessly with @ addressing
    x = normal_reparam(0.0, 1.0) @ "x"
    y = flip_enum(0.7) @ "y"
    return x + jnp.float32(y)

# Compatible with seed transformation
result = seed(model_with_adev.simulate)(jrand.key(42))
```

## Common Patterns and Best Practices

### 1. Choosing Gradient Estimators

**Decision Tree**:
1. Is the variable continuous and reparameterizable?
   - Yes → Use `*_reparam` (lowest variance)
   - No → Continue to 2
2. Is the variable discrete with small support (≤10 categories)?
   - Yes → Use `*_enum` (exact gradients)
   - No → Continue to 3
3. Use `*_reinforce` (most general, higher variance)

### 2. Variance Reduction

For high-variance estimators (REINFORCE):
- Use control variates (not yet implemented in GenJAX)
- Increase sample size in expectation
- Consider continuous relaxations for discrete variables

### 3. Numerical Stability

- For multivariate normal, ensure covariance is positive definite
- Use Cholesky parameterization for covariance matrices
- Check for finite gradients in optimization loops

### 4. Performance Considerations

- Enumeration has exponential cost in number of discrete variables
- Reparameterization is typically fastest per sample
- REINFORCE may need more samples for stable gradients
- Use `modular_vmap` for batched gradient estimation

## Error Handling

### Common Issues

1. **Non-finite gradients**:
   - Check distribution parameters are valid
   - Ensure objective function is differentiable
   - Consider gradient clipping in optimization

2. **High variance estimates**:
   - Use reparameterization when possible
   - Increase number of samples
   - Check for numerical issues in log-probabilities

3. **Type errors with discrete variables**:
   - Convert boolean to float: `jnp.float32(b)`
   - Use appropriate tangent types for discrete values

### Debugging Tips

1. **Test gradient estimates**:
   ```python
   # Compare different estimators
   grad_reparam = obj_reparam.grad_estimate(theta)
   grad_reinforce = obj_reinforce.grad_estimate(theta)
   ```

2. **Check expectation values**:
   ```python
   # Verify forward pass works
   value = objective.estimate(theta)
   assert jnp.isfinite(value)
   ```

3. **Validate with known solutions**:
   - Linear objectives should have constant gradients
   - Enumeration should give exact results for small discrete spaces

## Integration with Other GenJAX Modules

### With Inference Module

ADEV is the foundation for gradient-based inference in GenJAX:

- **Variational Inference**: Automatic gradient estimation for ELBO
- **Maximum Likelihood**: Gradient-based optimization of model parameters
- **Amortized Inference**: Training inference networks with gradient estimates

### With Core GFI

ADEV estimators are fully compatible with GenJAX's addressing system:
- Use `@` operator with any ADEV primitive
- Works with `seed` transformation
- Compatible with `modular_vmap` for batching

### With PJAX

ADEV primitives integrate with PJAX infrastructure:
- Proper handling of random key splitting
- Compatible with JAX transformations
- Supports addressing and trace operations

## Advanced Topics

### Custom Gradient Estimators

To implement a new gradient estimator:

1. Subclass `ADEVPrimitive`
2. Implement `sample` method
3. Implement `prim_jvp_estimate` with your gradient strategy
4. Wrap with `distribution` for GFI compatibility

Example:
```python
@Pytree.dataclass
class MyCustomEstimator(ADEVPrimitive):
    def sample(self, *args):
        # Your sampling logic
        pass

    def prim_jvp_estimate(self, dual_tree, konts):
        # Your gradient estimation strategy
        pass
```

### Theoretical Guarantees

ADEV provides several theoretical guarantees:

1. **Unbiasedness**: All gradient estimates satisfy E[∇̂f] = ∇E[f]
2. **Composability**: Different strategies compose correctly
3. **Soundness**: Proven via logical relations in the paper

### Performance Optimization

1. **JIT Compilation**:
   ```python
   @jax.jit
   def compute_gradient(theta):
       return objective.grad_estimate(theta)
   ```

2. **Vectorization**:
   ```python
   # Compute multiple gradient estimates
   grads = modular_vmap(objective.grad_estimate)(thetas)
   ```

3. **Caching**:
   - ADEV transformations are cached internally
   - Reuse `Expectation` objects when possible

## Limitations and Future Work

### Current Limitations

1. **Variance reduction**: Limited built-in variance reduction techniques
2. **Higher-order gradients**: Focus on first-order gradients
3. **Implicit reparameterization**: Not yet implemented
4. **Control variates**: No automatic control variate selection

### Planned Improvements

1. Advanced variance reduction methods
2. Support for implicit reparameterization gradients
3. Automatic selection of gradient strategies
4. Integration with more distribution families

## References

For detailed theoretical background and implementation details, see:
- `REFERENCES.md` in this directory for academic papers
- Source code in `__init__.py` for implementation details
- Test suite in `tests/test_adev.py` for usage examples
