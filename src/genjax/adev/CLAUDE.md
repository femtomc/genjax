# adev/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX automatic differentiation for variational estimates (ADEV) module.

## Overview

The `adev` module provides automatic differentiation capabilities specifically designed for variational inference in probabilistic programming. It implements various gradient estimators that enable efficient optimization of variational objectives.

## Module Structure

```
src/genjax/adev/
├── __init__.py          # Main ADEV implementation (moved from adev.py)
└── CLAUDE.md           # This file
```

## Core Concepts

### Gradient Estimators

ADEV provides several gradient estimators for different types of random variables:

#### Reparameterization Trick
- **Use case**: Continuous variables with reparameterizable distributions
- **Distributions**: Normal, Beta (with appropriate transformations)
- **Advantages**: Low variance, exact gradients for simple cases
- **Implementation**: `reparam` estimator

#### REINFORCE (Score Function)
- **Use case**: Discrete variables or non-reparameterizable continuous variables  
- **Distributions**: Categorical, Bernoulli, Geometric
- **Advantages**: General applicability
- **Disadvantages**: High variance, requires variance reduction
- **Implementation**: `reinforce` estimator

#### Enumeration
- **Use case**: Discrete variables with small support
- **Distributions**: Categorical with few categories, Bernoulli
- **Advantages**: Exact gradients, zero variance
- **Disadvantages**: Exponential complexity in number of variables
- **Implementation**: `enum_exact` estimator

#### Multi-Sample Variance Reduction (MVD)
- **Use case**: Variance reduction for discrete variables
- **Advantages**: Lower variance than standard REINFORCE
- **Implementation**: `mvd` estimator

### Estimator Selection

**Automatic Selection**: ADEV can automatically choose appropriate estimators based on distribution types:

```python
from genjax.adev import adev

# Automatically selects reparam for normal, reinforce for categorical
@adev(normal="reparam", categorical="reinforce")
def variational_model():
    mu = normal(0.0, 1.0) @ "mu"
    category = categorical(jnp.array([0.3, 0.7])) @ "category"
    return mu + category
```

**Manual Selection**: For fine-grained control:

```python
# Use enumeration for small discrete spaces
@adev(categorical="enum_exact")
def small_discrete_model():
    return categorical(jnp.array([0.25, 0.25, 0.25, 0.25])) @ "choice"

# Use MVD for variance reduction
@adev(categorical="mvd")
def variance_reduced_model():
    return categorical(large_logits_array) @ "choice"
```

### ELBO Computation

ADEV integrates with variational inference to compute Evidence Lower BOund (ELBO) gradients:

```python
from genjax.adev import elbo_loss

# Define variational and target models
@adev(normal="reparam")
def variational_model(params):
    return normal(params["mu"], params["sigma"]) @ "x"

@gen
def target_model():
    return normal(2.0, 1.0) @ "x"

# Compute ELBO loss and gradients
def loss_fn(params):
    return elbo_loss(variational_model, target_model, params, {}, n_samples=1000)

# Use with JAX optimization
import optax
optimizer = optax.adam(0.01)
params = {"mu": 0.0, "sigma": 1.0}
opt_state = optimizer.init(params)

for step in range(1000):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

## Integration with GenJAX Inference

### With Variational Inference Module

ADEV works seamlessly with the `genjax.inference.vi` module:

```python
from genjax.inference import MeanFieldNormalFamily, variational_inference
from genjax.adev import adev

# Define target model
@gen
def target():
    mu = normal(0.0, 2.0) @ "mu"
    return normal(mu, 1.0) @ "obs"

# Create variational family with ADEV
variational_family = MeanFieldNormalFamily(
    parameter_names=["mu"],
    estimator_mapping={"normal": "reparam"}  # Uses ADEV internally
)

# Run variational inference
result = variational_inference(
    target, 
    target_args=(), 
    constraints={"obs": 1.5},
    variational_family=variational_family,
    n_samples=const(100),
    n_steps=const(1000)
)
```

### Estimator-Specific Guidelines

#### Reparameterization (`reparam`)
- **Best for**: Normal, Beta, other location-scale families
- **Variance**: Low
- **Computational cost**: Low
- **Requirements**: Distribution must be reparameterizable

#### REINFORCE (`reinforce`)  
- **Best for**: Categorical, Bernoulli, discrete distributions
- **Variance**: High (use variance reduction techniques)
- **Computational cost**: Low per sample
- **Requirements**: Score function must be available

#### Enumeration (`enum_exact`)
- **Best for**: Small discrete spaces (≤10 categories typically)
- **Variance**: Zero (exact)
- **Computational cost**: Exponential in number of variables
- **Requirements**: Finite, small support

#### MVD (`mvd`)
- **Best for**: Categorical with medium-sized support
- **Variance**: Lower than REINFORCE
- **Computational cost**: Higher than REINFORCE
- **Requirements**: Multiple samples for variance reduction

## Advanced Usage

### Mixed Estimators

Combine different estimators in the same model:

```python
@adev(normal="reparam", categorical="enum_exact", geometric="reinforce")
def complex_model(params):
    # Continuous parameter - use reparameterization
    scale = normal(params["scale_mu"], params["scale_sigma"]) @ "scale"
    
    # Small discrete choice - use enumeration
    choice = categorical(jnp.array([0.3, 0.4, 0.3])) @ "choice"
    
    # Count data - use REINFORCE
    count = geometric(params["rate"]) @ "count"
    
    return scale * choice + count
```

### Custom Estimator Configuration

For specialized use cases:

```python
# High-precision requirements
@adev(normal="reparam", categorical="enum_exact")  # Zero variance where possible

# Memory-constrained environments  
@adev(normal="reparam", categorical="reinforce")   # Lower memory usage

# Balanced accuracy/speed
@adev(normal="reparam", categorical="mvd")         # Good variance-speed tradeoff
```

### Debugging and Diagnostics

Monitor gradient estimator performance:

```python
def diagnostic_loss_fn(params):
    # Compute loss with different sample sizes
    losses = []
    for n_samples in [100, 500, 1000]:
        loss = elbo_loss(variational_model, target_model, params, {}, n_samples)
        losses.append(loss)
    
    # Check convergence - losses should stabilize with more samples
    return losses[-1], {"sample_stability": jnp.std(jnp.array(losses))}
```

## Performance Considerations

### Estimator Selection Trade-offs

1. **Accuracy vs Speed**:
   - `enum_exact` > `reparam` > `mvd` > `reinforce` (accuracy)
   - `reinforce` > `reparam` > `mvd` > `enum_exact` (speed for large problems)

2. **Memory Usage**:
   - `enum_exact`: Exponential in number of discrete variables
   - `mvd`: Linear in sample size and support size
   - `reparam`, `reinforce`: Linear in sample size

3. **Convergence Rate**:
   - Low variance estimators converge faster
   - High variance estimators may need more samples or lower learning rates

### Optimization Tips

1. **Use appropriate sample sizes**:
   - Start with n_samples=100 for prototyping
   - Increase to 1000+ for final optimization
   - Monitor loss stability to determine sufficient sample size

2. **Learning rate tuning**:
   - Lower learning rates for high-variance estimators (REINFORCE)
   - Higher learning rates for low-variance estimators (reparam, enum_exact)

3. **Gradient clipping**:
   - Useful for REINFORCE-based estimators
   - Less critical for reparameterization

## Integration with JAX

### JIT Compilation

ADEV works with JAX JIT compilation:

```python
@jax.jit
def compiled_loss_fn(params):
    return elbo_loss(adev_model, target_model, params, constraints, n_samples=100)
```

### Vectorization

Compatible with JAX transformations:

```python
# Vectorize over different parameter initializations
batched_loss_fn = jax.vmap(loss_fn, in_axes=0)
batch_params = {"mu": jnp.array([0.0, 1.0, 2.0]), "sigma": jnp.array([1.0, 1.0, 1.0])}
batch_losses = batched_loss_fn(batch_params)
```

### Differentiation

Works with higher-order derivatives:

```python
# Hessian computation for second-order optimization
hessian_fn = jax.hessian(loss_fn)
hessian = hessian_fn(params)
```

## Common Patterns

### Variational Autoencoder Style

```python
@adev(normal="reparam")  # Reparameterizable encoder
def encoder(data, params):
    # Encode data to latent parameters
    mu = params["encoder_net"](data)
    sigma = jax.nn.softplus(params["encoder_sigma"])
    return normal(mu, sigma) @ "latent"

@gen
def decoder(latent, params):
    # Decode latent to reconstruction
    reconstruction_params = params["decoder_net"](latent)
    return normal(reconstruction_params, 1.0) @ "reconstruction"
```

### Hierarchical Models

```python
@adev(normal="reparam", categorical="enum_exact")
def hierarchical_model(params):
    # Global parameters
    global_mean = normal(params["global_mu"], params["global_sigma"]) @ "global_mean"
    
    # Group assignments (small number of groups)
    group = categorical(params["group_probs"]) @ "group"
    
    # Group-specific parameters
    group_effect = normal(global_mean, params["group_sigma"]) @ f"group_{group}_effect"
    
    return group_effect
```

## Testing and Validation

### Gradient Checking

Validate estimator implementations:

```python
def test_gradient_estimator():
    def loss_fn(params):
        return elbo_loss(variational_model, target_model, params, {}, n_samples=10000)
    
    # Check gradients with large sample size (should be accurate)
    numerical_grad = jax.test_util.check_grads(loss_fn, (params,), order=1)
    
    # Compare with analytical gradients
    analytical_grad = jax.grad(loss_fn)(params)
    
    assert jnp.allclose(numerical_grad, analytical_grad, rtol=0.1)
```

### Convergence Testing

Monitor optimization progress:

```python
def test_convergence():
    losses = []
    params = initial_params
    
    for step in range(1000):
        loss = loss_fn(params)
        losses.append(loss)
        
        # Update params...
    
    # Loss should generally decrease
    assert losses[-1] < losses[0]
    
    # Should converge (loss change becomes small)
    recent_losses = losses[-100:]
    assert jnp.std(recent_losses) < convergence_threshold
```

## Error Handling

### Common Issues

1. **High variance in REINFORCE**:
   ```python
   # Solution: Use variance reduction or switch estimators
   @adev(categorical="mvd")  # Instead of "reinforce"
   ```

2. **Memory issues with enumeration**:
   ```python
   # Solution: Limit to small discrete spaces or use approximation
   @adev(categorical="reinforce")  # Instead of "enum_exact" for large spaces
   ```

3. **Gradient explosion**:
   ```python
   # Solution: Use gradient clipping
   grads = jax.grad(loss_fn)(params)
   clipped_grads = optax.clip_by_global_norm(1.0).update(grads, {}, params)
   ```

## References

### Theoretical Background

- **Reparameterization Trick**: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- **REINFORCE**: Williams (1992), "Simple Statistical Gradient-Following Algorithms"
- **Variance Reduction**: Mnih & Gregor (2014), "Neural Variational Inference"

### Implementation Notes

ADEV builds on JAX's automatic differentiation capabilities while providing specialized handling for probabilistic programs. The estimators are designed to work seamlessly with GenJAX's generative function interface and JAX's compilation system.