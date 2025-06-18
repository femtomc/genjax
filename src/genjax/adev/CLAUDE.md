# adev/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX's Automatic Differentiation of Expected Values (ADEV) module.

**For core GenJAX concepts**, see `../CLAUDE.md`
**For inference algorithms using ADEV**, see `../inference/CLAUDE.md`
**For academic references**, see `REFERENCES.md`

## Overview

The `adev` module provides automatic differentiation capabilities specifically designed for unbiased gradient estimation of expected values.

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

## References

### Theoretical Background

- **Reparameterization Trick**: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- **REINFORCE**: Williams (1992), "Simple Statistical Gradient-Following Algorithms"
- **Variance Reduction**: Mnih & Gregor (2014), "Neural Variational Inference"

### Implementation Notes

ADEV builds on JAX's automatic differentiation capabilities while providing specialized handling for probabilistic programs which denotes expectations. The estimators are designed to work seamlessly with GenJAX's generative function interface and JAX's compilation system.
