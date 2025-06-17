# CLAUDE.md - Fair Coin Case Study

This file provides guidance to Claude Code when working with the fair coin (Beta-Bernoulli) timing comparison case study.

## Overview

The fair coin case study demonstrates probabilistic programming framework performance through a simple Beta-Bernoulli model comparing GenJAX, NumPyro, handcoded JAX, and optionally Pyro implementations.

## Directory Structure

```
examples/faircoin/
├── CLAUDE.md           # This file - guidance for Claude Code
├── README.md           # User documentation
├── __init__.py         # Python package marker
├── core.py             # Model definitions and timing functions
├── figs.py             # Visualization utilities
├── main.py             # Command-line interface
└── figs/               # Generated comparison plots
    └── *.pdf           # Parametrized filename plots
```

## Code Organization

### `core.py` - Model Implementations

- **`beta_ber()`**: GenJAX model using `@gen` decorator
- **`genjax_timing()`**: GenJAX importance sampling benchmark
- **`numpyro_timing()`**: NumPyro importance sampling benchmark
- **`handcoded_timing()`**: Direct JAX implementation benchmark
- **`pyro_timing()`**: Pyro importance sampling benchmark
- **`timing()`**: Core benchmarking utility function

### `figs.py` - Visualization

- **`timing_comparison_fig()`**: Generates horizontal bar chart comparisons
- **Research paper ready**: Large fonts, high DPI, professional formatting
- **Parametrized filenames**: Includes experimental parameters in output filename

### `main.py` - CLI Interface

- **Default parameters**: 50 obs, 1000 samples, 200 repeats
- **Pyro disabled by default**: Use `--comparison` flag to include Pyro
- **Configurable**: All timing parameters adjustable via command line

## Key Implementation Details

### Model Specification

```python
# Fair coin model: Beta(10, 10) prior, Bernoulli likelihood
@gen
def beta_ber():
    alpha0, beta0 = jnp.array(10.0), jnp.array(10.0)
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip(f) @ "obs"
```

### Importance Sampling Pattern

All frameworks implement the same importance sampling strategy:

1. Sample from prior Beta(10, 10) as proposal
2. Compute likelihood weights for observed data
3. Vectorized execution with JAX/framework primitives

### Timing Methodology

- **Warm-up runs**: 2 JIT compilation runs before timing
- **Multiple repeats**: Default 200 outer repeats for statistical reliability
- **Inner repeats**: 200 inner repeats per outer repeat, taking minimum
- **Block until ready**: Ensures GPU/async operations complete

## Visualization Features

### Research Paper Quality

- **Font sizes**: 14-18pt for publication readability
- **High DPI**: 300 DPI PDF output for crisp figures
- **Clean layout**: No title (suitable for figure captions)
- **Clear guidance**: "Smaller bar is better" in top right corner

### Parametrized Filenames

Format: `comparison_obs{N}_samples{M}_repeats{R}[_with_pyro].pdf`

- Enables experiment tracking and reproducibility
- Multiple configurations can coexist in same directory

## Usage Patterns

### Basic Comparison (GenJAX, NumPyro, Handcoded)

```bash
pixi run -e faircoin faircoin-timing
```

### Full Comparison (Including Pyro)

```bash
pixi run -e faircoin faircoin-comparison
```

### Custom Parameters

```bash
pixi run -e faircoin python -m examples.faircoin.main --num-obs 100 --repeats 50 --num-samples 2000
```

## Performance Expectations

### Typical Results

- **GenJAX**: ~97-103% of handcoded baseline (very competitive)
- **Handcoded JAX**: 100% baseline (theoretical optimum)
- **NumPyro**: ~400-500% of baseline (mature but higher overhead)
- **Pyro**: ~1000%+ of baseline (feature-rich but slower for simple models)

### Framework Characteristics

- **GenJAX**: Clean syntax with minimal performance overhead
- **NumPyro**: Mature ecosystem, good performance for complex models
- **Handcoded**: Raw JAX performance ceiling
- **Pyro**: Rich features, slower for simple models

## Development Guidelines

### When Modifying Timing Functions

1. **Maintain consistency**: All frameworks should implement identical algorithms
2. **Preserve warm-up**: Always include JIT warm-up runs before timing
3. **Use block_until_ready()**: Ensure accurate timing measurements
4. **Keep static parameters**: Avoid dynamic arguments that break JIT compilation

### When Updating Visualizations

1. **Research paper standards**: Maintain large fonts and high DPI
2. **Color consistency**: Use established color scheme for frameworks
3. **Filename parametrization**: Include all relevant parameters in filename
4. **Clear interpretation**: Maintain "smaller bar is better" guidance

### Testing Changes

```bash
# Quick test with minimal parameters
pixi run -e faircoin python -m examples.faircoin.main --repeats 10 --num-samples 100

# Full comparison test
pixi run -e faircoin python -m examples.faircoin.main --comparison --repeats 20 --num-samples 500
```

## Common Issues

### Pyro Timeout

- **Cause**: Pyro can be significantly slower than JAX-based frameworks
- **Solution**: Use fewer repeats for Pyro or increase timeout
- **Default**: Pyro disabled by default to avoid workflow disruption

### Import Errors

- **Cause**: Missing dependencies in faircoin environment
- **Solution**: Ensure `pixi install -e faircoin` completed successfully
- **Dependencies**: matplotlib, seaborn, torch, numpyro, pyro-ppl

### JIT Compilation Issues

- **Cause**: Dynamic arguments breaking JAX compilation
- **Solution**: Keep model parameters static, use closures for configuration
- **Pattern**: All timing functions use fixed model configurations

## Integration with Main GenJAX

This case study serves as:

1. **Performance benchmark**: Demonstrates GenJAX competitive performance
2. **Usage example**: Shows proper `@gen` function patterns
3. **Framework comparison**: Contextualizes GenJAX within PPL ecosystem
4. **Research validation**: Provides publication-ready performance comparisons

The case study should remain stable and serve as a reference implementation for GenJAX performance characteristics.
