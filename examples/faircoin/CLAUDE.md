# CLAUDE.md - Fair Coin Case Study

This file provides guidance to Claude Code when working with the fair coin (Beta-Bernoulli) timing comparison case study.

## Overview

The fair coin case study demonstrates probabilistic programming framework performance and accuracy through a Beta-Bernoulli model comparing GenJAX, NumPyro, and handcoded JAX implementations. It includes both timing comparisons and posterior sampling accuracy validation.

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

- **`beta_ber_multi(num_obs: Const[int])`**: GenJAX model for multiple coin flips using `@gen` decorator
- **`genjax_timing()`**: GenJAX importance sampling benchmark using `beta_ber_multi`
- **`numpyro_timing()`**: NumPyro importance sampling benchmark with multiple observations
- **`handcoded_timing()`**: Direct JAX implementation benchmark with multiple observations
- **`timing()`**: Core benchmarking utility function
- **Posterior sampling functions**: `genjax_posterior_samples()`, `handcoded_posterior_samples()`, `numpyro_posterior_samples()`
- **`exact_beta_posterior_stats()`**: Analytical Beta posterior computation for validation

### `figs.py` - Visualization

- **`timing_comparison_fig()`**: Generates horizontal bar chart comparisons
- **`posterior_comparison_fig()`**: Generates 2x2 grid of posterior histograms vs exact posterior
- **`combined_comparison_fig()`**: Generates 3x2 layout with posterior plots (top) and timing (bottom)
- **Research paper ready**: Large fonts, high DPI, professional formatting
- **Error bars**: Standard deviation whiskers on timing measurements
- **Parametrized filenames**: Includes experimental parameters in output filename

### `main.py` - CLI Interface

- **Default parameters**: 50 obs, 1000 samples, 200 repeats
- **Multiple figure types**: `--timing`, `--posterior`, `--combined`, `--all`
- **Three frameworks**: GenJAX, handcoded JAX, and NumPyro comparison
- **Configurable**: All timing and sampling parameters adjustable via command line

## Key Implementation Details

### Model Specification

```python
# Multi-observation Beta-Bernoulli model: Beta(10, 10) prior, multiple Bernoulli observations
@gen
def beta_ber_multi(num_obs: Const[int]):
    alpha0, beta0 = jnp.array(10.0), jnp.array(10.0)
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip.repeat(n=num_obs.value)(f) @ "obs"
```

### Data Pattern

All frameworks use **80% heads, 20% tails** observations:
- `num_heads = int(0.8 * num_obs)`
- `num_tails = num_obs - num_heads`
- `data = jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])`

This creates a realistic scenario where all frameworks recover the exact Beta posterior: **Beta(50, 20)** for 50 observations.

### Importance Sampling Pattern

All frameworks implement the same importance sampling strategy:

1. Sample from prior Beta(10, 10) as proposal
2. Compute likelihood weights for 50 observed coin flips (80% heads, 20% tails)
3. Vectorized execution with JAX/framework primitives
4. Different random seeds: GenJAX (123), handcoded JAX (456), NumPyro (42)

### Timing Methodology

- **Warm-up runs**: 2 JIT compilation runs before timing
- **Multiple repeats**: Default 200 outer repeats for statistical reliability
- **Inner repeats**: 200 inner repeats per outer repeat, taking minimum
- **Block until ready**: Ensures GPU/async operations complete

## Visualization Features

### Research Paper Quality

- **Font sizes**: 18-22pt for publication readability
- **High DPI**: 300 DPI PDF output for crisp figures
- **Clean layout**: No overall titles (suitable for figure captions)
- **Error bars**: Standard deviation whiskers on timing measurements
- **Professional styling**: No gridlines, minimal axis decoration

### Figure Types

1. **Timing comparison**: Horizontal bar chart with error bars, embedded framework labels
2. **Posterior comparison**: 2x2 grid of histograms vs exact posterior, no y-axis labels
3. **Combined comparison**: 3x2 layout with posterior plots (top row) and timing (bottom row)

### Parametrized Filenames

- Timing: `comparison_obs{N}_samples{M}_repeats{R}.pdf`
- Posterior: `posterior_comparison_obs{N}_samples{M}.pdf`
- Combined: `combined_3x2_obs{N}_samples{M}.pdf`

Enables experiment tracking and reproducibility with multiple configurations coexisting.

## Usage Patterns

### Timing Comparison Only

```bash
pixi run -e faircoin faircoin-timing
```

### Combined Figure (Recommended)

```bash
pixi run -e faircoin faircoin-combined
```

### Posterior Comparison Only

```bash
pixi run -e faircoin python -m examples.faircoin.main --posterior
```


### All Figures

```bash
pixi run -e faircoin python -m examples.faircoin.main --all
```

### Custom Parameters

```bash
pixi run -e faircoin python -m examples.faircoin.main --combined --num-obs 100 --num-samples 5000
```

## Performance Expectations

### Typical Results

- **GenJAX**: ~100% of handcoded baseline (identical performance)
- **Handcoded JAX**: 100% baseline (theoretical optimum)
- **NumPyro**: ~400% of baseline (mature framework with reasonable overhead)

### Framework Characteristics

- **GenJAX**: Clean syntax with zero performance overhead - compiles to identical performance as handcoded JAX
- **NumPyro**: Mature ecosystem, good performance for complex models, ~4x overhead for simple models
- **Handcoded**: Raw JAX performance ceiling - serves as the reference implementation

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
pixi run -e faircoin python -m examples.faircoin.main --combined --num-samples 1000

# Timing only test
pixi run -e faircoin python -m examples.faircoin.main --repeats 20

# Posterior only test
pixi run -e faircoin python -m examples.faircoin.main --posterior --num-samples 2000
```

## Common Issues

### Import Errors

- **Cause**: Missing dependencies in faircoin environment
- **Solution**: Ensure `pixi install -e faircoin` completed successfully
- **Dependencies**: matplotlib, seaborn, numpy, jax, numpyro

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
