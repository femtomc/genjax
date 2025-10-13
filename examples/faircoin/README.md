# Fair Coin (Beta-Bernoulli) Timing Comparison

This case study compares the performance of importance sampling for a simple Beta-Bernoulli model across different probabilistic programming frameworks.

## Model

The model is a simple hierarchical Beta-Bernoulli representing fair coin inference:

- Prior: `f ~ Beta(10, 10)` (coin fairness parameter)
- Likelihood: `obs[i] ~ Bernoulli(f)` for each coin flip observation

## Frameworks Compared

1. **GenJAX**: Using the `@gen` decorator and vectorized importance sampling
2. **NumPyro**: Using Numpyro's importance sampling with manual guide
3. **Handcoded**: Direct JAX implementation without PPL overhead

## Code Structure

The case study is organized into modular components:

- `core.py`: Model definitions and timing functions for all frameworks
- `figs.py`: Visualization utilities for generating comparison plots
- `main.py`: Command-line interface for running comparisons

## Usage

### Setup Environment

```bash
# Install the faircoin environment with all dependencies
pixi install -e faircoin
```

### Run Basic Comparison (GenJAX, NumPyro, Handcoded)

```bash
# Run with default settings
pixi run -e faircoin faircoin-timing

# Or with custom parameters
pixi run -e faircoin python -m examples.faircoin.main --num-obs 100 --repeats 50 --num-samples 2000
```

### Run Combined Comparison

```bash
# Recommended combined timing + posterior figure
pixi run -e faircoin faircoin-combined

# Manually specify parameters
pixi run -e faircoin python -m examples.faircoin.main --combined --num-obs 100 --num-samples 2000
```

### Posterior-Only Figure

```bash
# Compare posterior histograms only
pixi run -e faircoin python -m examples.faircoin.main --posterior --num-samples 5000
```

## Output

All figures are written to the repository-level `figs/` directory:

- **Timing only** (`faircoin-timing`): `faircoin_timing_performance_comparison_obs{N}_samples{M}_repeats{R}.pdf`
- **Posterior only** (`--posterior`): `faircoin_posterior_accuracy_comparison_obs{N}_samples{M}.pdf`
- **Combined** (`--combined`): `combined_3x2_obs{N}_samples{M}.pdf`

Timing statistics are printed to stdout alongside each run.

## Performance Notes

- **GenJAX** typically shows competitive performance with clean probabilistic programming syntax
- **Handcoded** represents the theoretical performance ceiling with minimal PPL overhead
- **NumPyro** provides a mature probabilistic programming interface with good performance

The comparison helps demonstrate GenJAX's performance characteristics relative to established frameworks in the probabilistic programming ecosystem.
