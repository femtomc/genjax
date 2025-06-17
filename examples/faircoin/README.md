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
4. **Pyro** (optional): Using Pyro's importance sampling interface

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

### Run Full Comparison (Including Pyro)

```bash
# Run full comparison including Pyro
pixi run -e faircoin faircoin-comparison

# Or manually with the --comparison flag
pixi run -e faircoin python -m examples.faircoin.main --comparison
```

## Output

The script will:

1. Run timing benchmarks for each framework
2. Print timing statistics to the console
3. Generate a horizontal bar chart comparison saved to `examples/faircoin/figs/`
   - `comparison.pdf`: GenJAX, NumPyro, Handcoded only
   - `comparison_with_pyro.pdf`: All frameworks including Pyro

The visualization shows relative performance as percentages compared to the handcoded JAX baseline, with absolute times in milliseconds.

## Performance Notes

- **GenJAX** typically shows competitive performance with clean probabilistic programming syntax
- **Handcoded** represents the theoretical performance ceiling with minimal PPL overhead
- **NumPyro** provides a mature probabilistic programming interface with good performance
- **Pyro** offers rich probabilistic programming features but may be slower for simple models like this

The comparison helps demonstrate GenJAX's performance characteristics relative to established frameworks in the probabilistic programming ecosystem.
