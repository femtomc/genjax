# Curve Fitting Case Study

This case study demonstrates Bayesian curve fitting using GenJAX for outlier-robust inference of sine wave parameters.

## Model

The model fits sine curves to noisy data with automatic outlier detection:

- **Sine wave**: `y = sin(2π * freq * x + offset)`
- **Parameters**: Frequency ~ Exponential(10), Offset ~ Uniform(0, 2π)
- **Noise**: Gaussian noise (σ = 0.2) with 8% outlier probability
- **Outliers**: Uniform(-3, 3) when present

## Code Structure

- `core.py`: Model definitions and inference functions
- `figs.py`: Visualization utilities for traces, inference results, and scaling studies
- `main.py`: Generates all figures for the case study

## Usage

### Setup Environment

```bash
# Install the curvefit environment with dependencies
pixi install -e curvefit
```

### Run Complete Case Study

```bash
# Generate all figures
pixi run -e curvefit curvefit
```

### Basic Inference Example

```python
from examples.curvefit.core import infer_latents, get_points_for_inference
import jax.random as jrand

# Generate test data
key = jrand.key(42)
curve, (xs, ys) = get_points_for_inference()

# Run Bayesian inference
samples, weights = infer_latents(key, ys, 1000)

# Extract parameter samples
freq_samples = samples.get_choices()['curve']['freq']
offset_samples = samples.get_choices()['curve']['off']
```

## Output

The case study generates several visualization types:

- **Trace visualizations**: Single and multi-point curve examples
- **Inference plots**: Posterior uncertainty bands over fitted curves
- **Scaling studies**: Performance analysis across different sample sizes
- **Density comparisons**: Log-probability evaluations

All figures are saved as high-resolution PDFs in `examples/curvefit/figs/`.

## Key Features

### Outlier Robustness

The mixture model automatically identifies and down-weights outliers, providing robust parameter estimates even with contaminated data.

### Bayesian Uncertainty

Full posterior distributions over parameters enable uncertainty quantification in predictions and parameter estimates.

### Scalable Inference

Uses GenJAX's SMC (Sequential Monte Carlo) library with importance sampling for efficient Bayesian inference.

## Technical Notes

This implementation demonstrates several important GenJAX patterns:

- **Factory pattern** for generative functions with static parameters
- **Proper SMC integration** using closure patterns with seed transformations
- **Outlier-robust modeling** through mixture distributions
- **Vectorized operations** for efficient inference

The case study serves as a reference for implementing similar Bayesian regression models in GenJAX.
