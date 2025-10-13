# Curve Fitting Case Study

This case study demonstrates Bayesian curve fitting using GenJAX for outlier-robust inference of quadratic polynomial coefficients.

## Model

The model fits degree-2 polynomials to noisy data with automatic outlier detection:

- **Polynomial**: `y = a + b x + c x^2`
- **Coefficients**: Independent Normal(0, 1) priors for `a`, `b`, and `c`
- **Noise**: Gaussian noise (σ = 0.05) on observations
- **Outliers**: Mixture branch that inflates noise (Gaussian or Uniform depending on variant)

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
# Generate POPL paper figures
pixi run -e curvefit python -m examples.curvefit.main paper
# (alias) pixi run -e curvefit curvefit-paper
```

### Basic Inference Example

```python
from genjax import Const
import jax.random as jrand
from examples.curvefit.core import infer_latents, get_points_for_inference

# Generate test data
key = jrand.key(42)
_, (xs, ys) = get_points_for_inference()

# Run Bayesian inference (importance sampling with Const-wrapped particle count)
traces, log_weights = infer_latents(xs, ys, Const(1000))

# Extract parameter samples
curve_choices = traces.get_choices()["curve"]
a_samples = curve_choices["a"]
b_samples = curve_choices["b"]
c_samples = curve_choices["c"]
```

## Output

`paper` mode writes a fixed set of publication-ready PDFs to the repository-level `figs/` directory:

- `curvefit_prior_multipoint_traces_density.pdf`
- `curvefit_single_multipoint_trace_density.pdf`
- `curvefit_scaling_performance.pdf`
- `curvefit_posterior_scaling_combined.pdf`
- `curvefit_outlier_detection_comparison.pdf`

The static illustration `curvefit_vectorization_illustration.pdf` is checked into the repository and not regenerated.

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
