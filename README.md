<p align="center">
<img width="450" src="./logo.png"/>
</p>

[![codecov](https://codecov.io/gh/femtomc/genjax/graph/badge.svg?token=V5W1YIYC5P)](https://codecov.io/gh/femtomc/genjax)

> **Note**: This is the research version of GenJAX. A [(more stable) community version can be found here](https://github.com/genjax-community/genjax).

## About GenJAX

### **Probabilistic Programming Language**

GenJAX is a probabilistic programming language (PPL): a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation for expected values, and more.

### **With Programmable Inference**

The design of GenJAX is centered on _programmable inference_: automation which allows users to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions: "_x_ affects _y_, and I observe _y_, what are my new beliefs about _x_?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

### **Core Concepts**

GenJAX's automation is based on two key concepts:
- **_Generative functions_** – GenJAX's version of probabilistic programs
- **_Traces_** – samples from probabilistic programs

GenJAX provides:

- **Modeling language automation** for constructing complex probability distributions from pieces
- **Inference automation** for constructing Monte Carlo samplers using convenient idioms (programs expressed by creating and editing traces), and [variational inference automation](https://dl.acm.org/doi/10.1145/3656463) using [new extensions to automatic differentation for expected values](https://dl.acm.org/doi/10.1145/3571198)

## POPL 2026 Artifact

This research branch powers the POPL'26 artifact submitted alongside the paper *Probabilistic Programming with Vectorized Programmable Inference*. It contains the GenJAX implementation and all case studies used in the empirical evaluation.

**Contents:**
- [Quick Example](#quick-example)
- [Getting Started](#getting-started)
- [Reproducing All Paper Figures](#reproducing-all-paper-figures)
- [Case Study Details](#case-study-details)
- [CPU vs GPU Execution](#cpu-vs-gpu-execution)
- [Generated Figures](#generated-figures)

## Quick Example

Here's a simple curve fitting model showing how to write importance sampling using GenJAX's generative function interface:

```python
from genjax import gen, normal
from genjax.pjax import modular_vmap as vmap
import jax.numpy as jnp

# Define a generative model for polynomial curve fitting
@gen
def polynomial():
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    return jnp.array([a, b, c])

@gen
def point(x, coeffs):
    y_det = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2
    y_obs = normal(y_det, 0.05) @ "obs"
    return y_obs

@gen
def npoint_curve(xs):
    coeffs = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, coeffs) @ "ys"
    return coeffs, (xs, ys)

# Generate test data
xs = jnp.linspace(0, 1, 10)
trace = npoint_curve.simulate(xs)
_, (_, ys_observed) = trace.get_retval()

# Write importance sampling using the generative function interface
def importance_sampling(model, args, observations, n_particles):
    """Importance sampling using generate."""

    def single_particle():
        # generate() samples from model with observations as constraints
        # Returns (trace, log_weight)
        trace, log_weight = model.generate(observations, *args)
        return trace, log_weight

    # Vectorize over particles - our vmap handles probabilistic sampling correctly
    # automatically
    vectorized = vmap(single_particle, in_axes=(), axis_size=n_particles)

    return vectorized()

# Run inference
observations = {"ys": {"obs": ys_observed}}
traces, log_weights = importance_sampling(npoint_curve, (xs,), observations, 1000)

# Extract posterior samples
curve_a = traces.get_choices()["curve"]["a"]
print(f"Posterior mean for 'a': {jnp.mean(curve_a):.3f}")
```

This example shows:
- **Generative functions** with `@gen` decorator
- **Named random choices** with `@` operator (e.g., `@ "a"`)
- **Composable vectorization** with `.vmap()` on generative functions
- **Programmable inference** write inference using generative function interface (here, the `generate()` interface)
- **modular_vmap** for vectorizing inference (handles seeding automatically)

## Getting Started

### Prerequisites

Install [pixi](https://pixi.sh/) (package manager). That's it.

### Setup

```bash
cd genjax
pixi install
```

This creates isolated conda environments for each case study.

### Quick Smoke Test

Verify setup with the simplest case study:

```bash
pixi run -e faircoin python -m examples.faircoin.main \
  --combined --num-obs 20 --num-samples 200 --repeats 5
```

Expected output: `figs/combined_3x2_obs20_samples200.pdf`

---

## Reproducing All Paper Figures

Generate all 10 paper figures with a single command:

```bash
# CPU execution
pixi run paper-figures

# GPU execution (requires CUDA 12)
pixi run paper-figures-gpu
```

All figures are saved to `genjax/figs/`:
- 1 faircoin figure
- 5 curvefit figures
- 2 GOL figures
- 2 localization figures

**Note**: One additional figure (`curvefit_vectorization_illustration.pdf`) is a static diagram already included in the repository.

---

## Case Study Details

### 1. Fair Coin (Beta-Bernoulli)

**What it does**: Compares GenJAX, handcoded JAX, and NumPyro on a simple conjugate inference problem.

**Command**:
```bash
pixi run -e faircoin python -m examples.faircoin.main \
  --combined --num-obs 50 --num-samples 2000 --repeats 10
```

**Outputs**: `figs/combined_3x2_obs50_samples2000.pdf`

---

### 2. Curve Fitting with Outlier Detection

**What it does**: Polynomial regression with robust outlier detection, demonstrating:
- Importance sampling with varying particle counts
- Gibbs sampling with HMC for mixture models
- Performance scaling analysis

**Command**:
```bash
pixi run -e curvefit python -m examples.curvefit.main paper
```

**Outputs**: 5 figures in `figs/`:
- `curvefit_prior_multipoint_traces_density.pdf`
- `curvefit_single_multipoint_trace_density.pdf`
- `curvefit_scaling_performance.pdf`
- `curvefit_posterior_scaling_combined.pdf`
- `curvefit_outlier_detection_comparison.pdf`

---

### 3. Game of Life Inverse Dynamics

**What it does**: Infers past Game of Life states from observed future states using Gibbs sampling on a 512×512 grid with 250 Gibbs steps.

**Command**:
```bash
pixi run -e gol gol-paper
```

**Outputs**: 2 figures in `figs/`:
- `gol_integrated_showcase_wizards_512.pdf` (3-panel inference showcase)
- `gol_gibbs_timing_bar_plot.pdf` (performance across grid sizes)

**Note**: Timing bar plot runs benchmarks at 64×64, 128×128, 256×256, and 512×512 grid sizes.

---

### 4. Robot Localization with SMC

**What it does**: Particle filter localization comparing bootstrap filter, SMC+HMC, and locally optimal proposals using 200 particles and 8-ray LIDAR.

**Command**:
```bash
pixi run -e localization python -m examples.localization.main paper \
  --include-basic-demo --include-smc-comparison \
  --n-particles 200 --n-steps 8 --timing-repeats 3 --n-rays 8 --output-dir figs
```

**Outputs**: 2 figures in `figs/`:
- `localization_r8_p200_basic_localization_problem_1x4_explanation.pdf`
- `localization_r8_p200_basic_comprehensive_4panel_smc_methods_analysis.pdf`

**Note**: Also generates experimental data in `examples/localization/data/` (regenerated each run).

---

## CPU vs GPU Execution

### Which Case Studies Benefit from GPU?

| Case Study | GPU Benefit | Notes |
|------------|-------------|-------|
| Faircoin | Minimal | Problem too small to amortize GPU overhead |
| Curvefit | Moderate | Vectorized importance sampling parallelizes well |
| GOL | Significant | Large grid operations (512×512) parallelize well |
| Localization | Significant | Particle filter (200 particles) vectorizes efficiently |

### Memory Requirements

- **CPU**: 8GB RAM sufficient for all case studies
- **GPU**: 8GB VRAM sufficient (tested on NVIDIA A100, RTX 3090)

### GPU Setup

For GPU execution, use `-cuda` environments:
```bash
pixi run -e faircoin-cuda python -m examples.faircoin.main ...
pixi run -e curvefit-cuda python -m examples.curvefit.main paper
pixi run -e gol-cuda gol-paper
pixi run -e localization-cuda python -m examples.localization.main paper ...
```

---

## Generated Figures

All figures are saved to `genjax/figs/`:

### Faircoin
- `combined_3x2_obs50_samples2000.pdf` - Framework comparison (timing + posterior accuracy)

### Curvefit
- `curvefit_prior_multipoint_traces_density.pdf` - Prior samples from generative model
- `curvefit_single_multipoint_trace_density.pdf` - Single trace with log density
- `curvefit_scaling_performance.pdf` - Inference scaling with particle count
- `curvefit_posterior_scaling_combined.pdf` - Posterior quality at different scales
- `curvefit_outlier_detection_comparison.pdf` - Robust inference with mixture model
- `curvefit_vectorization_illustration.pdf` - Static diagram (already in repo)

### Game of Life
- `gol_integrated_showcase_wizards_512.pdf` - Inverse dynamics inference (3 panels)
- `gol_gibbs_timing_bar_plot.pdf` - Performance scaling across grid sizes

### Localization
- `localization_r8_p200_basic_localization_problem_1x4_explanation.pdf` - Problem setup
- `localization_r8_p200_basic_comprehensive_4panel_smc_methods_analysis.pdf` - SMC method comparison
