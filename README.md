<p align="center">
<img width="450" src="./logo.png"/>
</p>

[![DOI](https://zenodo.org/badge/971731825.svg)](https://doi.org/10.5281/zenodo.17342547)

## What is it?

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

## As a POPL 2026 Artifact

This repository is also a POPL'26 artifact submitted alongside the paper *Probabilistic Programming with Vectorized Programmable Inference*.

**Canonical artifact version: [v1.0.8](https://github.com/femtomc/genjax/releases/tag/v1.0.8)** - Use this release for artifact evaluation.

It contains the GenJAX implementation (including source code and tests), extensive documentation, curated agentic context (see the `AGENTS.md` throughout the codebase) to allow users of Claude Code and Codex (or others) to quickly use the system, and several of the case studies used in the empirical evaluation.

**Contents:**
- [Quick Example](#quick-example)
- [Getting Started](#getting-started)
- [Reproducing Paper Figures](#reproducing-all-paper-figures)
- [Case Study Details](#case-study-details)
- [Generated Figures](#generated-figures)

## Quick Example

We mirror the curve fitting case study from the paper's overview section. The example walks through (1) defining quadratic generative functions, (2) simulating a dataset with explicit PRNG keys, and (3) running vectorized importance sampling.

### 1. Define the model
We compose three generative functions: `polynomial` samples quadratic coefficients, `point` emits a noisy observation, and `npoint_curve` maps `point` across an input vector using traced-aware `.vmap()`.

```python
import jax.numpy as jnp
from genjax import gen, normal

@gen
def polynomial():
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    return jnp.array([a, b, c])

@gen
def point(x, coeffs):
    y_det = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
    return normal(y_det, 0.05) @ "obs"

@gen
def npoint_curve(xs):
    coeffs = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, coeffs) @ "ys"
    return coeffs, (xs, ys)
```

### 2. Simulate a dataset with `seed`
The `seed` transformation wraps a probabilistic callable so its first argument becomes a JAX `PRNGKey`. We can now simulate noisy quadratic data deterministically given a key.

```python
import jax.random as jrand
from genjax.pjax import seed

xs = jnp.linspace(-1.0, 1.0, 64)
simulate_curve = seed(npoint_curve.simulate)
curve_trace = simulate_curve(jrand.key(0), xs)
coeffs_true, (_, ys_obs) = curve_trace.get_retval()
observations = {"ys": {"obs": ys_obs}}
```

### 3. Vectorized importance sampling
Importance sampling draws many traces in parallel. GenJAX's `modular_vmap` understands probabilistic primitives, so the only change from a single-particle sampler is batching the keys.

```python
import jax.nn as jnn
from genjax.pjax import modular_vmap

seeded_generate = seed(npoint_curve.generate)

def one_particle(key):
    trace, log_weight = seeded_generate(key, observations, xs)
    return trace, log_weight

particle_keys = jrand.split(jrand.key(1), 2048)
vectorized_importance = modular_vmap(one_particle, in_axes=0)
traces, log_weights = vectorized_importance(particle_keys)

weights = jnn.softmax(log_weights)
curve_choices = traces.get_choices()["curve"]
posterior_mean_a = jnp.sum(weights * curve_choices["a"])
print(f"Posterior mean for 'a': {posterior_mean_a:.3f}")
```

This mirrors the literate walkthrough in the paper: generative functions compose naturally, the same code vectorizes over data and particles, and explicit seeding keeps randomness under user control.

## Getting Started


### Prerequisites

Install [pixi](https://pixi.sh/) (package manager, which will allow you to build and run the case studies).

### Setup

```bash
cd genjax
pixi install
```

This creates isolated conda environments for each case study.

## Reproducing Paper Figures

*Note:* in our artifact, all figures involving code execution are provided below _with the exception of the multi-system benchmarking figure_. The code for the benchmarking figure (Fig 16, b) is available in the Git history of this repository (but requires a complex deployment setup, and won't run on CPUs).

To generate several of the case study paper figures, use the following commands:

```bash
# CPU execution
pixi run paper-figures

# GPU execution (requires CUDA 12)
pixi run paper-figures-gpu
```

All figures are saved to `genjax/figs/`.

### Execution properties vs. device

Here's a list of behaviors which should be expected when running the case studies on CPU:

- CPU takes longer than GPU: executed on an Apple M4 (Macbook Air) takes around 4 minutes.
- CPU won't exhibit the same vectorized scaling properties as GPU (in many cases, linear versus near-constant scaling).
- When running the artifact on CPU only, some of the timing figures may be missing comparisons between CPU and GPU.

Keep these behaviors in mind when interpreting figures generated via CPU execution.

### Devices that we tested the artifact on

We expect that any environment which supports JAX should allow you to run our artifact (using `pixi`) -- but for precision, here's a list of devices which we tested the artifact on (using `pixi run paper-figures` for CPU and `pixi run paper-figures-gpu` for GPU):

**Apple M4 (Macbook Air)**
- Model: MacBook Air (Mac16,12)
- Chip: Apple M4
- CPU Cores: 10 cores total (4 performance + 6 efficiency)
- Memory: 16 GB
- OS: macOS 15.6 (Sequoia)
- Build: 24G84
- Kernel: Darwin 24.6.0

**Linux machine with Nvidia RTX 4090**
- Pop!_OS 22.04 LTS
- Kernel: Linux 6.16.3-76061603-generic
- AMD Ryzen 7 7800X3D 8-Core Processor
- 16 threads (8 cores with SMT)
- Max frequency: 5.05 GHz
- 96 MiB L3 cache
- NVIDIA GeForce RTX 4090 (24GB VRAM)

## Case Study Details

In this section, we provide more details on the case studies. Each case study directory also contains a `README.md` file with more information. In each case study, we provide a reference to the figures in the paper which the case study supports.

### Fair Coin (Beta-Bernoulli)

**What it does**: Compares GenJAX, handcoded JAX, and NumPyro on a simple conjugate inference problem.

**Figures in the paper**: Figure 16 (a).

**Command**:
```bash
pixi run -e faircoin python -m examples.faircoin.main \
  --combined --num-obs 50 --num-samples 2000 --repeats 10
```

**Outputs**: `figs/faircoin_combined_posterior_and_timing_obs50_samples2000.pdf`

### Curve Fitting with Outlier Detection

**What it does**: Polynomial regression with robust outlier detection, demonstrating:
- GPU scaling of importance sampling with varying particle counts
- Gibbs sampling with HMC for an outlier mixture model

**Figures in the paper**: Figure 4, Figure 5, Figure 6.

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

### Game of Life Inverse Dynamics

**What it does**: Infers past Game of Life states from observed future states using Gibbs sampling on a 512×512 grid with 250 Gibbs steps.

**Figures in the paper**: Figure 18.

**Command**:
```bash
pixi run -e gol gol-paper
```

**Outputs**: 2 figures in `figs/`:
- `gol_integrated_showcase_wizards_512.pdf` (3-panel inference showcase)
- `gol_gibbs_timing_bar_plot.pdf` (performance across grid sizes)

**Note**: Timing bar plot runs benchmarks at 64×64, 128×128, 256×256, and 512×512 grid sizes.

### Robot Localization with SMC

**What it does**: Particle filter localization comparing bootstrap filter, SMC+HMC, and approximate (using grid enumeration) locally optimal proposals with 200 particles and a generative model with a simulated 8-ray LIDAR measurement.

**Figures in the paper**: Figure 19.

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

## Generated Figures

All figures are saved to `figs/`:

### Faircoin
- `faircoin_combined_posterior_and_timing_obs50_samples2000.pdf` - Framework comparison (timing + posterior accuracy)

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
