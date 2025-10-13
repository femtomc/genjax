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

**Canonical artifact version: [v1.0.10](https://github.com/femtomc/genjax/releases/tag/v1.0.10)** - Use this release for artifact evaluation.

It contains the GenJAX implementation (including source code and tests), extensive documentation, curated agentic context (see the `AGENTS.md` throughout the codebase) to allow users of Claude Code and Codex (or others) to quickly use the system, and several of the case studies used in the empirical evaluation.

**Contents:**
- [Quick Example](#quick-example)
- [Getting Started](#getting-started)
- [Reproducing Paper Figures](#reproducing-all-paper-figures)
- [Case Study Details](#case-study-details)
- [Generated Figures](#generated-figures)

## Quick Example

The snippets below develop the polynomial regression example from our paper's *Overview* section in GenJAX: we compose the model from generative functions, vectorize importance sampling to scale the number of particles, extend the model with stochastic branching to capture outliers, and finish with a programmable kernel that mixes enumerative Gibbs updates with Hamiltonian Monte Carlo. The full code can be found in `examples/curvefit`; the code here can be run as a linear notebook-style walkthrough.

### Vectorizing Generative Functions with vmap

We begin by expressing the quadratic regression model as a composition of generative functions (`@gen`-decorated Python functions).

Each random choice is tagged with a string address (`"a"`, `"b"`, `"c"`, `"obs"`), which is used to construct a structured representation of the model’s latent variables and observed data, called a _trace_.

Packaging the coefficients inside a callable `Lambda` Pytree mirrors the notion of sampling a function-valued random variable: downstream computations can call the curve directly while the trace retains access to its parameters.

```python
from genjax import gen, normal
from genjax.core import Const
from genjax import Pytree
import jax.numpy as jnp

@Pytree.dataclass
class Lambda(Pytree):
    # Wrap polynomial coefficients in a callable pytree so traces retain parameters.
    f: Const[object]
    dynamic_vals: jnp.ndarray
    static_vals: Const[tuple] = Const(())

    def __call__(self, *x):
        return self.f.value(*x, *self.static_vals.value, self.dynamic_vals)

def polyfn(x, coeffs):
    # Deterministic quadratic curve evaluated at x.
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    return a + b * x + c * x**2

@gen
def polynomial():
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    return Lambda(Const(polyfn), jnp.array([a, b, c]))

@gen
def point(x, curve):
    y_det = curve(x)
    y_obs = normal(y_det, 0.05) @ "obs"
    return y_obs

@gen
def npoint_curve(xs):
    curve = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)

xs = jnp.linspace(0.0, 1.0, 8)
trace = npoint_curve.simulate(xs)
print(trace.get_choices()["curve"].keys())
print(trace.get_choices()["ys"]["obs"].shape)
```

Vectorizing the `point` generative function with `vmap` mirrors the Overview’s Figure 3: the resulting trace preserves the hierarchical structure of the coefficients while lifting the observation site into an array-valued address. That “structure preserving” vectorization is what later enables us to reason about entire datasets and inference states in bulk.

### Vectorized Programmable Inference

The generative function interface supplies a small set of methods—`simulate`, `generate`, `assess`, `update`—that we can compose into inference algorithms.

Here we implement likelihood weighting (importance sampling): a single-particle routine constrains the observation site via the `generate` interface, while a vectorized wrapper scales out the number of particles. The logic of guessing (sampling) and checking (computing an importance weight) -- internally implemented in `generate` -- remains the same across particles, only the array dimensions vary with the particle count.

```python
from jax.scipy.special import logsumexp
from genjax.pjax import modular_vmap
import jax.numpy as jnp

def single_particle_importance(model, xs, ys_obs):
    # Draw a single constrained trace and compute its importance weight.
   trace, log_weight = model.generate({"ys": {"obs": ys_obs}}, xs)
   return trace, log_weight

def vectorized_importance_sampling(model, xs, ys_obs, num_particles):
    # Lift the single-particle routine across an explicit particle axis.
   sampler = modular_vmap(
        single_particle_importance,
        in_axes=(None, None, None),
        axis_size=num_particles,
    )
    return sampler(model, xs, ys_obs)

def log_marginal_likelihood(log_weights):
    return logsumexp(log_weights) - jnp.log(log_weights.shape[0])

xs = jnp.linspace(0.0, 1.0, 8)
trace = npoint_curve.simulate(xs)
_, (_, ys_obs) = trace.get_retval()

traces, log_weights = vectorized_importance_sampling(
    npoint_curve, xs, ys_obs, num_particles=512
)
print(traces.get_choices()["curve"]["a"].shape, log_marginal_likelihood(log_weights))
```

Running on hardware with ample parallel resources (e.g., a GPU) simply increases that axis size as far as memory allows, just as in the scaling curves shown in Figure 5.

### Improving Robustness using Stochastic Branching

Real datasets often include heterogeneous noise processes. Following the Overview, we enrich the observation model with stochastic branching that classifies each datapoint as an inlier or an outlier. The latent `is_outlier` switch feeds a `Cond` combinator that chooses between a tight Gaussian noise model and a broad uniform alternative; both branches write to the same observation address so later inference can target the entire `ys` subtree uniformly.

```python
from genjax import Cond, flip, uniform

@gen
def inlier_branch(mean, extra_noise):
    # Inlier observations stay near the quadratic trend.
    return normal(mean, 0.1) @ "obs"

@gen
def outlier_branch(_, extra_noise):
    # Outliers come from a broad, curve-independent distribution.
    return uniform(-2.0, 2.0) @ "obs"

@gen
def point_with_outliers(x, curve, outlier_rate=0.1, extra_noise=5.0):
    is_outlier = flip(outlier_rate) @ "is_outlier"
    cond_model = Cond(outlier_branch, inlier_branch)
    y_det = curve(x)
    return cond_model(is_outlier, y_det, extra_noise) @ "y"

@gen
def npoint_curve_with_outliers(xs, outlier_rate=0.1):
    curve = polynomial() @ "curve"
    ys = point_with_outliers.vmap(
        in_axes=(0, None, None, None)
    )(xs, curve, outlier_rate, 5.0) @ "ys"
    return curve, (xs, ys)

xs = jnp.linspace(0.0, 1.0, 8)
trace = npoint_curve_with_outliers.simulate(xs)
choices = trace.get_choices()["ys"]
print(choices["is_outlier"].shape, choices["y"]["obs"].shape)
```

The resulting trace contains a boolean vector of outlier indicators alongside the observations, matching the mixture-structured traces shown in Figure 6. Because the addresses are shared across branches, regenerating either the discrete switches or the continuous curve parameters becomes a matter of selecting the appropriate keys in the trace.

### Improving Inference Accuracy using Programmable Inference

To improve inference accuracy on the richer model we combine discrete and continuous updates within a single programmable kernel. Enumerative Gibbs updates each `is_outlier` choice by scoring the two possible values with `assess` before resampling, while Hamiltonian Monte Carlo refines the continuous parameters. Both steps operate on traces using a _selection_ (a way to target addresses within a trace), and they compose sequentially without requiring special-case code.

```python
import jax
import jax.numpy as jnp
from genjax.core import Const, sel
from genjax.distributions import categorical
from genjax.pjax import modular_vmap, seed
from genjax.inference import hmc, chain

def enumerative_gibbs_outliers(trace, xs, ys, outlier_rate=0.1):
    curve_params = trace.get_choices()["curve"]
    curve = Lambda(
        Const(polyfn),
        jnp.array([curve_params["a"], curve_params["b"], curve_params["c"]]),
    )

    def update_single_point(x, y_obs):
        chm_false = {"is_outlier": False, "y": {"obs": y_obs}}
        # Score the inlier explanation for the current observation.
        log_false, _ = point_with_outliers.assess(chm_false, x, curve, outlier_rate, 5.0)

        chm_true = {"is_outlier": True, "y": {"obs": y_obs}}
        # Score the outlier explanation for the same observation.
        log_true, _ = point_with_outliers.assess(chm_true, x, curve, outlier_rate, 5.0)

        logits = jnp.array([log_false, log_true])
        return categorical.sample(logits=logits) == 1

    new_outliers = modular_vmap(update_single_point)(xs, ys)
    gen_fn = trace.get_gen_fn()
    args = trace.get_args()
    new_trace, _, _ = gen_fn.update(
        trace, {"ys": {"is_outlier": new_outliers}}, *args[0], **args[1]
    )
    return new_trace

def mixed_gibbs_hmc_kernel(xs, ys, hmc_step_size=0.01, hmc_n_steps=10, outlier_rate=0.1):
    def kernel(trace):
        trace = enumerative_gibbs_outliers(trace, xs, ys, outlier_rate)
        return hmc(
            trace,
            sel(("curve", "a")) | sel(("curve", "b")) | sel(("curve", "c")),
            step_size=hmc_step_size,
            n_steps=hmc_n_steps,
        )
    return kernel

xs = jnp.linspace(0.0, 1.0, 8)
trace = npoint_curve_with_outliers.simulate(xs)
_, (_, ys) = trace.get_retval()
constraints = {"ys": {"y": {"obs": ys}}}
initial_trace, _ = npoint_curve_with_outliers.generate(constraints, xs, 0.1)

kernel = mixed_gibbs_hmc_kernel(xs, ys, hmc_step_size=0.02, hmc_n_steps=5)
runner = chain(kernel)
samples = seed(runner)(jax.random.key(0), initial_trace, n_steps=Const(10))

print(samples.traces.get_choices()["curve"]["a"].shape)
```

These moves yields a chain that captures both inlier/outlier classifications and posterior uncertainty over polynomial coefficients.

For the full treatment—including command-line tooling, figure generation, and alternative inference routines—see `examples/curvefit`.

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

**What it does**: Compares GenJAX, handcoded JAX, and NumPyro on a simple inference problem (with known exact inference).

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
