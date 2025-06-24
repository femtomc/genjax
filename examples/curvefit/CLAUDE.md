# CLAUDE.md - Curve Fitting Case Study

This file provides guidance to Claude Code when working with the curve fitting case study that demonstrates Bayesian inference for polynomial regression.

## Overview

The curvefit case study showcases Bayesian curve fitting using GenJAX, demonstrating polynomial regression (degree 2) with hierarchical modeling and both importance sampling and HMC inference. The case study has been simplified to focus on essential comparisons: IS with 1000 particles vs HMC methods.

## Directory Structure

```
examples/curvefit/
├── CLAUDE.md           # This file - guidance for Claude Code
├── README.md           # User documentation (if present)
├── main.py             # Main script to generate all figures
├── core.py             # Model definitions and inference functions
├── data.py             # Standardized test data generation across frameworks
├── figs.py             # Visualization and figure generation utilities
└── figs/               # Generated visualization outputs
    └── *.pdf           # Various curve fitting visualizations
```

## Code Organization

### `core.py` - Model Implementations

**GenJAX Models:**

- **`point(x, curve)`**: Single data point model with Gaussian noise (σ=0.05)
- **`polynomial()`**: Polynomial coefficient prior model (degree 2)
- **`onepoint_curve(x)`**: Single point curve fitting model
- **`npoint_curve(xs)`**: Multi-point curve model taking xs as input
- **`infer_latents()`**: SMC-based parameter inference using importance sampling
- **`get_points_for_inference()`**: Test data generation utility

**NumPyro Implementations (if numpyro available):**

- **`numpyro_npoint_model()`**: Equivalent NumPyro model with Gaussian likelihood
- **`numpyro_run_importance_sampling()`**: Importance sampling inference
- **`numpyro_run_hmc_inference()`**: Hamiltonian Monte Carlo inference
- **`numpyro_hmc_summary_statistics()`**: HMC diagnostics and summary stats

**Pyro Implementations (if torch and pyro-ppl available):**

- **`pyro_npoint_model()`**: Equivalent Pyro model with Gaussian likelihood
- **`pyro_run_importance_sampling()`**: Importance sampling inference
- **`pyro_run_variational_inference()`**: Stochastic variational inference (SVI)
- **`pyro_sample_from_variational_posterior()`**: Posterior sampling from fitted guide

### `data.py` - Standardized Test Data

**Cross-Framework Data Generation**:

- **`polyfn()`**: Core polynomial function evaluating degree 2 polynomials
- **`generate_test_dataset()`**: Creates standardized datasets with configurable parameters
- **`get_standard_datasets()`**: Generate pre-configured datasets for common benchmarks
- **`print_dataset_summary()`**: Display dataset statistics and true parameters

**Key Features**:

- **Consistent Parameters**: Standard polynomial coefficients across all frameworks
- **Reproducible Seeds**: Fixed random seeds ensure identical datasets for fair comparisons
- **Framework Compatibility**: JAX-based data generation compatible with NumPyro
- **Noise Modeling**: Standardized Gaussian noise (σ=0.05) for realistic observations
- **Benchmark Suites**: Pre-configured datasets for performance and accuracy comparisons

### `figs.py` - Comprehensive Visualization Suite

**IMPORTANT**: All visualization functions now use the shared GenJAX Research Visualization Standards (GRVS) from `examples.viz` module for consistent styling across case studies.

**Trace Visualizations:**
- **`save_onepoint_trace_viz()`**: Single curve from prior → `curvefit_prior_trace.pdf`
- **`save_multiple_onepoint_traces_with_density()`**: 3 prior curves with log density → `curvefit_prior_traces_density.pdf`
- **`save_multipoint_trace_viz()`**: Single posterior curve with data → `curvefit_posterior_trace.pdf`
- **`save_multiple_multipoint_traces_with_density()`**: 3 posterior curves with log density → `curvefit_posterior_traces_density.pdf`
- **`save_four_multipoint_trace_vizs()`**: 2x2 grid of posterior curves → `curvefit_posterior_traces_grid.pdf`

**Inference and Scaling:**
- **`save_inference_scaling_viz()`**: 3-panel scaling analysis → `curvefit_scaling_performance.pdf`
  - Runtime vs N (flat line showing vectorization benefit)
  - Log Marginal Likelihood estimates vs N
  - Effective Sample Size (ESS) vs N
  - Uses 100 trials per N for Monte Carlo noise reduction
  - Scientific notation on x-axis ($10^2$, $10^3$, $10^4$)
- **`save_inference_viz()`**: Posterior uncertainty bands from IS → `curvefit_posterior_curves.pdf`

**Method Comparisons:**
- **`save_genjax_posterior_comparison()`**: IS vs HMC comparison → `curvefit_posterior_comparison.pdf`
- **`save_framework_comparison_figure()`**: Main framework comparison → `curvefit_framework_comparison_n10.pdf`
  - **Methods compared**: GenJAX IS (1000), GenJAX HMC, NumPyro HMC
  - **Two-panel layout**: Posterior curves (top), timing comparison (bottom)

**Parameter Density Visualizations:**
- **`save_individual_method_parameter_density()`**: Main inference methods
  - GenJAX IS (N=1000) → `curvefit_params_is1000.pdf`
  - GenJAX HMC → `curvefit_params_hmc.pdf`
  - NumPyro HMC → `curvefit_params_numpyro.pdf`
- **`save_is_comparison_parameter_density()`**: IS variants with N=50, 500, 5000
  - N=50 → `curvefit_params_is50.pdf`
  - N=500 → `curvefit_params_is500.pdf`
  - N=5000 → `curvefit_params_is5000.pdf`
- **`save_is_single_resample_comparison()`**: Single particle resampling distributions
  - N=50 → `curvefit_params_resample50.pdf`
  - N=500 → `curvefit_params_resample500.pdf`
  - N=5000 → `curvefit_params_resample5000.pdf`

**Timing Comparisons:**
- **`save_is_only_timing_comparison()`**: Horizontal bar chart for IS methods only
- **`save_parameter_density_timing_comparison()`**: Timing for all parameter density methods

**Legends:**
- **`create_all_legends()`**: Complete method legend → `curvefit_legend_all.pdf`
- **`create_genjax_is_legend()`**: GenJAX IS-only legends
  - Horizontal → `curvefit_legend_is_horiz.pdf`
  - Vertical → `curvefit_legend_is_vert.pdf`
  - Consistent color scheme throughout

**Other Visualizations:**
- **`save_log_density_viz()`**: Log joint density surface → `curvefit_logprob_surface.pdf`
- **`save_multiple_curves_single_point_viz()`**: Posterior marginal at x → `curvefit_posterior_marginal.pdf`

### `main.py` - Entry Point with Multiple Modes

**Available Modes:**
- **`quick`**: Fast demonstration with basic visualizations
- **`full`**: Complete analysis with all visualizations
- **`benchmark`**: Framework comparison (IS vs HMC methods)
- **`is-only`**: IS-only comparison (N=5, 1000, 5000)
- **`scaling`**: Inference scaling analysis only
- **`outlier`**: Outlier model analysis (generative conditionals)

**Key Features:**
- **Consistent parameters**: Standard defaults for reproducibility
- **CUDA support**: Use `pixi run -e curvefit-cuda` for GPU acceleration
- **Flexible customization**: Command-line args for all parameters

## Key Implementation Details

### Model Specification

**Hierarchical Polynomial Model**:

```python
@gen
def polynomial():
    # Degree 2 polynomial: y = a + b*x + c*x^2
    a = normal(0.0, 1.0) @ "a"  # Constant term
    b = normal(0.0, 0.5) @ "b"  # Linear coefficient
    c = normal(0.0, 0.2) @ "c"  # Quadratic coefficient
    return Lambda(Const(polyfn), jnp.array([a, b, c]))

@gen
def point(x, curve):
    y_det = curve(x)                      # Deterministic curve value
    y_observed = normal(y_det, 0.05) @ "obs"  # Observation noise
    return y_observed
```

### Direct Model Implementation

**Current Pattern**: The `npoint_curve` model takes `xs` as input directly:

```python
@gen
def npoint_curve(xs):
    """N-point curve model with xs as input."""
    curve = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)
```

**Key Design**:

- `xs` passed as input avoids static parameter issues
- Direct model definition without factory pattern
- Vectorized observations using `vmap` for efficiency

### SMC Integration

**Current SMC Usage**:

```python
def infer_latents(xs, ys, n_samples: Const[int]):
    """Infer latent curve parameters using GenJAX SMC importance sampling."""
    from genjax.inference import init

    constraints = {"ys": {"obs": ys}}

    # Use SMC init for importance sampling
    result = init(
        npoint_curve,  # target generative function
        (xs,),  # target args with xs as input
        n_samples,  # already wrapped in Const
        constraints,  # constraints
    )

    return result.traces, result.log_weights
```

**Key Patterns**:

1. **Direct model usage**: No factory pattern needed with xs as input
2. **Const wrapper**: Use `Const[int]` for static n_samples parameter
3. **Input arguments**: Pass `(xs,)` as target args to the model

### Noise Modeling

**Simple Gaussian Noise**:

- **Observation model**: Polynomial evaluation with Gaussian noise
- **Noise level**: σ=0.05 for low observation noise
- **No outlier handling**: Clean data assumption
- **Parameter priors**: Hierarchical with decreasing variance for higher-order terms

### Lambda Utility for Dynamic Functions

**Dynamic Function Creation**:

```python
@Pytree.dataclass
class Lambda(Pytree):
    f: any = Pytree.static()
    dynamic_vals: jnp.ndarray
    static_vals: tuple = Pytree.static(default=())

    def __call__(self, *x):
        return self.f(*x, *self.static_vals, self.dynamic_vals)
```

**Purpose**: Allows generative functions to return callable objects with captured parameters.

## Visualization Features

### GenJAX Research Visualization Standards (GRVS)

All figures use the shared `examples.viz` module for consistent styling:

**Core Standards:**
- **Typography**: 18pt base fonts, bold axis labels, 16pt legends
- **3-Tick Standard**: Exactly 3 tick marks per axis for optimal readability (ENFORCED)
- **Colors**: Colorblind-friendly palette with consistent method colors
- **No Titles Policy**: Figures designed for LaTeX integration
- **Clean Grid**: 30% alpha grid lines, major lines only
- **Publication Quality**: 300 DPI PDF output with tight layout

**Usage Pattern:**
```python
from examples.viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, apply_standard_ticks, save_publication_figure
)

setup_publication_fonts()
fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
ax.plot(x, y, color=get_method_color("curves"), **LINE_SPECS["curve_main"])
apply_grid_style(ax)
apply_standard_ticks(ax)  # GRVS 3-tick standard
save_publication_figure(fig, "output.pdf")
```

### Research Quality Outputs

- **High DPI PDF generation**: Publication-ready figures
- **Multiple visualization types**: Traces, densities, inference results, scaling studies
- **Systematic organization**: Numbered figure outputs for paper inclusion
- **Consistent aesthetics**: All figures follow GRVS standards

### Scaling Studies

- **Performance analysis**: Timing across different sample sizes
- **Quality assessment**: Inference accuracy vs computational cost
- **Comparative visualization**: Shows convergence properties

## Usage Patterns

### Basic Inference

**GenJAX:**

```python
key = jrand.key(42)
curve, (xs, ys) = get_points_for_inference()
samples, weights = seed(infer_latents)(key, xs, ys, Const(1000))
```

**NumPyro (if available):**

```python
# Importance sampling
result = numpyro_run_importance_sampling(key, xs, ys, num_samples=5000)

# Hamiltonian Monte Carlo
hmc_result = numpyro_run_hmc_inference(key, xs, ys, num_samples=2000, num_warmup=1000)
summary = numpyro_hmc_summary_statistics(hmc_result)
```

**Pyro (if available):**

```python
# Importance sampling
result = pyro_run_importance_sampling(xs, ys, num_samples=5000)

# Variational inference
vi_result = pyro_run_variational_inference(xs, ys, num_iterations=500, learning_rate=0.01)
samples = pyro_sample_from_variational_posterior(xs, num_samples=1000)
```

### Custom Model Creation

```python
# Create model trace with specific input points
xs = jnp.linspace(0, 10, 15)
trace = npoint_curve.simulate(xs)
curve, (xs_ret, ys_ret) = trace.get_retval()
```

### Running Examples

```bash
# Quick demonstration (default)
pixi run curvefit
# or equivalently:
python -m examples.curvefit.main quick

# Full analysis
pixi run curvefit-full
# or:
python -m examples.curvefit.main full

# Framework benchmark comparison
pixi run curvefit-benchmark
# or:
python -m examples.curvefit.main benchmark

# With CUDA acceleration
pixi run cuda-curvefit          # Quick mode
pixi run cuda-curvefit-full     # Full analysis
pixi run cuda-curvefit-benchmark # Benchmark

# Customize parameters
python -m examples.curvefit.main benchmark --n-points 30 --timing-repeats 20
python -m examples.curvefit.main full --n-samples-is 2000 --n-samples-hmc 1500
```

## Development Guidelines

### When Adding New Models

1. **Pass data as inputs** to avoid static dependency issues
2. **Use Const wrapper** for parameters that must remain static
3. **Follow established patterns** from core.py implementation

### When Modifying Inference

1. **Use Const wrapper** for static parameters like n_samples
2. **Test with different data sizes** to ensure model flexibility
3. **Apply seed transformation** before JIT compilation

### When Adding Visualizations

1. **Use high DPI settings** for publication quality
2. **Follow systematic naming** (e.g., `050_inference_viz.pdf`)
3. **Include uncertainty visualization** for Bayesian results

## Common Patterns

### Input Parameter Pattern

```python
# ✅ CORRECT - Pass data as input arguments
@gen
def model(xs):
    # xs is passed as input, avoiding static issues
    ys = process(xs)
    return ys

# Alternative if static values needed - use Const wrapper
@gen
def model(n: Const[int]):
    xs = jnp.arange(0, n.value)  # Access static value
    return xs
```

### SMC with Const Pattern

```python
# ✅ CORRECT - Use Const wrapper for static parameters
def infer(xs, ys, n_samples: Const[int]):
    result = init(model, (xs,), n_samples, constraints)
    return result

# Call with Const wrapper
infer(xs, ys, Const(1000))
```

## Testing Patterns

### Model Validation

```python
# Test model with specific inputs
xs = jnp.linspace(0, 5, 20)
trace = npoint_curve.simulate(xs)
curve, (xs_ret, ys_ret) = trace.get_retval()
assert xs_ret.shape == (20,)
assert ys_ret.shape == (20,)
```

### Inference Validation

```python
# Test inference with proper seeding
xs, ys = get_points_for_inference(n_points=20)
samples, weights = seed(infer_latents)(key, xs, ys, Const(1000))
assert samples.get_choices()['curve']['a'].shape == (1000,)  # polynomial coefficients
assert samples.get_choices()['curve']['b'].shape == (1000,)
assert samples.get_choices()['curve']['c'].shape == (1000,)
assert weights.shape == (1000,)
```

## Performance Considerations

### JIT Compilation

GenJAX functions use JAX JIT compilation for performance, following the proper `seed()` → `jit()` order:

**Correct Pattern**:
```python
# Apply seed() before jit() for GenJAX functions
seeded_fn = seed(my_probabilistic_function)
jit_fn = jax.jit(seeded_fn)  # No static_argnums needed with Const pattern
```

**Available JIT-compiled functions**:
- `infer_latents_jit`: JIT-compiled GenJAX importance sampling (~5x speedup)
- `hmc_infer_latents_jit`: JIT-compiled GenJAX HMC inference (~4-5x speedup)
- `numpyro_run_importance_sampling_jit`: JIT-compiled NumPyro importance sampling
- `numpyro_run_hmc_inference_jit`: JIT-compiled NumPyro HMC with `jit_model_args=True`

**Key benefits**:
- **Const pattern**: Use `Const[int]`, `Const[float]` instead of `static_argnums`
- **Significant speedups**: 4-5x performance improvement for GenJAX inference
- **Factory benefits**: Eliminates repeated model compilation
- **Closure benefits**: Enables efficient SMC vectorization

### Memory Usage

- **Large sample sizes**: Monitor memory usage with >100k samples
- **Vectorized operations**: Prefer `point.vmap()` over Python loops
- **Trace storage**: Consider trace compression for very large inference runs

## Integration with Main GenJAX

This case study serves as:

1. **Input parameter pattern**: Shows how to pass data as model inputs
2. **SMC usage demonstration**: Illustrates importance sampling with Const wrapper
3. **Polynomial regression showcase**: Demonstrates hierarchical Bayesian curve fitting
4. **Visualization reference**: Provides examples of research-quality figure generation

## Common Issues

### Concrete Value Errors

- **Cause**: Using dynamic arguments in `jnp.arange`, `jnp.zeros`, etc.
- **Solution**: Pass data as input arguments or use Const wrapper
- **Example**: `npoint_curve(xs)` with xs as input

### SMC Parameter Issues

- **Cause**: Passing unwrapped integers to inference functions
- **Solution**: Use Const wrapper for static parameters
- **Pattern**: `infer_latents(xs, ys, Const(1000))`

### NumPyro JAX Transformation Issues

- **Issue**: NumPyro's HMC diagnostics contain format strings that fail when values are JAX tracers
- **Error**: `TypeError: unsupported format string passed to Array.__format__`
- **Root Cause**: JAX tracers cannot be directly formatted with Python string formatting
- **Solution**: Convert JAX arrays to Python floats before string formatting using `.item()` or `float()`
- **Context**: This is a known issue when running NumPyro under JAX transformations

**Example Fix**:
```python
# ❌ WRONG - JAX tracer formatting fails
f"Value: {jax_array:.2f}"

# ✅ CORRECT - Convert to Python float first
f"Value: {float(jax_array):.2f}"
```

### Cond Combinator for Mixture Models

The `Cond` combinator now fully supports mixture models with same addresses in both branches. The outlier model in this case study demonstrates this capability:

```python
# ✅ Mixture model with outliers using Cond combinator
# Define branch functions outside to avoid JAX local function comparison issues
@gen
def inlier_branch(y_det, outlier_std):
    # outlier_std is ignored for inliers but needed for consistent signatures
    return normal(y_det, 0.2) @ "obs"  # Same address in both branches

@gen  
def outlier_branch(y_det, outlier_std):
    return normal(y_det, outlier_std) @ "obs"  # Same address in both branches

@gen
def point_with_outliers(x, curve, outlier_rate=0.1, outlier_std=1.0):
    y_det = curve(x)
    is_outlier = flip(outlier_rate) @ "is_outlier"
    
    # Natural mixture model using Cond
    cond_model = Cond(outlier_branch, inlier_branch)
    y_observed = cond_model(is_outlier, y_det, outlier_std) @ "y"
    return y_observed
```

**Key Features**:
- **Natural syntax**: Express mixture models as you would mathematically
- **Full inference support**: Works with all GenJAX inference algorithms (IS, HMC)
- **JAX optimized**: Uses efficient `jnp.where` for conditional selection
- **Type safe**: Branches must have compatible return types and signatures

**Implementation Note**: Define branch functions at module level (not as local functions) to avoid JAX's local function comparison issues during tracing.

### Import Dependencies

- **Matplotlib required**: For figure generation in `figs.py`
- **NumPy compatibility**: Used alongside JAX for some visualizations
- **Environment**: Use `pixi run -e curvefit` for proper dependencies

## Recent Updates (June 2025)

### Enhanced Visualization Suite

The case study has been significantly enhanced with comprehensive visualizations:

**New Figure Types Added**:
- **Multiple trace figures**: 3-panel trace visualizations with log density values
- **IS comparison suite**: Comprehensive comparison of IS with N=50, 500, 5000
- **Parameter density plots**: 2D hexbin + 3D surface visualizations for all methods
- **Timing comparisons**: Horizontal bar charts for method performance
- **Legend figures**: Standalone legends for flexible LaTeX integration

**Inference Scaling Improvements**:
- **Monte Carlo noise reduction**: 100 trials per N for stable estimates
- **Scientific notation**: Clean x-axis labels ($10^2$, $10^3$, $10^4$)
- **Runtime analysis**: Shows flat performance due to GPU vectorization
- **No error bars**: Cleaner runtime plot focusing on the mean
- **Y-axis limits**: Zoomed to 0.2-0.3ms range to emphasize flatness

**Visual Consistency**:
- **No titles**: Figures designed to be understood from context
- **Consistent colors**: 
  - IS N=50: Light purple (#B19CD9)
  - IS N=500: Medium blue (#0173B2)
  - IS N=5000: Dark green (#029E73)
  - HMC: Orange (#DE8F05)
  - NumPyro: Green (#029E73)
- **Reduced noise**: Observation noise reduced from 0.2 to 0.05 for tighter posteriors
- **Vertical red lines**: Ground truth indicators in 3D parameter density plots

**Figure Naming Update**:
- **Descriptive names**: Replaced cryptic numbers with self-explanatory names
- **Clear prefixes**: `trace_`, `posterior_`, `params_`, `legend_` for grouping
- **Explicit particle counts**: `is50`, `is500`, `is5000` instead of generic numbers
- **Resample clarity**: `params_resample` instead of ambiguous "single"

**CUDA Integration**:
- **GPU acceleration**: All timing benchmarks run with CUDA when available
- **Proper environments**: Use `pixi run -e curvefit-cuda` for GPU support
- **Vectorization demonstration**: Runtime plots clearly show GPU benefits
