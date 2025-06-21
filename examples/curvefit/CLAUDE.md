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

### `figs.py` - Visualization

- **Trace visualizations**: Single and multi-point curve traces
- **Inference visualizations**: Posterior curve overlays with uncertainty
- **Scaling studies**: Performance and quality analysis across sample sizes
- **Density visualizations**: Log-density comparisons

### `figs.py` - Clean Visualization Suite

**Core Visualizations:**
- **`save_onepoint_trace_viz()`**: Single point curve trace
- **`save_multipoint_trace_viz()`**: Multi-point curve trace
- **`save_inference_viz()`**: Posterior curve overlay with uncertainty

**Framework Comparison:**
- **`save_framework_comparison_figure()`**: Clean comparison of IS (1000) vs HMC methods
  - **Methods compared**: GenJAX IS (1000 particles), GenJAX HMC, NumPyro HMC
  - **Two-panel layout**: Posterior curves (top), timing comparison (bottom)
  - **JIT-compiled performance**: Fair comparison with proper warm-up
  - **Professional styling**: Publication-ready with clear legends
  - **Focused metrics**: Essential timing and acceptance rate information

### `main.py` - Simplified Entry Point

- **Three clean modes**: quick (fast demo), full (complete analysis), benchmark (framework comparison)
- **Focused on essentials**: IS with 1000 particles vs HMC methods
- **Standard parameters**: Consistent defaults for reproducibility

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

### Research Quality Outputs

- **High DPI PDF generation**: Publication-ready figures
- **Multiple visualization types**: Traces, densities, inference results, scaling studies
- **Systematic organization**: Numbered figure outputs for paper inclusion

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

### Import Dependencies

- **Matplotlib required**: For figure generation in `figs.py`
- **NumPy compatibility**: Used alongside JAX for some visualizations
- **Environment**: Use `pixi run -e curvefit` for proper dependencies

## Recent Updates (2025)

### Simplified and Focused Structure

The case study has been streamlined to focus on essential comparisons:

**Key Changes**:
- **Clean `figs.py`**: Focused on core visualizations and framework comparison
- **Simplified `main.py`**: Three clear modes (quick, full, benchmark)
- **Essential benchmarks**: IS with 1000 particles vs HMC methods only
- **Reduced complexity**: From 2000+ lines to ~350 lines in figs.py

**Framework Comparison Focus**:
- **GenJAX IS**: 1000 particles (fixed) for consistent comparison
- **GenJAX HMC**: Standard parameters matching NumPyro
- **NumPyro HMC**: JIT-compiled for fair performance comparison
- **Clean visualization**: Two-panel figure with posterior curves and timing

**Benefits**:
- **Faster execution**: Reduced from minutes to seconds for benchmarks
- **Clearer focus**: Essential comparisons without overwhelming details
- **Better maintainability**: Simpler code structure following standards
- **Educational value**: Clear demonstration of key concepts
