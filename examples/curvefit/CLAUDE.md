# CLAUDE.md - Curve Fitting Case Study

This file provides guidance to Claude Code when working with the curve fitting case study that demonstrates Bayesian inference for sine wave parameter estimation.

## Overview

The curvefit case study showcases Bayesian curve fitting using GenJAX, demonstrating outlier-robust inference for sine wave parameters with hierarchical modeling and importance sampling.

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

- **`point(x, curve)`**: Single data point model with outlier handling
- **`sine()`**: Sine wave parameter prior model
- **`onepoint_curve(x)`**: Single point curve fitting model
- **`npoint_curve_factory(n)`**: Factory for multi-point curve models with static n
- **`_infer_latents()`**: SMC-based parameter inference using proper factory/closure patterns
- **`infer_latents()`**: JIT-compiled inference function
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

- **`sinfn()`**: Core sine function with frequency and offset parameters
- **`generate_test_dataset()`**: Creates standardized datasets with configurable parameters
- **`convert_to_torch()`**: Convert JAX datasets to PyTorch format for Pyro compatibility
- **`convert_to_numpy()`**: Convert JAX datasets to NumPy format for general use
- **`get_standard_datasets()`**: Generate pre-configured datasets for common benchmarks
- **`print_dataset_summary()`**: Display dataset statistics and true parameters

**Key Features**:

- **Consistent Parameters**: Default true_freq=0.3, true_offset=1.5 across all frameworks
- **Reproducible Seeds**: Fixed random seeds ensure identical datasets for fair comparisons
- **Framework Compatibility**: Automatic conversion between JAX, NumPy, and PyTorch formats
- **Noise Modeling**: Standardized Gaussian noise (σ=0.3) for realistic observations
- **Benchmark Suites**: Pre-configured datasets for performance and accuracy comparisons

### `figs.py` - Visualization

- **Trace visualizations**: Single and multi-point curve traces
- **Inference visualizations**: Posterior curve overlays with uncertainty
- **Scaling studies**: Performance and quality analysis across sample sizes
- **Density visualizations**: Log-density comparisons

### `figs.py` - Comprehensive Visualization Suite

**Standard Visualizations:**
- **Trace visualizations**: Single and multi-point curve traces
- **Inference visualizations**: Posterior curve overlays with uncertainty
- **Scaling studies**: Performance and quality analysis across sample sizes
- **Density visualizations**: Log-density comparisons
- **Comprehensive benchmark**: Cross-framework comparison with timing analysis

**Advanced Analysis Functions:**

**`save_genjax_scaling_benchmark()` - GenJAX Performance Analysis:**
- **Importance sampling scaling**: Performance and posterior quality vs N_samples
- **HMC scaling**: Performance and posterior quality vs chain length
- **Comprehensive metrics**: Timing, bias, variance, ESS, acceptance rates
- **JIT-compiled functions**: Uses optimized GenJAX functions for accurate benchmarks
- **Research-quality output**: 12-panel figure showing complete scaling analysis
- **Standard interface**: Integrated with main.py CLI system

**`save_genjax_posterior_comparison()` - Comprehensive Method Comparison:**
- **Three-method comparison**: GenJAX IS, GenJAX HMC, and NumPyro HMC with JIT compilation
- **Visual posterior comparison**: Side-by-side posterior curve visualizations for each method
- **Parameter space visualization**: Joint posterior distributions showing parameter estimates
- **Comprehensive overlay**: Combined view showing all three methods with observed data
- **Performance benchmarking**: JIT-compiled timing comparisons across all methods
- **Statistical diagnostics**: Posterior means, uncertainties, bias, and HMC acceptance rates
- **Professional visualization**: 4-row layout with publication-ready formatting
  - **Row 1**: Individual method posterior plots + parameter estimates scatter plot
  - **Row 2**: Combined overlay comparison spanning full width
  - **Row 3**: Performance comparison bar chart (unlabeled bars, color-coded)
  - **Row 4**: Comprehensive shared legend (3-column, 3-row layout, 20pt font)
- **JIT-compiled inference**: Uses optimized functions for fair performance comparison
- **Parameter alignment**: Ensures identical HMC parameters between GenJAX and NumPyro (step_size=0.01, num_steps=20)
- **Font standardization**: Matches faircoin publication standards (18-22pt fonts, bold titles)
- **Legend organization**: Groups elements logically (ground truth, posteriors, means)
- **Standard interface**: Integrated with main.py CLI system

### `main.py` - Figure Generation

- **Orchestrates all visualizations**: Calls figure generation functions in sequence
- **Produces complete paper-ready figures**: For research and documentation
- **Multiple modes**: Support for traces, inference, scaling, benchmark, genjax-scaling, and posterior-comparison

## Key Implementation Details

### Model Specification

**Hierarchical Sine Wave Model**:

```python
@gen
def sine():
    freq = exponential(10.0) @ "freq"      # Frequency parameter
    offset = uniform(0.0, 2.0 * pi) @ "off"  # Phase offset
    return Lambda(sinfn, jnp.array([freq, offset]))

@gen
def point(x, curve):
    y_det = curve(x)                       # Deterministic curve value
    is_outlier = flip(0.08) @ "is_out"     # 8% outlier probability
    y_out = uniform(-3.0, 3.0) @ "y_out"   # Outlier value
    y = jnp.where(is_outlier, y_out, y_det)  # Mixture model
    y_observed = normal(y, 0.2) @ "obs"    # Observation noise
    return y_observed
```

### Factory Pattern for Static Dependencies

**Critical Pattern**: The `npoint_curve_factory` demonstrates proper handling of static parameters:

```python
def npoint_curve_factory(n: int):
    """Factory function to create npoint_curve with static n parameter."""

    @gen
    def npoint_curve():
        curve = sine() @ "curve"
        xs = jnp.arange(0, n)  # n is now static from factory closure
        ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
        return curve, (xs, ys)

    return npoint_curve
```

**Why Factory Pattern is Necessary**:

- `jnp.arange(0, n)` requires concrete value for `n`
- Direct usage in `@gen` functions causes tracing issues with SMC
- Factory pattern captures `n` as static in closure, preventing tracer propagation

### SMC Integration with Closure Pattern

**Proper SMC Usage**:

```python
def _infer_latents(key, ys, n_samples):
    # Create model with static n using factory pattern
    npoint_curve_model = npoint_curve_factory(len(ys))

    # Create closure for default_importance_sampling that captures static arguments
    def default_importance_sampling_closure(target_gf, target_args, constraints):
        return default_importance_sampling(
            target_gf, target_args, n_samples, constraints
        )

    # Apply seed to the closure - follows test_smc.py pattern
    result = seed(default_importance_sampling_closure)(
        key, npoint_curve_model, (), constraints
    )

    return result.traces, result.log_weights
```

**Key Patterns**:

1. **Factory for static dependencies**: Handles `jnp.arange` concrete value requirement
2. **Closure for static arguments**: Captures `n_samples` for `seed` transformation
3. **Empty target args**: `()` because `n` is captured in factory, not passed as argument

### Outlier-Robust Modeling

**Mixture Model Approach**:

- **Primary model**: Sine wave with Gaussian noise
- **Outlier model**: Uniform distribution over reasonable range
- **Mixture probability**: 8% outlier rate (tunable parameter)
- **Robust inference**: Automatically identifies and down-weights outliers

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
samples, weights = infer_latents(key, ys, 1000)
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
# Create model for specific number of points
model = npoint_curve_factory(15)
trace = model.simulate()  # No args needed for this model
```

### Running Examples

```bash
# Generate all figures (complete case study)
pixi run -e curvefit curvefit
# or equivalently:
python -m examples.curvefit.main --mode all

# Specific figure types
python -m examples.curvefit.main --mode traces              # Basic trace visualizations
python -m examples.curvefit.main --mode inference          # Inference visualization
python -m examples.curvefit.main --mode scaling            # Original scaling analysis
python -m examples.curvefit.main --mode benchmark          # Cross-framework benchmark
python -m examples.curvefit.main --mode genjax-scaling     # GenJAX scaling analysis
python -m examples.curvefit.main --mode posterior-comparison # Three-method comparison

# Customize parameters
python -m examples.curvefit.main --mode posterior-comparison --n-points 20 --timing-repeats 5
python -m examples.curvefit.main --mode genjax-scaling --n-points 15 --timing-repeats 3

# Legacy compatibility (still supported)
pixi run -e curvefit curvefit-core
pixi run -e curvefit curvefit-figs
pixi run -e curvefit curvefit-all
```

## Development Guidelines

### When Adding New Models

1. **Use factory pattern** for any static dependencies (array sizes, loop bounds)
2. **Capture concrete values** before entering generative functions
3. **Follow SMC closure pattern** for inference integration

### When Modifying Inference

1. **Maintain factory + closure pattern** for SMC compatibility
2. **Test with different data sizes** to ensure static parameter handling works
3. **Verify JIT compilation** with `static_argnums` specification

### When Adding Visualizations

1. **Use high DPI settings** for publication quality
2. **Follow systematic naming** (e.g., `050_inference_viz.pdf`)
3. **Include uncertainty visualization** for Bayesian results

## Common Patterns

### Factory Pattern Usage

```python
# ✅ CORRECT - Static parameter in factory
def model_factory(n: int):
    @gen
    def model():
        xs = jnp.arange(0, n)  # n is static
        # ... rest of model
    return model

# ❌ WRONG - Dynamic parameter causes tracing issues
@gen
def model(n):
    xs = jnp.arange(0, n)  # n becomes tracer, causes concrete value error
```

### SMC Closure Pattern

```python
# ✅ CORRECT - Closure captures static arguments
def inference_closure(target_gf, target_args, constraints):
    return default_importance_sampling(target_gf, target_args, n_samples, constraints)

result = seed(inference_closure)(key, model, (), constraints)

# ❌ WRONG - Direct usage causes tracing issues
result = seed(default_importance_sampling)(key, model, (), n_samples, constraints)
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
assert samples.get_choices()['curve']['freq'].shape == (1000,)
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

1. **Factory pattern example**: Shows how to handle static dependencies properly
2. **SMC usage demonstration**: Illustrates correct closure patterns with seed
3. **Outlier modeling showcase**: Demonstrates robust Bayesian inference
4. **Visualization reference**: Provides examples of research-quality figure generation

## Common Issues

### Concrete Value Errors

- **Cause**: Using dynamic arguments in `jnp.arange`, `jnp.zeros`, etc.
- **Solution**: Use factory pattern to capture static values
- **Example**: `npoint_curve_factory(n)` instead of `npoint_curve(n)`

### SMC Tracing Issues

- **Cause**: Passing dynamic `n_samples` to `default_importance_sampling`
- **Solution**: Use closure pattern to capture static arguments
- **Pattern**: See `test_smc.py` lines 242-256 for reference

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

## Recent Developments

### Standard Case Study Integration (2025)

The case study has been fully integrated into the standard GenJAX examples format:

**Structural Improvements**:
- **Unified `figs.py`**: All visualization functions consolidated into single file following examples standards
- **Standard CLI interface**: Full integration with `main.py` supporting all standard modes and parameters
- **Removed standalone scripts**: Eliminated `genjax_scaling_benchmark.py` and `genjax_posterior_comparison.py`
- **Consistent API**: All functions follow `save_*()` naming convention with standardized parameters

**Advanced Analysis Integration**:
- **`save_genjax_scaling_benchmark()`**: 12-panel GenJAX performance analysis fully integrated
- **`save_genjax_posterior_comparison()`**: 4-row three-method comparison fully integrated
- **Parameter customization**: CLI arguments for sample sizes, timing repeats, and seeds
- **Error handling**: Robust exception handling with graceful degradation

### Comprehensive Framework Comparison (2025)

The case study includes a complete three-way comparison between GenJAX and NumPyro:

**Implementation Features**:
- **JIT compilation**: All methods use optimized JIT compilation for fair performance comparison
- **Parameter alignment**: GenJAX and NumPyro HMC use identical parameters (step_size=0.01, num_steps=20)
- **Chain length optimization**: Increased to 5000 samples + 3000 warmup for better mixing
- **Thinning for autocorrelation**: Every 5th sample used for visualization to reduce correlation
- **Professional visualization**: 4-row layout with publication-ready formatting and comprehensive legend

**Key Insights**:
- **GenJAX performance**: Competitive with NumPyro on JIT-compiled implementations
- **Method comparison**: Clear visual comparison between importance sampling and HMC approaches
- **Publication quality**: Matches font standards and formatting of other GenJAX examples

**Technical Achievements**:
- **Font standardization**: All figures match faircoin publication standards (18-22pt fonts)
- **Legend organization**: 3-column, 3-row layout grouping ground truth, posteriors, and means
- **Bar chart design**: Clean unlabeled bars with color-coding for method identification
- **Statistical rigor**: Proper timing benchmarks with warm-up and multiple repetitions
- **Standard compliance**: Follows all GenJAX examples directory standards

### Evolution Notes

The implementation has evolved from manual SMC replication to a standard-compliant case study:

- **Original**: Manual `modular_vmap` + `generate` pattern
- **SMC Integration**: Factory pattern + closure pattern + `genjax.smc.default_importance_sampling`
- **Framework Comparison**: Added NumPyro integration with JIT compilation and comprehensive visualization
- **Standard Integration**: Unified into standard examples format with consolidated `figs.py` and standard CLI
- **Benefits**: Better maintainability, consistency with GenJAX patterns, educational value, competitive benchmarking, and standard compliance

This case study now serves as both a reference for proper GenJAX patterns and a demonstration of competitive performance against established probabilistic programming frameworks, while fully conforming to GenJAX examples directory standards.
