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
├── figs.py             # Visualization and figure generation utilities
└── figs/               # Generated visualization outputs
    └── *.pdf           # Various curve fitting visualizations
```

## Code Organization

### `core.py` - Model Implementations

- **`point(x, curve)`**: Single data point model with outlier handling
- **`sine()`**: Sine wave parameter prior model
- **`onepoint_curve(x)`**: Single point curve fitting model
- **`npoint_curve_factory(n)`**: Factory for multi-point curve models with static n
- **`_infer_latents()`**: SMC-based parameter inference using proper factory/closure patterns
- **`infer_latents()`**: JIT-compiled inference function
- **`get_points_for_inference()`**: Test data generation utility

### `figs.py` - Visualization

- **Trace visualizations**: Single and multi-point curve traces
- **Inference visualizations**: Posterior curve overlays with uncertainty
- **Scaling studies**: Performance and quality analysis across sample sizes
- **Density visualizations**: Log-density comparisons

### `main.py` - Figure Generation

- **Orchestrates all visualizations**: Calls figure generation functions in sequence
- **Produces complete paper-ready figures**: For research and documentation

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

```python
key = jrand.key(42)
curve, (xs, ys) = get_points_for_inference()
samples, weights = infer_latents(key, ys, 1000)
```

### Custom Model Creation

```python
# Create model for specific number of points
model = npoint_curve_factory(15)
trace = model.simulate(())
```

### Visualization Generation

```bash
pixi run -e curvefit curvefit
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
# Test factory pattern
model = npoint_curve_factory(10)
trace = model.simulate(())
assert trace.get_retval()[1][0].shape == (10,)  # xs shape
assert trace.get_retval()[1][1].shape == (10,)  # ys shape
```

### Inference Validation

```python
# Test inference convergence
samples, weights = infer_latents(key, ys, 1000)
assert samples.get_choices()['curve']['freq'].shape == (1000,)
assert weights.shape == (1000,)
```

## Performance Considerations

### JIT Compilation

- **Static arguments**: Use `static_argnums=(2,)` for `n_samples` parameter
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

### Import Dependencies

- **Matplotlib required**: For figure generation in `figs.py`
- **NumPy compatibility**: Used alongside JAX for some visualizations
- **Environment**: Use `pixi run -e curvefit` for proper dependencies

## Evolution Notes

The implementation has evolved from manual SMC replication to proper SMC library usage:

- **Original**: Manual `modular_vmap` + `generate` pattern
- **Current**: Factory pattern + closure pattern + `genjax.smc.default_importance_sampling`
- **Benefits**: Better maintainability, consistency with GenJAX patterns, educational value

This case study should remain stable and serve as a reference for proper factory and SMC usage patterns in GenJAX.
