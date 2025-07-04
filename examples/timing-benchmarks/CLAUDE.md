# CLAUDE.md - Timing Benchmarks Case Study

This file provides guidance to Claude Code when working with the timing benchmarks case study that comprehensively compares GenJAX performance against other probabilistic programming systems.

## Overview

The timing-benchmarks case study provides systematic performance comparisons between GenJAX and other probabilistic programming frameworks (Pyro, NumPyro, Handcoded JAX, Gen.jl, Stan) across multiple benchmark problems and inference algorithms. This case study focuses on fair, reproducible comparisons with standardized data and algorithm parameters.

## Directory Structure

```
examples/timing-benchmarks/
├── CLAUDE.md           # This file - guidance for Claude Code
├── main.py             # CLI entry point with benchmark modes
├── core.py             # Benchmark implementations across frameworks
├── data.py             # Standardized benchmark data generation
├── figs.py             # Performance visualization utilities
├── export.py           # Benchmark results export/import
├── julia_interface.py  # Python-Julia bridge for Gen.jl benchmarks
├── julia/              # Julia/Gen.jl benchmark implementations
│   ├── Project.toml    # Julia project configuration
│   └── src/
│       ├── TimingBenchmarks.jl      # Main module
│       ├── polynomial_regression.jl  # Polynomial benchmarks
│       └── utils.jl                 # Data I/O utilities
├── data/               # Experimental timing data storage
│   └── benchmark_*/    # Timestamped benchmark results
└── figs/               # Generated performance visualizations
    └── *.pdf           # Timing comparison figures
```

## Benchmark Categories

### Category 1: Polynomial Regression (No Outliers)

**Model**: Degree 2 polynomial regression with Gaussian likelihood
- **Parameters**: a, b, c ~ Normal priors
- **Observation model**: y ~ Normal(a + b*x + c*x², σ)
- **Data sizes**: 10, 50, 100, 500, 1000 observations

**Inference Methods**:
1. **Importance Sampling**:
   - Particle counts: 100, 1000, 10000
   - Frameworks: GenJAX, Pyro, NumPyro, Gen.jl
   
2. **Hamiltonian Monte Carlo (HMC)**:
   - Chain lengths: 1000 samples (500 warmup)
   - Step size: 0.01, Leapfrog steps: 20
   - Frameworks: GenJAX, NumPyro, Stan

### Category 2: Hierarchical Models (Planned)

**Beta-Bernoulli Model**:
- Hierarchical inference with conjugacy
- Frameworks: GenJAX, Pyro, NumPyro, Handcoded JAX, Gen.jl

### Category 3: State Space Models (Planned)

**Hidden Markov Models**:
- Sequential inference benchmarks
- SMC vs Kalman filtering
- Frameworks: GenJAX, NumPyro, Gen.jl

## Code Organization

### `core.py` - Benchmark Implementations

**GenJAX Implementations**:
```python
# Polynomial regression model (reuse from curvefit)
@gen
def polynomial():
    a = normal(0.0, 1.0) @ "a"
    b = normal(0.0, 1.0) @ "b"
    c = normal(0.0, 1.0) @ "c"
    return Lambda(Const(polyfn), jnp.array([a, b, c]))

# Timing functions
def genjax_polynomial_is_timing(xs, ys, n_particles, repeats=100):
    """GenJAX importance sampling timing."""
    # Implementation with proper warm-up
    
def genjax_polynomial_hmc_timing(xs, ys, n_samples, repeats=100):
    """GenJAX HMC timing."""
    # Implementation
```

**Framework Adapters**:
- **NumPyro**: Direct JAX-based implementation
- **Pyro**: PyTorch-based with device management
- **Gen.jl**: Julia interop via subprocess (see julia_interface.py)
- **Stan**: PyStan or CmdStanPy interface
- **Handcoded JAX**: Pure JAX implementation for baseline

### `data.py` - Standardized Benchmark Data

**Key Functions**:
```python
def generate_polynomial_data(n_points, seed=42, noise_std=0.05):
    """Generate polynomial regression data for all frameworks."""
    # Consistent data generation
    
def get_benchmark_datasets():
    """Return standard benchmark dataset configurations."""
    return {
        "small": generate_polynomial_data(10),
        "medium": generate_polynomial_data(50),
        "large": generate_polynomial_data(100),
        "xlarge": generate_polynomial_data(500),
        "xxlarge": generate_polynomial_data(1000),
    }
```

### `figs.py` - Performance Visualizations

**Visualization Types**:

1. **Scaling Plots**:
   - Runtime vs data size (log-log scale)
   - Runtime vs particle count
   - Memory usage comparisons

2. **Framework Comparisons**:
   - Grouped bar charts by method
   - Speedup ratios (relative to baseline)
   - Statistical significance indicators

3. **Quality vs Speed**:
   - Pareto frontiers
   - ESS/second metrics
   - Accuracy-runtime tradeoffs

### `export.py` - Benchmark Data Management

**Data Export Structure**:
```
data/benchmark_polynomial_is_20250702_143022/
├── metadata.json              # Benchmark configuration
├── summary.csv               # Overview results
├── genjax_is_timing.csv     # Detailed timings
├── pyro_is_timing.csv        # Per-framework results
├── numpyro_is_timing.csv
└── gen_is_timing.csv
```

### `main.py` - CLI Interface

**Benchmark Modes**:
```bash
# Run specific benchmark
python -m examples.timing-benchmarks.main polynomial-is --frameworks genjax pyro numpyro

# Run all polynomial benchmarks
python -m examples.timing-benchmarks.main polynomial-all

# Plot from saved data
python -m examples.timing-benchmarks.main plot --data data/benchmark_polynomial_is_*

# Quick test mode
python -m examples.timing-benchmarks.main test --repeats 5
```

## Implementation Plan

### Phase 1: Core Infrastructure (Current)
1. ✅ Directory structure
2. ✅ CLAUDE.md planning
3. 🔄 data.py for polynomial regression
4. 🔄 core.py with GenJAX implementation
5. 🔄 Basic main.py CLI

### Phase 2: Framework Integration
1. NumPyro polynomial regression
2. Pyro polynomial regression  
3. Basic timing comparisons
4. Initial visualizations

### Phase 3: Extended Benchmarks
1. Gen.jl integration (if feasible)
2. Stan integration
3. Handcoded JAX baseline
4. HMC benchmarks

### Phase 4: Analysis & Reporting
1. Comprehensive visualizations
2. Statistical analysis
3. Performance profiling
4. Memory usage tracking

## Technical Considerations

### Fair Comparison Principles

1. **Identical Data**: All frameworks use exact same datasets
2. **Warm-up Runs**: JIT compilation warm-up for fair timing
3. **Parameter Matching**: Same algorithm parameters across frameworks
4. **Hardware Consistency**: Document GPU/CPU usage per framework
5. **Version Tracking**: Record framework versions in metadata

### Framework-Specific Notes

**GenJAX**:
- Use `seed()` transformation before `jit()`
- Const[int] pattern for static parameters
- Leverage vectorization where possible

**NumPyro**:
- JIT compilation with `numpyro.jit()`
- Same JAX backend as GenJAX
- Fair comparison due to shared infrastructure

**Pyro**:
- PyTorch backend differences
- GPU device management
- Consider torch.compile() for newer versions

**Gen.jl**:
- Julia interop via subprocess (julia_interface.py)
- Requires Julia installation via juliaup
- Project dependencies managed in julia/Project.toml
- Data exchange via CSV for simplicity

**Stan**:
- Compilation overhead considerations
- Separate warmup timing
- CmdStanPy for stability

### Performance Metrics

1. **Wall Clock Time**: Primary metric
2. **Compilation Time**: Separate JIT/Stan compilation
3. **Memory Usage**: Peak memory consumption
4. **Throughput**: Samples/second for MCMC
5. **Quality Metrics**: ESS, R-hat, log marginal likelihood

## Julia/Gen.jl Setup

### Prerequisites

1. **Install Julia via juliaup**:
   ```bash
   curl -fsSL https://install.julialang.org | sh
   juliaup add 1.9  # or latest stable
   juliaup default 1.9
   ```

2. **Setup Gen.jl environment**:
   ```bash
   cd examples/timing-benchmarks/julia
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

3. **Test Julia interface**:
   ```python
   from julia_interface import setup_gen_jl
   setup_gen_jl()  # One-time setup
   ```

### Julia Benchmark Implementation

The Gen.jl benchmarks are implemented in `julia/src/`:
- **polynomial_regression.jl**: Matching model and inference algorithms
- **utils.jl**: Data I/O and timing utilities
- **TimingBenchmarks.jl**: Main module exports

Data exchange uses CSV format for simplicity and compatibility.

## Usage Patterns

### Running Benchmarks

```bash
# Full polynomial regression benchmark suite
pixi run -e timing-benchmarks python -m examples.timing-benchmarks.main polynomial-all \
    --repeats 100 --export-data

# Specific comparison
pixi run -e timing-benchmarks python -m examples.timing-benchmarks.main polynomial-is \
    --frameworks genjax numpyro --n-particles 1000 --data-size 100

# Test mode for development
pixi run -e timing-benchmarks python -m examples.timing-benchmarks.main test
```

### Visualization Generation

```bash
# Generate all figures from latest benchmark
pixi run -e timing-benchmarks python -m examples.timing-benchmarks.main plot

# Specific visualization
pixi run -e timing-benchmarks python -m examples.timing-benchmarks.main plot \
    --plot-type scaling --methods is hmc
```

## Development Guidelines

### When Adding New Benchmarks

1. **Define in data.py**: Standardized data generation
2. **Implement in core.py**: Framework-specific versions
3. **Add CLI support**: New command in main.py
4. **Create visualizations**: Appropriate plots in figs.py
5. **Document thoroughly**: Update this CLAUDE.md

### Testing Strategy

1. **Unit tests**: Each framework implementation
2. **Integration tests**: Full benchmark pipeline
3. **Smoke tests**: Quick runs with minimal data
4. **Validation**: Ensure inference results are correct

### Performance Tips

1. **Batch operations**: Run multiple configurations together
2. **Parallel execution**: Use multiprocessing for independent runs
3. **Caching**: Avoid recomputing reference data
4. **Profiling**: Include optional detailed profiling

## Common Issues

### Framework Installation

- **NumPyro**: Included in pixi environment
- **Pyro**: May need separate PyTorch environment
- **Gen.jl**: Requires Julia installation via juliaup
- **Stan**: Requires C++ toolchain

### Timing Accuracy

- **Warm-up**: Always include JIT warm-up runs
- **System load**: Run on quiet system
- **Variance**: Use sufficient repetitions (100+)
- **Outliers**: Consider median vs mean

### Memory Management

- **Large benchmarks**: Monitor memory usage
- **GPU memory**: Clear between runs
- **Data copies**: Minimize framework conversions

## Future Extensions

1. **More Models**: Gaussian processes, neural networks
2. **More Algorithms**: VI, SMC, NUTS
3. **Scaling Studies**: Multi-GPU, distributed
4. **Automated CI**: Regular benchmark tracking
5. **Web Dashboard**: Interactive results visualization