# CLAUDE.md - Game of Life Case Study

This file provides guidance to Claude Code when working with the Game of Life (GoL) inference case study.

## Overview

The Game of Life case study demonstrates probabilistic inference for Conway's Game of Life using Gibbs sampling with GenJAX. It showcases inverse dynamics - given an observed next state, infer the most likely previous state that would generate it, while accounting for possible rule violations through a "softness" parameter.

## Directory Structure

```
examples/gol/
├── CLAUDE.md           # This file - guidance for Claude Code
├── core.py             # Game of Life model, Gibbs sampler, and animation utilities
├── data.py             # Test patterns and image loading utilities
├── figs.py             # Timing benchmarks and figure generation
├── main.py             # Command-line interface and timing studies
├── assets/             # Image assets for patterns
│   ├── mit.png         # MIT logo pattern
│   └── popl.png        # POPL logo pattern
└── figs/               # Generated visualizations
    ├── gibbs_on_blinker.pdf     # Blinker pattern reconstruction
    ├── gibbs_on_logo_*.pdf      # Logo pattern reconstructions
    └── timing_scaling.pdf       # Performance scaling analysis
```

## Code Organization

### `core.py` - Game of Life Model and Inference

**Game of Life Implementation**:

- **`get_cell_from_window()`**: Single cell state transition with probabilistic rule violations
- **`get_windows()`**: Extract 3×3 neighborhoods for all cells using JAX dynamic slicing
- **`generate_next_state()`**: Apply GoL rules to entire grid using vectorized operations
- **`generate_state_pair()`**: Generate random initial state and its GoL successor

**Gibbs Sampling Infrastructure**:

- **`gibbs_move_on_cell_fast()`**: Single-cell Gibbs update in O(1) time
- **`get_gibbs_probs_fast()`**: Compute conditional probabilities for cell state
- **`gibbs_move_on_all_cells_at_offset()`**: Update cells at specific grid offset
- **`full_gibbs_sweep()`**: Complete Gibbs sweep over all cells with proper ordering

**Sampler State Management**:

- **`GibbsSamplerState`**: Container for inference trace and derived quantities
- **`GibbsSampler`**: Main sampler class with initialization and update methods
- **`run_sampler_and_get_summary()`**: Execute sampling with progress tracking

**Visualization and Animation**:

- **`get_gol_figure_and_updater()`**: Create matplotlib figure with multiple subplots
- **`get_gol_sampler_anim()`**: Generate animated visualization of sampling process
- **`get_gol_sampler_lastframe_figure()`**: Static final result visualization

### `data.py` - Pattern Generation and Assets

**Built-in Patterns**:

- **`get_blinker_4x4()`**: Small blinker pattern for quick testing
- **`get_blinker_10x10()`**: Larger blinker in 10×10 grid
- **`get_blinker_n()`**: Parameterized blinker generator for any grid size

**Image Pattern Loading**:

- **`get_popl_logo()`**: Load and process POPL conference logo from PNG
- **`get_mit_logo()`**: Load and process MIT logo from PNG
- **Image preprocessing**: Convert RGBA to binary patterns with proper thresholding

### `figs.py` - Performance Analysis

**Timing Infrastructure**:

- **`timing()`**: Robust timing utility with multiple repeats and statistical analysis
- **`task()`**: Single Gibbs sampling task for performance measurement

**Benchmark Generation**:

- **`save_blinker_gibbs_figure()`**: Generate and save blinker reconstruction figure
- **`save_logo_gibbs_figure()`**: Generate logo reconstruction with error metrics
- **Error reporting**: Track prediction accuracy and bit reconstruction errors

### `main.py` - Timing Studies and CLI

**Performance Scaling**:

- **`timing()`**: Core timing utility with JIT compilation handling
- **`task()`**: Parameterized Gibbs sampling for different grid sizes
- **`timing_figure()`**: Generate scaling analysis plots

**Command Line Interface**:

- **Grid size scaling**: Test performance across different board dimensions
- **Statistical analysis**: Multiple repeats for reliable timing measurements
- **Publication ready**: Large font sizes and professional figure formatting

## Key Implementation Details

### Game of Life Model with Softness

**Probabilistic Rule Violations**:

```python
@gen
def get_cell_from_window(window, flip_prob):
    # Standard Conway's Game of Life rules
    deterministic_next_bit = jnp.where(
        (grid == 1) & ((neighbors == 2) | (neighbors == 3)), 1,
        jnp.where((grid == 0) & (neighbors == 3), 1, 0)
    )
    # Allow probabilistic rule violations
    p_is_one = jnp.where(deterministic_next_bit == 1, 1 - flip_prob, flip_prob)
    bit = flip(p_is_one) @ "bit"
    return bit
```

**Key Patterns**:

- **Softness Parameter**: `flip_prob` controls probability of violating GoL rules
- **Deterministic Core**: Standard Conway's rules as baseline behavior
- **Probabilistic Deviations**: Allow model flexibility for noisy or partial observations

### Efficient Gibbs Sampling

**O(1) Single Cell Updates**:

- **Local Computation**: Only consider 3×3 neighborhood around target cell
- **Vectorized Probabilities**: Compute P(cell=0) and P(cell=1) in parallel
- **Conditional Independence**: Exploit GoL locality for efficient updates

**Full Grid Sweep Strategy**:

- **9-Coloring Pattern**: Update cells in 3×3 offset pattern to avoid conflicts
- **Parallel Updates**: Cells at same offset can be updated simultaneously
- **Proper Ordering**: Ensures all cells updated exactly once per sweep

### PJAX and Vectorization

**Critical Patterns**:

- **Seed Transformation**: Apply `seed()` to eliminate PJAX primitives for JIT compilation
- **Modular Vmap**: Use `trace()` function with vectorized operations
- **Static Arguments**: Grid dimensions must be compile-time constants

### Animation and Visualization

**Multi-Panel Display**:

- **Predictive Posterior Score**: Track likelihood of inferred state over time
- **Softness Parameter**: Monitor flip probability during sampling
- **Target State**: Show ground truth pattern being reconstructed
- **Inferred Previous State**: Display current best estimate
- **One-Step Rollout**: Show what inferred state would generate

**Research Quality Output**:

- **High DPI PDFs**: Publication-ready figures at 300 DPI
- **Professional Layout**: Clean grid arrangement with proper labels
- **Animation Support**: Full matplotlib animation with customizable parameters

## Usage Patterns

### Basic Pattern Reconstruction

```python
# Load target pattern
target = get_blinker_4x4()

# Create sampler with softness parameter
sampler = GibbsSampler(target, p_flip=0.03)

# Run inference
key = jrand.key(42)
run_summary = run_sampler_and_get_summary(key, sampler, n_steps=250, n_steps_per_summary_frame=1)

# Generate visualization
fig = get_gol_sampler_lastframe_figure(target, run_summary, 1)
fig.savefig("reconstruction.pdf")
```

### Performance Benchmarking

```python
# Test different grid sizes
ns = [10, 100, 200, 300, 400]
times = []

for n in ns:
    _, (mean_time, _) = timing(lambda: task(n), repeats=5, inner_repeats=3)
    times.append(mean_time)

# Generate scaling plot
timing_figure(ns, times)
```

### Custom Pattern Loading

```python
# Load custom image pattern
pattern = get_mit_logo()  # or get_popl_logo()

# Run reconstruction
sampler = GibbsSampler(pattern, p_flip=0.03)
result = run_sampler_and_get_summary(key, sampler, 500, 1)

# Check reconstruction quality
n_errors = result.n_incorrect_bits_in_reconstructed_image(pattern)
print(f"Final reconstruction errors: {n_errors}")
```

## Development Commands

```bash
# Run basic GoL demo (requires CUDA environment for assets)
pixi run -e cuda python -m examples.gol.main

# Generate timing figures
pixi run -e cuda python -m examples.gol.figs

# Custom grid size testing
pixi run -e cuda python -c "
from examples.gol.main import timing_figure
timing_figure([50, 100, 150])
"
```

## Performance Characteristics

### Scaling Properties

- **Grid Size**: O(n²) complexity for n×n grids
- **Gibbs Steps**: Linear scaling with number of sampling steps
- **Memory Usage**: Efficient JAX array operations
- **JIT Compilation**: First run slower due to compilation overhead

### Typical Results

- **Small Patterns (10×10)**: ~0.1 seconds for 250 Gibbs steps
- **Medium Patterns (100×100)**: ~2-5 seconds for 250 steps
- **Large Patterns (300×300)**: ~30-60 seconds for 250 steps
- **Reconstruction Accuracy**: Typically 95%+ bit accuracy for well-posed problems

## Common Issues

### Asset Loading

- **Missing Images**: Ensure `examples/gol/assets/*.png` files exist
- **Path Issues**: Run from project root or use proper relative paths
- **Image Format**: Assets should be RGBA PNG format

### PJAX Primitives

- **Compilation Errors**: Apply `seed()` transformation before JIT compilation
- **Key Management**: Split random keys appropriately for vectorized operations
- **Static Arguments**: Grid dimensions must be known at compile time

### Performance Optimization

- **JIT Warmup**: First run includes compilation time - use multiple repeats for timing
- **Memory Management**: Large grids may require GPU memory management
- **Vectorization**: Prefer JAX operations over Python loops

## Integration with Main GenJAX

This case study demonstrates:

1. **Gibbs Sampling**: Proper MCMC implementation using GenJAX primitives
2. **Inverse Problems**: Inferring causes from observed effects
3. **Vectorized Operations**: Efficient computation using JAX and GenJAX combinators
4. **Animation**: Dynamic visualization of sampling progress
5. **Performance Analysis**: Systematic benchmarking and scaling studies

The Game of Life case study showcases GenJAX capabilities for discrete probabilistic models, MCMC inference, and complex visualization beyond continuous parameter estimation problems.

## Research Applications

### Cellular Automata Inference

- **Rule Discovery**: Infer GoL-like rules from state transitions
- **Noise Modeling**: Handle partial or corrupted observations
- **Pattern Completion**: Reconstruct missing parts of cellular automata patterns

### Methodological Contributions

- **Efficient Gibbs**: O(1) single-cell updates for large grid inference
- **Probabilistic Rules**: Soft constraints allow model flexibility
- **Visualization**: Comprehensive animation and analysis tools

The GoL case study represents a sophisticated application of probabilistic programming to discrete dynamical systems with practical relevance to pattern recognition and rule inference problems.
