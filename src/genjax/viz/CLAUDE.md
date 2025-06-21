# CLAUDE.md - GenJAX Visualization Module

This module provides visualization utilities for GenJAX, focusing on statistical visualizations for probabilistic programming applications.

## Overview

The viz module currently provides raincloud plots - a combination of violin plots, box plots, and scatter plots that effectively visualize distributions. The primary implementation is adapted from the PtitPrince library to avoid seaborn compatibility issues.

## Module Contents

### Core Visualization Functions

**Function**: `horizontal_raincloud(data, labels=None, ax=None, figsize=(10, 6), colors=None, width_violin=0.4, width_box=0.15, jitter=0.05, point_size=20, alpha=0.7, orient="h") -> plt.Axes`
**Location**: `raincloud.py:274-393`
- Creates horizontal/vertical raincloud plots combining three visualization layers:
  1. Cloud (half violin plot): Shows kernel density estimation
  2. Umbrella (box plot): Shows quartiles and outliers
  3. Rain (strip plot): Shows individual data points with jitter
- Accepts JAX arrays, NumPy arrays, or lists of arrays
- Automatically converts JAX arrays to NumPy for matplotlib compatibility
- Returns matplotlib Axes object for further customization

**Function**: `raincloud(x=None, y=None, data=None, orient="h", **kwargs)`
**Location**: `raincloud.py:397-417`
- Convenience function with pandas DataFrame support
- Simplified API matching PtitPrince interface
- Extracts data from DataFrame using groupby when x/y/data provided

### Diagnostic Visualization

**Function**: `diagnostic_raincloud(ax, timestep_data, position, color="#606060", width=0.3, n_particles=None, ess_thresholds=None) -> tuple[float, str]`
**Location**: `raincloud.py:420-543`
- Specialized raincloud for particle filter diagnostics
- Creates three-layer visualization optimized for weight distributions:
  1. Bottom: Scatter plot of individual weights
  2. Middle: Histogram-based density lines
  3. Top: Smooth KDE density curve with fill
- Computes and returns Effective Sample Size (ESS) with quality coloring
- ESS quality thresholds (as fractions of n_particles):
  - Good: ≥ 0.5 * n_particles (green)
  - Medium: ≥ 0.25 * n_particles (orange)
  - Bad: < 0.25 * n_particles (red)

### Internal Helper Functions

**Function**: `_estimate_density(data, bw="scott", cut=2.0, gridsize=100)`
**Location**: `raincloud.py:16-43`
- Estimates kernel density using scipy.stats.gaussian_kde
- Handles edge cases: empty data, single unique value
- Returns support grid and density values

**Function**: `_scale_density(density, scale="area")`
**Location**: `raincloud.py:46-59`
- Normalizes density curves to maximum value
- Supports different scaling methods (area, width, count)

**Function**: `_draw_half_violin(ax, position, support, density, width, color, alpha, orient, offset)`
**Location**: `raincloud.py:61-93`
- Draws the cloud portion (half violin plot)
- Supports horizontal and vertical orientations
- Uses fill_between for the density area

**Function**: `_draw_boxplot(ax, position, data, width, color, orient, offset)`
**Location**: `raincloud.py:95-225`
- Draws the umbrella portion (box plot)
- Calculates quartiles and whisker limits (1.5 * IQR)
- Creates box, median line, and whiskers using matplotlib patches

**Function**: `_draw_stripplot(ax, position, data, jitter, size, color, alpha, orient, offset)`
**Location**: `raincloud.py:227-272`
- Draws the rain portion (strip plot)
- Adds random jitter to avoid overplotting
- Positions points below/beside the violin and box

## Usage Examples

### Basic Raincloud Plot

```python
from genjax.viz import horizontal_raincloud
import jax.numpy as jnp

# Single distribution
data = jnp.array(np.random.normal(0, 1, 100))
ax = horizontal_raincloud(data)

# Multiple categories
data = [
    jnp.array(np.random.normal(0, 1, 100)),
    jnp.array(np.random.normal(2, 1.5, 150)),
    jnp.array(np.random.normal(-1, 0.5, 80))
]
labels = ["Control", "Treatment A", "Treatment B"]
ax = horizontal_raincloud(data, labels=labels)
```

### Particle Filter Diagnostics

The `diagnostic_raincloud` function is used in the localization example to visualize particle weights over time:

**Location**: `examples/localization/figs.py`
```python
from genjax.viz.raincloud import diagnostic_raincloud

# Visualize particle weights with ESS coloring
ess, ess_color = diagnostic_raincloud(
    ax_weights,
    linear_weights,  # Normalized particle weights
    position=plot_pos,
    color=grayscale_colors[method_name],
    width=0.3,
    n_particles=N  # Total particle count for ESS assessment
)
```

## Design Philosophy

1. **Framework Compatibility**: Adapted from PtitPrince to avoid seaborn version conflicts
2. **JAX Integration**: Seamlessly handles JAX arrays by converting to NumPy for matplotlib
3. **Statistical Focus**: Designed for probabilistic programming visualizations
4. **Modular Components**: Separate functions for each visual element (violin, box, strip)
5. **Diagnostic Support**: Specialized functions for particle filter and SMC diagnostics

## Dependencies

- matplotlib: Core plotting functionality
- numpy: Array operations and statistics
- scipy: Kernel density estimation
- jax.numpy: JAX array support

## Future Extensions

Consider adding:
- Vertical raincloud plots (current implementation supports both orientations)
- Additional diagnostic visualizations for MCMC chains
- Trace plots and autocorrelation functions
- Posterior predictive check visualizations
- Integration with arviz for standard Bayesian diagnostics
