# CLAUDE.md - Examples Directory Standards

This file provides guidance to Claude Code when working with case studies in the `examples/` directory.

## HIGH PRIORITY: Usage Pattern Reference

**When working on new case studies or examples:**

1. **FIRST**: Check existing tests in `tests/` for usage patterns
   - `tests/test_*.py` files show correct API usage
   - Tests demonstrate edge cases and proper error handling
   - Tests provide minimal working examples

2. **SECOND**: Review existing examples in `examples/` for implementation patterns
   - Look for similar models or inference approaches
   - Study how data generation is handled
   - Observe visualization standards

3. **THIRD**: Refer to source code only for detailed understanding
   - After understanding usage from tests/examples
   - When needing to understand internal implementation
   - For performance optimization details

## Overview

The `examples/` directory contains case studies that demonstrate GenJAX capabilities across different domains. Each case study follows a standardized structure to ensure consistency, maintainability, and ease of development.

## Shared Utilities

The `examples/utils.py` module provides shared utilities for all case studies:

- **`timing()`**: Standard benchmarking function with consistent methodology
- **`benchmark_with_warmup()`**: Automatic JIT warm-up before timing
- **`compare_timings()`**: Formatted comparison of multiple timing results

**Always use `examples.utils.timing()` instead of duplicating timing code.**

## Standard Case Study Structure

Every case study MUST follow this exact directory structure:

```
examples/{case_study_name}/
├── CLAUDE.md           # Case study guidance for Claude Code (REQUIRED)
├── README.md           # User documentation (optional, create only if requested)
├── __init__.py         # Python package marker (optional)
├── main.py             # CLI entry point (REQUIRED)
├── core.py             # Model definitions and core logic (REQUIRED)
├── data.py             # Data generation and loading utilities (REQUIRED)
├── figs.py             # Visualization and figure generation (REQUIRED)
├── export.py           # Data export/import utilities (OPTIONAL)
├── data/               # Experimental data storage (OPTIONAL)
│   ├── README.md       # Data format documentation
│   └── experiment_*/   # Timestamped experiment directories
└── figs/               # Generated figure outputs (REQUIRED)
    └── *.pdf           # Research-quality PDF outputs
```

## File Responsibilities

### `CLAUDE.md` (REQUIRED)

- **Purpose**: Provides Claude Code with case study-specific guidance
- **Template**: Follow the pattern established in existing case studies
- **Sections**: Overview, Directory Structure, Code Organization, Key Implementation Details, Usage Patterns, Development Guidelines
- **Critical**: Must document model specifications, data patterns, and performance expectations

### `main.py` (REQUIRED)

- **Purpose**: Command-line interface for the case study
- **Pattern**: Use `argparse` for CLI arguments
- **Default parameters**: Provide sensible defaults for quick testing
- **Multiple modes**: Support different figure types (timing, comparison, all)
- **Example**: Support `--all`, `--timing`, `--comparison` flags

### `core.py` (REQUIRED)

- **Purpose**: Model definitions, inference algorithms, timing utilities
- **GenJAX models**: Use `@gen` decorator with proper type annotations
- **Timing functions**: Include benchmarking utilities with proper warm-up
- **Framework comparisons**: Implement identical algorithms across frameworks
- **Consistent naming**: `{framework}_timing()`, `{framework}_inference()` patterns

### `data.py` (REQUIRED)

- **Purpose**: Standardized data generation across all frameworks
- **Consistency**: Same random seeds and data patterns for fair comparison
- **Reusability**: Functions that can be imported by other case studies
- **Documentation**: Clear docstrings explaining data generation process

### `figs.py` (REQUIRED)

- **Purpose**: All visualization and figure generation code
- **Research quality**: 300 DPI, large fonts (18-22pt), publication-ready
- **Parametrized filenames**: Include experimental parameters in output names
- **Multiple figure types**: Support timing, accuracy, and combined visualizations
- **Consistent styling**: Use established color schemes and formatting

### `export.py` (OPTIONAL)

- **Purpose**: Data export/import utilities for reproducible research
- **CSV format**: Export experimental results to CSV files with metadata
- **Timestamped directories**: Organize experiments with unique timestamps
- **Plot-from-data**: Enable visualization without recomputation
- **Example functions**: `save_benchmark_results()`, `load_benchmark_results()`

### `data/` directory (OPTIONAL)

- **Purpose**: Storage for experimental data to enable experiment/plotting separation
- **Structure**: Timestamped experiment directories with standardized CSV format
- **Benefit**: Run expensive experiments once, iterate on plots separately
- **Documentation**: Include `README.md` explaining data format and usage
- **Reproducibility**: Complete experimental record with configuration metadata

### `figs/` directory (REQUIRED)

- **Purpose**: Output directory for generated figures
- **Format**: Prefer PDF for research publications
- **Naming**: Parametrized filenames with experimental configuration
- **Git**: Directory should exist but figures may be gitignored

## Implementation Standards

### Model Specifications

```python
# Use proper type annotations with Const pattern
@gen
def model_name(param: Const[type]):
    # Clear docstring explaining the model
    """Model description with mathematical specification."""
    # Implementation
```

### Timing Benchmarks

**Use `examples.utils.timing()` or `examples.utils.benchmark_with_warmup()` instead of duplicating timing code.**

```python
from examples.utils import timing, benchmark_with_warmup

def framework_timing(num_obs=50, repeats=200, num_samples=1000):
    """Standard timing function signature using shared utilities."""
    # Setup computation
    jitted_fn = jax.jit(my_function)

    # Use shared timing utility with automatic warm-up
    times, (mean, std) = benchmark_with_warmup(
        lambda: jitted_fn(args),
        repeats=repeats
    )
    return times, (mean, std)
```

### Data Generation

```python
def generate_standard_data(num_obs=50, seed=42):
    """Generate standardized data for framework comparison."""
    # Use fixed seeds for reproducibility
    # Return consistent data structures
```

### Visualization Standards

**IMPORTANT**: All case studies must follow the GenJAX Research Visualization Standards (GRVS). Use the shared `examples.viz` module for consistent styling.

#### Core Aesthetic Principles

1. **No Titles Policy**: Figures designed for LaTeX integration without axis titles
2. **Large Typography**: 18pt base fonts for publication readability  
3. **Bold Axis Labels**: Clear variable identification
4. **3-Tick Standard**: Exactly 3 tick marks per axis for optimal readability (ENFORCED)
5. **Colorblind-Friendly Palette**: Accessible to all readers
6. **Consistent Sizing**: Standardized figure dimensions
7. **Publication Quality**: 300 DPI vector output

#### Typography Standards

```python
# Use examples.viz for consistent font settings
from examples.viz import setup_publication_fonts, FONT_HIERARCHY

setup_publication_fonts()  # Applies GRVS typography

# Font hierarchy (applied automatically):
# - Base text: 18pt
# - Axis labels: 18pt bold  
# - Tick labels: 16pt
# - Legends: 16pt
# - Titles: 20pt (when used)
```

#### Figure Sizing

```python
from examples.viz import FIGURE_SIZES

# Standard sizes for consistent LaTeX integration
fig = plt.figure(figsize=FIGURE_SIZES["single_medium"])  # 6.5×4.875"

# Available sizes:
# - single_small: (4.33, 3.25) - 1/3 textwidth
# - single_medium: (6.5, 4.875) - 1/2 textwidth  
# - single_large: (8.66, 6.5) - 2/3 textwidth
# - two_panel_horizontal: (12, 5) - side-by-side
# - framework_comparison: (12, 8) - comparison plots
```

#### Color Standards

```python
from examples.viz import PRIMARY_COLORS, get_method_color

# Use consistent method colors
genjax_color = get_method_color("genjax_is")     # #0173B2 (blue)
hmc_color = get_method_color("genjax_hmc")       # #DE8F05 (orange)
numpyro_color = get_method_color("numpyro_hmc")  # #029E73 (green)
data_color = get_method_color("data_points")     # #CC3311 (red)
```

#### Visual Elements

```python
from examples.viz import MARKER_SPECS, LINE_SPECS, apply_grid_style, apply_standard_ticks

# Consistent marker sizes
ax.scatter(x, y, **MARKER_SPECS["data_points"])  # s=120, proper edges

# Standard line weights  
ax.plot(x, y, **LINE_SPECS["curve_main"])        # linewidth=3, alpha=0.9

# Clean grid styling
apply_grid_style(ax)  # alpha=0.3, major only

# GRVS 3-tick standard (ENFORCED)
apply_standard_ticks(ax)  # Exactly 3 ticks per axis
```

#### Research-Quality Output

```python
from examples.viz import save_publication_figure

# Standard save configuration (300 DPI, tight layout, PDF)
save_publication_figure(fig, "figure_name.pdf")

# Automatic cleanup and optimization
# - Applies tight_layout()
# - Saves with bbox_inches="tight" 
# - Closes figure to prevent memory leaks
```

#### Complete Example

```python
from examples.viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, apply_standard_ticks, save_publication_figure
)

# Setup
setup_publication_fonts()
fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])

# Plot with GRVS standards
ax.scatter(x_data, y_data, color=get_method_color("data_points"), 
           s=120, zorder=10, edgecolor="white", linewidth=2)
ax.plot(x_curve, y_curve, color=get_method_color("genjax_is"),
        linewidth=3, alpha=0.9, label="GenJAX IS")

# Styling
ax.set_xlabel("X Variable", fontweight='bold')
ax.set_ylabel("Y Variable", fontweight='bold')
apply_grid_style(ax)
apply_standard_ticks(ax)  # GRVS 3-tick standard
ax.legend(fontsize=16)

# Save
save_publication_figure(fig, "comparison_figure.pdf")
```

#### Migration Status

**Completed Migrations:**
- **curvefit**: Fully migrated to shared `examples.viz` module
- **localization**: Fully migrated with additional clean room visualization and shared axes
- **gol**: Fully migrated with GRVS styling for showcase figures and performance plots

**Pending Migrations:**
- **faircoin**: Uses local `plt.rcParams.update()` calls, should migrate to GRVS
- **gen2d**: May have local styling (needs assessment)
- **state_space**: May have local styling (needs assessment)

**Migration Priority**: Major case studies (curvefit, localization, gol) are complete. Remaining case studies should be migrated as they are actively developed.

#### Migration Guide for Future Case Studies

To update new case studies to use the shared `examples.viz` module:

**1. Replace Local Styling**
```python
# OLD: Local matplotlib settings
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, ...})

# NEW: Import and apply GRVS
from examples.viz import setup_publication_fonts
setup_publication_fonts()
```

**2. Replace Local Figure Sizes**
```python
# OLD: Local FIGURE_SIZES dictionary
FIGURE_SIZES = {"single_medium": (6.5, 4.875), ...}

# NEW: Import shared sizes
from examples.viz import FIGURE_SIZES
```

**3. Replace Local Colors**
```python
# OLD: Hardcoded colors
ax.scatter(x, y, color="red", s=80)

# NEW: Standardized colors and markers
from examples.viz import get_method_color, MARKER_SPECS
ax.scatter(x, y, color=get_method_color("data_points"), **MARKER_SPECS["data_points"])
```

**4. Replace Manual Grid Styling**
```python
# OLD: Manual grid configuration
ax.grid(True, alpha=0.3)

# NEW: Standardized grid styling
from examples.viz import apply_grid_style
apply_grid_style(ax)
```

**5. Replace Manual Save Configuration**
```python
# OLD: Manual save parameters
fig.savefig("output.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# NEW: Standardized save with cleanup
from examples.viz import save_publication_figure
save_publication_figure(fig, "output.pdf")
```

#### GRVS Compliance Checklist

When creating or updating case studies, ensure:

- [ ] Import `examples.viz` for all styling needs
- [ ] Call `setup_publication_fonts()` early in module
- [ ] Use `FIGURE_SIZES` for all figure dimensions
- [ ] Use `get_method_color()` for consistent color palette
- [ ] Apply `apply_grid_style()` to all plots
- [ ] Use `MARKER_SPECS` and `LINE_SPECS` for visual elements
- [ ] Remove all figure titles (no `ax.set_title()` calls)
- [ ] Use `fontweight='bold'` for axis labels
- [ ] Save with `save_publication_figure()` for consistency
- [ ] Test with `validate_grvs_compliance()` if needed

## CLI Standards

### Environment Selection

**IMPORTANT**: Case studies may require specific environments for dependencies:

```bash
# CPU environment (default for most examples)
pixi run -e {name} python -m examples.{name}.main

# CUDA environment (for GPU acceleration + visualization dependencies)
pixi run -e cuda python -m examples.{name}.main

# Using predefined tasks (automatically selects correct environment)
pixi run {name}           # CPU version
pixi run cuda-{name}      # CUDA version
```

**Environment Requirements by Case Study**:
- **localization**: Requires `cuda` environment for matplotlib dependencies
- **faircoin, curvefit, gol**: Work in both CPU and CUDA environments
- **Other examples**: Generally use CPU environment unless GPU is needed

### Command Line Interface

Every `main.py` should support:

```bash
# Default behavior (usually timing)
pixi run -e {environment} python -m examples.{name}.main

# All figures
pixi run -e {environment} python -m examples.{name}.main --all

# Specific figure types
pixi run -e {environment} python -m examples.{name}.main --timing
pixi run -e {environment} python -m examples.{name}.main --comparison
pixi run -e {environment} python -m examples.{name}.main --posterior  # if applicable

# Parameter customization
pixi run -e {environment} python -m examples.{name}.main --num-obs 100 --num-samples 5000

# Data export/import (RECOMMENDED for complex experiments)
pixi run -e {environment} python -m examples.{name}.main --experiment --export-data  # Run and save
pixi run -e {environment} python -m examples.{name}.main --plot-from-data path/to/data  # Plot only
```

## Pixi Task Integration

Each case study should integrate with the top-level `pyproject.toml`:

```toml
[tool.pixi.environments]
{name} = ["base", "{name}"]

[tool.pixi.feature.{name}.tasks]
{name}-timing = "python -m examples.{name}.main"
{name}-all = "python -m examples.{name}.main --all"
```

## Experiment/Plotting Separation (RECOMMENDED)

**Best Practice**: Separate expensive experiments from iterative plotting for efficient development.

### Benefits

- **Efficiency**: Run costly experiments once, iterate on visualizations quickly
- **Reproducibility**: Complete experimental record with metadata and configuration
- **Collaboration**: Share experimental data without requiring full computational environment
- **Flexibility**: Generate different visualizations from same experimental data
- **Development**: Rapid iteration on plot aesthetics without recomputation

### Implementation Pattern

1. **Export Data**: Add `--export-data` flag to save experimental results
2. **Import Data**: Add `--plot-from-data` flag to load saved results
3. **CSV Format**: Use structured CSV files with metadata for portability
4. **Timestamped Directories**: Organize experiments with unique timestamps
5. **Documentation**: Include `data/README.md` explaining format and usage

### Example Workflow

```bash
# Step 1: Run expensive experiment once and save data
python -m examples.localization.main --experiment --export-data \
    --n-particles 200 --timing-repeats 50

# Step 2: Iterate on plotting without recomputation
python -m examples.localization.main \
    --plot-from-data data/experiment_20250620_123456

# Step 3: Modify visualization code in figs.py
# Step 4: Re-run plotting quickly
python -m examples.localization.main \
    --plot-from-data data/experiment_20250620_123456
```

### Data Export Structure

```
data/experiment_YYYYMMDD_HHMMSS/
├── experiment_metadata.json     # Configuration and parameters
├── benchmark_summary.csv        # Method comparison overview
├── ground_truth_*.csv          # True trajectory and observations
└── method_name/                # Results for each method
    ├── timing.csv              # Performance statistics
    ├── diagnostic_weights.csv  # Method-specific diagnostics
    └── particles/              # Particle trajectories
        └── timestep_*.csv      # Per-timestep particle states
```

## Development Workflow

When creating or modifying case studies:

### 1. **Read existing CLAUDE.md**

- Understand case study-specific patterns and constraints
- Follow established model specifications and data patterns

### 2. **Follow standard structure**

- Create all required files (`main.py`, `core.py`, `data.py`, `figs.py`)
- Create `figs/` directory for outputs
- Consider adding `export.py` and `data/` for complex experiments
- Write comprehensive `CLAUDE.md` with case study guidance

### 3. **Implement consistent patterns**

- Use established naming conventions
- Follow timing benchmark standards
- Generate research-quality figures
- Support standard CLI arguments including `--export-data` and `--plot-from-data`

### 4. **Test thoroughly**

- Run all figure generation modes
- Test data export/import functionality if implemented
- Verify framework comparison fairness
- Check research paper quality of outputs

### 5. **Document properly**

- Update case study `CLAUDE.md` with implementation details
- Document data export format in `data/README.md` if applicable
- Document any special requirements or dependencies

## Common Patterns

### Framework Comparisons

- All frameworks should implement identical algorithms for fair comparison

### Data Consistency

- Use identical random seeds across frameworks
- Generate same data patterns for meaningful comparisons
- Document any framework-specific data transformations

## Quality Standards

### Research Publication Ready

- **Figures**: 300 DPI PDF output with large fonts (18-22pt)
- **Documentation**: Clear mathematical model specifications
- **Reproducibility**: Fixed seeds and documented parameters
- **Performance**: Statistical rigor with multiple timing runs

### Code Quality

- **Type hints**: Use proper GenJAX patterns (`Const[int]`, etc.)
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Verify outputs across different parameter settings
- **Consistency**: Follow established patterns from successful case studies

## Integration Guidelines

### Adding New Case Studies

1. Create directory structure following the standard format
2. Implement all required files with proper patterns
3. Add pixi task integration to top-level `pyproject.toml`
4. Write comprehensive `CLAUDE.md` following existing examples
5. Test thoroughly across all supported modes

### Modifying Existing Case Studies

1. Read the case study's `CLAUDE.md` for specific guidance
2. Maintain backward compatibility with existing CLI arguments
3. Follow established model specifications and data patterns
4. Update documentation to reflect any changes
5. Ensure research paper quality is maintained

This standardized structure ensures that all GenJAX case studies are consistent, maintainable, and provide high-quality demonstrations of the framework's capabilities.
