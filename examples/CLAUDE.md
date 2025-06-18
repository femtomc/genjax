# CLAUDE.md - Examples Directory Standards

This file provides guidance to Claude Code when working with case studies in the `examples/` directory.

## Overview

The `examples/` directory contains case studies that demonstrate GenJAX capabilities across different domains. Each case study follows a standardized structure to ensure consistency, maintainability, and ease of development.

## Shared Utilities

The `examples/utils.py` module provides shared utilities for all case studies:

- **`timing()`**: Standard benchmarking function with consistent methodology
- **`benchmark_with_warmup()`**: Automatic JIT warm-up before timing
- **`compare_timings()`**: Formatted comparison of multiple timing results
- **`timing_legacy()`**: Backward compatibility for existing case studies

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

```python
def comparison_fig(num_obs=50, num_samples=1000, **kwargs):
    """Research-quality figure generation."""
    # 300 DPI, large fonts
    # Parametrized filename
    # Professional styling
```

## CLI Standards

Every `main.py` should support:

```bash
# Default behavior (usually timing)
python -m examples.{name}.main

# All figures
python -m examples.{name}.main --all

# Specific figure types
python -m examples.{name}.main --timing
python -m examples.{name}.main --comparison
python -m examples.{name}.main --posterior  # if applicable

# Parameter customization
python -m examples.{name}.main --num-obs 100 --num-samples 5000
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

## Development Workflow

When creating or modifying case studies:

### 1. **Read existing CLAUDE.md**

- Understand case study-specific patterns and constraints
- Follow established model specifications and data patterns

### 2. **Follow standard structure**

- Create all required files (`main.py`, `core.py`, `data.py`, `figs.py`)
- Create `figs/` directory for outputs
- Write comprehensive `CLAUDE.md` with case study guidance

### 3. **Implement consistent patterns**

- Use established naming conventions
- Follow timing benchmark standards
- Generate research-quality figures
- Support standard CLI arguments

### 4. **Test thoroughly**

- Run all figure generation modes
- Verify framework comparison fairness
- Check research paper quality of outputs

### 5. **Document properly**

- Update case study `CLAUDE.md` with implementation details
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
