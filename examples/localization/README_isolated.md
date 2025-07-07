# Localization Case Study

This is an isolated Pixi project for the particle filter localization case study demonstrating probabilistic robot localization using Sequential Monte Carlo (SMC) with GenJAX.

## Setup

This directory is configured as an isolated Pixi environment. To use it:

```bash
# From this directory
pixi install              # Install default environment
pixi install -e cuda      # Install CUDA environment (recommended for this case study)
```

**Note**: The localization case study is designed to run with the CUDA environment for proper dependencies and performance.

## Available Tasks

### Standard Tasks (use cuda environment)
- `pixi run -e cuda localization-test` - Quick test with minimal parameters
- `pixi run -e cuda localization-quick` - Medium test run
- `pixi run -e cuda localization-generate-data` - Generate full experimental data
- `pixi run -e cuda localization-plot` - Generate plots from saved data
- `pixi run -e cuda localization-all` - Complete pipeline (generate + plot)

### Shortcuts (automatically use cuda environment)
- `pixi run -e cuda cuda-localization-test` - Quick test
- `pixi run -e cuda cuda-localization-experiment` - Standard experiment
- `pixi run -e cuda cuda-localization-all` - Full pipeline

### Custom Parameters

Use `localization-custom` task with additional arguments:

```bash
pixi run -e cuda localization-custom generate-data --n-particles 200 --k-rejuv 30
pixi run -e cuda localization-custom plot-figures --experiment-name localization_r8_p200_basic_20250706_123456
```

## Two-Step Workflow

The case study follows a two-step workflow for efficiency:

### 1. Generate Data
```bash
pixi run -e cuda localization-generate-data
```

This creates experimental data in the `data/` directory with timestamped experiment names.

### 2. Plot Figures
```bash
pixi run -e cuda localization-plot
```

This generates all visualizations from the most recent experiment data.

## Output

- **Data**: Saved to `data/localization_r{rays}_p{particles}_{world}_YYYYMMDD_HHMMSS/`
- **Figures**: Saved to `figs/` with descriptive filenames

## Key Features

- **Drift-only dynamics**: Simplified state space (x, y, θ) for better convergence
- **4 SMC methods**: Basic, MH rejuvenation, HMC rejuvenation, Locally optimal
- **8-ray LIDAR**: Vectorized sensor model
- **Multi-room world**: 3-room layout with walls and doorways
- **Complete data export**: CSV format with metadata preservation

## Dependencies

The project uses:
- JAX/JAXlib with CUDA support
- NumPy for array operations
- Matplotlib for visualization
- GenJAX (from parent directory) for probabilistic programming

GPU support is recommended for performance but not required.