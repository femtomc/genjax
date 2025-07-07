# Game of Life Case Study

This is an isolated Pixi project for the Game of Life (GoL) case study demonstrating probabilistic inference for Conway's Game of Life using Gibbs sampling with GenJAX.

## Setup

This directory is configured as an isolated Pixi environment. To use it:

```bash
# From this directory
pixi install              # Install default environment
pixi install -e cuda      # Install CUDA environment for GPU support
```

## Available Tasks

### CPU Tasks (default environment)
- `pixi run gol-blinker` - Quick blinker pattern reconstruction
- `pixi run gol-logo` - Logo pattern reconstruction (computationally intensive)
- `pixi run gol-timing` - Performance scaling analysis on CPU
- `pixi run gol-all` - Generate all figures
- `pixi run gol-quick` - Quick test with reduced chain length

### GPU Tasks (cuda environment)
- `pixi run -e cuda cuda-gol-blinker` - Blinker reconstruction on GPU
- `pixi run -e cuda cuda-gol-logo` - Logo reconstruction on GPU
- `pixi run -e cuda cuda-gol-timing` - Performance analysis on GPU
- `pixi run -e cuda cuda-gol-all` - All figures on GPU
- `pixi run -e cuda cuda-info` - Check CUDA/JAX configuration

## Custom Parameters

Use `gol-custom` task with additional arguments:

```bash
pixi run gol-custom --mode blinker --chain-length 1000 --flip-prob 0.05
pixi run gol-custom --mode timing --grid-sizes 10 50 100 --device both
```

## Output

All figures are saved to the `figs/` directory with descriptive filenames that include experimental parameters.

## Dependencies

The project uses:
- JAX/JAXlib for computation
- NumPy for array operations
- Matplotlib for visualization
- GenJAX (from parent directory) for probabilistic programming

CUDA support requires an NVIDIA GPU with CUDA 12 support.