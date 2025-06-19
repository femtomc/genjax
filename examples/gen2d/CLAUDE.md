# gen2d/CLAUDE.md

This file provides guidance for Claude Code when working with the gen2d case study.

## Overview

The gen2d case study demonstrates tracking objects in Conway's Game of Life using a state-space Gaussian mixture model with Sequential Monte Carlo inference. This combines:

1. **State-space dynamics**: Position and velocity evolution for Gaussian component centers
2. **Gaussian mixture model**: Multiple components with evolving means
3. **Binary grid observations**: Game of Life frames as boolean arrays
4. **Rejuvenation SMC**: Particle filtering with block MCMC rejuvenation

## Directory Structure

```
examples/gen2d/
├── CLAUDE.md           # This file
├── main.py            # CLI entry point
├── core.py            # Model definitions and inference
├── data.py            # Game of Life generation
├── figs.py            # Visualization code
└── figs/              # Generated figure outputs
```

## Model Specification

### State Variables
- **Positions**: (n_components, 2) - 2D positions of Gaussian centers
- **Velocities**: (n_components, 2) - 2D velocities for dynamics
- **Weights**: (n_components,) - Mixture weights (Dirichlet prior)

### Dynamics
- Linear Gaussian dynamics: x[t] = x[t-1] + v[t-1] * dt + noise
- Constant velocity model with process noise

### Observation Model
- Game of Life grid: (N, N) boolean array
- Active pixels (True values) are observations
- Each pixel independently assigned to a mixture component
- Pixel locations sampled from assigned Gaussian

### Inference
- Sequential Monte Carlo with rejuvenation
- Block MCMC kernel updating positions, velocities, weights, assignments
- Automatic resampling based on ESS

## Key Implementation Details

### Block Rejuvenation MCMC
The custom MCMC kernel chains together updates for different variable blocks:
1. Update positions (Metropolis-Hastings)
2. Update velocities (Metropolis-Hastings)
3. Update weights (Metropolis-Hastings with Dirichlet proposal)
4. Update assignments (Gibbs sampling or MH)

### Game of Life Patterns
- Periodic configurations (oscillators, gliders)
- Grid boundary handling (toroidal/periodic)
- Pattern detection and tracking

### Visualization
- Clean Game of Life frame sequences with bottom axis labels
- Gaussian ellipses showing component locations and spread
- Trajectory paths showing motion over time
- Professional seaborn styling with consistent color palettes

## Usage Patterns

### Running from Directory
```bash
# Data visualization (Game of Life patterns only)
python main.py --data

# Basic tracking with default parameters (oscillators pattern)
python main.py --tracking

# Achim's p4 pattern tracking
python main.py --pattern achims_p4 --tracking --n-frames 8

# Generate all inference figures
python main.py --all --n-particles 50 --n-frames 10

# Custom parameters
python main.py --pattern glider --grid-size 64 --n-components 3 --n-frames 15
```

### Available Patterns
- **oscillators**: Multiple period-2 oscillators (blinker, toad, beacon, pulsar)
- **glider**: Moving glider pattern
- **achims_p4**: Achim's p4 oscillator (complex period ~145 pattern)
- **random**: Random initial configuration

### CLI Options
- `--data`: Generate Game of Life data visualizations only (no inference)
- `--tracking`: Generate tracking visualization with SMC results
- `--trajectories`: Plot component trajectories over time
- `--diagnostics`: SMC diagnostics (ESS, log marginal likelihood)
- `--all`: Generate all visualization types

### Parameter Guidelines
- **Grid size**: 32-128 (larger for more complex patterns)
- **Components**: 3-10 (based on expected number of objects)
- **Particles**: 100-1000 (more for better tracking accuracy)
- **Rejuvenation moves**: 1-5 (more for better mixing)

## Development Guidelines

### Model Extensions
- Variable covariance matrices
- Birth/death processes for components
- More sophisticated dynamics (e.g., interacting objects)
- Learned transition proposals

### Performance Considerations
- JAX compilation for Game of Life updates
- Vectorized pixel processing
- Efficient resampling strategies

### Testing
- Verify tracking on known periodic patterns
- Compare with ground truth trajectories
- Assess convergence with different particle counts
