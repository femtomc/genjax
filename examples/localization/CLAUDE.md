# CLAUDE.md - Localization Case Study

This file provides guidance to Claude Code when working with the localization (robot particle filtering) case study.

## Overview

The localization case study demonstrates probabilistic robot localization using particle filtering with GenJAX. It showcases sequential inference, state estimation, vectorized LIDAR sensor modeling, and initial + step model patterns for complex multi-room navigation.

## Directory Structure

```
examples/localization/
├── CLAUDE.md           # This file - guidance for Claude Code
├── README.md           # User documentation
├── __init__.py         # Python package marker
├── core.py             # Model definitions and particle filter implementation
├── data.py             # Synthetic data generation and waypoint trajectories
├── figs.py             # Visualization utilities
├── main.py             # Command-line interface
└── figs/               # Generated visualization plots
    └── *.png           # Various localization visualizations
```

## Code Organization

### `core.py` - Models and Inference

**Data Structures**:
- **`Pose`**: Robot pose with position (x, y) and heading (theta)
- **`Control`**: Robot control command (velocity, angular_velocity)
- **`Wall`**: Wall segment defined by two endpoints (x1, y1) to (x2, y2)
- **`World`**: Multi-room world with internal walls, doorways, and complex geometry

**Generative Models**:
- **`step_model()`**: Single-step motion model with control inference and wall bouncing
- **`sensor_model()`**: Vectorized 8-ray LIDAR sensor using Vmap combinator
- **`sensor_model_single_ray()`**: Individual LIDAR ray with Gaussian noise
- **`initial_model()`**: Broad initial pose distribution for multi-room scenarios

**Inference**:
- **`particle_filter_step()`**: Single particle filter update with vectorized LIDAR
- **`run_particle_filter()`**: Complete particle filtering using initial + step model pattern
- **`resample_particles()`**: Systematic resampling
- **`initialize_particles()`**: Broad particle initialization across multi-room world

### `figs.py` - Visualization
- **`plot_world()`**: Draw world boundaries
- **`plot_pose()`**: Visualize robot pose with heading arrow
- **`plot_trajectory()`**: Show trajectory path with poses
- **`plot_particles()`**: Display particle distribution with weights
- **`plot_particle_filter_step()`**: Single filter step visualization
- **`plot_particle_filter_evolution()`**: Time series of particle evolution
- **`plot_estimation_error()`**: Position and heading error over time
- **`plot_sensor_observations()`**: 8-ray LIDAR sensor readings vs true distances
- **`plot_ground_truth_trajectory()`**: Comprehensive trajectory analysis
- **`plot_multiple_trajectories()`**: Comparison of different trajectory types

### `main.py` - Demo Script
- **Complete workflow**: Data generation → filtering → visualization
- **Error handling**: Graceful degradation if particle filter fails
- **Multiple plots**: Ground truth, particle evolution, error analysis

## Key Implementation Details

### Model Specification

**Key Patterns**:
- **Control Inference**: Step model samples velocity and angular_velocity as random variables instead of requiring fixed controls
- **Wall Bouncing**: Physics-based collision detection using JAX conditionals (jnp.where) for boundary reflection
- **Multi-Room Navigation**: Increased position noise (σ=0.5) for cross-room exploration
- **LIDAR Sensor**: Single-ray model with Gaussian noise, vectorized using GenJAX Vmap combinator
- **Joint Observations**: Vmap combines 8 individual ray models into vectorized LIDAR sensor

### Initial + Step Model Pattern

**Key Patterns**:
- **Separate Models**: Initial model for broad particle initialization, step model for sequential updates
- **Control-Free Interface**: Particle filter doesn't require external control commands
- **ESS-Based Resampling**: Resample when effective sample size < n_particles/8 for diversity preservation
- **Sequential Updates**: Standard predict-update cycle with systematic resampling
- **Weight Normalization**: Robust handling of zero weights with uniform fallback

### PJAX and Seed Usage

**CRITICAL**: Use `seed` transformation for PJAX primitives in particle filtering:

**Key Patterns**:
- **Seed Transformation**: Apply `seed(gen_fn.simulate)` before using generative functions in particle prediction
- **Key Management**: Split random keys appropriately for vectorized operations
- **Waypoint Generation**: Use strategic waypoint placement instead of control-based trajectory simulation
- **Weight Computation**: Structure choices to match Vmap combinator expectations: `{"measurements": {"distance": observation}}`
- **Numerical Stability**: Use JAX conditionals and minimum thresholds for weight computation

### Vectorization with JAX and GenJAX

**Key Patterns**:
- **JAX vmap**: Used for vectorizing ray-wall intersection computations across 8 LIDAR directions
- **GenJAX Vmap**: Combines individual sensor ray models into joint LIDAR sensor
- **Fixed Vmap.assess**: Modified core GenJAX to sum individual densities for proper joint likelihood
- **Vectorized Geometry**: Boundary and wall intersection calculations use element-wise JAX operations

### Multi-Room World Geometry

**Key Patterns**:
- **JAX Array Storage**: Walls stored as coordinate arrays for vectorized intersection computations
- **3-Room Layout**: Rooms connected by strategic doorway placement at specific y-coordinates
- **Complex Obstacles**: Internal walls create alcoves and rectangular obstacles for navigation challenges
- **Validation**: World enforces minimum 1 wall constraint with shape validation

## Common Patterns

### Particle Representation
- **List of Pose objects**: Easy to work with, good for debugging
- **Vectorized operations**: Use JAX arrays where possible for performance
- **Weight normalization**: Always normalize weights before resampling

### Visualization Workflow
1. **Ground truth**: Show true trajectory with sensor observations
2. **Particle evolution**: Display particles over first few time steps
3. **Final estimation**: Compare final particle distribution to true pose
4. **Error analysis**: Plot position and heading errors over time
5. **Sensor validation**: Compare observed vs true distances

### Error Handling
- **Graceful degradation**: Continue with available data if particle filter fails
- **Debug output**: Print shapes and types for JAX array issues
- **Fallback visualizations**: Always generate some plots even if inference fails

## Development Commands

```bash
# Run localization demo
pixi run -e cuda python -m examples.localization.main

# Development with different environments
pixi run -e cuda python -m examples.localization.main  # For GPU acceleration (if available)
```

## Common Issues

### PJAX Primitive Lowering
- **Cause**: Using PJAX primitives inside JAX transformations without `seed`
- **Solution**: Apply `seed` transformation to generative functions before simulation
- **Pattern**: `seeded_fn = seed(gen_fn.simulate); result = seeded_fn(key, args)`

### Vectorized Outputs from Scan
- **Issue**: Scan returns vectorized Pose with array fields, not list of poses
- **Solution**: Convert to list: `[Pose(poses.x[i], poses.y[i], poses.theta[i]) for i in range(len(poses.x))]`
- **Pattern**: Check for vectorized structure and convert appropriately

### Particle Filter Performance
- **Current**: Sequential processing for compatibility
- **Future**: Vectorize particle operations for better performance
- **Trade-off**: Correctness vs speed in current implementation

### Import Paths
- **Use relative imports**: `from .core import ...` in package files
- **Run as module**: `python -m examples.localization.main` from project root
- **Environment**: Use `-e cuda` for visualization dependencies

## Integration with Main GenJAX

This case study demonstrates:
1. **Sequential inference**: Particle filtering for time series
2. **Scan combinator**: Proper usage for trajectory modeling
3. **PJAX transformations**: Correct `seed` usage patterns
4. **State estimation**: Probabilistic robotics applications
5. **Visualization**: Comprehensive plotting for analysis

The case study showcases GenJAX capabilities for robotics and time series applications beyond static inference problems.

## Performance Characteristics

### Current Status
- **Fully Vectorized**: LIDAR distance computation and sensor model use JAX vmap and GenJAX Vmap
- **Multi-Room Navigation**: Successfully navigates complex 3-room environment with doorways and obstacles
- **Realistic Sensor Model**: 8-ray LIDAR with appropriate noise levels (σ=0.8)
- **Robust Particle Filtering**: Achieves ~4.5 unit final position error over 16-step cross-room trajectory
- **Effective Resampling**: ESS-based resampling with diversity preservation (threshold: n_particles/8)

### Key Achievements
- **Control Inference**: Step model infers velocity and angular velocity as random variables
- **Wall Bouncing**: Physics-based collision detection and response in multi-room environment
- **Waypoint Navigation**: Strategic waypoint-based trajectory generation for reliable cross-room movement
- **Joint Density**: Fixed Vmap.assess to return proper joint log density for multi-ray observations
- **Enhanced Visualizations**: 8-subplot LIDAR sensor visualization showing individual ray measurements

### Performance Metrics
- **Localization Accuracy**: ~4.5 unit final error in 12×10 unit multi-room world
- **Particle Efficiency**: 200 particles with effective resampling based on ESS < 25
- **Weight Diversity**: Meaningful particle weights (range 1e-10 to 0.11) instead of uniform fallback
- **Cross-Room Success**: Reliable navigation from Room 1 lower-left to Room 3 upper-right

### Technical Improvements Over Basic Implementation
1. **Vectorized LIDAR**: JAX vmap for 8-ray distance computation (8× more sensor information)
2. **GenJAX Vmap Fix**: Corrected assess method to sum individual densities for joint likelihood
3. **Realistic Noise Models**: Sensor noise σ=0.8 instead of σ=0.15 for particle diversity
4. **Improved Resampling**: Reduced frequency (ESS < n/8) to maintain particle diversity
5. **Multi-Room Geometry**: 9 internal walls creating complex 3-room environment with doorways

The localization case study demonstrates advanced GenJAX capabilities including vectorized sensor modeling, complex geometry handling, and robust sequential inference in challenging multi-room navigation scenarios.
