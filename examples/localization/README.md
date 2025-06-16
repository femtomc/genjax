# GenJAX Localization Case Study

This case study demonstrates probabilistic robot localization using particle filtering implemented in GenJAX. It showcases sequential inference, state estimation, vectorized LIDAR sensor modeling, and advanced multi-room navigation scenarios.

## Overview

The localization problem involves estimating a robot's position and orientation (pose) as it moves through an environment, given noisy sensor observations. This implementation uses a particle filter to maintain a probability distribution over possible robot poses.

## Key Features

- **Vectorized LIDAR Sensor**: 8-directional distance measurements using GenJAX Vmap combinator
- **Multi-Room Navigation**: Complex 3-room environment with doorways and obstacles
- **Control Inference**: Robot velocity and angular velocity as random variables
- **Wall Bouncing Physics**: Realistic collision detection and response
- **Initial + Step Models**: Modern particle filtering approach instead of Scan
- **Enhanced Visualizations**: Individual LIDAR ray plots and cross-room trajectory tracking

## Model Description

### Robot Motion
The robot navigates a complex multi-room environment with:
- **Pose**: Position (x, y) and heading angle (θ)
- **Inferred Controls**: Velocity and angular velocity as random variables
- **Wall Bouncing**: Physics-based collision detection with internal walls
- **Multi-Room Geometry**: 3 rooms with doorways, alcoves, and obstacles

### LIDAR Sensor Model
- **8-ray LIDAR**: Directional distance measurements at regular angular intervals
- **Vectorized Implementation**: GenJAX Vmap combinator for efficient computation
- **Realistic Noise**: Gaussian noise (σ=0.5) appropriate for meter-scale measurements
- **Ray-wall Intersection**: Vectorized geometric computation for all walls

### Trajectory Generation
Uses waypoint-based approach for reliable cross-room navigation:
- **Strategic Waypoints**: Carefully placed coordinates for Room 1→2→3 trajectory
- **Synthetic Data**: Ground truth LIDAR observations with realistic noise
- **Cross-Room Success**: Reliable navigation through doorways and around obstacles

## Usage

### Basic Demo
```bash
# Run the complete localization demo
pixi run -e cuda python -m examples.localization.main
```

### Generated Outputs

The demo generates several visualization files:

1. **`ground_truth_detailed.png`**: Comprehensive 4-panel analysis with trajectory, controls, position/heading, and sensor data
2. **`ground_truth.png`**: Simple trajectory overview with observations
3. **`particle_evolution.png`**: 16-step particle filter evolution in grid layout
4. **`final_step.png`**: Final particle distribution vs true pose
5. **`estimation_error.png`**: Position and heading errors over time
6. **`sensor_observations.png`**: 8-ray LIDAR measurements showing individual directional sensors
7. **`multiple_trajectories.png`**: Comparison of room navigation, exploration, and wall bouncing trajectories

### Example Output
```
GenJAX Localization Case Study
========================================
Creating multi-room world...
World dimensions: 12.0 x 10.0
Internal walls: 9
Generating ground truth trajectory...
Generated trajectory with 16 steps
Initial pose: x=0.50, y=0.50, theta=0.00
Initializing particles...
Initialized 200 particles
Running particle filter...
  Step 1/16
    Weight range: [1.00e-10, 4.79e-06]
    Effective sample size: 1.1
    Resampling particles (ESS: 1.1)
  Step 2/16
    Weight range: [1.00e-10, 6.35e-03]
    Effective sample size: 1.8
    Resampling particles (ESS: 1.8)
  ...
  Step 16/16
Particle filter completed successfully!

Results Summary:
Final true pose: x=11.50, y=9.50, theta=0.00
Final estimated pose: x=9.81, y=5.35, theta=-0.03
Final position error: 4.481

Generating visualizations...
Saved: figs/ground_truth_detailed.png
Saved: figs/ground_truth.png
Saved: figs/particle_evolution.png
Saved: figs/final_step.png
Saved: figs/estimation_error.png
Saved: figs/sensor_observations.png
Saved: figs/multiple_trajectories.png

Localization demo completed!
```

## Implementation Details

### Core Components

- **`core.py`**: Data structures, generative models, and particle filter implementation
- **`data.py`**: Synthetic data generation and waypoint-based trajectory creation
- **`figs.py`**: Comprehensive visualization functions including LIDAR ray plots
- **`main.py`**: Demo script with multi-room navigation scenarios

### Key Patterns

**Vectorized LIDAR Sensor**:
- **Single Ray Model**: Individual LIDAR ray with Gaussian noise (σ=0.5)
- **GenJAX Vmap**: Combines 8 individual ray models into joint LIDAR sensor
- **Vector Observations**: Returns array of 8 distance measurements per timestep
- **Joint Density**: Vmap.assess sums individual ray log densities for proper likelihood

**Initial + Step Model Pattern**:
- **Separate Models**: Initial model for broad particle initialization, step model for sequential updates
- **Control Inference**: Step model samples velocity and angular_velocity as random variables
- **ESS Resampling**: Resample when effective sample size < n_particles/8 for diversity
- **Sequential Updates**: Standard predict-update cycle with systematic resampling

**Vectorized Distance Computation**:
- **JAX Vectorization**: Ray directions and boundary intersections computed element-wise
- **vmap for Walls**: Individual ray-wall intersections vectorized using jax.vmap
- **Geometric Computation**: Ray-line segment intersection with parametric equations
- **Distance Clipping**: Results clamped to maximum sensor range (10.0 units)

## Parameters

### World Configuration
- **Dimensions**: 12.0 x 10.0 units (3-room layout)
- **Rooms**: Room 1 (0,0)-(4,10), Room 2 (4,0)-(8,10), Room 3 (8,0)-(12,10)
- **Doorways**: Room 1↔2 at (4, y∈[3,5]), Room 2↔3 at (8, y∈[4,6])
- **Internal Walls**: 9 wall segments creating complex geometry
- **Obstacles**: Rectangular obstacle in Room 3, alcove in Room 2

### Motion Model
- **Control Inference**: Velocity ~ N(1.5, 0.5), Angular velocity ~ N(0, 0.3)
- **Position noise**: σ = 0.5 units (increased for cross-room exploration)
- **Heading noise**: σ = 0.2 radians
- **Wall Bouncing**: Physics-based collision detection and response
- **Boundary Constraints**: Clipping with bounce margin = 0.3 units

### LIDAR Sensor Model
- **Ray Count**: 8 directional measurements (0° to 315° in 45° increments)
- **Distance noise**: σ = 0.8 units per ray (increased for better particle diversity)
- **Max Range**: 10.0 units
- **Vectorization**: JAX vmap for boundary/wall intersection computation
- **Joint Density**: Vmap.assess sums individual ray log densities

### Particle Filter
- **Particles**: 200 particles (increased for multi-room complexity)
- **Initialization**: Uniform distribution across entire 3-room world
- **Resampling**: When effective sample size < n_particles/8 (25 particles)
- **Resampling method**: Systematic resampling with diversity preservation
- **Weight Computation**: Joint likelihood across 8 LIDAR rays

### Cross-Room Trajectory
- **Pattern**: Room 1 → Room 2 → Room 3 (16 waypoints)
- **Navigation**: Strategic waypoint placement for doorway traversal
- **Start**: Lower-left Room 1 (0.5, 0.5)
- **End**: Upper-right Room 3 (11.5, 9.5)
- **Challenges**: Doorway navigation, obstacle avoidance, room transitions

## Extensions

This case study can be extended in several ways:

1. **Enhanced Sensor Models**: IMU, GPS, camera features, or multi-modal sensor fusion
2. **Advanced Motion Models**: Non-holonomic constraints, wheel slip, or dynamic obstacles
3. **Larger Environments**: Multi-floor buildings, outdoor scenarios, or semantic mapping
4. **Algorithm Comparisons**: Extended Kalman Filter, FastSLAM, or modern neural approaches
5. **Real Robot Integration**: ROS interface, actual LIDAR data, or hardware deployment
6. **Performance Optimization**: Full particle vectorization, GPU acceleration, or distributed filtering

## Dependencies

- **GenJAX**: Core probabilistic programming
- **JAX**: Array operations and transformations
- **Matplotlib**: Visualization and plotting
- **NumPy**: Array utilities

## References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*
- Gordon, N.J., Salmond, D.J., & Smith, A.F.M. (1993). "Novel approach to nonlinear/non-Gaussian Bayesian state estimation"
- Doucet, A., Godsill, S., & Andrieu, C. (2000). "On sequential Monte Carlo sampling methods for Bayesian filtering"
