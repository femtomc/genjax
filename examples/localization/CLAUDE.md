# CLAUDE.md - Localization Case Study

Probabilistic robot localization using particle filtering with GenJAX. Demonstrates SMC with MCMC rejuvenation, vectorized LIDAR sensing, and drift-only dynamics for improved convergence.

## Environment Setup

**IMPORTANT**: This case study requires the CUDA environment for proper execution:

```bash
# Always use the cuda environment for localization
pixi run -e cuda python -m examples.localization.main [command]

# Or use the predefined tasks (which automatically use cuda environment):
pixi run cuda-localization-generate-data
pixi run cuda-localization-plot-figures
pixi run cuda-localization  # Full pipeline
```

The `cuda` environment includes:
- JAX with GPU support (if available)
- Matplotlib for visualization
- All required dependencies for the case study

## Directory Structure

```
examples/localization/
├── core.py             # SMC methods, drift-only model, world geometry
├── data.py             # Trajectory generation
├── figs.py             # Visualization (4-row SMC comparison, error plots)
├── main.py             # CLI with data export/import
├── export.py           # CSV data export/import system
├── data/               # Experimental data (CSV + JSON metadata)
└── figs/               # Generated PDF plots
```

## Core Implementation

### Drift-Only Model Design
The localization model uses **drift-only dynamics** without velocity variables for improved SMC convergence:
- **State space**: Only (x, y, θ) - no velocity or angular velocity
- **Dynamics**: Simple positional drift `x_t ~ Normal(x_{t-1}, σ)`
- **Benefits**: Better particle diversity, stable convergence, faster computation

### SMC Methods (`core.py`)
- **`run_smc_basic()`**: Bootstrap filter (no rejuvenation)
- **`run_smc_with_mh()`**: SMC + Metropolis-Hastings rejuvenation
- **`run_smc_with_hmc()`**: SMC + Hamiltonian Monte Carlo rejuvenation
- **`run_smc_with_locally_optimal()`**: SMC + Locally optimal proposal using grid evaluation
- **K parameter**: Uses `Const[int]` pattern for JAX compilation: `K: Const[int] = const(10)`

### Models
- **`localization_model()`**: Drift-only dynamics with no velocity variables
  - Initial distribution: Centered at (1.5, 1.5) near true start with σ=0.5
  - Drift noise: σ=0.15 for position, σ=0.05 for heading
  - Sensor noise: σ=0.3 (reduced from 1.5 for better tracking)
- **`sensor_model_single_ray()`**: LIDAR ray model with Gaussian noise
- **`initial_model()`**: Initial pose distribution near true starting position

### Locally Optimal Proposal (`core.py`)
- **`create_locally_optimal_proposal()`**: Creates transition proposal using grid evaluation
- **Grid Evaluation**: 15×15×15 grid over (x, y, θ) space (no velocity dimensions)
- **Vectorized Assessment**: Uses `jax.vmap` to evaluate `localization_model.assess()` at all grid points
- **Optimal Selection**: Finds `argmax` of log probabilities across grid
- **Noise Injection**: Adds Gaussian noise around selected point (σ=0.1 for position, σ=0.05 for angle)
- **JAX Compatible**: Fully vectorized implementation using JAX primitives

### World Geometry
- **3-room layout**: 12×10 world with 9 internal walls and doorways
- **JAX arrays**: Wall coordinates stored as `walls_x1`, `walls_y1`, `walls_x2`, `walls_y2`
- **Vectorized intersections**: Ray-wall calculations use JAX vmap

## Data Export System (`export.py`)

### Structure
```
data/localization_r{rays}_p{particles}_{world_type}_{timestamp}/
├── experiment_metadata.json          # All config parameters
├── benchmark_summary.csv            # Method comparison
├── ground_truth_poses.csv           # timestep,x,y,theta
├── ground_truth_observations.csv    # timestep,ray_0,...,ray_7
├── smc_basic/timing.csv              # mean_time_sec,std_time_sec
├── smc_basic/diagnostic_weights.csv # ESS computation data
├── smc_basic/particles/timestep_*.csv # particle_id,x,y,theta,weight
├── smc_mh/...                       # Same structure
├── smc_hmc/...                      # Same structure
└── smc_locally_optimal/...          # Same structure
```

### API
- **Export**: `save_benchmark_results(data_dir, results, config)`
- **Import**: `load_benchmark_results(data_dir)` → identical plot generation
- **Ground truth**: `save_ground_truth_data()`, `load_ground_truth_data()`

## Visualization (`figs.py`)

### GenJAX Research Visualization Standards (GRVS)

All visualization functions use the shared `examples.viz` module for consistent styling across case studies:

**Core Standards:**
- **Typography**: 18pt base fonts, bold axis labels, 16pt legends
- **Colors**: Colorblind-friendly palette with consistent SMC method colors
- **Clean Room Visualization**: LIDAR and trajectory plots remove axes/grid for cleaner spatial view
- **Shared Axes**: Distance readings plots use shared y-axis with single "Distance" label
- **Publication Quality**: 300 DPI PDF output with GRVS compliance

**Usage Pattern:**
```python
from examples.viz import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, save_publication_figure
)
```

### SMC Method Comparison Plot
**4-row layout** (`plot_smc_method_comparison()`):
1. **Initial particles** with "Start" label (left side)
2. **Final particles** with "End" label (left side)
3. **Raincloud plots** - ESS diagnostics with color coding (good/medium/bad)
4. **Timing comparison** - horizontal bars with error bars

**Visualization features**:
- **Color coding**: Bootstrap filter (blue), SMC+MH (orange), SMC+HMC (green), SMC+Locally Optimal (red)
- **ESS thresholds**: Good ≥50% particles, Medium ≥25%, Bad <25%
- **Ground truth**: Marked with 'x' symbols
- **Particle blending**: Shows temporal evolution with alpha transparency

### Clean Room Visualization
**LIDAR and trajectory plots** (`plot_lidar_demo()`, `plot_ground_truth_trajectory()`):
- **No axes, ticks, or grid**: Clean spatial visualization focusing on room layout
- **Hidden spines**: All axis borders removed for uncluttered view
- **Room-only focus**: Emphasizes spatial relationships without distracting elements
- **Enhanced visibility**: Legend placed inside room (lower left), doubled marker sizes (160 vs 80), increased legend font size (20pt), larger robot arrows with thicker lines
- **Larger room labels**: "Room _" labels increased from 14pt to 20pt for better readability in LIDAR demo

### Shared Distance Plots
**Sensor observations** (`plot_sensor_observations()`):
- **Shared y-axis**: All subplots use consistent distance scaling
- **Single label**: "Distance" label only on leftmost column
- **Ray-specific plots**: Each LIDAR ray shown in separate subplot with shared scaling
- **Improved label positioning**: "Ray _" labels moved from (0.02, 0.98) to (0.15, 0.85) for better visibility within subplots

### Other Plots
- **`plot_particle_filter_evolution()`**: 4×4 grid showing particle evolution over 16 timesteps
- **`plot_multi_method_estimation_error()`**: Position and heading error comparison across methods
- **`plot_smc_timing_comparison()`**: Horizontal bar chart with confidence intervals

## CLI Usage (`main.py`)

### Two-Step Workflow
```bash
# Step 1: Generate all experimental data
pixi run cuda-localization-generate-data

# Step 2: Plot all figures from saved data
pixi run cuda-localization-plot-figures

# Or run full pipeline:
pixi run cuda-localization
```

### Direct Environment Usage
```bash
# Generate data with specific parameters
pixi run -e cuda python -m examples.localization.main generate-data \
    --n-particles 100 --k-rejuv 20 --timing-repeats 5 \
    --include-basic-demo --include-smc-comparison

# Plot from specific experiment
pixi run -e cuda python -m examples.localization.main plot-figures \
    --experiment-name localization_r8_p100_basic_20250620_123456
```

### Key Arguments for `generate-data`
- **`--include-basic-demo`**: Include basic particle filter demo
- **`--include-smc-comparison`**: Include 4-method SMC comparison (adds computation time)
- **`--n-particles N`**: Particle count (default: 200)
- **`--k-rejuv K`**: MCMC rejuvenation steps (default: 20)
- **`--timing-repeats R`**: Timing repetitions (default: 20)
- **`--experiment-name NAME`**: Custom experiment name (defaults to timestamped)

### Key Arguments for `plot-figures`
- **`--experiment-name NAME`**: Experiment to plot (defaults to most recent)
- **`--no-lidar-rays`**: Disable LIDAR ray visualization in plots
- **`--output-dir DIR`**: Output directory for figures (default: figs)

## Technical Details

### JAX Patterns
- **rejuvenation_smc usage**: `seed(rejuvenation_smc)(key, model, observations=obs, n_particles=const(N))`
- **Const[...] pattern**: Static parameters use `K: Const[int] = const(10)` for proper JIT compilation
- **Vmap integration**: Sensor model uses GenJAX `Vmap` for 8-ray LIDAR vectorization
- **Key management**: Use `seed()` transformation at top level, avoid explicit keys in @gen functions

### Performance (Drift-Only Model)
- **LIDAR rays**: 8 rays provide good accuracy vs speed tradeoff
- **Particle counts**: 50-200 particles for real-time performance
- **Timing (100 particles)**:
  - Basic SMC: ~22ms
  - SMC + MH: ~26ms
  - SMC + HMC: ~53ms
  - SMC + Locally Optimal: ~30ms
- **Convergence**: Excellent tracking with average position error < 0.2

### Drift-Only Model Parameters
- **Initial distribution**: (1.5, 1.5) with σ=0.5 (near true start at 1.2, 1.2)
- **Drift noise**: σ_x=0.15, σ_y=0.15, σ_θ=0.05
- **Sensor noise**: σ=0.3 for LIDAR measurements
- **No velocity variables**: Simplified state space improves convergence

### Common Issues
- **Environment**: Always use `pixi run -e cuda` for proper dependencies
- **Const[...] errors**: Ensure `from genjax import const` in imports
- **PJAX primitives**: Apply `seed()` before JAX transformations
- **Observation format**: Ground truth must match 8-element LIDAR array structure

## Current Status (June 20, 2025)

### ✅ Production Ready
- **Drift-only model**: Simplified dynamics for excellent SMC convergence
- **Enhanced visualization**: 4-row SMC comparison with particle blending
- **Complete data export**: CSV system with metadata preservation
- **Plot-from-data**: Generate visualizations without recomputation
- **GPU acceleration**: CUDA environment for fast computation

### 🎯 Model Improvements
1. **Removed velocity variables**: Simplified state space to (x, y, θ) only
2. **Centered initial distribution**: Near true start position (1.5, 1.5)
3. **Reduced sensor noise**: From σ=1.5 to σ=0.3 for better tracking
4. **Updated locally optimal proposal**: 3D grid search without velocity dimensions
5. **Achieved excellent convergence**: All methods track ground truth effectively

### 📊 Data Export Benefits
- **Reproducibility**: Complete experimental record with metadata
- **Efficiency**: Avoid rerunning expensive experiments for plot adjustments
- **Sharing**: CSV format enables external analysis (R, MATLAB, pandas)
- **Comparison**: Easy parameter studies across experimental conditions

### 🚀 Ready for Research
- **Four SMC methods** fully implemented and benchmarked
- **Drift-only dynamics** provide stable, interpretable results
- **Complete experimental pipeline** with data export/import
- **Publication-ready visualizations** with method comparison plots
- **Fast performance** suitable for real-time applications

All functionality tested and verified with the drift-only model providing excellent convergence properties.

## Figure Generation and Naming

### 📊 Generated Figures
The localization case study generates 15 distinct visualizations, each with descriptive filenames:

1. **`localization_r8_p200_basic_true_robot_path_with_lidar_observations.pdf`**
   - Ground truth robot trajectory through the multi-room environment
   - LIDAR sensor readings annotated along the path
   - Shows actual robot movement for comparison with estimates

2. **`localization_r8_p200_basic_trajectory_types_exploration_vs_navigation.pdf`**
   - Compares different movement strategies: exploration vs room navigation
   - Multiple trajectories overlaid on the same world map
   - Demonstrates variety in robot movement patterns

3. **`localization_r8_p200_basic_lidar_8ray_wall_detection_visualization.pdf`**
   - Single robot pose with 8 LIDAR rays extending to walls
   - Shows how distance measurements are computed
   - Illustrates sensor model used in particle filtering

4. **`localization_r8_p200_basic_particle_filter_temporal_evolution_16steps.pdf`**
   - 4×4 grid showing particle cloud evolution over 16 timesteps
   - Visualizes convergence from initial uncertainty to final estimate
   - Particles colored by viridis colormap for visual distinction

5. **`localization_r8_p200_basic_final_particle_distribution_at_convergence.pdf`**
   - Final timestep particles showing convergence to true location
   - Weighted particle sizes indicate importance
   - Demonstrates successful localization

6. **`localization_r8_p200_basic_position_and_heading_error_over_time.pdf`**
   - Two subplots: position error (meters) and heading error (radians)
   - Tracks estimation accuracy throughout trajectory
   - Shows convergence behavior over time

7. **`localization_r8_p200_basic_particle_weights_and_ess_percentage_timeline.pdf`**
   - Raincloud plots of particle weight distributions
   - ESS (Effective Sample Size) shown as percentage of total particles
   - Color-coded by ESS quality: good (>50%), fair (10-50%), poor (<10%)

8. **`localization_r8_p200_basic_lidar_distance_readings_along_trajectory.pdf`**
   - LIDAR distance measurements plotted over trajectory steps
   - 8 lines representing each LIDAR ray's readings
   - Shows sensor data variation along the path

9. **`localization_r8_p200_basic_inference_runtime_performance_comparison.pdf`**
   - Horizontal bar chart comparing SMC method execution times
   - Error bars show standard deviation across timing runs
   - Methods sorted by speed (fastest to slowest)

10. **`localization_r8_p200_basic_comprehensive_4panel_smc_methods_analysis.pdf`**
    - Main 4-row comparison figure with all SMC variants
    - Row 1: Initial particle distributions with method titles
    - Row 2: Particle evolution showing trajectory blending
    - Row 3: Raincloud plots with ESS percentages
    - Row 4: Timing comparison bars
    - Legend at bottom with method descriptions

11-14. **Per-method particle evolution timelines**:
    - `smc-basic_particle_evolution_timeline.pdf` - Bootstrap filter
    - `smc-hmc_particle_evolution_timeline.pdf` - SMC with HMC rejuvenation
    - `smc-locally-optimal_particle_evolution_timeline.pdf` - SMC with locally optimal proposal
    - `smc-locally-optimal-big-grid_particle_evolution_timeline.pdf` - SMC with 5 particles

15. **`localization_r8_p200_basic_all_methods_tracking_accuracy_comparison.pdf`**
    - Side-by-side position and heading error plots for all methods
    - Direct comparison of tracking accuracy
    - Shows which methods maintain lowest error

### 🎨 Visualization Features
- **ESS Display**: Shows as percentage (e.g., "ESS: 75%") for intuitive understanding
- **Legend Design**: 
  - Large colored squares (3×2 units) for visibility
  - 18pt font at bottom of figure
  - Line breaks in method labels for readability
- **Method Titles**: Each subplot shows method name with parameters
- **Consistent Prefix**: All figures start with "localization_" for clear identification
- **Parametric Naming**: Configuration encoded in filename (r8=8 rays, p200=200 particles, basic=world type)
