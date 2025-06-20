# CLAUDE.md - Localization Case Study

Probabilistic robot localization using particle filtering with GenJAX. Demonstrates SMC with MCMC rejuvenation, vectorized LIDAR sensing, and multi-room navigation.

## Directory Structure

```
examples/localization/
├── core.py             # SMC methods, models, world geometry
├── data.py             # Trajectory generation
├── figs.py             # Visualization (4-row SMC comparison, raincloud plots)
├── main.py             # CLI with data export/import
├── export.py           # CSV data export/import system
├── data/               # Experimental data (CSV + JSON metadata)
└── figs/               # Generated PDF plots
```

## Core Implementation

### SMC Methods (`core.py`)
- **`run_smc_basic()`**: Bootstrap filter (no rejuvenation)
- **`run_smc_with_mh()`**: SMC + Metropolis-Hastings rejuvenation
- **`run_smc_with_hmc()`**: SMC + Hamiltonian Monte Carlo rejuvenation
- **`run_smc_with_locally_optimal()`**: SMC + Locally optimal proposal using grid evaluation
- **K parameter**: Uses `Const[int]` pattern for JAX compilation: `K: Const[int] = const(10)`

### Models
- **`localization_model()`**: Autoregressive step model with control inference
- **`sensor_model()`**: Vectorized 8-ray LIDAR using GenJAX `Vmap`
- **`initial_model()`**: Broad initialization across 3-room world

### Locally Optimal Proposal (`core.py`)
- **`create_locally_optimal_proposal()`**: Creates transition proposal using grid evaluation
- **Grid Evaluation**: 15×15×15 grid over (x, y, θ) space
- **Vectorized Assessment**: Uses `jax.vmap` to evaluate `localization_model.assess()` at all grid points
- **Optimal Selection**: Finds `argmax` of log probabilities across grid
- **Noise Injection**: Adds Gaussian noise around selected point (σ=0.15 for position, σ=0.075 for angle)
- **JAX Compatible**: Fully vectorized implementation using JAX primitives

### World Geometry
- **3-room layout**: 12×10 world with 9 internal walls and doorways
- **JAX arrays**: Wall coordinates stored as `walls_x1`, `walls_y1`, `walls_x2`, `walls_y2`
- **Vectorized intersections**: Ray-wall calculations use JAX vmap

## Data Export System (`export.py`)

### Structure
```
data/smc_comparison_r{rays}_p{particles}_{world_type}_{timestamp}/
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

### SMC Method Comparison Plot
**4-row layout** (`plot_smc_method_comparison()`):
1. **Initial particles** with "Start" label (left side)
2. **Final particles** with "End" label (left side)
3. **Raincloud plots** - ESS diagnostics with color coding (good/medium/bad)
4. **Timing comparison** - horizontal bars with error bars

**Color coding**: Bootstrap filter (blue), SMC+MH (orange), SMC+HMC (green), SMC+Locally Optimal (red)
**ESS thresholds**: Good ≥50% particles, Medium ≥25%, Bad <25%

### Other Plots
- **`plot_particle_filter_evolution()`**: Grid of timesteps showing particle evolution
- **`plot_smc_timing_comparison()`**: Horizontal bar chart with confidence intervals
- **Raincloud plots**: Use `genjax.viz.raincloud.diagnostic_raincloud()`

## CLI Usage (`main.py`)

### Experimental Workflow
```bash
# Run experiment and export data
pixi run -e localization-cuda python -m examples.localization.main \
    --experiment --export-data --n-particles 100 --k-rejuv 10 --timing-repeats 5

# Generate plots from existing data (no recomputation)
pixi run -e localization-cuda python -m examples.localization.main \
    --plot-from-data data/smc_comparison_r8_p100_basic_20250620_123456
```

### Key Arguments
- **`--experiment`**: Run 4-method SMC comparison (basic, MH, HMC, locally optimal)
- **`--export-data`**: Save all results to timestamped CSV directory
- **`--plot-from-data PATH`**: Generate plots from exported data
- **`--n-particles N`**: Particle count (default: 200)
- **`--k-rejuv K`**: MCMC rejuvenation steps (default: 10)
- **`--timing-repeats R`**: Timing repetitions (default: 20)

## Technical Details

### JAX Patterns
- **rejuvenation_smc usage**: `seed(rejuvenation_smc)(key, model, observations=obs, n_particles=const(N))`
- **Const[...] pattern**: Static parameters use `K: Const[int] = const(10)` for proper JIT compilation
- **Vmap integration**: Sensor model uses GenJAX `Vmap` for 8-ray LIDAR vectorization
- **Key management**: Use `seed()` transformation at top level, avoid explicit keys in @gen functions

### Performance
- **LIDAR rays**: 8 rays provide good accuracy vs speed tradeoff
- **Particle counts**: 50-200 particles for real-time performance
- **ESS resampling**: Trigger when ESS < n_particles/8
- **Timing**: Basic SMC ~4ms, +MH ~6ms, +HMC ~36ms, +Locally Optimal ~4ms (for N=50)

### SMC API Enhancement
- **Fixed transition_proposal signature**: Now correctly passes `(obs, prev_choices, *args)` to proposal functions
- **Improved extensibility**: Proposals can access current observations and previous particle state
- **Backward compatibility**: Existing SMC code continues to work unchanged

### Common Issues
- **Import path**: Always use `pixi run -e localization-cuda` for matplotlib dependencies
- **Const[...] errors**: Ensure `from genjax import const` in imports
- **PJAX primitives**: Apply `seed()` before JAX transformations
- **Observation format**: Ground truth must match 8-element LIDAR array structure

## Current Status (June 20, 2025)

### ✅ Production Ready
- **Enhanced visualization**: 4-row SMC comparison with Start/End labels
- **Complete data export**: CSV system with metadata preservation
- **Plot-from-data**: Generate visualizations without recomputation
- **Const[...] pattern**: Proper JAX compilation for all parameters
- **Environment**: Uses `localization-cuda` for GPU acceleration + visualization

### 🎯 Latest Session Accomplishments
1. **Implemented Locally Optimal Proposal**: Added fourth SMC method using grid evaluation
2. **Fixed SMC transition_proposal API**: Updated to pass `(obs, prev_choices, *args)` correctly
3. **Added vectorized assess calls**: Uses `jax.vmap` for efficient grid evaluation
4. **Updated visualization**: All plots now support 4 methods with proper color coding
5. **Enhanced performance**: Locally optimal method achieves ~4ms timing, comparable to bootstrap filter

### 📊 Data Export Benefits
- **Reproducibility**: Complete experimental record with metadata
- **Efficiency**: Avoid rerunning expensive experiments for plot adjustments
- **Sharing**: CSV format enables external analysis (R, MATLAB, pandas)
- **Comparison**: Easy parameter studies across experimental conditions

### 🚀 Ready for Research
- **Four SMC methods** fully implemented and benchmarked
- **Locally optimal proposal** demonstrates vectorized assess capabilities
- **Complete experimental pipeline** with data export/import
- **Publication-ready visualizations** with method comparison plots
- **Extensible framework** for adding new proposal methods

All functionality tested and verified. Ready for immediate research use or future enhancement.
