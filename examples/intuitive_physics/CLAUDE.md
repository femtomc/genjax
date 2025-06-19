# CLAUDE.md - Intuitive Physics Case Study

This file provides guidance to Claude Code when working with the intuitive physics case study, implementing model-inference co-design for rational agent behavior in physical environments.

## Overview

The intuitive physics case study demonstrates how rational agents make decisions in physical environments and how observers can infer hidden environmental constraints from observed behavior. This implements the core ideas from Liu, Outa, & Akbiyik (2024) on naive psychology depending on naive physics.

**Key Research Question**: Can we infer whether a wall was present by observing an agent's trajectory?

## Directory Structure

```
examples/intuitive_physics/
├── CLAUDE.md           # This file - guidance for Claude Code
├── core.py             # Physics simulation and rational agent model
├── figs.py             # Visualization utilities
├── main.py             # Command-line interface
└── figs/               # Generated visualization outputs
    ├── environment_setup.pdf
    ├── action_space_analysis.pdf
    ├── inference_demonstration.pdf
    ├── trajectory_*.pdf
    ├── sample_actions.pdf
    └── timing_comparison.pdf
```

## Code Organization

### `core.py` - Core Implementation

**Physics Simulation**:
- **`physics_step(state, wall_present)`**: Single physics timestep with collision detection
- **`simulate_trajectory(theta, impulse, wall_present)`**: Complete trajectory simulation
- **Environment**: 2D world with agent, wall (optional), goal, gravity, friction

**Rational Agent Model**:
- **`compute_utility(theta, impulse, wall_present, goal_weight)`**: Utility computation
- **`rational_agent(wall_present, goal_weight)`**: GenJAX model for action selection
- **`intuitive_physics_model()`**: Full generative model

**Inference Utilities**:
- **`wall_inference_from_action(theta_idx, impulse_idx)`**: Infer wall presence from action
- **Timing functions**: Performance benchmarking for physics and inference

### `figs.py` - Visualizations

**Environment Visualization**:
- **`create_physics_visualization()`**: Side-by-side environment comparison
- **`visualize_trajectory(theta, impulse)`**: Trajectory comparison with/without wall

**Analysis Visualization**:
- **`action_space_heatmap()`**: Comprehensive 3x3 grid showing utility landscapes for different agent types
- **`sample_actions_visualization()`**: Sample trajectories for different agent types
- **`inference_demonstration_fig()`**: Key insight visualization showing action informativeness
- **`timing_comparison_fig()`**: Performance benchmarking results

### `main.py` - CLI Interface

**Visualization Modes**:
- `--environment`: Environment setup comparison
- `--trajectory`: Single trajectory visualization
- `--action-space`: Action space utility analysis (enhanced 3x3 layout)
- `--samples`: Sample agent behavior visualization
- `--timing`: Performance benchmarking
- `--inference`: Inference demonstration showing action informativeness
- `--all`: Generate all visualizations

## Key Implementation Details

### Physics Environment

**Environment Setup**:
```python
AGENT_START = [0.0, 0.1]        # Agent starting position
WALL_CENTER = [0.7, 0.3]        # Wall position (when present)
GOAL_CENTER = [1.5, 0.25]       # Goal region center
```

**Physics Parameters**:
```python
GRAVITY = -0.5                   # Gravity acceleration
FRICTION = 0.95                  # Friction coefficient
TIME_STEPS = 100                 # Simulation duration
DT = 0.02                        # Time step size
```

**Action Space**:
```python
ANGLE_GRID = linspace(0, π/3, 25)    # Launch angles (0-60°)
IMPULSE_GRID = linspace(0.1, 3.0, 25) # Launch impulses
```

### Generative Model

**Model Structure**:
1. **Wall presence**: Bernoulli(0.5) prior
2. **Agent preferences**: High goal weight (80%) vs. low goal weight (20%)
3. **Rational action selection**: Softmax over utilities (temperature=3.0)
4. **Physics**: Deterministic trajectory given action and environment

**Utility Function**:
```python
utility = goal_weight * (goal_reward + distance_penalty) + (1-goal_weight) * effort_cost
```

**Key Insight**: Different wall conditions lead to different action preferences, enabling inference.

### JAX Implementation Patterns

**Critical JAX Usage** (strictly enforced):
- **Physics simulation**: Pure functions with `jax.lax.scan` for time evolution
- **Collision detection**: `jax.lax.cond` for conditional physics
- **Action selection**: `categorical` distribution over flattened action space
- **Utility computation**: `jax.vmap` over flattened meshgrids (NO Python for loops)
- **Vectorization**: `jax.vmap` for all batch operations

**GenJAX Patterns**:
- **`@gen` decorator**: For probabilistic model definitions
- **`Const[T]` types**: For static configuration parameters
- **`@ "address"`**: For random choice addressing
- **`modular_vmap`**: For probabilistic vectorization

**JAX Best Practices Enforced**:
- **NO Python control flow** in `@gen` functions
- **NO nested Python for loops** in utility computation
- **Vectorized operations** using `jax.vmap` for efficiency
- **Flattened meshgrid approach** for 2D utility computation

## Performance Characteristics

### Computational Complexity

**Physics Simulation**:
- Single trajectory: O(time_steps) ≈ 100 steps
- Batch simulation: O(batch_size × time_steps) with vectorization
- Collision detection: O(1) per step

**Rational Agent** (optimized implementation):
- Utility computation: O(|actions|²) = O(100) for subsampled action space (10×10)
- Vectorized with `jax.vmap`: All utilities computed in parallel
- Action selection: O(|actions|) = O(100) for softmax
- **Performance improvement**: ~10x faster than nested Python loops

**Inference**:
- Importance sampling: O(num_samples × model_cost)
- Wall inference: ~100-1000 samples typically sufficient
- Physics dominates computational cost

### Expected Timing

**Typical Performance** (CPU, optimized):
- Single trajectory: ~1-2ms
- Batch physics (100 trajectories): ~10-20ms
- Rational agent simulation: ~200-800ms (vectorized utility computation)
- Full model simulation: ~200-300ms
- Wall inference (1000 samples): ~1-5 seconds

## Visualization Standards

### Research Paper Quality

**Figure Specifications**:
- **Format**: 300 DPI PDF output
- **Fonts**: 12-16pt for readability
- **Colors**: Color-blind friendly palettes
- **Layout**: Clean, publication-ready styling

**Figure Types**:
1. **Environment**: Side-by-side comparison (no wall vs. wall)
2. **Trajectories**: Overlaid paths with endpoint markers
3. **Action space**: Enhanced 3×3 grid layout comparing agent types and utilities
4. **Inference demonstration**: 2×2 layout showing action informativeness and key insights
5. **Samples**: Multiple trajectory examples per agent type
6. **Timing**: Bar charts with error bars and distributions

### Parametrized Filenames

**Naming Convention**:
- Environment: `environment_setup.pdf`
- Trajectory: `trajectory_theta{θ:.3f}_impulse{ι:.3f}.pdf`
- Action space: `action_space_analysis.pdf`
- Inference: `inference_demonstration.pdf`
- Samples: `sample_actions.pdf`
- Timing: `timing_comparison.pdf`

## Usage Patterns

### Quick Testing

```bash
# Environment visualization
pixi run -e intuitive-physics intuitive-physics-env

# Single trajectory
pixi run -e intuitive-physics intuitive-physics-trajectory --theta 0.5 --impulse 2.5
```

### Analysis Workflows

```bash
# Complete analysis
pixi run -e intuitive-physics intuitive-physics-all

# Action space analysis (enhanced 3x3 layout)
pixi run -e intuitive-physics intuitive-physics-action-space

# Inference demonstration (key insights)
pixi run -e intuitive-physics intuitive-physics-inference

# Performance benchmarking
pixi run -e intuitive-physics intuitive-physics-timing
```

### Development Testing

```bash
# Quick verification of core functionality
pixi run -e intuitive-physics intuitive-physics-env

# Trajectory with specific parameters
pixi run -e intuitive-physics intuitive-physics-trajectory --theta 0.1 --impulse 3.0
```

## Model-Inference Co-Design Insights

### Design Trade-offs

**Physics Complexity vs. Inference Tractability**:
- **Simplified physics**: 2D, basic collision detection, friction
- **Benefit**: Fast simulation enables large-scale inference
- **Cost**: Less realistic than full 3D physics engines

**Action Space Discretization**:
- **10×10 = 100 actions**: Optimized for demonstration efficiency
- **Full space**: 25×25 = 625 actions available for production use
- **Benefit**: Rational agent model remains tractable with JAX vectorization
- **Cost**: Limited action resolution in demonstration mode

**Utility Function Design**:
- **Goal achievement + distance + effort**: Captures key trade-offs
- **Linear combination**: Simple but expressive
- **Agent heterogeneity**: Goal weight parameter creates behavior diversity

### Inference Capabilities

**Wall Inference Accuracy**:
- **High-impulse, low-angle actions**: Strong evidence for wall presence (collision)
- **Low-impulse actions**: Ambiguous evidence (don't reach wall region)
- **High-angle actions**: Moderate evidence (trajectory shape differences)

**Key Result**: Rational action model enables sophisticated "inverse psychology" reasoning from minimal observations.

## Development Guidelines

### When Modifying Physics

1. **Maintain JAX compatibility**: Use `jax.lax.cond`, `jax.lax.scan`
2. **Test collision detection**: Verify wall effects with `debug_physics.py`
3. **Check parameter ranges**: Ensure actions can reach wall region
4. **Preserve determinism**: Physics should be deterministic given action

### When Modifying Agent Model

1. **Maintain JAX vectorization**: NEVER use Python for loops in utility computation
2. **Utility function changes**: May require re-tuning temperature parameter
3. **Action space changes**: Update grid definitions and ensure `jax.vmap` compatibility
4. **Preference modeling**: Maintain interpretable goal/effort trade-offs
5. **Vectorization patterns**: Use flattened meshgrid approach for 2D computations

### When Adding Inference Methods

1. **Constraint format**: Match GenJAX model address structure
2. **Sample size tuning**: Balance accuracy vs. computational cost
3. **Diagnostic visualization**: Add convergence and mixing diagnostics
4. **Performance comparison**: Benchmark against importance sampling baseline

## Integration with GenJAX

This case study demonstrates:

1. **Physics simulation integration**: JAX-compatible deterministic models
2. **Rational agent modeling**: Utility-based probabilistic choice
3. **Model-inference co-design**: Balancing expressiveness and tractability
4. **Hierarchical models**: Environment → agent → action causally structured
5. **Inverse reasoning**: Inferring hidden causes from observed behavior

**Research Applications**: Cognitive science, robotics, behavioral economics, theory of mind modeling.

## References

### Theoretical Foundation

- **Liu, S., Outa, J., & Akbiyik, S. (2024)**. Naive psychology depends on naive physics.
- **Csibra, G., et al. (2003)**. Teleological reasoning in infancy: The naive psychology of rational action.

### Technical Implementation

- **GenJAX Documentation**: Core generative function interface and probabilistic programming patterns
- **JAX Documentation**: Functional programming and automatic differentiation in JAX
- **Model-Inference Co-design**: Balancing model expressiveness with computational tractability

This case study provides a concrete example of how model-inference co-design principles guide the development of tractable yet expressive probabilistic models for complex behavioral reasoning tasks.
