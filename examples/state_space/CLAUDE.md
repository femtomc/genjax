# State Space Model Case Study - CLAUDE.md

This file provides guidance for Claude Code when working with the state space model case study.

## Overview

This case study demonstrates advanced SMC techniques using GenJAX's inference algorithms. It compares approximate SMC inference with exact baselines on two canonical state space models.

## Key Components

### Core Module (`core.py`)

**Models**:
- `discrete_hmm_initial` / `discrete_hmm_step`: Factored HMM for SMC
- `linear_gaussian_initial` / `linear_gaussian_step`: Factored linear Gaussian SSM

**Inference Algorithms**:
- `discrete_hmm_rejuvenation_smc`: SMC with MH rejuvenation
- `linear_gaussian_rejuvenation_smc`: SMC with MALA rejuvenation

**Experiments**:
- `run_discrete_hmm_experiment`: Full experimental pipeline for HMMs
- `run_linear_gaussian_experiment`: Full experimental pipeline for linear Gaussian

### Visualization Module (`figs.py`)

**Plotting Functions**:
- `plot_convergence_comparison`: Log-log error plots
- `plot_ess_evolution`: ESS over time
- `plot_state_trajectories`: Sample paths and distributions
- `plot_log_marginal_comparison`: SMC vs exact estimates

### Main Script (`main.py`)

Orchestrates experiments with command-line interface and result management.

## Critical Implementation Details

### PJAX Seeding Pattern

**ALWAYS use seed transformation** before calling functions with random sampling:

```python
# ✅ CORRECT
seeded_fn = seed(lambda: run_experiment(...))
result = seeded_fn(key)

# ❌ WRONG - will cause LoweringSamplePrimitiveToMLIRException
result = run_experiment(...)
```

### SMC Implementation Notes

1. **Particle Management**: Track ESS and resample when below threshold
2. **Rejuvenation Kernels**: MH for discrete, MALA for continuous
3. **Weight Normalization**: Always work in log space for numerical stability

### Model Factorization

SMC requires separate initial and transition models:

```python
# Initial: p(x_0, y_0)
initial_model(params) → sample x_0, y_0

# Transition: p(x_t, y_t | x_{t-1})
transition_model(x_prev, params) → sample x_t, y_t
```

## Common Issues and Solutions

### Shape Mismatches

Ensure consistency between:
- Observation dimensions and observation matrix shape
- State dimensions across initial/transition models
- Time series length and model expectations

### Numerical Stability

- Use log probabilities throughout
- Check covariance matrices are positive definite
- Ensure transition matrices have stable eigenvalues

### Performance

- JIT compile the main inference loops
- Vectorize particle operations with `vmap`
- Use appropriate particle counts (start small, scale up)

## Testing the Case Study

Quick test:
```bash
pixi run -e state-space state-space-quick
```

Full experiment:
```bash
pixi run -e state-space state-space-all
```

## Comprehensive Features

### Challenging Inference Scenarios
- **Standard**: Baseline noise and drift for algorithm comparison
- **Challenging**: 3× observation noise, drift > 1σ process noise
- **Extreme**: 5× observation noise, nonlinear spiral drift

### New Pixi Tasks
```bash
pixi run -e state-space state-space-comparison      # Standard rejuvenation comparison
pixi run -e state-space state-space-challenging     # Challenging scenario
pixi run -e state-space state-space-extreme         # Extreme scenario
pixi run -e state-space state-space-difficulty      # Cross-scenario analysis
pixi run -e state-space state-space-comprehensive   # All scenarios
```

### Descriptive Figure Names
Figures auto-generate with format: `[type]_[model]_[scenario]_[strategy]_[particles]_[frames]_[info].pdf`

Example: `rejuvenation_smc_linear_gaussian_2d_challenging_n500_frames0_4_8_12_all_strategies.pdf`

## Extension Ideas

1. **Additional Models**: Nonlinear dynamics, switching state space models
2. **Advanced SMC**: Adaptive resampling, tempering, backward simulation
3. **Diagnostics**: Particle genealogy, acceptance rates, weight entropy
4. **Comparisons**: Bootstrap filter, auxiliary particle filter
