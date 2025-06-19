# Rejuvenation SMC Case Study

This case study demonstrates Sequential Monte Carlo (SMC) with rejuvenation on discrete Hidden Markov Models (HMMs) and linear Gaussian state space models. We compare the SMC approximations against exact inference baselines.

## Overview

The case study implements:

1. **Discrete HMM**: SMC with Metropolis-Hastings rejuvenation steps
2. **Linear Gaussian Model**: SMC with MALA (gradient-based) rejuvenation steps

Both models are tested against exact inference algorithms:
- HMM: Forward filtering backward sampling (FFBS)
- Linear Gaussian: Kalman filtering and smoothing

## Running the Experiments

### Basic Usage

Run all experiments with default parameters:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main
```

### Specific Experiments

Run only the discrete HMM experiment:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main --hmm
```

Run only the linear Gaussian experiment:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main --lg
```

### Custom Parameters

Adjust model dimensions and time steps:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main \
    --hmm-states 5 \
    --hmm-obs 10 \
    --lg-dim-state 3 \
    --lg-dim-obs 2 \
    --time-steps 30
```

Test with different particle counts:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main \
    --particles 50,100,200,500,1000,2000,5000
```

### Output Options

Save results and skip figure generation:
```bash
pixi run -e rejuv-smc python -m examples.rejuvenation_smc.main \
    --save-results \
    --no-figs \
    --output-dir my_results
```

## Results

The experiments produce:

1. **Convergence plots**: Log-log plots showing error vs. number of particles
2. **ESS evolution**: Effective sample size over time
3. **State trajectories**: Sample trajectories and posterior distributions
4. **Log marginal comparison**: SMC estimates vs. exact values

All figures are saved to `results/figs/` by default.

## Implementation Details

### Rejuvenation SMC Algorithm

1. **Initialize** particles at t=0 with importance sampling
2. For each time step t=1 to T:
   - **Extend** particles with new observations
   - Check effective sample size (ESS)
   - If ESS < threshold:
     - **Resample** particles
     - **Rejuvenate** with MCMC moves

### Model Specifications

**Discrete HMM**:
- States: Categorical distribution
- Transitions: Discrete Markov chain
- Observations: Categorical emissions
- Rejuvenation: Metropolis-Hastings on state variables

**Linear Gaussian**:
- States: Multivariate Gaussian
- Transitions: Linear dynamics with Gaussian noise
- Observations: Linear projection with Gaussian noise
- Rejuvenation: MALA (gradient-based) on continuous states

## Key Insights

1. **Convergence**: Both models show power-law convergence of errors with increasing particles
2. **ESS Management**: Rejuvenation helps maintain particle diversity
3. **Accuracy**: SMC approximations converge to exact log marginal likelihoods
4. **Efficiency**: Rejuvenation SMC scales well with time series length

## References

- Del Moral et al. (2006): "Sequential Monte Carlo samplers"
- Chopin (2002): "A sequential particle filter method for static models"
- Gilks & Berzuini (2001): "Following a moving target—Monte Carlo inference for dynamic Bayesian models"
