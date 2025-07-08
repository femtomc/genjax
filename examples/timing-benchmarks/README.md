# Timing Benchmarks

Performance benchmarks comparing GenJAX against other probabilistic programming frameworks.

## Quick Start - GMM Benchmarks

The Gaussian Mixture Model (GMM) benchmarks compare GenJAX, handcoded JAX, and handcoded PyTorch implementations for computing posterior component assignments P(z|x).

```bash
# Run quick GMM benchmarks (small data sizes, few repeats)
pixi run gmm-quick

# Run full GMM benchmarks (all data sizes, many repeats)
pixi run gmm-all

# Just generate the comparison plot from existing data
pixi run gmm-plot
```

## Overview

We benchmark two main categories:

1. **GMM (Gaussian Mixture Models)**: Comparing handcoded implementations for posterior inference
2. **Curvefit (Polynomial Regression)**: Comparing importance sampling performance across frameworks

## Key Figures Generated

- `gmm_all_frameworks_comparison.pdf`: GMM performance comparison showing GenJAX achieves native JAX performance
- `all_frameworks_comparison.pdf`: Curvefit comparison showing scaling performance vs hand-optimized JAX

## Directory Structure

```
timing-benchmarks/
├── combine_results.py          # Combines curvefit results and generates main plot
├── combine_handcoded_results.py # Combines GMM results
├── src/timing_benchmarks/
│   ├── curvefit-benchmarks/    # Polynomial regression benchmarks
│   │   ├── genjax.py          # GenJAX implementation
│   │   ├── numpyro.py         # NumPyro implementation
│   │   ├── handcoded_tfp.py   # TensorFlow Probability baseline
│   │   ├── handcoded_torch.py # PyTorch baseline
│   │   ├── pyro.py            # Pyro implementation
│   │   └── genjl.py           # Gen.jl wrapper
│   ├── handcoded_benchmarks/   # GMM benchmarks
│   │   ├── genjax_handcoded.py
│   │   ├── jax_handcoded.py
│   │   └── torch_handcoded.py
│   ├── generate_curvefit_data.py # Data generation for curvefit
│   └── julia_interface.py      # Python-Julia bridge for Gen.jl
├── julia/                      # Gen.jl implementations
│   └── src/
│       ├── polynomial_regression.jl
│       └── polynomial_regression_dynamic.jl
├── data/                       # Benchmark results
│   └── curvefit/              # Results by framework
└── figs/                       # Generated figures
```

## Environments

The benchmarks require different pixi environments for different frameworks:

- **default**: GenJAX and NumPyro (CPU)
- **cuda**: GenJAX, NumPyro, and TensorFlow Probability (GPU)
- **pyro**: PyTorch and Pyro (GPU)

## Running Benchmarks

### Full Pipeline for Paper Figures

To generate the main comparison figure (`all_frameworks_comparison.pdf`):

```bash
# 1. Generate test data
pixi run generate-data

# 2. Run benchmarks (in separate environments)
pixi run genjax                    # GenJAX
pixi run numpyro                   # NumPyro
pixi run -e cuda cuda-tfp          # TensorFlow Probability (GPU)
pixi run -e pyro pyro-curvefit    # Pyro (GPU)
pixi run -e pyro pyro-torch       # PyTorch baseline (GPU)
pixi run genjl                     # Gen.jl (dynamic)

# 3. Combine results and generate plot
pixi run combine-curvefit
```

The final plot will be in `figs/all_frameworks_comparison.pdf`.

### Quick Testing

For development/testing with fewer iterations:

```bash
# Generate data
pixi run generate-data

# Run quick benchmarks
pixi run python -m timing_benchmarks.curvefit-benchmarks.genjax --repeats 10 --n-particles 1000

# Generate plot
pixi run combine-curvefit
```

### GMM Benchmarks

The GMM benchmarks test the performance of computing posterior probabilities for component assignments given observations from a 1D Gaussian mixture model with 3 components.

#### What's Being Benchmarked

Each framework implements the same computation: given observations x and GMM parameters (means, stds, weights), compute the posterior probability P(z|x) for component assignments. This is equivalent to one step of importance sampling with the exact posterior as the proposal.

#### Running GMM Benchmarks

```bash
# Quick test (small data, few repeats)
pixi run gmm-quick

# Full benchmarks (all data sizes, many repeats)
pixi run gmm-all

# Individual framework runs
pixi run -e cuda cuda-gmm-genjax   # GenJAX
pixi run -e cuda cuda-gmm-jax      # Handcoded JAX
pixi run -e torch torch-gmm        # Handcoded PyTorch

# Generate plot only
pixi run gmm-plot
```

#### Results Summary

- **GenJAX**: 1.0-1.1x compared to handcoded JAX (essentially native performance)
- **Handcoded JAX**: Baseline implementation
- **Handcoded PyTorch**: ~5x slower than JAX implementations

The results demonstrate that GenJAX achieves native JAX performance while providing a high-level probabilistic programming interface.

## Julia/Gen.jl Setup

Gen.jl benchmarks require Julia to be installed:

```bash
# Install Julia via juliaup
curl -fsSL https://install.julialang.org | sh
juliaup add 1.9
juliaup default 1.9

# Install Gen.jl dependencies
cd julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Important Notes

1. **Environment Separation**: JAX and PyTorch frameworks must run in separate CUDA environments to avoid conflicts
2. **Warm-up**: All benchmarks include proper JIT warm-up runs before timing
3. **Fair Comparison**: All frameworks implement identical algorithms with same parameters
4. **Gen.jl Performance**: The dynamic DSL version performs better than static+Map for this use case

## Cleaning Up

```bash
# Remove generated data
pixi run clean-data

# Remove figures
pixi run clean-figs

# Clean everything
pixi run clean-all
```

## Troubleshooting

- **CUDA Errors**: Ensure you're using the correct environment (`-e cuda` or `-e pyro`)
- **Julia Errors**: Check that Julia and Gen.jl are properly installed
- **Missing Data**: Run the generate-data task before benchmarking