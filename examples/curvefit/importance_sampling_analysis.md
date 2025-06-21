# Importance Sampling Analysis for Curvefit Example

## Problem Summary

The GenJAX importance sampling implementation shows zero variance because of severe weight degeneracy. The debug analysis reveals:

### Key Findings

1. **Extreme Weight Concentration**
   - 100 samples: 100% weight on 1 particle (ESS = 1.00)
   - 1000 samples: 95% weight on 1 particle (ESS = 1.10)
   - 5000 samples: 89% weight on 1 particle (ESS = 1.26)

2. **Log Weight Statistics**
   - Enormous range: up to 99,200 in log space
   - This translates to weight ratios of e^99200 ≈ 10^43000
   - Most particles have effectively zero weight

3. **Resampling Collapse**
   - When resampling, only 1-2 unique particles are selected
   - This creates the appearance of zero variance
   - The posterior mean is just the value of the single dominant particle

## Root Cause

The issue is that GenJAX's `init` function uses importance sampling from the prior by default. For this curve fitting problem:

- **Prior**: Normal distributions with relatively large variances (σ_a=1.0, σ_b=1.5, σ_c=0.8)
- **Posterior**: Much more concentrated around true values due to 20 data points
- **Mismatch**: Prior is too diffuse compared to posterior

When sampling from such a broad prior:
- Most samples fall in low-probability regions of the posterior
- A tiny fraction of samples happen to land near the true values
- These rare "lucky" samples get astronomical weights
- Resampling selects only these particles

## Mathematical Explanation

For importance sampling with proposal q and target p:
- Weight: w_i = p(x_i)/q(x_i)
- Here: q = prior, p = posterior ∝ prior × likelihood

With 20 observations, the likelihood is very peaked. Samples far from the true parameters have likelihood ≈ 0, while samples near truth have reasonable likelihood. This creates weight ratios of 10^40000 or more.

## Solutions

1. **Use Better Proposals**: Design proposals closer to the posterior
2. **Use Sequential Methods**: Build up observations gradually (SMC)
3. **Use MCMC**: Start from a reasonable point and explore locally
4. **Adaptive Importance Sampling**: Learn better proposals iteratively

## Comparison with NumPyro

NumPyro's importance sampling likely shows the same issue, as both frameworks sample from the prior. The difference in reported statistics may be due to:
- Different random seeds hitting different "lucky" particles
- Numerical differences in weight calculations
- Different resampling implementations

## Conclusion

The zero variance is not a bug but a fundamental limitation of importance sampling from the prior for this problem. The method is working correctly but is extremely inefficient for problems where the posterior is much more concentrated than the prior.
