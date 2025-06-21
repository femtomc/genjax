# Inference Algorithm References

This document provides references to papers and resources for the inference algorithms implemented in GenJAX.

## Markov Chain Monte Carlo (MCMC)

### Metropolis-Hastings
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). **Equation of state calculations by fast computing machines**. The Journal of Chemical Physics, 21(6), 1087-1092.
  - **Used in**: `mcmc.py:14-16` - Module docstring, fundamental MCMC algorithm
- Hastings, W. K. (1970). **Monte Carlo sampling methods using Markov chains and their applications**. Biometrika, 57(1), 97-109.
  - **Used in**: `mcmc.py:17-18` - Module docstring, generalization of Metropolis algorithm
- Robert, C., & Casella, G. (2004). **Monte Carlo Statistical Methods**. Springer.
  - **Used in**: General MCMC theory and implementation patterns

### MALA (Metropolis-Adjusted Langevin Algorithm)
- Roberts, G. O., & Tweedie, R. L. (1996). **Exponential convergence of Langevin distributions and their discrete approximations**. Bernoulli, 341-363.
  - **Used in**: `mcmc.py:20-21` - Module docstring, theoretical foundation for MALA
- Roberts, G. O., & Rosenthal, J. S. (1998). **Optimal scaling of discrete approximations to Langevin diffusions**. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 60(1), 255-268.
  - **Used in**: `mcmc.py:22-24` - Module docstring, optimal step size theory

### Hamiltonian Monte Carlo (HMC)
- Neal, R. M. (2011). **MCMC Using Hamiltonian Dynamics**, Handbook of Markov Chain Monte Carlo, pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf
  - **Used in**: `mcmc.py:27` - Module docstring, comprehensive HMC tutorial
- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). **Hybrid Monte Carlo**. Physics Letters B, 195(2), 216-222.
  - **Used in**: `mcmc.py:28-29` - Module docstring, original HMC paper

### Implementation References
- Gen.jl MALA implementation: https://github.com/probcomp/Gen.jl/blob/master/src/inference/mala.jl
  - **Used in**: `mcmc.py:32` - Implementation patterns for MALA
- Gen.jl HMC implementation: https://github.com/probcomp/Gen.jl/blob/master/src/inference/hmc.jl
  - **Used in**: `mcmc.py:33` - Implementation patterns for HMC

### Chain Implementation and Diagnostics
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). **Bayesian Data Analysis**. CRC Press.
  - **Used in**: General MCMC chain methodology
- Geyer, C. J. (1992). **Practical Markov chain Monte Carlo**. Statistical Science, 7(4), 473-483.
  - **Used in**: `mcmc.py:201-204` - Effective sample size computation
- Stan Development Team (2023). **Stan Reference Manual: Effective Sample Size**. Version 2.33. Section 15.4.
  - **Used in**: `mcmc.py:207-208` - ESS implementation reference

## Sequential Monte Carlo (SMC)

### Core SMC Theory
- Del Moral, P., Doucet, A., & Jasra, A. (2006). **Sequential Monte Carlo samplers**. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3), 411-436.
  - **Used in**: `smc.py:8-11` - Module docstring, fundamental SMC reference

### Particle Filtering
- Doucet, A., De Freitas, N., & Gordon, N. (Eds.). (2001). **Sequential Monte Carlo Methods in Practice**. Springer.
  - **Used in**: `smc.py:61-64` - Effective sample size theory
- Doucet, A., & Johansen, A. M. (2009). **A tutorial on particle filtering and smoothing: Fifteen years later**. Handbook of Nonlinear Filtering, 12(656-704), 3.
  - **Used in**: `smc.py:112-115` - Systematic resampling methods

### Effective Sample Size
- Kong, A., Liu, J. S., & Wong, W. H. (1994). **Sequential imputations and Bayesian missing data problems**. Journal of the American Statistical Association, 89(425), 278-288.
  - **Used in**: `smc.py:61-62` - ESS for particle filters
- Liu, J. S. (2001). **Monte Carlo strategies in scientific computing**. Springer, Chapter 3.
  - **Used in**: `smc.py:62-63` - ESS theory and computation

### Resampling Methods
- Kitagawa, G. (1996). **Monte Carlo filter and smoother for non-Gaussian nonlinear state space models**. Journal of Computational and Graphical Statistics, 5(1), 1-25.
  - **Used in**: `smc.py:112-113` - Systematic resampling algorithm
- Douc, R., & Cappé, O. (2005). **Comparison of resampling schemes for particle filtering**. In ISPA 2005. Proceedings of the 4th International Symposium on Image and Signal Processing and Analysis, 2005. (pp. 64-69). IEEE.
  - **Used in**: General resampling theory and comparison
- Hol, J. D., Schon, T. B., & Gustafsson, F. (2006). **On resampling algorithms for particle filters**. In 2006 IEEE Nonlinear Statistical Signal Processing Workshop (pp. 79-82). IEEE.
  - **Used in**: `smc.py:115-117` - Resampling algorithm implementations

### SMC Samplers and Rejuvenation
- Del Moral, P., Doucet, A., & Jasra, A. (2006). **Sequential Monte Carlo samplers**. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3), 411-436.
- Chopin, N. (2002). **A sequential particle filter method for static models**. Biometrika, 89(3), 539-552.

## Variational Inference (VI)

### General VI Theory
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). **Variational inference: A review for statisticians**. Journal of the American Statistical Association, 112(518), 859-877.
- Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). **An introduction to variational methods for graphical models**. Machine Learning, 37(2), 183-233.

### ELBO and Optimization
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). **Stochastic variational inference**. The Journal of Machine Learning Research, 14(1), 1303-1347.
- Ranganath, R., Gerrish, S., & Blei, D. (2014). **Black box variational inference**. In Artificial Intelligence and Statistics (pp. 814-822).

### Integration with ADEV
- Lew, A. K., Huot, M., Staton, S., & Mansinghka, V. K. (2023). **ADEV: Sound Automatic Differentiation of Expected Values of Probabilistic Programs**. Proceedings of the ACM on Programming Languages, 7(POPL), 121-148.

## JAX-Specific Implementation

### JAX Transformations for Inference
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Zhang, Q. (2018). **JAX: composable transformations of Python+NumPy programs**. http://github.com/google/jax

### Vectorization and Performance
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). **Composable effects for flexible and accelerated probabilistic programming in NumPyro**. arXiv preprint arXiv:1912.11554.

## Related Work

### Probabilistic Programming Systems
- Cusumano-Towner, M. F., Saad, F. A., Lew, A. K., & Mansinghka, V. K. (2019). **Gen: a general-purpose probabilistic programming system with programmable inference**. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (pp. 221-236).
- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., ... & Goodman, N. (2019). **Pyro: Deep universal probabilistic programming**. The Journal of Machine Learning Research, 20(1), 973-978.

### Convergence Diagnostics
- Gelman, A., & Rubin, D. B. (1992). **Inference from iterative simulation using multiple sequences**. Statistical Science, 7(4), 457-472.
  - **Used in**: `mcmc.py:123-126` - R-hat convergence diagnostic computation
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). **Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC**. Bayesian Analysis, 16(2), 667-718.
  - **Used in**: `mcmc.py:126-133` - Improved R-hat implementation
  - **Used in**: `mcmc.py:205-206` - ESS computation methods
