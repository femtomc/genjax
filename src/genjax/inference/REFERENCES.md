# Inference Algorithm References

This document provides references to papers and resources for the inference algorithms implemented in GenJAX.

## Markov Chain Monte Carlo (MCMC)

### Metropolis-Hastings
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). **Equation of state calculations by fast computing machines**. The Journal of Chemical Physics, 21(6), 1087-1092.
- Hastings, W. K. (1970). **Monte Carlo sampling methods using Markov chains and their applications**. Biometrika, 57(1), 97-109.
- Robert, C., & Casella, G. (2004). **Monte Carlo Statistical Methods**. Springer.

### MALA (Metropolis-Adjusted Langevin Algorithm)
- Roberts, G. O., & Tweedie, R. L. (1996). **Exponential convergence of Langevin distributions and their discrete approximations**. Bernoulli, 341-363.
- Roberts, G. O., & Rosenthal, J. S. (1998). **Optimal scaling of discrete approximations to Langevin diffusions**. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 60(1), 255-268.

### Chain Implementation
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). **Bayesian Data Analysis**. CRC Press.
- Geyer, C. J. (1992). **Practical Markov chain Monte Carlo**. Statistical Science, 7(4), 473-483.

## Sequential Monte Carlo (SMC)

### Particle Filtering
- Doucet, A., De Freitas, N., & Gordon, N. (Eds.). (2001). **Sequential Monte Carlo Methods in Practice**. Springer.
- Doucet, A., & Johansen, A. M. (2009). **A tutorial on particle filtering and smoothing: Fifteen years later**. Handbook of Nonlinear Filtering, 12(656-704), 3.

### Resampling Methods
- Douc, R., & Cappé, O. (2005). **Comparison of resampling schemes for particle filtering**. In ISPA 2005. Proceedings of the 4th International Symposium on Image and Signal Processing and Analysis, 2005. (pp. 64-69). IEEE.
- Hol, J. D., Schon, T. B., & Gustafsson, F. (2006). **On resampling algorithms for particle filters**. In 2006 IEEE Nonlinear Statistical Signal Processing Workshop (pp. 79-82). IEEE.

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
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). **Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC**. Bayesian Analysis, 16(2), 667-718.