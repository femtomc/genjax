# ADEV (Automatic Differentiation of Expected Values) References

This document provides references for the gradient estimation techniques and automatic differentiation methods implemented in the ADEV module.

## Core ADEV Theory

### Primary Reference
- Lew, A. K., Huot, M., Staton, S., & Mansinghka, V. K. (2023). **ADEV: Sound Automatic Differentiation of Expected Values of Probabilistic Programs**. Proceedings of the ACM on Programming Languages, 7(POPL), 121-148. [arXiv:2212.06386](https://arxiv.org/abs/2212.06386)

### Theoretical Foundations
- Staton, S. (2017). **Commutative semantics for probabilistic programming**. In European Symposium on Programming (pp. 855-879). Springer.
- Ścibior, A., Ghahramani, Z., & Gordon, A. D. (2015). **Practical probabilistic programming with monads**. In Proceedings of the 2015 ACM SIGPLAN Symposium on Haskell (pp. 165-176).

## Gradient Estimation Strategies

### Score Function (REINFORCE) Estimator
- Williams, R. J. (1992). **Simple statistical gradient-following algorithms for connectionist reinforcement learning**. Machine Learning, 8(3-4), 229-256.
- Glynn, P. W. (1990). **Likelihood ratio gradient estimation for stochastic systems**. Communications of the ACM, 33(10), 75-84.
- Fu, M. C. (2006). **Gradient estimation**. Handbooks in Operations Research and Management Science, 13, 575-616.

### Reparameterization Trick
- Kingma, D. P., & Welling, M. (2013). **Auto-encoding variational bayes**. arXiv preprint arXiv:1312.6114.
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). **Stochastic backpropagation and approximate inference in deep generative models**. In International Conference on Machine Learning (pp. 1278-1286).
- Figurnov, M., Mohamed, S., & Mnih, A. (2018). **Implicit reparameterization gradients**. In Advances in Neural Information Processing Systems (pp. 441-452).

### Enumeration for Discrete Distributions
- Obermeyer, F., Bingham, E., Jankowiak, M., Chiu, J., Pradhan, N., Rush, A., & Goodman, N. (2019). **Tensor variable elimination for plated factor graphs**. In International Conference on Machine Learning (pp. 4871-4880).
- Lew, A. K., Matheos, G., Zhi-Xuan, T., Ghavamizadeh, M., Gothoskar, N., Russell, S., & Mansinghka, V. K. (2023). **SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals**. In International Conference on Artificial Intelligence and Statistics (pp. 7061-7088).

### Measure-Valued Derivatives
- Arya, G., Schauer, M., Schäfer, F., & Rackauckas, C. (2022). **Automatic differentiation of programs with discrete randomness**. In Advances in Neural Information Processing Systems, 35, 10435-10447.
- Vaikuntanathan, S., Qin, Z., & Maddison, C. J. (2023). **Unbiased gradient estimation for differentiable surface splatting via Poisson sampling**. arXiv preprint arXiv:2304.09161.

## Related Gradient Estimation Work

### Control Variates and Variance Reduction
- Tucker, G., Mnih, A., Maddison, C. J., Lawson, J., & Sohl-Dickstein, J. (2017). **REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models**. In Advances in Neural Information Processing Systems (pp. 2627-2636).
- Grathwohl, W., Choi, D., Wu, Y., Roeder, G., & Duvenaud, D. (2018). **Backpropagation through the void: Optimizing control variates for black-box gradient estimation**. arXiv preprint arXiv:1711.00123.
- Miller, A., Foti, N., D'Amour, A., & Adams, R. P. (2017). **Reducing reparameterization gradient variance**. In Advances in Neural Information Processing Systems (pp. 3708-3718).

### Continuous Relaxations
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). **The concrete distribution: A continuous relaxation of discrete random variables**. arXiv preprint arXiv:1611.00712.
- Jang, E., Gu, S., & Poole, B. (2017). **Categorical reparameterization with gumbel-softmax**. arXiv preprint arXiv:1611.01144.

## Implementation and Systems

### Automatic Differentiation Systems
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Zhang, Q. (2018). **JAX: composable transformations of Python+NumPy programs**. http://github.com/google/jax
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). **PyTorch: An imperative style, high-performance deep learning library**. In Advances in Neural Information Processing Systems (pp. 8026-8037).

### Probabilistic Programming and AD
- van de Meent, J. W., Paige, B., Yang, H., & Wood, F. (2018). **An introduction to probabilistic programming**. arXiv preprint arXiv:1809.10756.
- Baudart, G., Burroni, J., Hirzel, M., Mandel, L., & Shinnar, A. (2021). **Compiling stan to generative probabilistic languages and extension to deep probabilistic programming**. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (pp. 966-979).

## Mathematical Background

### Measure Theory and Probability
- Çınlar, E. (2011). **Probability and Stochastics**. Springer.
- Kallenberg, O. (2017). **Random Measures, Theory and Applications**. Springer.

### Category Theory and Effects
- Ścibior, A., Kammar, O., Vákár, M., Staton, S., Yang, H., Cai, Y., ... & Ghahramani, Z. (2018). **Denotational validation of higher-order Bayesian inference**. Proceedings of the ACM on Programming Languages, 2(POPL), 1-29.
- Heunen, C., Kammar, O., Staton, S., & Yang, H. (2017). **A convenient category for higher-order probability theory**. In 2017 32nd Annual ACM/IEEE Symposium on Logic in Computer Science (LICS) (pp. 1-12). IEEE.
