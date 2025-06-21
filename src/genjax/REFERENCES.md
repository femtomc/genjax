# References for GenJAX Core Module

This document provides comprehensive references for the core GenJAX module, including generative function interface (GFI) implementations, probability distributions, and probabilistic JAX (PJAX) infrastructure.

## Generative Function Interface and Gen.jl

## [1] Cusumano-Towner, M. F., Saad, F. A., Lew, A. K., & Mansinghka, V. K. (2019). "Gen: a general-purpose probabilistic programming system with programmable inference"
**Conference**: Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pp. 221-236
**DOI**: 10.1145/3314221.3314642
**Abstract**: Introduces Gen, a probabilistic programming system with the Generative Function Interface enabling programmable inference

**Used in**:
- `core.py:1030-1285` - Complete GFI implementation (simulate, assess, generate, update, regenerate)
- `core.py:537-630` - Trace interface design
- General architecture throughout the module

**Key Concepts**:
- Generative Function Interface (GFI) for composable probabilistic models
- Trace-based execution model
- Programmable inference through GFI methods

---

## Probability Distributions

## [2] Johnson, N. L., Kotz, S., & Kemp, A. W. (1992). "Univariate Discrete Distributions"
**Publisher**: John Wiley & Sons, New York, 2nd edition
**ISBN**: 0-471-54897-9
**Abstract**: Comprehensive reference on discrete probability distributions

**Used in**:
- `distributions.py:42-44` - Bernoulli distribution
- `distributions.py:149-151` - Binomial distribution
- `distributions.py:287-290` - Poisson distribution
- `distributions.py:402-404` - Geometric distribution

**Key Concepts**:
- Bernoulli distribution (Chapter 3)
- Binomial distribution (Chapter 5)
- Poisson distribution (Chapter 4)
- Geometric distribution (Chapter 3)

---

## [3] Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). "Continuous Univariate Distributions, Volume 1"
**Publisher**: John Wiley & Sons, New York, 2nd edition
**ISBN**: 0-471-58495-9
**Abstract**: Comprehensive reference on continuous probability distributions

**Used in**:
- `distributions.py:442-444` - Uniform distribution (Chapter 17)

---

## [4] Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). "Continuous Univariate Distributions, Volume 2"
**Publisher**: John Wiley & Sons, New York, 2nd edition
**ISBN**: 0-471-58494-0
**Abstract**: Continuation of continuous distribution reference

**Used in**:
- `distributions.py:220-222` - Gamma distribution (Chapter 26)

---

## [5] Gupta, A. K., & Nadarajah, S. (2004). "Handbook of Beta Distribution and Its Applications"
**Publisher**: CRC Press
**ISBN**: 0-8247-5396-6
**Abstract**: Comprehensive treatment of beta distribution theory and applications

**Used in**:
- `distributions.py:83-85` - Beta distribution implementation

**Key Concepts**:
- Properties and parameterizations of beta distribution
- Relationships to other distributions
- Applications in Bayesian inference

---

## [6] Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
**Publisher**: Springer, New York
**ISBN**: 978-0-387-31073-2
**Abstract**: Comprehensive textbook on machine learning with probabilistic foundations

**Used in**:
- `distributions.py:115-117` - Categorical distribution (Section 2.2)

**Key Concepts**:
- Discrete probability distributions
- Multinomial and categorical distributions
- Bayesian inference foundations

---

## [7] Patel, J. K., & Read, C. B. (1996). "Handbook of the Normal Distribution"
**Publisher**: Marcel Dekker, 2nd edition
**ISBN**: 0-8247-9342-0
**Abstract**: Comprehensive reference on normal distribution

**Used in**:
- `distributions.py:185-187` - Normal (Gaussian) distribution

**Key Concepts**:
- Properties of univariate normal distribution
- Computational methods
- Statistical applications

---

## [8] Haight, F. A. (1967). "Handbook of the Poisson Distribution"
**Publisher**: John Wiley & Sons, New York
**ISBN**: 978-0471333326
**Abstract**: Dedicated reference for Poisson distribution

**Used in**:
- `distributions.py:287-290` - Poisson distribution implementation

---

## [9] Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). "Multivariate Analysis"
**Publisher**: Academic Press, London
**ISBN**: 0-12-471250-9
**Abstract**: Classic text on multivariate statistical analysis

**Used in**:
- `distributions.py:325-328` - Multivariate normal distribution (Chapter 3)

**Key Concepts**:
- Multivariate normal theory
- Covariance matrix properties
- Multivariate statistical inference

---

## [10] Tong, Y. L. (1990). "The Multivariate Normal Distribution"
**Publisher**: Springer-Verlag, New York
**ISBN**: 0-387-97062-0
**Abstract**: Comprehensive treatment of multivariate normal distribution

**Used in**:
- `distributions.py:325-328` - Multivariate normal distribution

---

## [11] Kotz, S., Balakrishnan, N., & Johnson, N. L. (2000). "Continuous Multivariate Distributions, Volume 1"
**Publisher**: John Wiley & Sons, New York, 2nd edition
**ISBN**: 0-471-18387-3
**Abstract**: Reference on continuous multivariate distributions

**Used in**:
- `distributions.py:362-365` - Dirichlet distribution (Chapter 49)

**Key Concepts**:
- Dirichlet distribution properties
- Relationships to beta distribution
- Applications in compositional data

---

## [12] Ng, K. W., Tian, G. L., & Tang, M. L. (2011). "Dirichlet and Related Distributions: Theory, Methods and Applications"
**Publisher**: John Wiley & Sons
**ISBN**: 978-0-470-68819-9
**Abstract**: Modern comprehensive treatment of Dirichlet distribution

**Used in**:
- `distributions.py:362-365` - Dirichlet distribution

---

## [13] Crow, E. L., & Shimizu, K. (Eds.). (1988). "Lognormal Distributions: Theory and Applications"
**Publisher**: Marcel Dekker, New York
**ISBN**: 0-8247-7803-0
**Abstract**: Comprehensive reference on log-normal distribution

**Used in**:
- `distributions.py:479-482` - Log-normal distribution

---

## [14] Limpert, E., Stahel, W. A., & Abbt, M. (2001). "Log-normal Distributions across the Sciences: Keys and Clues"
**Journal**: BioScience, 51(5), 341-352
**DOI**: 10.1641/0006-3568(2001)051[0341:LNDATS]2.0.CO;2
**Abstract**: Review of log-normal distribution applications across scientific disciplines

**Used in**:
- `distributions.py:479-482` - Log-normal distribution

---

## [15] Lange, K. L., Little, R. J., & Taylor, J. M. (1989). "Robust statistical modeling using the t distribution"
**Journal**: Journal of the American Statistical Association (JASA), 84(408), 881-896
**DOI**: 10.1080/01621459.1989.10478852
**Abstract**: Robust statistical methods using Student's t distribution

**Used in**:
- `distributions.py:520-523` - Student's t distribution

**Key Concepts**:
- Robust alternatives to normal distribution
- Heavy-tailed distributions
- Parameter estimation methods

---

## [16] Kotz, S., & Nadarajah, S. (2004). "Multivariate t-distributions and their applications"
**Publisher**: Cambridge University Press
**ISBN**: 0-521-82654-3
**Abstract**: Comprehensive treatment of multivariate t-distributions

**Used in**:
- `distributions.py:520-523` - Student's t distribution

---

## JAX and Implementation

## [17] JAX Development Team. "How JAX primitives work"
**Documentation**: JAX official documentation
**URL**: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
**Abstract**: Tutorial on implementing custom JAX primitives and transformations

**Used in**:
- `pjax.py:75-78` - PJAX primitive implementation reference
- `pjax.py` - General JAX transformation patterns

**Key Concepts**:
- Custom JAX primitives (sample_p, log_density_p)
- Transformation rules for automatic differentiation
- Integration with JAX's compilation infrastructure

---

## Related Work

## [18] van de Meent, J. W., Paige, B., Yang, H., & Wood, F. (2018). "An introduction to probabilistic programming"
**ArXiv**: arXiv:1809.10756
**Abstract**: Comprehensive introduction to probabilistic programming concepts and systems

**Related to**:
- General probabilistic programming concepts throughout GenJAX
- Design patterns for probabilistic primitives
- Inference algorithm integration

---

## Implementation Notes

The GenJAX core module implements the Generative Function Interface (GFI) from Gen.jl [1] in JAX, providing:

1. **Full GFI implementation** with all required methods (simulate, assess, generate, update, regenerate)
2. **Comprehensive distribution library** with proper citations to statistical literature
3. **PJAX infrastructure** for probabilistic JAX transformations
4. **State interpreter** for diagnostic and debugging support

All distributions follow the parameterizations specified in their respective references, with adaptations for JAX's computational model and automatic differentiation requirements.
