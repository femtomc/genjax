"""Standard probability distributions for GenJAX.

This module provides a collection of common probability distributions
wrapped as GenJAX Distribution objects. All distributions are built
using TensorFlow Probability as the backend.
"""

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax.core import (
    tfp_distribution,
)

tfd = tfp.distributions

# Discrete distributions
bernoulli = tfp_distribution(
    tfd.Bernoulli,
    name="Bernoulli",
)
"""Bernoulli distribution for binary outcomes.

Mathematical Formulation:
    PMF: P(X = k) = p^k × (1-p)^(1-k) for k ∈ {0, 1}

    Where p is the probability of success.

    Mean: 𝔼[X] = p
    Variance: Var[X] = p(1-p)
    Support: {0, 1}

Parameterization:
    Can be specified via:
    - probs: p ∈ [0, 1] (probability of success)
    - logits: log(p/(1-p)) ∈ ℝ (log-odds)

Args:
    logits: Log-odds of success, or
    probs: Probability of success.

References:
    .. [1] Johnson, N. L., Kotz, S., & Kemp, A. W. (1992). "Univariate
           Discrete Distributions". Wiley, Chapter 3.
"""

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    name="Flip",
)
"""Flip distribution (Bernoulli with boolean output).

Args:
    p: Probability of True outcome.
"""

# Continuous distributions
beta = tfp_distribution(
    tfd.Beta,
    name="Beta",
)
"""Beta distribution on the interval [0, 1].

Mathematical Formulation:
    PDF: f(x; α, β) = Γ(α+β)/(Γ(α)Γ(β)) × x^(α-1) × (1-x)^(β-1)

    Where Γ is the gamma function, α > 0, β > 0.

    Mean: 𝔼[X] = α/(α+β)
    Variance: Var[X] = αβ/((α+β)²(α+β+1))
    Mode: (α-1)/(α+β-2) for α,β > 1
    Support: [0, 1]

Special Cases:
    - Beta(1, 1) = Uniform(0, 1)
    - Beta(α, α) is symmetric about 0.5
    - As α,β → ∞ with α/(α+β) fixed, approaches Normal

Args:
    concentration1: Alpha parameter α (> 0).
    concentration0: Beta parameter β (> 0).

References:
    .. [1] Gupta, A. K., & Nadarajah, S. (2004). "Handbook of Beta
           Distribution and Its Applications". CRC Press.
"""

categorical = tfp_distribution(
    lambda logits: tfd.Categorical(logits),
    name="Categorical",
)
"""Categorical distribution over discrete outcomes.

Mathematical Formulation:
    PMF: P(X = k) = p_k for k ∈ {0, 1, ..., K-1}

    Where ∑_k p_k = 1 and p_k ≥ 0.

    Mean: 𝔼[X] = ∑_k k × p_k
    Variance: Var[X] = ∑_k k² × p_k - (𝔼[X])²
    Entropy: H[X] = -∑_k p_k log(p_k)
    Support: {0, 1, ..., K-1}

Parameterization:
    - logits: θ_k ∈ ℝ, where p_k = exp(θ_k) / ∑_j exp(θ_j)
    - Softmax transformation ensures valid probabilities

Connection to Other Distributions:
    - K=2: Equivalent to Bernoulli
    - Generalization of multinomial for single trial

Args:
    logits: Log-probabilities θ for each category.

References:
    .. [1] Bishop, C. M. (2006). "Pattern Recognition and Machine Learning".
           Springer, Section 2.2.
"""

geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)
"""Geometric distribution (number of trials until first success).

Mathematical Formulation:
    PMF: P(X = k) = (1-p)^(k-1) × p for k ∈ {1, 2, 3, ...}

    Where p ∈ (0, 1] is the probability of success.

    Mean: 𝔼[X] = 1/p
    Variance: Var[X] = (1-p)/p²
    CDF: F(k) = 1 - (1-p)^k
    Support: {1, 2, 3, ...}

Memoryless Property:
    P(X > m + n | X > m) = P(X > n)

    The only discrete distribution with this property.

Alternative Parameterization:
    Some define X as failures before first success:
    P(X = k) = (1-p)^k × p for k ∈ {0, 1, 2, ...}

Args:
    logits: Log-odds of success log(p/(1-p)), or
    probs: Probability of success p.

References:
    .. [1] Johnson, N. L., Kotz, S., & Kemp, A. W. (1992). "Univariate
           Discrete Distributions". Wiley, Chapter 5.
"""


normal = tfp_distribution(
    tfd.Normal,
    name="Normal",
)
"""Normal (Gaussian) distribution.

Mathematical Formulation:
    PDF: f(x; μ, σ) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

    Where μ ∈ ℝ is the mean, σ > 0 is the standard deviation.

    Mean: 𝔼[X] = μ
    Variance: Var[X] = σ²
    MGF: M(t) = exp(μt + σ²t²/2)
    Support: ℝ

Standard Normal:
    Z = (X - μ)/σ ~ N(0, 1)

    Φ(z) = P(Z ≤ z) = ∫_{-∞}^z (1/√(2π)) exp(-t²/2) dt

Properties:
    - Maximum entropy distribution for fixed mean and variance
    - Stable under convolution: X₁ + X₂ ~ N(μ₁+μ₂, σ₁²+σ₂²)
    - Central Limit Theorem: Sample means converge to Normal

Args:
    loc: Mean of the distribution μ.
    scale: Standard deviation σ (> 0).

References:
    .. [1] Patel, J. K., & Read, C. B. (1996). "Handbook of the Normal
           Distribution". Marcel Dekker, 2nd edition.
"""

uniform = tfp_distribution(
    tfd.Uniform,
    name="Uniform",
)
"""Uniform distribution on an interval.

Mathematical Formulation:
    PDF: f(x; a, b) = 1/(b-a) for x ∈ [a, b], 0 otherwise

    Where a < b define the support interval.

    Mean: 𝔼[X] = (a + b)/2
    Variance: Var[X] = (b - a)²/12
    CDF: F(x) = (x - a)/(b - a) for x ∈ [a, b]
    Support: [a, b]

Properties:
    - Maximum entropy distribution on bounded interval
    - All moments exist: 𝔼[X^n] = (b^(n+1) - a^(n+1))/((n+1)(b-a))
    - Order statistics have Beta distributions

Connection to Other Distributions:
    - Standard uniform U(0,1) generates other distributions
    - -log(U) ~ Exponential(1)
    - U^(1/α) ~ Power distribution

Args:
    low: Lower bound a of the distribution.
    high: Upper bound b of the distribution (> low).

References:
    .. [1] Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995).
           "Continuous Univariate Distributions". Wiley, Vol. 2, Chapter 26.
"""

exponential = tfp_distribution(
    tfd.Exponential,
    name="Exponential",
)
"""Exponential distribution for positive continuous values.

Mathematical Formulation:
    PDF: f(x; λ) = λ exp(-λx) for x ≥ 0

    Where λ > 0 is the rate parameter.

    Mean: 𝔼[X] = 1/λ
    Variance: Var[X] = 1/λ²
    CDF: F(x) = 1 - exp(-λx)
    Support: [0, ∞)

Memoryless Property:
    P(X > s + t | X > s) = P(X > t)

    The only continuous distribution with this property.

Connection to Other Distributions:
    - Special case of Gamma(1, λ)
    - -log(U) ~ Exponential(1) where U ~ Uniform(0,1)
    - Minimum of n Exponential(λ) ~ Exponential(nλ)
    - Sum of n Exponential(λ) ~ Gamma(n, λ)

Args:
    rate: Rate parameter (> 0), or
    scale: Scale parameter (1/rate).
"""

poisson = tfp_distribution(
    tfd.Poisson,
    name="Poisson",
)
"""Poisson distribution for count data.

Mathematical Formulation:
    PMF: P(X = k) = (λ^k / k!) × exp(-λ) for k ∈ {0, 1, 2, ...}

    Where λ > 0 is the rate parameter (expected count).

    Mean: 𝔼[X] = λ
    Variance: Var[X] = λ
    MGF: M(t) = exp(λ(e^t - 1))
    Support: {0, 1, 2, ...}

Properties:
    - Mean equals variance (equidispersion)
    - Sum of Poissons: X₁ ~ Pois(λ₁), X₂ ~ Pois(λ₂) ⇒ X₁+X₂ ~ Pois(λ₁+λ₂)
    - Limit of Binomial: Bin(n,p) → Pois(np) as n→∞, p→0, np=λ

Connection to Other Distributions:
    - Poisson process: Inter-arrival times ~ Exponential(λ)
    - Large λ: Approximately Normal(λ, λ)
    - Conditional on rate: If λ ~ Gamma(α,β), then X ~ NegBin(α, β/(1+β))

Args:
    rate: Expected number of events λ (> 0), or
    log_rate: Log of the rate parameter log(λ).

References:
    .. [1] Johnson, N. L., Kotz, S., & Kemp, A. W. (1992). "Univariate
           Discrete Distributions". Wiley, Chapter 4.
    .. [2] Haight, F. A. (1967). "Handbook of the Poisson Distribution".
           Wiley.
"""

multivariate_normal = tfp_distribution(
    tfd.MultivariateNormalFullCovariance,
    name="MultivariateNormal",
)
"""Multivariate normal distribution.

Mathematical Formulation:
    PDF: f(x; μ, Σ) = (2π)^(-k/2) |det(Σ)|^(-1/2) exp(-½(x-μ)^T Σ^(-1) (x-μ))

    Where μ ∈ ℝ^k is the mean vector, Σ is k×k positive definite covariance.

    Mean: 𝔼[X] = μ
    Covariance: Cov[X] = Σ
    MGF: M(t) = exp(t^Tμ + ½t^TΣt)
    Support: ℝ^k

Properties:
    - Linear transformations: If Y = AX + b, then Y ~ N(Aμ + b, AΣA^T)
    - Marginals are Normal: X_i ~ N(μ_i, Σ_{ii})
    - Conditional distributions are Normal with closed-form parameters
    - Maximum entropy for fixed mean and covariance

Special Cases:
    - Σ = σ²I: Spherical/isotropic Gaussian
    - Σ diagonal: Independent components
    - k = 1: Univariate normal

Args:
    loc: Mean vector μ of shape (k,).
    covariance_matrix: Covariance matrix Σ of shape (k, k) (positive definite).

References:
    .. [1] Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). "Multivariate
           Analysis". Academic Press, Chapter 3.
    .. [2] Tong, Y. L. (1990). "The Multivariate Normal Distribution".
           Springer-Verlag.
"""

dirichlet = tfp_distribution(
    tfd.Dirichlet,
    name="Dirichlet",
)
"""Dirichlet distribution for probability vectors.

Mathematical Formulation:
    PDF: f(x; α) = [Γ(∑ᵢαᵢ)/∏ᵢΓ(αᵢ)] × ∏ᵢ xᵢ^(αᵢ-1)

    Where x ∈ δ_{k-1} (probability simplex), αᵢ > 0 are concentrations.

    Mean: 𝔼[Xᵢ] = αᵢ / ∑ⱼαⱼ
    Variance: Var[Xᵢ] = [αᵢ(α₀-αᵢ)] / [α₀²(α₀+1)], where α₀ = ∑ⱼαⱼ
    Support: δ_{k-1} = {x ∈ ℝ^k : xᵢ ≥ 0, ∑ᵢxᵢ = 1}

Properties:
    - Conjugate prior for categorical/multinomial
    - Marginals: Xᵢ ~ Beta(αᵢ, ∑ⱼ≠ᵢαⱼ)
    - Aggregation property: (Xᵢ + Xⱼ, X_rest) follows lower-dim Dirichlet
    - Neutral element: Dir(1, 1, ..., 1) = Uniform on simplex

Connection to Other Distributions:
    - k=2: Dir(α₁, α₂) equivalent to Beta(α₁, α₂)
    - Gamma construction: If Yᵢ ~ Gamma(αᵢ, 1), then Y/∑Y ~ Dir(α)
    - Log-normal approximation for large α

Args:
    concentration: Concentration parameters α (all > 0).
                  Shape (k,) determines the dimension of the distribution.

References:
    .. [1] Kotz, S., Balakrishnan, N., & Johnson, N. L. (2000). "Continuous
           Multivariate Distributions". Wiley, Vol. 1, Chapter 49.
    .. [2] Ng, K. W., Tian, G. L., & Tang, M. L. (2011). "Dirichlet and
           Related Distributions". Wiley.
"""

# High-priority additional distributions

binomial = tfp_distribution(
    tfd.Binomial,
    name="Binomial",
)
"""Binomial distribution for count data with fixed number of trials.

Mathematical Formulation:
    PMF: P(X = k) = C(n,k) × p^k × (1-p)^(n-k) for k ∈ {0, 1, ..., n}

    Where n is the number of trials, p is success probability,
    and C(n,k) = n!/(k!(n-k)!) is the binomial coefficient.

    Mean: 𝔼[X] = np
    Variance: Var[X] = np(1-p)
    MGF: M(t) = (1 - p + pe^t)^n
    Support: {0, 1, 2, ..., n}

Properties:
    - Sum of Bernoulli: X = ∑ᵢ Yᵢ where Yᵢ ~ Bernoulli(p)
    - Additivity: Bin(n₁,p) + Bin(n₂,p) = Bin(n₁+n₂,p)
    - Symmetry: If p = 0.5, then P(X = k) = P(X = n-k)

Approximations:
    - Normal: For large n, np(1-p) > 10, approximately N(np, np(1-p))
    - Poisson: For large n, small p, np = λ moderate, approximately Pois(λ)

Args:
    total_count: Number of trials n (≥ 0, integer).
    logits: Log-odds of success log(p/(1-p)), or
    probs: Probability of success per trial p ∈ [0,1].

References:
    .. [1] Johnson, N. L., Kotz, S., & Kemp, A. W. (1992). "Univariate
           Discrete Distributions". Wiley, Chapter 3.
"""

gamma = tfp_distribution(
    tfd.Gamma,
    name="Gamma",
)
"""Gamma distribution for positive continuous values.

Mathematical Formulation:
    PDF: f(x; α, β) = (β^α / Γ(α)) × x^(α-1) × exp(-βx) for x > 0

    Where α > 0 is the shape, β > 0 is the rate (or θ = 1/β is scale).

    Mean: 𝔼[X] = α/β = αθ
    Variance: Var[X] = α/β² = αθ²
    Mode: (α-1)/β for α ≥ 1
    Support: (0, ∞)

Special Cases:
    - α = 1: Exponential(β)
    - α = k/2, β = 1/2: Chi-squared(k)
    - Integer α: Erlang distribution

Properties:
    - Additivity: Gamma(α₁,β) + Gamma(α₂,β) = Gamma(α₁+α₂,β)
    - Scaling: cX ~ Gamma(α, β/c) for c > 0
    - Conjugate prior for Poisson rate, exponential rate

Connection to Other Distributions:
    - If Xᵢ ~ Gamma(αᵢ, 1), then Xᵢ/∑Xⱼ ~ Dirichlet(α)
    - Inverse: 1/X ~ InverseGamma(α, β)

Args:
    concentration: Shape parameter α (> 0).
    rate: Rate parameter β (> 0), or
    scale: Scale parameter θ = 1/β.

References:
    .. [1] Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). "Continuous
           Univariate Distributions". Wiley, Vol. 1, Chapter 17.
"""

log_normal = tfp_distribution(
    tfd.LogNormal,
    name="LogNormal",
)
"""Log-normal distribution (exponential of normal random variable).

Mathematical Formulation:
    If Y ~ N(μ, σ²), then X = exp(Y) ~ LogNormal(μ, σ²)

    PDF: f(x; μ, σ) = (1/(xσ√(2π))) × exp(-(ln(x)-μ)²/(2σ²)) for x > 0

    Mean: 𝔼[X] = exp(μ + σ²/2)
    Variance: Var[X] = (exp(σ²) - 1) × exp(2μ + σ²)
    Mode: exp(μ - σ²)
    Support: (0, ∞)

Properties:
    - Multiplicative: If Xᵢ ~ LogN(μᵢ, σᵢ²) independent, then ∏Xᵢ is log-normal
    - Not closed under addition (sum of log-normals is not log-normal)
    - Heavy right tail: all moments exist but grow rapidly
    - Median: exp(μ)

Applications:
    - Income distributions
    - Stock prices (geometric Brownian motion)
    - Particle sizes
    - Species abundance

Args:
    loc: Mean μ of underlying normal distribution.
    scale: Standard deviation σ of underlying normal (> 0).

References:
    .. [1] Crow, E. L., & Shimizu, K. (Eds.). (1988). "Lognormal Distributions:
           Theory and Applications". Marcel Dekker.
    .. [2] Limpert, E., Stahel, W. A., & Abbt, M. (2001). "Log-normal
           distributions across the sciences". BioScience, 51(5), 341-352.
"""

student_t = tfp_distribution(
    tfd.StudentT,
    name="StudentT",
)
"""Student's t-distribution with specified degrees of freedom.

Mathematical Formulation:
    PDF: f(x; ν, μ, σ) = Γ((ν+1)/2)/(Γ(ν/2)√(νπ)σ) × [1 + ((x-μ)/σ)²/ν]^(-(ν+1)/2)

    Where ν > 0 is degrees of freedom, μ is location, σ > 0 is scale.

    Mean: 𝔼[X] = μ for ν > 1 (undefined for ν ≤ 1)
    Variance: Var[X] = σ²ν/(ν-2) for ν > 2 (infinite for 1 < ν ≤ 2)
    Support: ℝ

Properties:
    - Heavier tails than normal (polynomial vs exponential decay)
    - ν → ∞: Converges to Normal(μ, σ²)
    - ν = 1: Cauchy distribution (no mean)
    - ν = 2: Finite mean but infinite variance
    - Symmetric about μ

Standardized Form:
    If T ~ t(ν), then X = μ + σT ~ t(ν, μ, σ)

Connection to Other Distributions:
    - Ratio of normal to chi: If Z ~ N(0,1), V ~ χ²(ν), then Z/√(V/ν) ~ t(ν)
    - F-distribution: T² ~ F(1, ν) if T ~ t(ν)

Args:
    df: Degrees of freedom ν (> 0).
    loc: Location parameter μ (default 0).
    scale: Scale parameter σ (> 0, default 1).

References:
    .. [1] Lange, K. L., Little, R. J., & Taylor, J. M. (1989). "Robust
           statistical modeling using the t distribution". JASA, 84(408), 881-896.
    .. [2] Kotz, S., & Nadarajah, S. (2004). "Multivariate t-distributions
           and their applications". Cambridge University Press.
"""

laplace = tfp_distribution(
    tfd.Laplace,
    name="Laplace",
)
"""Laplace (double exponential) distribution.

Args:
    loc: Location parameter (median).
    scale: Scale parameter (> 0).
"""

half_normal = tfp_distribution(
    tfd.HalfNormal,
    name="HalfNormal",
)
"""Half-normal distribution (positive half of normal distribution).

Args:
    scale: Scale parameter (> 0).
"""

inverse_gamma = tfp_distribution(
    tfd.InverseGamma,
    name="InverseGamma",
)
"""Inverse gamma distribution for positive continuous values.

Args:
    concentration: Shape parameter (alpha > 0).
    rate: Rate parameter (beta > 0), or
    scale: Scale parameter (1/rate).
"""

weibull = tfp_distribution(
    tfd.Weibull,
    name="Weibull",
)
"""Weibull distribution for modeling survival times and reliability.

Args:
    concentration: Shape parameter (k > 0).
    scale: Scale parameter (lambda > 0).
"""

cauchy = tfp_distribution(
    tfd.Cauchy,
    name="Cauchy",
)
"""Cauchy distribution with heavy tails.

Args:
    loc: Location parameter (median).
    scale: Scale parameter (> 0).
"""

chi2 = tfp_distribution(
    tfd.Chi2,
    name="Chi2",
)
"""Chi-squared distribution.

Args:
    df: Degrees of freedom (> 0).
"""

multinomial = tfp_distribution(
    tfd.Multinomial,
    name="Multinomial",
)
"""Multinomial distribution over count vectors.

Args:
    total_count: Total number of trials.
    logits: Log-probabilities for each category, or
    probs: Probabilities for each category (must sum to 1).
"""

negative_binomial = tfp_distribution(
    tfd.NegativeBinomial,
    name="NegativeBinomial",
)
"""Negative binomial distribution for overdispersed count data.

Args:
    total_count: Number of successes (> 0).
    logits: Log-odds of success, or
    probs: Probability of success per trial.
"""

zipf = tfp_distribution(
    tfd.Zipf,
    name="Zipf",
)
"""Zipf distribution for power-law distributed discrete data.

Args:
    power: Power parameter (> 1).
    dtype: Integer dtype for samples (default int32).
"""
