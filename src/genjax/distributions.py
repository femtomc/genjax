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

Args:
    logits: Log-odds of success, or
    probs: Probability of success.
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

Args:
    concentration1: Alpha parameter (> 0).
    concentration0: Beta parameter (> 0).
"""

categorical = tfp_distribution(
    lambda logits: tfd.Categorical(logits),
    name="Categorical",
)
"""Categorical distribution over discrete outcomes.

Args:
    logits: Log-probabilities for each category.
"""

geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)
"""Geometric distribution (number of trials until first success).

Args:
    logits: Log-odds of success, or
    probs: Probability of success.
"""


normal = tfp_distribution(
    tfd.Normal,
    name="Normal",
)
"""Normal (Gaussian) distribution.

Args:
    loc: Mean of the distribution.
    scale: Standard deviation (> 0).
"""

uniform = tfp_distribution(
    tfd.Uniform,
    name="Uniform",
)
"""Uniform distribution on an interval.

Args:
    low: Lower bound of the distribution.
    high: Upper bound of the distribution.
"""

exponential = tfp_distribution(
    tfd.Exponential,
    name="Exponential",
)
"""Exponential distribution for positive continuous values.

Args:
    rate: Rate parameter (> 0), or
    scale: Scale parameter (1/rate).
"""

poisson = tfp_distribution(
    tfd.Poisson,
    name="Poisson",
)
"""Poisson distribution for count data.

Args:
    rate: Expected number of events (lambda parameter), or
    log_rate: Log of the rate parameter.
"""

multivariate_normal = tfp_distribution(
    tfd.MultivariateNormalFullCovariance,
    name="MultivariateNormal",
)
"""Multivariate normal distribution.

Args:
    loc: Mean vector.
    covariance_matrix: Covariance matrix (positive definite).
"""

dirichlet = tfp_distribution(
    tfd.Dirichlet,
    name="Dirichlet",
)
"""Dirichlet distribution for probability vectors.

Args:
    concentration: Concentration parameters (all > 0).
                  Shape determines the dimension of the distribution.
"""

# High-priority additional distributions

binomial = tfp_distribution(
    tfd.Binomial,
    name="Binomial",
)
"""Binomial distribution for count data with fixed number of trials.

Args:
    total_count: Number of trials (non-negative integer).
    logits: Log-odds of success, or
    probs: Probability of success per trial.
"""

gamma = tfp_distribution(
    tfd.Gamma,
    name="Gamma",
)
"""Gamma distribution for positive continuous values.

Args:
    concentration: Shape parameter (alpha > 0).
    rate: Rate parameter (beta > 0), or
    scale: Scale parameter (1/rate).
"""

log_normal = tfp_distribution(
    tfd.LogNormal,
    name="LogNormal",
)
"""Log-normal distribution (exponential of normal random variable).

Args:
    loc: Mean of underlying normal distribution.
    scale: Standard deviation of underlying normal distribution (> 0).
"""

student_t = tfp_distribution(
    tfd.StudentT,
    name="StudentT",
)
"""Student's t-distribution with specified degrees of freedom.

Args:
    df: Degrees of freedom (> 0).
    loc: Location parameter (default 0).
    scale: Scale parameter (> 0, default 1).
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
