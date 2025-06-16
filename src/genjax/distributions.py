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
