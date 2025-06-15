import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax.core import (
    tfp_distribution,
)

tfd = tfp.distributions

bernoulli = tfp_distribution(
    tfd.Bernoulli,
    name="Bernoulli",
)

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    name="Flip",
)

beta = tfp_distribution(
    tfd.Beta,
    name="Beta",
)

categorical = tfp_distribution(
    lambda logits: tfd.Categorical(logits),
    name="Categorical",
)

geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)


normal = tfp_distribution(
    tfd.Normal,
    name="Normal",
)

uniform = tfp_distribution(
    tfd.Uniform,
    name="Uniform",
)

exponential = tfp_distribution(
    tfd.Exponential,
    name="Exponential",
)

poisson = tfp_distribution(
    tfd.Poisson,
    name="Poisson",
)

multivariate_normal = tfp_distribution(
    tfd.MultivariateNormalFullCovariance,
    name="MultivariateNormal",
)
