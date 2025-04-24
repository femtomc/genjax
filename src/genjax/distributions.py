import jax.numpy as jnp
from jax import vmap
from tensorflow_probability.substrates import jax as tfp

from genjax.core import (
    Callable,
    Distribution,
    X,
    distribution,
    tfp_distribution,
    wrap_logpdf,
    wrap_sampler,
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
    tfd.Categorical,
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
