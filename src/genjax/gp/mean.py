"""Mean functions for Gaussian processes."""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float, Array
from genjax import Pytree


class MeanFunction(Pytree, ABC):
    """Abstract base class for GP mean functions."""

    @abstractmethod
    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        """Compute mean function at input locations."""
        pass


@Pytree.dataclass
class Zero(MeanFunction):
    """Zero mean function.

    m(x) = 0
    """

    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        return jnp.zeros(x.shape[0])


@Pytree.dataclass
class Constant(MeanFunction):
    """Constant mean function.

    m(x) = c
    """

    value: Float[Array, ""]

    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        return jnp.full(x.shape[0], self.value)


@Pytree.dataclass
class Linear(MeanFunction):
    """Linear mean function.

    m(x) = wᵀx + b
    """

    weights: Float[Array, "d"]
    bias: Float[Array, ""]

    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n"]:
        return jnp.dot(x, self.weights) + self.bias
