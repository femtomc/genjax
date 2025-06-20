"""Kernel functions for Gaussian processes."""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float, Array
from genjax import Pytree


class Kernel(Pytree, ABC):
    """Abstract base class for GP kernel functions."""

    @abstractmethod
    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        """Compute kernel matrix between two sets of inputs."""
        pass


@Pytree.dataclass
class RBF(Kernel):
    """Radial Basis Function (RBF/Squared Exponential) kernel.

    k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""] | Float[Array, "d"]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        # Compute squared distances
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        sq_dists = jnp.sum(
            (x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=-1
        )
        return self.variance * jnp.exp(-0.5 * sq_dists)


@Pytree.dataclass
class Matern12(Kernel):
    """Matérn 1/2 kernel (Exponential kernel).

    k(x, x') = σ² exp(-||x - x'|| / ℓ)
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""] | Float[Array, "d"]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        dists = jnp.sqrt(
            jnp.sum((x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=-1)
        )
        return self.variance * jnp.exp(-dists)


@Pytree.dataclass
class Matern32(Kernel):
    """Matérn 3/2 kernel.

    k(x, x') = σ² (1 + √3||x - x'|| / ℓ) exp(-√3||x - x'|| / ℓ)
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""] | Float[Array, "d"]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        dists = jnp.sqrt(
            jnp.sum((x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=-1)
        )
        sqrt3_dists = jnp.sqrt(3.0) * dists
        return self.variance * (1.0 + sqrt3_dists) * jnp.exp(-sqrt3_dists)


@Pytree.dataclass
class Matern52(Kernel):
    """Matérn 5/2 kernel.

    k(x, x') = σ² (1 + √5||x - x'|| / ℓ + 5||x - x'||² / (3ℓ²)) exp(-√5||x - x'|| / ℓ)
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""] | Float[Array, "d"]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale
        sq_dists = jnp.sum(
            (x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=-1
        )
        dists = jnp.sqrt(sq_dists)
        sqrt5_dists = jnp.sqrt(5.0) * dists
        return (
            self.variance
            * (1.0 + sqrt5_dists + 5.0 * sq_dists / 3.0)
            * jnp.exp(-sqrt5_dists)
        )


@Pytree.dataclass
class Linear(Kernel):
    """Linear kernel.

    k(x, x') = σ² (x - c)ᵀ(x' - c)
    """

    variance: Float[Array, ""]
    offset: Float[Array, "d"]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        x1_centered = x1 - self.offset
        x2_centered = x2 - self.offset
        return self.variance * jnp.dot(x1_centered, x2_centered.T)


@Pytree.dataclass
class Polynomial(Kernel):
    """Polynomial kernel.

    k(x, x') = (σ² xᵀx' + c)^d
    """

    variance: Float[Array, ""]
    offset: Float[Array, ""]
    degree: int = Pytree.static()

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        return (self.variance * jnp.dot(x1, x2.T) + self.offset) ** self.degree


@Pytree.dataclass
class White(Kernel):
    """White noise kernel.

    k(x, x') = σ² δ(x, x')
    """

    variance: Float[Array, ""]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        if x1.shape == x2.shape and jnp.allclose(x1, x2):
            return self.variance * jnp.eye(x1.shape[0])
        else:
            return jnp.zeros((x1.shape[0], x2.shape[0]))


@Pytree.dataclass
class Constant(Kernel):
    """Constant kernel.

    k(x, x') = σ²
    """

    variance: Float[Array, ""]

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        return self.variance * jnp.ones((x1.shape[0], x2.shape[0]))


@Pytree.dataclass
class Sum(Kernel):
    """Sum of two kernels.

    k(x, x') = k1(x, x') + k2(x, x')
    """

    k1: Kernel
    k2: Kernel

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        return self.k1(x1, x2) + self.k2(x1, x2)


@Pytree.dataclass
class Product(Kernel):
    """Product of two kernels.

    k(x, x') = k1(x, x') * k2(x, x')
    """

    k1: Kernel
    k2: Kernel

    def __call__(
        self, x1: Float[Array, "n1 d"], x2: Float[Array, "n2 d"]
    ) -> Float[Array, "n1 n2"]:
        return self.k1(x1, x2) * self.k2(x1, x2)
