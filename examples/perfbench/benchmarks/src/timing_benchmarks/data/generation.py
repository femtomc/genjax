"""Polynomial regression dataset used across the perfbench benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import jax.random as jrand


@dataclass
class PolynomialDataset:
    xs: jnp.ndarray
    ys: jnp.ndarray
    true_a: float
    true_b: float
    true_c: float
    noise_std: float
    n_points: int


def polyfn(xs: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the quadratic polynomial a + b x + c x^2."""
    a, b, c = coeffs
    return a + b * xs + c * xs**2


def generate_polynomial_data(
    n_points: int = 50,
    seed: int = 42,
    coeffs: Optional[jnp.ndarray] = None,
    noise_std: float = 0.05,
) -> PolynomialDataset:
    """Create the shared polynomial regression dataset."""
    key = jrand.key(seed)
    xs = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float32)
    if coeffs is None:
        coeffs = jnp.array([0.5, -1.25, 2.0], dtype=jnp.float32)
    clean = polyfn(xs, coeffs)
    noise = noise_std * jrand.normal(key, shape=(n_points,), dtype=jnp.float32)
    ys = clean + noise
    return PolynomialDataset(
        xs=xs,
        ys=ys,
        true_a=float(coeffs[0]),
        true_b=float(coeffs[1]),
        true_c=float(coeffs[2]),
        noise_std=noise_std,
        n_points=n_points,
    )
