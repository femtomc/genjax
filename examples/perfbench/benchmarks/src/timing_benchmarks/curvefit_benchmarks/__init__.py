"""Benchmark implementations for different frameworks."""

from .genjax import genjax_polynomial_is_timing, genjax_polynomial_hmc_timing
from .numpyro import numpyro_polynomial_is_timing
from .handcoded_jax import handcoded_jax_polynomial_is_timing

# Pyro requires separate environment due to PyTorch dependencies
# Import will be handled at runtime when using pyro-specific tasks

__all__ = [
    "genjax_polynomial_is_timing",
    "genjax_polynomial_hmc_timing",
    "numpyro_polynomial_is_timing",
    "handcoded_jax_polynomial_is_timing",
]
