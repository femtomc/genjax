"""Benchmark implementations for different frameworks."""

from .genjax import (
    genjax_polynomial_is_timing,
    genjax_polynomial_hmc_timing,
    genjax_polynomial_is_timing_simple,
)

try:
    from .pyro import (
        pyro_polynomial_is_timing,
        pyro_polynomial_hmc_timing,
    )
except ImportError:
    # Pyro is optional
    pass

__all__ = [
    "genjax_polynomial_is_timing",
    "genjax_polynomial_hmc_timing",
    "genjax_polynomial_is_timing_simple",
    "pyro_polynomial_is_timing",
    "pyro_polynomial_hmc_timing",
]