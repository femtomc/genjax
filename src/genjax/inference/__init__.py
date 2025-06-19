"""Inference algorithms for GenJAX.

This module provides implementations of standard inference algorithms including
MCMC, SMC, and variational inference methods.
"""

from .mcmc import mh, mala, hmc, chain, MCMCResult
from .smc import (
    init,
    change,
    extend,
    rejuvenate,
    resample,
    rejuvenation_smc,
    ParticleCollection,
)
from .vi import (
    elbo_factory,
    VariationalApproximation,
    mean_field_normal_family,
    full_covariance_normal_family,
    elbo_vi,
)

__all__ = [
    # MCMC
    "mh",
    "mala",
    "hmc",
    "chain",
    "MCMCResult",
    # SMC
    "init",
    "change",
    "extend",
    "rejuvenate",
    "resample",
    "rejuvenation_smc",
    "ParticleCollection",
    # VI
    "elbo_factory",
    "VariationalApproximation",
    "mean_field_normal_family",
    "full_covariance_normal_family",
    "elbo_vi",
]
