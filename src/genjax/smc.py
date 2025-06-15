"""
Standard library of programmable inference algorithms for GenJAX.

This module provides implementations of common inference algorithms that can be
composed with generative functions through the GFI (Generative Function Interface).
Uses GenJAX distributions and modular_vmap for efficient vectorized computation.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special

from .core import GFI, Trace, modular_vmap, Pytree, X, R, Any, Weight
from .distributions import categorical, uniform
import jax.tree_util as jtu


def effective_sample_size(log_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the effective sample size from log importance weights.

    Args:
        log_weights: Array of log importance weights

    Returns:
        Effective sample size in [1, num_samples]
    """
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    return 1.0 / jnp.sum(weights_normalized**2)


def systematic_resample(log_weights: jnp.ndarray, n_samples: int) -> jnp.ndarray:
    """
    Systematic resampling from importance weights using GenJAX distributions.

    Args:
        log_weights: Log importance weights
        n_samples: Number of samples to draw

    Returns:
        Array of indices for resampling
    """
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_normalized)

    # Use uniform distribution for systematic resampling offset
    u = uniform.sample(0.0, 1.0)
    positions = (jnp.arange(n_samples) + u) / n_samples
    cumsum = jnp.cumsum(weights)

    indices = jnp.searchsorted(cumsum, positions)
    return indices


def resample_vectorized_trace(
    trace: Trace[X, R],
    log_weights: jnp.ndarray,
    n_samples: int,
    method: str = "categorical",
) -> Trace[X, R]:
    """
    Resample a vectorized trace using importance weights.

    Uses categorical or systematic sampling to select indices and jax.tree_util.tree_map
    to index into the Pytree leaves.

    Args:
        trace: Vectorized trace to resample
        log_weights: Log importance weights
        n_samples: Number of samples to draw
        method: Resampling method - "categorical" or "systematic"

    Returns:
        Resampled vectorized trace
    """
    if method == "categorical":
        # Sample indices using categorical distribution
        indices = categorical(logits=log_weights).sample(sample_shape=(n_samples,))
    elif method == "systematic":
        # Use systematic resampling
        indices = systematic_resample(log_weights, n_samples)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    # Use tree_map to index into all leaves of the trace Pytree
    def index_leaf(leaf):
        # Index into the first dimension (batch dimension) of each leaf
        return leaf[indices]

    resampled_trace = jtu.tree_map(index_leaf, trace)
    return resampled_trace


@Pytree.dataclass
class ParticleCollection(Pytree):
    """Result of importance sampling containing traces, weights, and statistics."""

    traces: Trace[X, R]  # Vectorized trace containing all samples
    log_weights: jnp.ndarray
    n_samples: int = Pytree.static()

    def effective_sample_size(self) -> jnp.ndarray:
        return effective_sample_size(self.log_weights)

    def log_marginal_likelihood(self) -> jnp.ndarray:
        """
        Estimate log marginal likelihood using importance sampling.

        Returns:
            Log marginal likelihood estimate using log-sum-exp of importance weights
        """
        return jax.scipy.special.logsumexp(self.log_weights) - jnp.log(self.n_samples)


def default_importance_sampling(
    target_gf: GFI[X, R],
    target_args: tuple,
    n_samples: int,
    constraints: X,
) -> ParticleCollection:
    """
    Importance sampling using the target's default internal proposal.

    Uses the target generative function's built-in `generate` method, which
    uses its default internal proposal to fill in unconstrained choices.

    Args:
        target_gf: Target generative function (model)
        target_args: Arguments for target generative function
        n_samples: Number of importance samples to draw
        constraints: Dictionary of constrained random choices

    Returns:
        ParticleCollection with traces, weights, and diagnostics
    """

    def _single_default_importance_sample(
        target_gf: GFI[X, R],
        target_args: tuple,
        constraints: X,
    ) -> tuple[Trace[X, R], Weight]:
        """Single importance sampling step using target's default proposal."""
        # Use target's generate method with constraints
        # This will use the target's internal proposal to fill in missing choices
        target_trace, log_weight = target_gf.generate(target_args, constraints)
        return target_trace, log_weight

    # Vectorize the single importance sampling step
    vectorized_sample = modular_vmap(
        _single_default_importance_sample,
        in_axes=(None, None, None),
        axis_size=n_samples,
    )

    # Run vectorized importance sampling
    traces, log_weights = vectorized_sample(target_gf, target_args, constraints)

    return ParticleCollection(
        traces=traces,  # vectorized
        log_weights=log_weights,
        n_samples=n_samples,
    )


def _single_importance_sample(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    constraints: X,
) -> tuple[Trace[X, R], Weight]:
    """
    Single importance sampling step for use with modular_vmap.

    Args:
        target_gf: Target generative function
        proposal_gf: Proposal generative function
        target_args: Arguments for target
        proposal_args: Arguments for proposal
        constraints: Optional constraints

    Returns:
        Tuple of (target_trace, log_weight)
    """
    # Sample from proposal using simulate
    proposal_trace = proposal_gf.simulate(proposal_args)
    proposal_choices = proposal_trace.get_choices()

    # Get proposal score: log(1/P_proposal)
    proposal_score = proposal_trace.get_score()

    # Merge proposal choices with constraints
    merged_choices = target_gf.merge(proposal_choices, constraints)

    # Generate from target using merged choices
    target_trace, target_weight = target_gf.generate(target_args, merged_choices)

    # Compute importance weight: P/Q
    # target_weight is the weight from generate (density of model at merged choices)
    # proposal_score is log(1/P_proposal)
    # importance_weight = target_weight + proposal_score
    log_weight = target_weight + proposal_score

    return target_trace, log_weight


def importance_sampling(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    n_samples: int,
    constraints: X,
) -> ParticleCollection:
    """
    Basic importance sampling using proposal and target generative functions.

    Uses modular_vmap for efficient vectorized computation without explicit loops.

    Args:
        target_gf: Target generative function (model)
        proposal_gf: Proposal generative function
        target_args: Arguments for target generative function
        proposal_args: Arguments for proposal generative function
        n_samples: Number of importance samples to draw
        constraints: Optional dictionary of constrained random choices

    Returns:
        ParticleCollection with traces, weights, and diagnostics
    """
    # Vectorize the single importance sampling step
    vectorized_sample = modular_vmap(
        _single_importance_sample,
        in_axes=(None, None, None, None, None),
        axis_size=n_samples,
    )

    # Run vectorized importance sampling
    traces, log_weights = vectorized_sample(
        target_gf, proposal_gf, target_args, proposal_args, constraints
    )

    return ParticleCollection(
        traces=traces,  # vectorized
        log_weights=log_weights,
        n_samples=n_samples,
    )
