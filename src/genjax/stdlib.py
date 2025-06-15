"""
Standard library of programmable inference algorithms for GenJAX.

This module provides implementations of common inference algorithms that can be
composed with generative functions through the GFI (Generative Function Interface).
Uses GenJAX distributions and modular_vmap for efficient vectorized computation.
"""

import jax
import jax.numpy as jnp

from .core import GFI, Trace, modular_vmap, Pytree, X, R, Callable, Any
from .distributions import categorical, uniform
import jax.tree_util as jtu


@Pytree.dataclass
class ParticleCollection(Pytree):
    """Result of importance sampling containing traces, weights, and statistics."""

    traces: Trace[X, R]  # Vectorized trace containing all samples
    log_weights: jnp.ndarray
    normalized_weights: jnp.ndarray
    log_marginal_likelihood: jnp.ndarray
    effective_sample_size: jnp.ndarray


def effective_sample_size(log_weights: jnp.ndarray) -> float:
    """
    Compute the effective sample size from log importance weights.

    Args:
        log_weights: Array of log importance weights

    Returns:
        Effective sample size in [1, num_samples]
    """
    log_weights_normalized = log_weights - jnp.logsumexp(log_weights)
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
    log_weights_normalized = log_weights - jnp.logsumexp(log_weights)
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


def _single_importance_sample(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    constraints: X,
) -> tuple[Trace[X, R], float]:
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
        ImportanceSampleResult with traces, weights, and diagnostics
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

    # Compute normalized weights
    log_marginal_likelihood = jnp.logsumexp(log_weights) - jnp.log(n_samples)
    normalized_log_weights = log_weights - jnp.logsumexp(log_weights)
    normalized_weights = jnp.exp(normalized_log_weights)

    # Compute effective sample size
    ess = effective_sample_size(log_weights)

    return ParticleCollection(
        traces=traces,  # traces is already vectorized
        log_weights=log_weights,
        normalized_weights=normalized_weights,
        log_marginal_likelihood=log_marginal_likelihood,
        effective_sample_size=ess,
    )


def _sequential_step(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    prev_trace: Trace[X, R],
    observation: X,
) -> tuple[Trace[X, R], float]:
    """
    Single step of sequential importance sampling.

    Args:
        target_gf: Target generative function
        proposal_gf: Proposal generative function
        target_args: Target arguments
        proposal_args: Proposal arguments
        prev_trace: Previous trace to update
        observation: Current observation

    Returns:
        Tuple of (new_trace, log_weight)
    """
    # Propose new state using simulate
    proposal_trace = proposal_gf.simulate(proposal_args)
    proposal_choices = proposal_trace.get_choices()
    proposal_score = proposal_trace.get_score()  # log(1/P_proposal)

    # Merge proposal choices with observation constraints
    merged_choices = target_gf.merge(proposal_choices, observation)

    # Generate new trace from target with merged choices
    new_trace, target_weight = target_gf.generate(target_args, merged_choices)

    # Compute importance weight: P/Q
    # target_weight is the weight from generate
    # proposal_score is log(1/P_proposal)
    log_weight = target_weight + proposal_score
    return new_trace, log_weight


def sequential_importance_sampling(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    observations: X,  # Vectorized observations
    n_particles: int,
    resample_threshold: float = 0.5,
) -> ParticleCollection:
    """
    Sequential importance sampling (particle filtering) using jax.lax.scan.

    Args:
        target_gf: Target generative function (state space model)
        proposal_gf: Proposal generative function for transitions
        target_args: Arguments for target
        proposal_args: Arguments for proposal
        observations: Vectorized observations (time series)
        n_particles: Number of particles to maintain
        resample_threshold: Resample when ESS/n_particles < threshold

    Returns:
        Final ParticleCollection after processing all observations
    """

    def scan_step(carry, observation):
        current_particles, current_weights = carry

        # First time step: initialize particles
        if current_particles is None:
            result = importance_sampling(
                target_gf,
                proposal_gf,
                target_args,
                proposal_args,
                n_particles,
                observation,
            )
        else:
            # Vectorized particle propagation
            vectorized_step = modular_vmap(
                _sequential_step, in_axes=(None, None, None, None, 0, None)
            )

            # Resample particles if needed
            def resample_branch():
                return resample_vectorized_trace(
                    current_particles, jnp.log(current_weights), n_particles
                )

            def no_resample_branch():
                return current_particles

            resampled_particles = jax.lax.cond(
                effective_sample_size(jnp.log(current_weights))
                < resample_threshold * n_particles,
                resample_branch,
                no_resample_branch,
            )

            # Propagate all particles forward
            new_traces, new_log_weights = vectorized_step(
                target_gf,
                proposal_gf,
                target_args,
                proposal_args,
                resampled_particles,
                observation,
            )

            result = ParticleCollection(
                traces=new_traces,
                log_weights=new_log_weights,
                normalized_weights=jnp.exp(
                    new_log_weights - jnp.logsumexp(new_log_weights)
                ),
                log_marginal_likelihood=jnp.logsumexp(new_log_weights)
                - jnp.log(n_particles),
                effective_sample_size=effective_sample_size(new_log_weights),
            )

        # Update particle state for next iteration
        def resample_branch():
            resampled_particles = resample_vectorized_trace(
                result.traces, result.log_weights, n_particles
            )
            reset_weights = jnp.ones(n_particles) / n_particles
            return resampled_particles, reset_weights

        def no_resample_branch():
            return result.traces, result.normalized_weights

        new_particles, new_weights = jax.lax.cond(
            result.effective_sample_size < resample_threshold * n_particles,
            resample_branch,
            no_resample_branch,
        )

        return (new_particles, new_weights), result

    # Initialize carry state
    initial_carry = (None, None)

    # Run scan over observations
    final_carry, all_results = jax.lax.scan(scan_step, initial_carry, observations)

    # Return the final result (could also return all_results for full trajectory)
    return all_results[-1]


def adaptive_importance_sampling(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    n_samples: int,
    adaptation_steps: int = 5,
    min_ess_ratio: float = 0.1,
) -> ParticleCollection:
    """
    Adaptive importance sampling that iteratively improves the proposal.

    Args:
        target_gf: Target generative function
        proposal_gf: Proposal generative function
        target_args: Target arguments
        proposal_args: Initial proposal arguments
        n_samples: Number of samples per iteration
        adaptation_steps: Maximum adaptation iterations
        min_ess_ratio: Minimum acceptable ESS/n_samples ratio

    Returns:
        Final ImportanceSampleResult
    """
    current_proposal_args = proposal_args

    for step in range(adaptation_steps):
        # Run importance sampling
        result = importance_sampling(
            target_gf, proposal_gf, target_args, current_proposal_args, n_samples
        )

        ess_ratio = result.effective_sample_size / n_samples

        # Check if adaptation is needed
        if ess_ratio >= min_ess_ratio:
            break

        # Simple adaptation: adjust proposal parameters based on
        # weighted sample statistics (problem-specific)
        if hasattr(proposal_gf, "adapt"):
            current_proposal_args = proposal_gf.adapt(
                current_proposal_args, result.traces, result.normalized_weights
            )
        else:
            # Default: no adaptation available
            break

    return result


def self_normalized_importance_sampling(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    n_samples: int,
    constraints: X,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Self-normalized importance sampling for expectation estimation.

    Computes E_target[f(X)] H � w_i f(x_i) / � w_i where w_i are importance weights.

    Args:
        target_gf: Target generative function
        proposal_gf: Proposal generative function
        target_args: Target arguments
        proposal_args: Proposal arguments
        n_samples: Number of samples
        constraints: Optional constraints

    Returns:
        Tuple of (function_values, normalized_weights) for expectation computation
    """
    result = importance_sampling(
        target_gf, proposal_gf, target_args, proposal_args, n_samples, constraints
    )

    # Extract function values (return values from target)
    # Since traces is already vectorized, we can directly get return values
    function_values = result.traces.get_retval()

    return function_values, result.normalized_weights


# Convenience functions for common use cases


def estimate_marginal_likelihood(
    model_gf: GFI[X, R],
    prior_gf: GFI[X, Any],
    model_args: tuple,
    prior_args: tuple,
    observations: X,
    n_samples: int = 1000,
) -> tuple[float, float]:
    """
    Estimate marginal likelihood p(observations) using importance sampling.

    Args:
        model_gf: Model generative function
        prior_gf: Prior generative function (proposal)
        model_args: Model arguments
        prior_args: Prior arguments
        observations: Observed data constraints
        n_samples: Number of importance samples

    Returns:
        Tuple of (log_marginal_likelihood_estimate, standard_error)
    """
    result = importance_sampling(
        model_gf, prior_gf, model_args, prior_args, n_samples, observations
    )

    # Standard error estimate for log marginal likelihood
    log_weights_centered = result.log_weights - jnp.mean(result.log_weights)
    variance = jnp.var(log_weights_centered)
    standard_error = jnp.sqrt(variance / n_samples)

    return result.log_marginal_likelihood, standard_error


def posterior_predictive_sampling(
    model_gf: GFI[X, R],
    prior_gf: GFI[X, Any],
    model_args: tuple,
    prior_args: tuple,
    observations: X,
    n_posterior_samples: int = 100,
    n_predictive_samples: int = 10,
) -> Trace[X, R]:
    """
    Generate posterior predictive samples using importance sampling.

    Args:
        model_gf: Model generative function
        prior_gf: Prior generative function
        model_args: Model arguments
        prior_args: Prior arguments
        observations: Training observations
        n_posterior_samples: Number of posterior samples
        n_predictive_samples: Predictive samples per posterior sample

    Returns:
        Vectorized trace containing predictive samples
    """
    # Get posterior samples via importance sampling
    posterior_result = importance_sampling(
        model_gf, prior_gf, model_args, prior_args, n_posterior_samples, observations
    )

    # Generate predictive samples using proper weighted resampling
    n_total_predictive = n_posterior_samples * n_predictive_samples

    # Resample posterior samples according to their importance weights
    resampled_posterior_traces = resample_vectorized_trace(
        posterior_result.traces, posterior_result.log_weights, n_total_predictive
    )

    # Generate predictive samples using vectorized operations
    def _generate_predictive_from_posterior(posterior_trace):
        posterior_choices = posterior_trace.get_choices()
        predictive_trace, _ = model_gf.generate(model_args, posterior_choices)
        return predictive_trace

    # Apply predictive generation to each resampled posterior sample
    predictive_traces = modular_vmap(_generate_predictive_from_posterior, in_axes=0)(
        resampled_posterior_traces
    )

    return predictive_traces


def importance_weighted_expectation(
    target_gf: GFI[X, R],
    proposal_gf: GFI[X, Any],
    target_args: tuple,
    proposal_args: tuple,
    test_function: Callable[[Trace[X, R]], float],
    n_samples: int,
    constraints: X,
) -> tuple[float, float]:
    """
    Compute importance-weighted expectation of a test function.

    Estimates E_target[test_function(trace)] using importance sampling.

    Args:
        target_gf: Target generative function
        proposal_gf: Proposal generative function
        target_args: Target arguments
        proposal_args: Proposal arguments
        test_function: Function to compute expectation of
        n_samples: Number of importance samples
        constraints: Optional constraints

    Returns:
        Tuple of (expectation_estimate, standard_error)
    """
    result = importance_sampling(
        target_gf, proposal_gf, target_args, proposal_args, n_samples, constraints
    )

    # Compute test function values - apply function to vectorized traces
    test_values = modular_vmap(test_function, in_axes=0)(result.traces)

    # Compute weighted expectation
    expectation = jnp.sum(test_values * result.normalized_weights)

    # Estimate standard error (delta method approximation)
    weighted_variance = jnp.sum(
        result.normalized_weights * (test_values - expectation) ** 2
    )
    standard_error = jnp.sqrt(weighted_variance / result.effective_sample_size)

    return expectation, standard_error
