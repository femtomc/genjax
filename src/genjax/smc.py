"""
Standard library of programmable inference algorithms for GenJAX.

This module provides implementations of common inference algorithms that can be
composed with generative functions through the GFI (Generative Function Interface).
Uses GenJAX distributions and modular_vmap for efficient vectorized computation.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special

from .core import GFI, Trace, Pytree, X, R, Any, Weight, Const, Callable, const
from .pjax import modular_vmap
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
        indices = categorical.sample(log_weights, sample_shape=(n_samples,))
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
    n_samples: Const[int]
    log_marginal_estimate: jnp.ndarray = Pytree.field(
        default_factory=lambda: jnp.array(0.0)
    )  # Accumulated log marginal likelihood estimate

    def effective_sample_size(self) -> jnp.ndarray:
        return effective_sample_size(self.log_weights)

    def log_marginal_likelihood(self) -> jnp.ndarray:
        """
        Estimate log marginal likelihood using importance sampling.

        Returns:
            Log marginal likelihood estimate using log-sum-exp of importance weights
            plus any accumulated marginal estimate from previous resampling steps
        """
        current_marginal = jax.scipy.special.logsumexp(self.log_weights) - jnp.log(
            self.n_samples.value
        )
        return self.log_marginal_estimate + current_marginal


def init(
    target_gf: GFI[X, R],
    target_args: tuple,
    n_samples: Const[int],
    constraints: X,
    proposal_gf: GFI[X, Any] = None,
) -> ParticleCollection:
    """
    Initialize particle collection using importance sampling.

    Uses either the target's default internal proposal or a custom proposal.
    Proposals use signature (constraints, *target_args).

    Args:
        target_gf: Target generative function (model)
        target_args: Arguments for target generative function
        n_samples: Number of importance samples to draw (static value)
        constraints: Dictionary of constrained random choices
        proposal_gf: Optional custom proposal generative function.
                    If None, uses target's default internal proposal.

    Returns:
        ParticleCollection with traces, weights, and diagnostics
    """
    if proposal_gf is None:
        # Use default importance sampling with target's internal proposal
        def _single_default_importance_sample(
            target_gf: GFI[X, R],
            target_args: tuple,
            constraints: X,
        ) -> tuple[Trace[X, R], Weight]:
            """Single importance sampling step using target's default proposal."""
            # Use target's generate method with constraints
            # This will use the target's internal proposal to fill in missing choices
            target_trace, log_weight = target_gf.generate(constraints, *target_args)
            return target_trace, log_weight

        # Vectorize the single importance sampling step
        vectorized_sample = modular_vmap(
            _single_default_importance_sample,
            in_axes=(None, None, None),
            axis_size=n_samples.value,
        )

        # Run vectorized importance sampling
        traces, log_weights = vectorized_sample(target_gf, target_args, constraints)
    else:
        # Use custom proposal importance sampling
        def _single_importance_sample(
            target_gf: GFI[X, R],
            proposal_gf: GFI[X, Any],
            target_args: tuple,
            constraints: X,
        ) -> tuple[Trace[X, R], Weight]:
            """
            Single importance sampling step using custom proposal.

            Proposal uses signature (constraints, *target_args).
            """
            # Sample from proposal using new signature
            proposal_trace = proposal_gf.simulate(constraints, *target_args)
            proposal_choices = proposal_trace.get_choices()

            # Get proposal score: log(1/P_proposal)
            proposal_score = proposal_trace.get_score()

            # Merge proposal choices with constraints
            merged_choices = target_gf.merge(proposal_choices, constraints)

            # Generate from target using merged choices
            target_trace, target_weight = target_gf.generate(
                merged_choices, *target_args
            )

            # Compute importance weight: P/Q
            # target_weight is the weight from generate (density of model at merged choices)
            # proposal_score is log(1/P_proposal)
            # importance_weight = target_weight + proposal_score
            log_weight = target_weight + proposal_score

            return target_trace, log_weight

        # Vectorize the single importance sampling step
        vectorized_sample = modular_vmap(
            _single_importance_sample,
            in_axes=(None, None, None, None),
            axis_size=n_samples.value,
        )

        # Run vectorized importance sampling
        traces, log_weights = vectorized_sample(
            target_gf, proposal_gf, target_args, constraints
        )

    return ParticleCollection(
        traces=traces,  # vectorized
        log_weights=log_weights,
        n_samples=const(n_samples.value),
        log_marginal_estimate=jnp.array(0.0),
    )


def change(
    particles: ParticleCollection,
    new_target_gf: GFI[X, R],
    new_target_args: tuple,
    choice_map_fn: Callable[[X], X],
) -> ParticleCollection:
    """
    Change target move for particle collection.

    Translates particles from one model to another by:
    1. Mapping each particle's choices using choice_map_fn
    2. Using generate with the new model to get new weights
    3. Accumulating importance weights

    Args:
        particles: Current particle collection
        new_target_gf: New target generative function
        new_target_args: Arguments for new target
        choice_map_fn: Function mapping choices X -> X

    Returns:
        New ParticleCollection with translated particles
    """

    def _single_change_target(
        old_trace: Trace[X, R], old_log_weight: jnp.ndarray
    ) -> tuple[Trace[X, R], jnp.ndarray]:
        # Map choices to new space
        old_choices = old_trace.get_choices()
        mapped_choices = choice_map_fn(old_choices)

        # Generate with new model using mapped choices as constraints
        new_trace, log_weight = new_target_gf.generate(mapped_choices, *new_target_args)

        # Accumulate importance weight
        new_log_weight = old_log_weight + log_weight

        return new_trace, new_log_weight

    # Vectorize across particles
    vectorized_change = modular_vmap(
        _single_change_target,
        in_axes=(0, 0),
        axis_size=particles.n_samples.value,
    )

    new_traces, new_log_weights = vectorized_change(
        particles.traces, particles.log_weights
    )

    return ParticleCollection(
        traces=new_traces,
        log_weights=new_log_weights,
        n_samples=particles.n_samples,
        log_marginal_estimate=particles.log_marginal_estimate,
    )


def extend(
    particles: ParticleCollection,
    proposal_gf: GFI[X, Any],
    extended_target_gf: GFI[X, R],
    extended_target_args: tuple,
) -> ParticleCollection:
    """
    Extension move for particle collection.

    Extends each particle by:
    1. Sampling an extension from the proposal using (old_choices, *extended_target_args)
    2. Merging with existing particle choices
    3. Evaluating under the extended target model

    Args:
        particles: Current particle collection
        proposal_gf: Proposal for the extension using signature (old_choices, *extended_target_args)
        extended_target_gf: Extended target generative function
        extended_target_args: Arguments for extended target

    Returns:
        New ParticleCollection with extended particles
    """

    def _single_extension(
        old_trace: Trace[X, R], old_log_weight: jnp.ndarray
    ) -> tuple[Trace[X, R], jnp.ndarray]:
        # Get existing choices to pass to proposal
        old_choices = old_trace.get_choices()

        # Sample extension from proposal using new signature
        proposal_trace = proposal_gf.simulate(old_choices, *extended_target_args)
        extension_choices = proposal_trace.get_choices()
        proposal_score = proposal_trace.get_score()

        # Merge with existing choices
        merged_choices = extended_target_gf.merge(old_choices, extension_choices)

        # Generate with extended target
        new_trace, log_weight = extended_target_gf.generate(
            merged_choices, *extended_target_args
        )

        # Importance weight: target_weight + proposal_score + old_weight
        new_log_weight = old_log_weight + log_weight + proposal_score

        return new_trace, new_log_weight

    # Vectorize across particles
    vectorized_extension = modular_vmap(
        _single_extension,
        in_axes=(0, 0),
        axis_size=particles.n_samples.value,
    )

    new_traces, new_log_weights = vectorized_extension(
        particles.traces, particles.log_weights
    )

    return ParticleCollection(
        traces=new_traces,
        log_weights=new_log_weights,
        n_samples=particles.n_samples,
        log_marginal_estimate=particles.log_marginal_estimate,
    )


def rejuvenate(
    particles: ParticleCollection,
    mcmc_kernel: Callable[[Trace[X, R]], tuple[Trace[X, R], Any]],
) -> ParticleCollection:
    """
    Rejuvenate move for particle collection.

    Applies an MCMC kernel to each particle independently to improve
    particle diversity and reduce degeneracy.

    Args:
        particles: Current particle collection
        mcmc_kernel: MCMC kernel function that takes a trace and returns
                    (new_trace, diagnostics). Should be compatible with
                    kernels from mcmc.py like metropolis_hastings_step.

    Returns:
        New ParticleCollection with rejuvenated particles
    """

    def _single_rejuvenate(
        old_trace: Trace[X, R], old_log_weight: jnp.ndarray
    ) -> tuple[Trace[X, R], jnp.ndarray]:
        # Apply MCMC kernel
        new_trace, _ = mcmc_kernel(old_trace)

        # Weights remain unchanged for MCMC moves (detailed balance)
        return new_trace, old_log_weight

    # Vectorize across particles
    vectorized_rejuvenate = modular_vmap(
        _single_rejuvenate,
        in_axes=(0, 0),
        axis_size=particles.n_samples.value,
    )

    new_traces, new_log_weights = vectorized_rejuvenate(
        particles.traces, particles.log_weights
    )

    return ParticleCollection(
        traces=new_traces,
        log_weights=new_log_weights,
        n_samples=particles.n_samples,
        log_marginal_estimate=particles.log_marginal_estimate,
    )


def resample(
    particles: ParticleCollection,
    method: str = "categorical",
) -> ParticleCollection:
    """
    Resample particle collection to combat degeneracy.

    After resampling, weights are reset to uniform (zero in log space)
    and the marginal likelihood estimate is updated to include the
    average weight before resampling.

    Args:
        particles: Current particle collection
        method: Resampling method - "categorical" or "systematic"

    Returns:
        New ParticleCollection with resampled particles and updated marginal estimate
    """
    # Compute current marginal contribution before resampling
    current_marginal = jax.scipy.special.logsumexp(particles.log_weights) - jnp.log(
        particles.n_samples.value
    )

    # Update accumulated marginal estimate
    new_log_marginal_estimate = particles.log_marginal_estimate + current_marginal

    # Resample traces using existing function
    resampled_traces = resample_vectorized_trace(
        particles.traces,
        particles.log_weights,
        particles.n_samples.value,
        method=method,
    )

    # Reset weights to uniform (zero in log space)
    uniform_log_weights = jnp.zeros(particles.n_samples.value)

    return ParticleCollection(
        traces=resampled_traces,
        log_weights=uniform_log_weights,
        n_samples=particles.n_samples,
        log_marginal_estimate=new_log_marginal_estimate,
    )


def rejuvenation_smc(
    initial_model: GFI[X, R],
    extended_model: GFI[X, R],
    transition_proposal: GFI[X, Any],
    mcmc_kernel: Callable[[Trace[X, R]], Trace[X, R]],
    observations: X,
    change_mapping: Callable[[X], X],
    n_particles: Const[int] = const(1000),
) -> ParticleCollection:
    """
    Complete SMC algorithm with rejuvenation using jax.lax.scan.

    Implements sequential Monte Carlo with particle extension, resampling,
    and MCMC rejuvenation. Handles observations as Pytree structures.

    Args:
        initial_model: Starting generative function for first timestep
        extended_model: Extended generative function for subsequent timesteps
        transition_proposal: Proposal for extending particles at each timestep
        mcmc_kernel: MCMC kernel for particle rejuvenation
        observations: Sequence of observations (can be Pytree structure)
        change_mapping: Function to map initial particles to extended model address space
        n_particles: Number of particles to maintain

    Returns:
        Final ParticleCollection after processing all observations
    """
    # Extract first observation using tree_map (handles Pytree structure)
    first_obs = jtu.tree_map(lambda x: x[0], observations)

    # Initialize with first observation
    particles = init(initial_model, (), n_particles, {"obs": first_obs})

    # Map initial particles to extended model address space
    particles = change(particles, extended_model, (), change_mapping)

    def smc_step(particles, obs):
        # Extend particles with transition proposal
        particles = extend(particles, transition_proposal, extended_model, obs)

        # Resample if needed
        ess = particles.effective_sample_size()
        particles = jax.lax.cond(
            ess < n_particles.value // 2, lambda p: resample(p), lambda p: p, particles
        )

        # Optional rejuvenation
        particles = rejuvenate(particles, mcmc_kernel)

        return particles, particles  # (carry, output)

    # Sequential updates using scan over remaining observations
    # Slice remaining observations using tree_map (handles Pytree structure)
    remaining_obs = jtu.tree_map(lambda x: x[1:], observations)

    final_particles, all_particles = jax.lax.scan(smc_step, particles, remaining_obs)

    return final_particles
