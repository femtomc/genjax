"""
MCMC (Markov Chain Monte Carlo) inference algorithms for GenJAX.

This module provides implementations of standard MCMC algorithms including
Metropolis-Hastings and Hamiltonian Monte Carlo (HMC). All algorithms use
the GFI (Generative Function Interface) for efficient trace operations.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .core import (
    Trace,
    Pytree,
    X,
    R,
    Score,
    FloatArray,
    modular_vmap,
    Selection,
    Const,
)
from .distributions import uniform, normal


@Pytree.dataclass
class MCMCResult(Pytree):
    """Result of MCMC sampling containing chain traces and diagnostics."""

    traces: Trace[X, R]  # Vectorized over chain steps
    n_steps: int = Pytree.static()
    acceptance_rate: FloatArray


def metropolis_hastings_step(
    current_trace: Trace[X, R],
    selection: Selection,
) -> tuple[Trace[X, R], jnp.ndarray]:
    """
    Single Metropolis-Hastings step using GFI.regenerate.

    Uses the trace's generative function regenerate method to propose
    new values for selected addresses and computes MH accept/reject ratio.

    Args:
        current_trace: Current trace state
        selection: Addresses to regenerate (subset of choices)

    Returns:
        Tuple of (new_trace, accepted) where accepted is JAX boolean array
    """
    target_gf = current_trace.get_gen_fn()
    args = current_trace.get_args()

    # Regenerate selected addresses - weight is log acceptance probability
    new_trace, log_weight, _ = target_gf.regenerate(args, current_trace, selection)

    # MH acceptance step
    log_alpha = jnp.minimum(0.0, log_weight)  # min(1, exp(log_weight))
    accept_prob = jnp.exp(log_alpha)

    # Accept or reject using GenJAX uniform distribution
    u = uniform.sample(0.0, 1.0)
    accept = u < accept_prob

    # Use tree_map to apply select across all leaves of the traces
    final_trace = jtu.tree_map(
        lambda new_leaf, old_leaf: jax.lax.select(accept, new_leaf, old_leaf),
        new_trace,
        current_trace,
    )

    return final_trace, accept


def metropolis_hastings(
    initial_trace: Trace[X, R],
    selection: Selection,
    n_steps: Const[int],
) -> MCMCResult:
    """
    Metropolis-Hastings MCMC sampling using GFI.regenerate.

    Args:
        initial_trace: Starting trace (contains target generative function)
        selection: Addresses to update in each step
        n_steps: Number of MCMC steps (wrapped in Const to stay static)

    Returns:
        MCMCResult containing chain traces and diagnostics
    """

    def mh_scan_fn(trace, _):
        new_trace, accepted = metropolis_hastings_step(trace, selection)
        return new_trace, (new_trace, accepted)

    # Run MCMC chain using scan
    final_trace, (traces, acceptances) = jax.lax.scan(
        mh_scan_fn, initial_trace, jnp.arange(n_steps.value)
    )

    # Compute acceptance rate
    acceptance_rate = jnp.mean(acceptances)

    return MCMCResult(
        traces=traces, n_steps=n_steps.value, acceptance_rate=acceptance_rate
    )


# HMC uses the same MCMCResult but manages momentum internally


def compute_kinetic_energy(momentum: X) -> Score:
    """Compute kinetic energy from momentum (0.5 * p^T * p)."""

    def sum_squares(x):
        return 0.5 * jnp.sum(x**2)

    # Sum kinetic energy across all momentum components
    kinetic_energies = jax.tree_map(sum_squares, momentum)
    return jax.tree_util.tree_reduce(jnp.add, kinetic_energies, 0.0)


def leapfrog_step(
    trace: Trace[X, R],
    momentum: X,
    selection: Selection,
    step_size: float,
) -> tuple[Trace[X, R], X]:
    """
    Single leapfrog integration step for HMC.

    Args:
        trace: Current trace (contains target generative function)
        momentum: Current momentum
        selection: Addresses being sampled
        step_size: Leapfrog step size

    Returns:
        Updated trace and momentum after leapfrog step
    """
    target_gf = trace.get_gen_fn()
    args = trace.get_args()

    # Get current choices and compute gradients
    choices = trace.get_choices()

    # Compute gradients of log density w.r.t. continuous choices
    # This requires the generative function to be differentiable
    def log_density_fn(selected_choices):
        # Merge selected choices back into full choices
        full_choices = jax.tree_map(
            lambda _, sel: sel,
            choices,
            selected_choices,
            is_leaf=lambda x: x in selection,
        )
        density, _ = target_gf.assess(args, full_choices)
        return density

    # Extract selected choices for gradient computation
    selected_choices = jax.tree_map(
        lambda x: x, choices, is_leaf=lambda x: x in selection
    )

    grad_log_density = jax.grad(log_density_fn)(selected_choices)

    # Half-step momentum update
    momentum_half = jax.tree_map(
        lambda p, g: p + 0.5 * step_size * g, momentum, grad_log_density
    )

    # Full-step position update
    new_selected_choices = jax.tree_map(
        lambda x, p: x + step_size * p, selected_choices, momentum_half
    )

    # Create new trace with updated choices
    new_choices = jax.tree_map(
        lambda _, new_sel: new_sel,
        choices,
        new_selected_choices,
        is_leaf=lambda x: x in selection,
    )

    # Update trace with new choices
    new_trace, _, _ = target_gf.update(args, trace, new_choices)

    # Recompute gradients at new position
    new_grad_log_density = jax.grad(log_density_fn)(new_selected_choices)

    # Half-step momentum update
    new_momentum = jax.tree_map(
        lambda p, g: p + 0.5 * step_size * g, momentum_half, new_grad_log_density
    )

    return new_trace, new_momentum


def hmc_step(
    current_trace: Trace[X, R],
    selection: Selection,
    n_leapfrog: int = 10,
    step_size: float = 0.1,
) -> tuple[Trace[X, R], jnp.ndarray]:
    """
    Single HMC step with leapfrog integration.

    Args:
        current_trace: Current trace (contains target generative function)
        selection: Addresses to sample
        n_leapfrog: Number of leapfrog steps
        step_size: Leapfrog step size

    Returns:
        Tuple of (new_trace, accepted) where accepted is JAX boolean array
    """

    # Sample fresh momentum using GenJAX normal distribution
    def sample_momentum(x):
        if jnp.isscalar(x):
            return normal.sample(0.0, 1.0)
        else:
            shape = x.shape
            return jnp.array(
                [normal.sample(0.0, 1.0) for _ in range(jnp.prod(shape))]
            ).reshape(shape)

    momentum = jax.tree_map(sample_momentum, selection)

    # Compute initial energy
    initial_kinetic = compute_kinetic_energy(momentum)
    initial_potential = -current_trace.get_score()  # -log p(choices)
    initial_energy = initial_kinetic + initial_potential

    # Perform leapfrog integration
    trace, momentum = current_trace, momentum
    for _ in range(n_leapfrog):
        trace, momentum = leapfrog_step(trace, momentum, selection, step_size)

    # Negate momentum for reversibility
    momentum = jax.tree_map(lambda p: -p, momentum)

    # Compute final energy
    final_kinetic = compute_kinetic_energy(momentum)
    final_potential = -trace.get_score()
    final_energy = final_kinetic + final_potential

    # Accept/reject based on energy difference
    log_accept_prob = initial_energy - final_energy
    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_prob))

    # Accept or reject using GenJAX uniform distribution
    u = uniform.sample(0.0, 1.0)
    accept = u < accept_prob

    # Use tree_map to apply select across all leaves of the traces
    final_trace = jtu.tree_map(
        lambda new_leaf, old_leaf: jax.lax.select(accept, new_leaf, old_leaf),
        trace,
        current_trace,
    )

    return final_trace, accept


def hmc(
    initial_trace: Trace[X, R],
    selection: Selection,
    n_steps: int,
    n_leapfrog: int = 10,
    step_size: float = 0.1,
) -> MCMCResult:
    """
    Hamiltonian Monte Carlo sampling.

    Args:
        initial_trace: Starting trace (contains target generative function)
        selection: Continuous addresses to sample
        n_steps: Number of HMC steps
        n_leapfrog: Leapfrog steps per HMC step
        step_size: Leapfrog step size

    Returns:
        MCMCResult containing chain traces and diagnostics
    """

    def hmc_scan_fn(trace, _):
        new_trace, accepted = hmc_step(trace, selection, n_leapfrog, step_size)
        return new_trace, (new_trace, accepted)

    # Run HMC chain
    final_trace, (traces, acceptances) = jax.lax.scan(
        hmc_scan_fn, initial_trace, jnp.arange(n_steps)
    )

    # Compute acceptance rate
    acceptance_rate = jnp.mean(acceptances)

    return MCMCResult(traces=traces, n_steps=n_steps, acceptance_rate=acceptance_rate)


# Vectorized versions using modular_vmap
def metropolis_hastings_vectorized(
    initial_traces: Trace[X, R],  # Vectorized initial traces
    selection: Selection,
    n_steps: Const[int],
) -> MCMCResult:
    """
    Vectorized Metropolis-Hastings for multiple chains.

    Args:
        initial_traces: Vectorized initial traces (batch of chains)
        selection: Addresses to update
        n_steps: Steps per chain (wrapped in Const to stay static)

    Returns:
        MCMCResult with vectorized chain traces
    """
    # Get number of chains from vectorized choices
    sample_choice = next(iter(initial_traces.get_choices().values()))
    n_chains = sample_choice.shape[0] if hasattr(sample_choice, "shape") else 1

    # Vectorize over chains
    vectorized_mh = modular_vmap(
        metropolis_hastings, in_axes=(0, None, None), axis_size=n_chains
    )

    return vectorized_mh(initial_traces, selection, n_steps)
