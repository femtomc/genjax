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
    FloatArray,
    Selection,
    Const,
)
from .distributions import uniform


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
    if isinstance(args, tuple) and len(args) == 2 and isinstance(args[1], dict):
        # Handle (args, kwargs) tuple format
        new_trace, log_weight, _ = target_gf.regenerate(
            current_trace, selection, *args[0], **args[1]
        )
    else:
        # Handle legacy args format
        new_trace, log_weight, _ = target_gf.regenerate(current_trace, selection, args)

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
        traces=traces,
        n_steps=n_steps.value,
        acceptance_rate=acceptance_rate,
    )
