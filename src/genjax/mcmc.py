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
    const,
    Callable,
)
from .distributions import uniform
from .state import save, state


def compute_rhat(samples: jnp.ndarray) -> FloatArray:
    """
    Compute potential scale reduction factor (R-hat) for MCMC convergence.

    R-hat compares between-chain and within-chain variance to assess convergence.
    Values close to 1.0 indicate good convergence.

    Args:
        samples: Array of shape (n_chains, n_samples) containing MCMC samples

    Returns:
        R-hat statistic. Values < 1.01 typically indicate convergence.
    """
    n_chains, n_samples = samples.shape

    # For R-hat, we need at least 2 chains and enough samples
    if n_chains < 2:
        return jnp.nan

    # Use all samples for simpler computation
    # Compute chain means
    chain_means = jnp.mean(samples, axis=1)  # (n_chains,)

    # Between-chain variance
    B = n_samples * jnp.var(chain_means, ddof=1)

    # Within-chain variance
    chain_vars = jnp.var(samples, axis=1, ddof=1)  # (n_chains,)
    W = jnp.mean(chain_vars)

    # Pooled variance estimate
    var_plus = ((n_samples - 1) * W + B) / n_samples

    # R-hat statistic
    rhat = jnp.sqrt(var_plus / W)

    return rhat


def compute_ess(samples: jnp.ndarray, kind: str = "bulk") -> FloatArray:
    """
    Compute effective sample size for MCMC chains.

    Args:
        samples: Array of shape (n_chains, n_samples) containing MCMC samples
        kind: Type of ESS to compute ("bulk" or "tail")

    Returns:
        Effective sample size estimate
    """
    n_chains, n_samples = samples.shape

    if kind == "tail":
        # For tail ESS, use quantile-based approach
        # Transform samples to focus on tails
        quantiles = jnp.array([0.05, 0.95])
        tail_samples = jnp.quantile(samples, quantiles, axis=1)
        # Use difference between quantiles as the statistic
        samples_for_ess = tail_samples[1] - tail_samples[0]
        samples_for_ess = samples_for_ess.reshape(1, -1)
    else:
        # For bulk ESS, use all samples
        samples_for_ess = samples.reshape(1, -1)

    # Simple ESS approximation based on autocorrelation
    # This is a simplified version - a full implementation would compute
    # autocorrelation function and find cutoff

    # Compute autocorrelation at lag 1 as rough approximation
    flat_samples = samples_for_ess.flatten()

    # Autocorrelation at lag 1
    lag1_corr = jnp.corrcoef(flat_samples[:-1], flat_samples[1:])[0, 1]
    lag1_corr = jnp.clip(lag1_corr, 0.0, 0.99)  # Avoid division issues

    # Simple ESS approximation: N / (1 + 2*rho)
    # where rho is the sum of positive autocorrelations
    effective_chains = n_chains if kind == "bulk" else 1
    total_samples = effective_chains * n_samples
    ess = total_samples / (1 + 2 * lag1_corr)

    return ess


@Pytree.dataclass
class MCMCResult(Pytree):
    """Result of MCMC chain sampling containing traces and diagnostics."""

    traces: Trace[X, R]  # Vectorized over chain steps and chains (if multiple)
    accepts: jnp.ndarray  # Individual acceptance decisions (boolean)
    acceptance_rate: FloatArray  # Overall acceptance rate (per chain if multiple)
    n_steps: Const[int]  # Total number of steps (after any burn-in/thinning)
    n_chains: Const[int]  # Number of parallel chains

    # Between-chain diagnostics (only computed when n_chains > 1)
    # These have the same pytree structure as X but with scalar diagnostics
    rhat: X | None = None  # R-hat per parameter (same structure as X)
    ess_bulk: X | None = None  # Bulk ESS per parameter (same structure as X)
    ess_tail: X | None = None  # Tail ESS per parameter (same structure as X)


def mh(
    current_trace: Trace[X, R],
    selection: Selection,
) -> Trace[X, R]:
    """
    Single Metropolis-Hastings step using GFI.regenerate.

    Uses the trace's generative function regenerate method to propose
    new values for selected addresses and computes MH accept/reject ratio.

    Args:
        current_trace: Current trace state
        selection: Addresses to regenerate (subset of choices)

    Returns:
        Updated trace after MH step

    State:
        accept: Boolean indicating whether the proposal was accepted
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

    # Save acceptance as auxiliary state (can be accessed via state decorator)
    save(accept=accept)

    return final_trace


def chain(mcmc_kernel: Callable[[Trace[X, R]], Trace[X, R]]):
    """
    Higher-order function that creates MCMC chain algorithms from simple kernels.

    This function transforms simple MCMC moves (like metropolis_hastings_step)
    into full-fledged MCMC algorithms with burn-in, thinning, and parallel chains.
    The kernel should save acceptances via state for diagnostics.

    Args:
        mcmc_kernel: MCMC kernel function that takes and returns a trace

    Returns:
        Function that runs MCMC chains with burn-in, thinning, and diagnostics

    Note:
        The mcmc_kernel should use save(accept=...) to record acceptances
        for proper diagnostics collection.
    """

    def run_chain(
        initial_trace: Trace[X, R],
        n_steps: Const[int],
        *,
        burn_in: Const[int] = const(0),
        autocorrelation_resampling: Const[int] = const(1),
        n_chains: Const[int] = const(1),
    ) -> MCMCResult:
        """
        Run MCMC chain with the configured kernel.

        Args:
            initial_trace: Starting trace
            n_steps: Total number of steps to run (before burn-in/thinning)
            burn_in: Number of initial steps to discard as burn-in
            autocorrelation_resampling: Keep every N-th sample (thinning)
            n_chains: Number of parallel chains to run

        Returns:
            MCMCResult with traces, acceptances, and diagnostics
        """

        def scan_fn(trace, _):
            new_trace = mcmc_kernel(trace)
            return new_trace, new_trace

        if n_chains.value == 1:
            # Single chain case
            @state  # Use state decorator to collect acceptances
            def run_scan():
                final_trace, all_traces = jax.lax.scan(
                    scan_fn, initial_trace, jnp.arange(n_steps.value)
                )
                return all_traces

            # Run chain and collect state (including accepts)
            all_traces, chain_state = run_scan()

            # Extract accepts from state
            accepts = chain_state.get("accept", jnp.zeros(n_steps.value))

            # Apply burn-in and thinning
            start_idx = burn_in.value
            end_idx = n_steps.value
            indices = jnp.arange(start_idx, end_idx, autocorrelation_resampling.value)

            # Apply selection to traces and accepts
            final_traces = jax.tree_util.tree_map(
                lambda x: x[indices] if hasattr(x, "shape") and len(x.shape) > 0 else x,
                all_traces,
            )
            final_accepts = accepts[indices]

            # Compute final acceptance rate
            acceptance_rate = jnp.mean(final_accepts)
            final_n_steps = len(indices)

            return MCMCResult(
                traces=final_traces,
                accepts=final_accepts,
                acceptance_rate=acceptance_rate,
                n_steps=const(final_n_steps),
                n_chains=n_chains,
            )

        else:
            # Multiple chains case - use vmap to run parallel chains
            from .pjax import modular_vmap

            # Vectorize the scan function over chains
            vectorized_run = modular_vmap(
                lambda trace: run_chain(
                    trace,
                    n_steps,
                    burn_in=burn_in,
                    autocorrelation_resampling=autocorrelation_resampling,
                    n_chains=const(1),  # Each vectorized call runs 1 chain
                ),
                in_axes=0,
            )

            # Create multiple initial traces by repeating the single trace
            # This creates independent starting points
            initial_traces = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x[None, ...], n_chains.value, axis=0),
                initial_trace,
            )

            # Run multiple chains in parallel
            multi_chain_results = vectorized_run(initial_traces)

            # Combine results from multiple chains
            # Traces shape: (n_chains, n_steps, ...)
            combined_traces = multi_chain_results.traces
            combined_accepts = multi_chain_results.accepts  # (n_chains, n_steps)

            # Per-chain acceptance rates
            acceptance_rates = jnp.mean(combined_accepts, axis=1)  # (n_chains,)
            overall_acceptance_rate = jnp.mean(acceptance_rates)

            final_n_steps = multi_chain_results.n_steps.value

            # Compute between-chain diagnostics using Pytree utilities
            rhat_values = None
            ess_bulk_values = None
            ess_tail_values = None

            if n_chains.value > 1:
                # Extract choices for diagnostics computation
                choices = combined_traces.get_choices()

                # Helper function to compute all diagnostics for scalar arrays
                def compute_all_diagnostics(samples):
                    """Compute all diagnostics if samples are scalar over (chains, steps)."""
                    if samples.ndim == 2:  # (n_chains, n_steps) - scalar samples
                        rhat_val = compute_rhat(samples)
                        ess_bulk_val = compute_ess(samples, kind="bulk")
                        ess_tail_val = compute_ess(samples, kind="tail")
                        # Return as JAX array so we can index into it
                        return jnp.array([rhat_val, ess_bulk_val, ess_tail_val])
                    else:
                        # For non-scalar arrays, return NaN for all diagnostics
                        return jnp.array([jnp.nan, jnp.nan, jnp.nan])

                # Compute all diagnostics in one tree_map pass
                all_diagnostics = jax.tree_util.tree_map(
                    compute_all_diagnostics, choices
                )

                # Extract individual diagnostics using indexing
                rhat_values = jax.tree_util.tree_map(lambda x: x[0], all_diagnostics)
                ess_bulk_values = jax.tree_util.tree_map(
                    lambda x: x[1], all_diagnostics
                )
                ess_tail_values = jax.tree_util.tree_map(
                    lambda x: x[2], all_diagnostics
                )

            return MCMCResult(
                traces=combined_traces,
                accepts=combined_accepts,
                acceptance_rate=overall_acceptance_rate,
                n_steps=const(final_n_steps),
                n_chains=n_chains,
                rhat=rhat_values,
                ess_bulk=ess_bulk_values,
                ess_tail=ess_tail_values,
            )

    return run_chain
