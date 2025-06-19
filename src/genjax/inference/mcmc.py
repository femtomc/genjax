"""
MCMC (Markov Chain Monte Carlo) inference algorithms for GenJAX.

This module provides implementations of standard MCMC algorithms including
Metropolis-Hastings and MALA (Metropolis-Adjusted Langevin Algorithm).
All algorithms use the GFI (Generative Function Interface) for efficient
trace operations.

References
----------

**Metropolis-Hastings Algorithm:**
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953).
  "Equation of state calculations by fast computing machines."
  The Journal of Chemical Physics, 21(6), 1087-1092.
- Hastings, W. K. (1970). "Monte Carlo sampling methods using Markov chains and their applications."
  Biometrika, 57(1), 97-109.

**MALA (Metropolis-Adjusted Langevin Algorithm):**
- Roberts, G. O., & Tweedie, R. L. (1996). "Exponential convergence of Langevin distributions
  and their discrete approximations." Bernoulli, 2(4), 341-363.
- Roberts, G. O., & Rosenthal, J. S. (1998). "Optimal scaling of discrete approximations to
  Langevin diffusions." Journal of the Royal Statistical Society: Series B, 60(1), 255-268.

**Implementation Reference:**
- Gen.jl MALA implementation: https://github.com/probcomp/Gen.jl/blob/master/src/inference/mala.jl
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax.core import (
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
from genjax.distributions import uniform, normal
from genjax.state import save, state
from genjax.pjax import modular_vmap

# Type alias for MCMC kernel functions
MCMCKernel = Callable[[Trace[X, R]], Trace[X, R]]


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
    new_trace, log_weight, _ = target_gf.regenerate(
        current_trace, selection, *args[0], **args[1]
    )

    # MH acceptance step in log space
    log_alpha = jnp.minimum(0.0, log_weight)  # log(min(1, exp(log_weight)))

    # Accept or reject using GenJAX uniform distribution in log space
    log_u = jnp.log(uniform.sample(0.0, 1.0))
    accept = log_u < log_alpha

    # Use tree_map to apply select across all leaves of the traces
    final_trace = jtu.tree_map(
        lambda new_leaf, old_leaf: jax.lax.select(accept, new_leaf, old_leaf),
        new_trace,
        current_trace,
    )

    # Save acceptance as auxiliary state (can be accessed via state decorator)
    save(accept=accept)

    return final_trace


def mala(
    current_trace: Trace[X, R],
    selection: Selection,
    step_size: float,
) -> Trace[X, R]:
    """
    Single MALA (Metropolis-Adjusted Langevin Algorithm) step.

    MALA uses gradient information to make more efficient proposals than
    standard Metropolis-Hastings. The proposal distribution is:

    x_proposed = x_current + step_size^2/2 * ∇log(p(x)) + step_size * ε

    where ε ~ N(0, I) is standard Gaussian noise.

    This implementation follows the approach from Gen.jl, computing both
    forward and backward proposal probabilities to account for the asymmetric
    drift term in the MALA proposal.

    Args:
        current_trace: Current trace state
        selection: Addresses to regenerate (subset of choices)
        step_size: Step size parameter (τ) controlling proposal variance

    Returns:
        Updated trace after MALA step

    State:
        accept: Boolean indicating whether the proposal was accepted
    """
    target_gf = current_trace.get_gen_fn()
    args = current_trace.get_args()
    current_choices = current_trace.get_choices()

    # Use the new GFI.filter method to extract selected choices
    selected_choices, unselected_choices = target_gf.filter(current_choices, selection)

    if selected_choices is None:
        # No choices selected, return current trace unchanged
        save(accept=True)
        return current_trace

    # Create closure to compute gradients with respect to only selected choices
    def log_density_wrt_selected(selected_choices_only):
        # Reconstruct full choices by merging selected with unselected
        if unselected_choices is None:
            # All choices were selected
            full_choices = selected_choices_only
        else:
            # Use the GFI's merge method for all choice structures
            full_choices = target_gf.merge(unselected_choices, selected_choices_only)

        log_density, _ = target_gf.assess(full_choices, *args[0], **args[1])
        return log_density

    # Get gradients with respect to selected choices only
    selected_gradients = jax.grad(log_density_wrt_selected)(selected_choices)

    # Generate MALA proposal for selected choices using tree operations
    def mala_proposal_fn(current_val, grad_val):
        # MALA drift term: step_size^2/2 * gradient
        drift = (step_size**2 / 2.0) * grad_val

        # Gaussian noise term: step_size * N(0,1)
        noise = step_size * normal.sample(0.0, 1.0)

        # Proposed value
        return current_val + drift + noise

    def mala_log_prob_fn(current_val, proposed_val, grad_val):
        # MALA proposal log probability: N(current + drift, step_size)
        drift = (step_size**2 / 2.0) * grad_val
        mean = current_val + drift
        log_probs = normal.logpdf(proposed_val, mean, step_size)
        # Sum over all dimensions to get scalar log probability
        return jnp.sum(log_probs)

    # Apply MALA proposal to all selected choices
    proposed_selected = jtu.tree_map(
        mala_proposal_fn, selected_choices, selected_gradients
    )

    # Compute forward proposal log probabilities
    forward_log_probs = jtu.tree_map(
        mala_log_prob_fn, selected_choices, proposed_selected, selected_gradients
    )

    # Update trace with only the proposed selected choices
    # This ensures discard only contains the keys that were actually changed
    proposed_trace, model_weight, discard = target_gf.update(
        current_trace, proposed_selected, *args[0], **args[1]
    )

    # Get gradients at proposed point with respect to selected choices
    backward_gradients = jax.grad(log_density_wrt_selected)(proposed_selected)

    # Filter discard to only the selected addresses (in case update includes extra keys)
    discarded_selected, _ = target_gf.filter(discard, selection)

    # Compute backward proposal log probabilities using the same function
    backward_log_probs = jtu.tree_map(
        mala_log_prob_fn,
        proposed_selected,
        discarded_selected,
        backward_gradients,
    )

    # Sum up log probabilities using tree_reduce
    forward_log_prob_total = jtu.tree_reduce(jnp.add, forward_log_probs)
    backward_log_prob_total = jtu.tree_reduce(jnp.add, backward_log_probs)

    # MALA acceptance probability
    # Alpha = model_weight + log P(x_old | x_new) - log P(x_new | x_old)
    log_alpha = model_weight + backward_log_prob_total - forward_log_prob_total
    log_alpha = jnp.minimum(0.0, log_alpha)  # min(1, exp(log_alpha))

    # Accept or reject using numerically stable log comparison
    log_u = jnp.log(uniform.sample(0.0, 1.0))
    accept = log_u < log_alpha

    # Select final trace
    final_trace = jtu.tree_map(
        lambda new_leaf, old_leaf: jax.lax.select(accept, new_leaf, old_leaf),
        proposed_trace,
        current_trace,
    )

    # Save acceptance for diagnostics
    save(accept=accept)

    return final_trace


def chain(mcmc_kernel: MCMCKernel):
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
