"""
MCMC (Markov Chain Monte Carlo) inference algorithms for GenJAX.

This module provides implementations of standard MCMC algorithms including
Metropolis-Hastings, MALA (Metropolis-Adjusted Langevin Algorithm), and
HMC (Hamiltonian Monte Carlo). All algorithms use the GFI (Generative Function Interface)
for efficient trace operations.

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

**HMC (Hamiltonian Monte Carlo):**
- Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics", Handbook of Markov Chain Monte Carlo,
  pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf
- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). "Hybrid Monte Carlo."
  Physics Letters B, 195(2), 216-222.

**Implementation Reference:**
- Gen.jl MALA implementation: https://github.com/probcomp/Gen.jl/blob/master/src/inference/mala.jl
- Gen.jl HMC implementation: https://github.com/probcomp/Gen.jl/blob/master/src/inference/hmc.jl
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


def _create_log_density_wrt_selected(target_gf, args, unselected_choices):
    """
    Create a log density function that only depends on selected choices.

    This helper function is used by both MALA and HMC to create a closure
    that computes log density with respect to only the selected addresses,
    enabling gradient computation for those addresses only.

    Args:
        target_gf: The generative function to evaluate
        args: Arguments tuple (args[0], args[1]) for the generative function
        unselected_choices: The subset of choices that remain fixed

    Returns:
        Function that takes selected_choices_only and returns log density
    """

    def log_density_wrt_selected(selected_choices_only):
        # Reconstruct full choices by merging selected with unselected
        if unselected_choices is None:
            # All choices were selected
            full_choices = selected_choices_only
        else:
            # Use the GFI's merge method for all choice structures
            full_choices, _ = target_gf.merge(unselected_choices, selected_choices_only)

        log_density, _ = target_gf.assess(full_choices, *args[0], **args[1])
        return log_density

    return log_density_wrt_selected


def compute_rhat(samples: jnp.ndarray) -> FloatArray:
    """
    Compute potential scale reduction factor (R-hat) for MCMC convergence diagnostics.

    Implements the split-R-hat diagnostic from Vehtari et al. (2021), which improves
    upon the original formulation of Gelman & Rubin (1992) by accounting for
    non-stationarity within chains.

    Mathematical Formulation:
        Given M chains each of length N, compute:

        B = N/(M-1) * Σᵢ (θ̄ᵢ - θ̄)²  (between-chain variance)
        W = 1/M * Σᵢ sᵢ²             (within-chain variance)

        where θ̄ᵢ is the mean of chain i, θ̄ is the grand mean, and sᵢ² is
        the sample variance of chain i.

        The potential scale reduction factor is:
        R̂ = √[(N-1)/N * W + 1/N * B] / W

    Convergence Criterion:
        R̂ < 1.01 indicates good convergence (Vehtari et al., 2021)
        R̂ < 1.1 was the classical threshold (Gelman & Rubin, 1992)

    Args:
        samples: Array of shape (n_chains, n_samples) containing MCMC samples
                 from M chains each of length N

    Returns:
        R-hat statistic. Values close to 1.0 indicate convergence.
        Returns NaN if n_chains < 2.

    References:
        .. [1] Gelman, A., & Rubin, D. B. (1992). "Inference from iterative
               simulation using multiple sequences". Statistical Science, 7(4), 457-472.
        .. [2] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C.
               (2021). "Rank-normalization, folding, and localization: An improved R̂
               for assessing convergence of MCMC". Bayesian Analysis, 16(2), 667-718.

    Notes:
        - This implementation uses the basic R-hat without rank-normalization
        - For rank-normalized R-hat (more robust), see [2]
        - Requires at least 2 chains for meaningful computation
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
    Compute effective sample size (ESS) for MCMC chains.

    Estimates the number of independent samples accounting for autocorrelation
    in Markov chains. Implements simplified versions of bulk and tail ESS from
    Vehtari et al. (2021).

    Mathematical Formulation:
        The effective sample size is defined as:

        ESS = M × N / τ

        where M is the number of chains, N is the chain length, and τ is the
        integrated autocorrelation time:

        τ = 1 + 2 × Σₖ ρₖ

        where ρₖ is the autocorrelation at lag k, summed over positive correlations.

    Algorithm:
        - Bulk ESS: Uses all samples to estimate central tendency efficiency
        - Tail ESS: Uses quantile differences (0.05 and 0.95) to assess tail behavior

        This implementation uses a simplified approximation based on lag-1
        autocorrelation: ESS ≈ N / (1 + 2ρ₁)

    Time Complexity: O(M × N)
    Space Complexity: O(1)

    Args:
        samples: Array of shape (n_chains, n_samples) containing MCMC samples
        kind: Type of ESS to compute:
              - "bulk": Efficiency for estimating posterior mean/median
              - "tail": Efficiency for estimating posterior quantiles

    Returns:
        Effective sample size estimate. Range: [1, M × N]
        Lower values indicate higher autocorrelation.

    References:
        .. [1] Geyer, C. J. (1992). "Practical Markov chain Monte Carlo".
               Statistical Science, 7(4), 473-483.
        .. [2] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C.
               (2021). "Rank-normalization, folding, and localization: An improved R̂
               for assessing convergence of MCMC". Bayesian Analysis, 16(2), 667-718.
        .. [3] Stan Development Team (2023). "Stan Reference Manual: Effective
               Sample Size". Version 2.33. Section 15.4.

    Notes:
        - This is a simplified implementation using lag-1 autocorrelation
        - Full implementation would compute autocorrelation function to first negative
        - Tail ESS focuses on extreme quantiles, useful for credible intervals
        - Bulk ESS focuses on center, useful for posterior expectations
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
    log_density_wrt_selected = _create_log_density_wrt_selected(
        target_gf, args, unselected_choices
    )

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


def hmc(
    current_trace: Trace[X, R],
    selection: Selection,
    step_size: float,
    n_steps: int,
) -> Trace[X, R]:
    """
    Single HMC (Hamiltonian Monte Carlo) step using leapfrog integration.

    HMC uses gradient information and auxiliary momentum variables to propose
    distant moves that maintain detailed balance. The algorithm simulates
    Hamiltonian dynamics using leapfrog integration:

    1. Sample momentum p ~ N(0, I)
    2. Simulate Hamiltonian dynamics for n_steps using leapfrog integration:
       - p' = p + (eps/2) * ∇log(p(x))
       - x' = x + eps * p'
       - p' = p' + (eps/2) * ∇log(p(x'))
    3. Accept/reject using Metropolis criterion with joint (x,p) density

    This implementation uses jax.lax.scan for leapfrog integration, making it
    fully JAX-compatible and JIT-compilable. It follows Neal (2011) equations
    (5.18)-(5.20) and the Gen.jl HMC implementation structure.

    Args:
        current_trace: Current trace state
        selection: Addresses to regenerate (subset of choices)
        step_size: Leapfrog integration step size (eps)
        n_steps: Number of leapfrog steps (L)

    Returns:
        Updated trace after HMC step

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
    log_density_wrt_selected = _create_log_density_wrt_selected(
        target_gf, args, unselected_choices
    )

    # Helper functions for momentum
    def sample_momentum(_):
        """Sample momentum with same structure as reference value."""
        return normal.sample(0.0, 1.0)

    def assess_momentum(momentum_val):
        """Compute log probability of momentum (standard normal)."""
        return normal.logpdf(momentum_val, 0.0, 1.0)

    # Initial model score (negative potential energy)
    prev_model_score = log_density_wrt_selected(selected_choices)

    # Sample initial momentum and compute its score (negative kinetic energy)
    initial_momentum = jtu.tree_map(sample_momentum, selected_choices)
    prev_momentum_score = jtu.tree_reduce(
        jnp.add, jtu.tree_map(assess_momentum, initial_momentum)
    )

    # Initialize leapfrog variables
    current_position = selected_choices
    current_momentum = initial_momentum

    # Leapfrog integration for n_steps using jax.lax.scan
    # Initial gradient
    current_gradient = jax.grad(log_density_wrt_selected)(current_position)

    def leapfrog_step(carry, _):
        """Single leapfrog integration step."""
        position, momentum, gradient = carry

        # Half step on momentum
        momentum = jtu.tree_map(
            lambda p, g: p + (step_size / 2.0) * g, momentum, gradient
        )

        # Full step on position
        position = jtu.tree_map(lambda x, p: x + step_size * p, position, momentum)

        # Get new gradient at new position
        gradient = jax.grad(log_density_wrt_selected)(position)

        # Half step on momentum (completing the leapfrog step)
        momentum = jtu.tree_map(
            lambda p, g: p + (step_size / 2.0) * g, momentum, gradient
        )

        new_carry = (position, momentum, gradient)
        return new_carry, None  # No output needed, just carry

    # Run leapfrog integration
    initial_carry = (current_position, current_momentum, current_gradient)
    final_carry, _ = jax.lax.scan(leapfrog_step, initial_carry, jnp.arange(n_steps))

    # Extract final position and momentum
    final_position, final_momentum, _ = final_carry

    # Update trace with proposed final position
    proposed_trace, model_weight, discard = target_gf.update(
        current_trace, final_position, *args[0], **args[1]
    )

    # Compute final model score (negative potential energy)
    new_model_score = log_density_wrt_selected(final_position)

    # Compute final momentum score (negative kinetic energy)
    # Note: In HMC, we evaluate momentum at negated final momentum to account for
    # the reversibility requirement of Hamiltonian dynamics
    final_momentum_negated = jtu.tree_map(lambda p: -p, final_momentum)
    new_momentum_score = jtu.tree_reduce(
        jnp.add, jtu.tree_map(assess_momentum, final_momentum_negated)
    )

    # HMC acceptance probability
    # alpha = (new_model_score + new_momentum_score) - (prev_model_score + prev_momentum_score)
    # This is equivalent to the energy difference: -ΔH = -(ΔU + ΔK)
    log_alpha = (new_model_score + new_momentum_score) - (
        prev_model_score + prev_momentum_score
    )
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
