"""
Discrete Hidden Markov Model with exact inference for testing approximate algorithms.

This module provides forward filtering backward sampling (FFBS) as an exact inference
algorithm that can be used to test approximate inference implementations in GenJAX.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from genjax.core import gen, Pytree, get_choices, Scan, Const
from genjax.distributions import categorical


@Pytree.dataclass
class DiscreteHMMTrace(Pytree):
    """Trace for discrete HMM containing latent states and observations."""

    states: jnp.ndarray  # Shape: (T,) - latent state sequence
    observations: jnp.ndarray  # Shape: (T,) - observation sequence
    log_prob: jnp.ndarray  # Log probability of the sequence


@gen
def hmm_step(carry, x):
    """
    Single step of HMM generation.

    Args:
        carry: Previous state and model parameters
        x: Scan input (unused but required by Scan interface)

    Returns:
        (new_state, observation): Next state and observation
    """
    prev_state, transition_matrix, emission_matrix = carry

    # Sample next state given previous state (using static addresses)
    next_state = categorical(jnp.log(transition_matrix[prev_state])) @ "state"

    # Sample observation given current state (using static addresses)
    obs = categorical(jnp.log(emission_matrix[next_state])) @ "obs"

    # New carry includes state and fixed matrices
    new_carry = (next_state, transition_matrix, emission_matrix)

    return new_carry, obs


@gen
def discrete_hmm_model(
    T: Const[int],  # Number of time steps (static parameter)
    initial_probs: jnp.ndarray,  # Shape: (K,) - initial state probabilities
    transition_matrix: jnp.ndarray,  # Shape: (K, K) - transition probabilities
    emission_matrix: jnp.ndarray,  # Shape: (K, M) - emission probabilities
):
    """
    Discrete HMM generative model using Scan combinator with Const parameters.

    Args:
        T: Number of time steps wrapped in Const (must be > 1)
        initial_probs: Initial state distribution (K states)
        transition_matrix: State transition probabilities (K x K)
        emission_matrix: Observation emission probabilities (K x M observations)

    Returns:
        Sequence of observations
    """
    # Sample initial state
    initial_state = categorical(jnp.log(initial_probs)) @ "state_0"

    # Sample initial observation
    initial_obs = categorical(jnp.log(emission_matrix[initial_state])) @ "obs_0"

    # Use Scan for remaining steps (T.value is static)
    scan_fn = Scan(hmm_step, length=T.value - 1)
    init_carry = (initial_state, transition_matrix, emission_matrix)
    final_carry, remaining_obs = scan_fn(init_carry, None) @ "scan_steps"

    # Combine initial and remaining observations
    all_obs = jnp.concatenate([jnp.array([initial_obs]), remaining_obs])
    return all_obs


def discrete_hmm_model_factory(T: int):
    """
    Factory function to create HMM models with static length.

    Deprecated: Use discrete_hmm_model() with Const[int] parameter instead.

    Args:
        T: Number of time steps (must be static, > 1)

    Returns:
        Generative function for HMM with fixed length T
    """

    # Create a generative function that uses the new model with Const parameter
    @gen
    def hmm_model_closure(initial_probs, transition_matrix, emission_matrix):
        result = (
            discrete_hmm_model(
                Const(T), initial_probs, transition_matrix, emission_matrix
            )
            @ "model"
        )
        return result

    return hmm_model_closure


def forward_filter(
    observations: jnp.ndarray,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward filtering algorithm for discrete HMM.

    Computes forward filter distributions α_t(x_t) = p(x_t | y_{1:t})
    using log-space calculations for numerical stability.

    Args:
        observations: Observed sequence (T,)
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)

    Returns:
        alpha: Forward filter distributions (T, K) in log space
        log_marginal: Log marginal likelihood of observations
    """
    T = len(observations)
    K = len(initial_probs)

    # Convert to log space
    log_initial = jnp.log(initial_probs)
    log_transition = jnp.log(transition_matrix)
    log_emission = jnp.log(emission_matrix)

    # Initialize alpha
    alpha = jnp.zeros((T, K))

    # Initial step: α_0(x_0) = p(y_0 | x_0) * p(x_0)
    alpha = alpha.at[0].set(log_emission[:, observations[0]] + log_initial)

    def scan_step(carry, t):
        """Scan step for forward filtering."""
        prev_alpha = carry

        # Compute α_t(x_t) = p(y_t | x_t) * Σ_{x_{t-1}} α_{t-1}(x_{t-1}) * p(x_t | x_{t-1})
        # In log space: log α_t(x_t) = log p(y_t | x_t) + logsumexp(log α_{t-1} + log p(x_t | x_{t-1}))
        transition_scores = prev_alpha[:, None] + log_transition  # (K, K)
        prediction = jax.scipy.special.logsumexp(transition_scores, axis=0)  # (K,)

        current_alpha = log_emission[:, observations[t]] + prediction

        return current_alpha, current_alpha

    # Run forward pass for t = 1, ..., T-1
    if T > 1:
        _, alphas = jax.lax.scan(scan_step, alpha[0], jnp.arange(1, T))
        alpha = alpha.at[1:].set(alphas)

    # Compute log marginal likelihood
    log_marginal = jax.scipy.special.logsumexp(alpha[-1])

    # Normalize alpha to get proper probabilities
    alpha_normalized = alpha - jax.scipy.special.logsumexp(alpha, axis=1, keepdims=True)

    return alpha_normalized, log_marginal


def backward_sample(
    alpha: jnp.ndarray,
    transition_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backward sampling algorithm for discrete HMM.

    Samples a latent state sequence given forward filter distributions
    using distributions directly as samplers.

    Args:
        alpha: Forward filter distributions (T, K) in log space (normalized)
        transition_matrix: Transition probabilities (K, K)

    Returns:
        states: Sampled latent state sequence (T,)
    """
    T, K = alpha.shape
    log_transition = jnp.log(transition_matrix)

    states = jnp.zeros(T, dtype=jnp.int32)

    # Sample final state from final alpha
    final_state = categorical.sample(logits=alpha[-1])
    states = states.at[-1].set(final_state)

    # Sample remaining states backwards using scan
    def scan_step(next_state, t):
        # p(x_t | x_{t+1}, y_{1:t}) ∝ α_t(x_t) * p(x_{t+1} | x_t)
        log_probs = alpha[t] + log_transition[:, next_state]
        state = categorical.sample(logits=log_probs)
        return state, state

    if T > 1:
        # Run scan over time indices in reverse order
        time_indices = jnp.arange(T - 2, -1, -1)
        _, sampled_states = jax.lax.scan(scan_step, final_state, time_indices)

        # Update states array with sampled states
        states = states.at[:-1].set(sampled_states[::-1])

    return states


def forward_filtering_backward_sampling(
    observations: jnp.ndarray,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> DiscreteHMMTrace:
    """
    Complete forward filtering backward sampling algorithm.

    Performs exact inference in a discrete HMM by computing forward filter
    distributions and then sampling a latent state sequence.

    Args:
        observations: Observed sequence (T,)
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)

    Returns:
        DiscreteHMMTrace containing sampled states and log probability
    """
    # Forward filtering
    alpha, log_marginal = forward_filter(
        observations, initial_probs, transition_matrix, emission_matrix
    )

    # Backward sampling
    states = backward_sample(alpha, transition_matrix)

    # Compute log probability of sampled sequence
    log_prob = compute_sequence_log_prob(
        states, observations, initial_probs, transition_matrix, emission_matrix
    )

    return DiscreteHMMTrace(
        states=states,
        observations=observations,
        log_prob=log_prob,
    )


def compute_sequence_log_prob(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log probability of a state-observation sequence using scan.

    Args:
        states: State sequence (T,)
        observations: Observation sequence (T,)
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)

    Returns:
        Log probability of the sequence
    """
    T = len(states)

    # Initial state and observation probabilities
    log_prob = jnp.log(initial_probs[states[0]])
    log_prob += jnp.log(emission_matrix[states[0], observations[0]])

    # Use scan for remaining steps (assume T > 1)
    def scan_step(carry_log_prob, t):
        """Accumulate log probabilities for transition and emission."""
        # Add transition probability
        transition_log_prob = jnp.log(transition_matrix[states[t - 1], states[t]])
        # Add emission probability
        emission_log_prob = jnp.log(emission_matrix[states[t], observations[t]])

        new_log_prob = carry_log_prob + transition_log_prob + emission_log_prob
        return new_log_prob, new_log_prob

    # Run scan over remaining time steps
    time_indices = jnp.arange(1, T)
    final_log_prob, _ = jax.lax.scan(scan_step, log_prob, time_indices)

    return final_log_prob


def sample_hmm_dataset(
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
    T: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Sample a dataset from the discrete HMM model.

    Args:
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)
        T: Number of time steps

    Returns:
        Tuple of (true_states, observations, constraints)
    """
    # Use the new discrete_hmm_model with Const[...] pattern directly
    from genjax.core import const

    trace = discrete_hmm_model.simulate(
        const(T), initial_probs, transition_matrix, emission_matrix
    )

    # Extract states and observations from trace
    choices = get_choices(trace)
    observations = trace.get_retval()

    # Extract states: initial state + states from scan (assume T > 1)
    initial_state = choices["state_0"]
    if T > 1:
        # Get vectorized states from scan
        scan_choices = choices["scan_steps"]
        scan_states = scan_choices["state"]  # This will be a vector of states
        states = jnp.concatenate([jnp.array([initial_state]), scan_states])
    else:
        states = jnp.array([initial_state])

    # Create constraints from observations
    if T == 1:
        constraints = {"obs_0": observations[0]}
    else:
        constraints = {
            "obs_0": observations[0],
            "scan_steps": {"obs": observations[1:]},
        }

    return states, observations, constraints
