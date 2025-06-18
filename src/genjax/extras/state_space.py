"""
State space models with exact inference for testing approximate algorithms.

This module provides exact inference algorithms for discrete and continuous state space models
including forward filtering backward sampling (FFBS) for discrete HMMs and Kalman filtering
for linear Gaussian models. These serve as baselines for testing approximate inference methods.

References
----------

Hidden Markov Models:
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications
  in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Chapter 13.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective, Chapter 17.

Kalman Filtering and Linear Gaussian State Space Models:
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.
  Journal of Basic Engineering, 82(1), 35-45.
- Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective, Chapter 18.

Forward Filtering Backward Sampling (FFBS):
- Carter, C. K., & Kohn, R. (1994). On Gibbs sampling for state space models.
  Biometrika, 81(3), 541-553.
- Frühwirth-Schnatter, S. (1994). Data augmentation and dynamic linear models.
  Journal of Time Series Analysis, 15(2), 183-202.

Rauch-Tung-Striebel Smoothing:
- Rauch, H. E., Striebel, C. T., & Tung, F. (1965). Maximum likelihood estimates of
  linear dynamic systems. AIAA Journal, 3(8), 1445-1450.
- Anderson, B. D., & Moore, J. B. (2012). Optimal Filtering. Dover Publications.
"""

import jax
import jax.numpy as jnp

from genjax.core import gen, Pytree, get_choices, Const, const
from genjax.distributions import categorical, multivariate_normal
from genjax.pjax import modular_vmap


@Pytree.dataclass
class DiscreteHMMTrace(Pytree):
    """Trace for discrete HMM containing latent states and observations."""

    states: jnp.ndarray  # Shape: (T,) - latent state sequence
    observations: jnp.ndarray  # Shape: (T,) - observation sequence
    log_prob: jnp.ndarray  # Log probability of the sequence


def _discrete_hmm(
    prev_state,  # Previous state (dummy for t=0) - can be jnp.ndarray or Const
    time_index,  # Current time step - can be jnp.ndarray or Const
    initial_probs: jnp.ndarray,  # Shape: (K,) - initial state probabilities
    transition_matrix: jnp.ndarray,  # Shape: (K, K) - transition probabilities
    emission_matrix: jnp.ndarray,  # Shape: (K, M) - emission probabilities
):
    """
    Discrete HMM kernel generative function.

    This function can be iteratively applied to generate state-observation sequences.
    For t=0, it samples from initial distribution; for t>0, it transitions from prev_state.

    Args:
        prev_state: Previous latent state (ignored when time_index == 0)
        time_index: Current time step (0 for initial, >0 for transitions)
        initial_probs: Initial state distribution (K states)
        transition_matrix: State transition probabilities (K x K)
        emission_matrix: Observation emission probabilities (K x M observations)

    Returns:
        tuple: (new_state, time_index + 1, initial_probs, transition_matrix, emission_matrix)
    """
    # For t=0: sample from initial distribution
    # For t>0: sample from transition distribution
    is_initial = time_index == 0

    # Sample current state
    # Select between initial and transition logits
    initial_logits = jnp.log(initial_probs)
    transition_logits = jnp.log(transition_matrix[prev_state])
    current_logits = jax.lax.select(is_initial, initial_logits, transition_logits)
    current_state = categorical(current_logits) @ "state"

    # Sample observation given current state (accessible via GFI)
    categorical(jnp.log(emission_matrix[current_state])) @ "obs"

    # Return new state and carry all other arguments
    return (
        current_state,
        time_index + 1,
        initial_probs,
        transition_matrix,
        emission_matrix,
    )


# Apply the @gen decorator explicitly
discrete_hmm = gen(_discrete_hmm)


def forward_filter(
    observations: jnp.ndarray,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    T: Const[int],
    K: Const[int] = const(1),  # Number of sequences to generate
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Sample a dataset from the discrete HMM model using kernel function.

    Args:
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)
        T: Number of time steps
        K: Number of sequences to generate

    Returns:
        Tuple of (true_states, observations, constraints)
        - true_states: Shape (K, T) if K > 1, else (T,)
        - observations: Shape (K, T) if K > 1, else (T,)
    """

    def sample_single_sequence():
        def scan_step(carry, _):
            trace = discrete_hmm.simulate(*carry)
            # Extract the new carry from the trace return value
            new_carry = trace.get_retval()
            return new_carry, trace

        # Initial carry: dummy state, time=0, and all parameters
        dummy_state = jnp.array(0)
        init_carry = (
            dummy_state,
            jnp.array(0),
            initial_probs,
            transition_matrix,
            emission_matrix,
        )

        # Use jax.lax.scan to generate T timesteps
        final_carry, traces = jax.lax.scan(scan_step, init_carry, None, length=T.value)

        # Extract states and observations from traces using tree operations
        all_choices = jax.tree.map(get_choices, traces)
        states = all_choices["state"]  # Shape: (T,)
        observations = all_choices["obs"]  # Shape: (T,)

        return states, observations

    if K.value == 1:
        # Single sequence
        states, observations = sample_single_sequence()
        constraints = {"obs": observations}
    else:
        # Multiple sequences using modular_vmap
        vectorized_sample = modular_vmap(
            sample_single_sequence, in_axes=(), axis_size=K.value
        )
        # Generate K sequences
        all_states, all_observations = vectorized_sample()
        states = all_states  # Shape: (K, T)
        observations = all_observations  # Shape: (K, T)
        constraints = {"obs": observations}

    return states, observations, constraints


# =============================================================================
# LINEAR GAUSSIAN STATE SPACE MODEL
# =============================================================================


@Pytree.dataclass
class LinearGaussianTrace(Pytree):
    """Trace for linear Gaussian state space model containing latent states and observations."""

    states: jnp.ndarray  # Shape: (T, d_state) - latent state sequence
    observations: jnp.ndarray  # Shape: (T, d_obs) - observation sequence
    log_prob: jnp.ndarray  # Log probability of the sequence


def _linear_gaussian(
    prev_state: jnp.ndarray,  # Previous state (dummy for t=0)
    time_index: jnp.ndarray,  # Current time step
    initial_mean: jnp.ndarray,  # Shape: (d_state,) - initial state mean
    initial_cov: jnp.ndarray,  # Shape: (d_state, d_state) - initial state covariance
    A: jnp.ndarray,  # Shape: (d_state, d_state) - transition matrix
    Q: jnp.ndarray,  # Shape: (d_state, d_state) - process noise covariance
    C: jnp.ndarray,  # Shape: (d_obs, d_state) - observation matrix
    R: jnp.ndarray,  # Shape: (d_obs, d_obs) - observation noise covariance
):
    """
    Linear Gaussian state space model kernel generative function.

    This function can be iteratively applied to generate state-observation sequences.
    For t=0, it samples from initial distribution; for t>0, it transitions from prev_state.

    Model:
        x_0 ~ N(initial_mean, initial_cov)
        x_t = A @ x_{t-1} + w_t, w_t ~ N(0, Q)
        y_t = C @ x_t + v_t, v_t ~ N(0, R)

    Args:
        prev_state: Previous latent state (ignored when time_index == 0)
        time_index: Current time step (0 for initial, >0 for transitions)
        initial_mean: Initial state mean
        initial_cov: Initial state covariance
        A: State transition matrix
        Q: Process noise covariance
        C: Observation matrix
        R: Observation noise covariance

    Returns:
        tuple: (new_state, time_index + 1, initial_mean, initial_cov, A, Q, C, R)
    """
    # For t=0: sample from initial distribution
    # For t>0: sample from transition distribution
    is_initial = time_index == 0

    # Sample current state
    # Select between initial and transition parameters
    transition_mean = A @ prev_state
    current_mean = jax.lax.select(is_initial, initial_mean, transition_mean)
    current_cov = jax.lax.select(is_initial, initial_cov, Q)
    current_state = multivariate_normal(current_mean, current_cov) @ "state"

    # Sample observation given current state (accessible via GFI)
    obs_mean = C @ current_state
    multivariate_normal(obs_mean, R) @ "obs"

    # Return new state and carry all other arguments
    return current_state, time_index + 1, initial_mean, initial_cov, A, Q, C, R


# Apply the @gen decorator explicitly
linear_gaussian = gen(_linear_gaussian)


def kalman_filter(
    observations: jnp.ndarray,  # Shape: (T, d_obs)
    initial_mean: jnp.ndarray,  # Shape: (d_state,)
    initial_cov: jnp.ndarray,  # Shape: (d_state, d_state)
    A: jnp.ndarray,  # Shape: (d_state, d_state)
    Q: jnp.ndarray,  # Shape: (d_state, d_state)
    C: jnp.ndarray,  # Shape: (d_obs, d_state)
    R: jnp.ndarray,  # Shape: (d_obs, d_obs)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Kalman filtering for linear Gaussian state space model.

    Computes the filtered state distributions p(x_t | y_{1:t}) and
    log marginal likelihood of observations.

    Args:
        observations: Observed sequence (T, d_obs)
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)

    Returns:
        filtered_means: Filtered state means (T, d_state)
        filtered_covs: Filtered state covariances (T, d_state, d_state)
        log_marginal: Log marginal likelihood of observations
    """
    T, d_obs = observations.shape
    d_state = len(initial_mean)

    # Initialize storage
    filtered_means = jnp.zeros((T, d_state))
    filtered_covs = jnp.zeros((T, d_state, d_state))
    log_marginal = 0.0

    # Initial step
    # Predict: x_0|∅ ~ N(initial_mean, initial_cov)
    predicted_mean = initial_mean
    predicted_cov = initial_cov

    # Update with first observation
    innovation = observations[0] - C @ predicted_mean
    innovation_cov = C @ predicted_cov @ C.T + R
    kalman_gain = predicted_cov @ C.T @ jnp.linalg.inv(innovation_cov)

    filtered_mean = predicted_mean + kalman_gain @ innovation
    filtered_cov = predicted_cov - kalman_gain @ C @ predicted_cov

    filtered_means = filtered_means.at[0].set(filtered_mean)
    filtered_covs = filtered_covs.at[0].set(filtered_cov)

    # Add to log marginal likelihood
    log_marginal += jax.scipy.stats.multivariate_normal.logpdf(
        innovation, jnp.zeros_like(innovation), innovation_cov
    )

    def scan_step(carry, t):
        """Scan step for Kalman filtering."""
        prev_filtered_mean, prev_filtered_cov, cum_log_marginal = carry

        # Predict: x_t|y_{1:t-1} ~ N(A @ x_{t-1|t-1}, A @ P_{t-1|t-1} @ A^T + Q)
        predicted_mean = A @ prev_filtered_mean
        predicted_cov = A @ prev_filtered_cov @ A.T + Q

        # Update with observation y_t
        innovation = observations[t] - C @ predicted_mean
        innovation_cov = C @ predicted_cov @ C.T + R
        kalman_gain = predicted_cov @ C.T @ jnp.linalg.inv(innovation_cov)

        filtered_mean = predicted_mean + kalman_gain @ innovation
        filtered_cov = predicted_cov - kalman_gain @ C @ predicted_cov

        # Update log marginal likelihood
        new_log_marginal = (
            cum_log_marginal
            + jax.scipy.stats.multivariate_normal.logpdf(
                innovation, jnp.zeros_like(innovation), innovation_cov
            )
        )

        new_carry = (filtered_mean, filtered_cov, new_log_marginal)
        return new_carry, (filtered_mean, filtered_cov)

    # Run filtering for t = 1, ..., T-1
    if T > 1:
        init_carry = (filtered_mean, filtered_cov, log_marginal)
        final_carry, step_results = jax.lax.scan(
            scan_step, init_carry, jnp.arange(1, T)
        )
        final_means, final_covs = step_results
        filtered_means = filtered_means.at[1:].set(final_means)
        filtered_covs = filtered_covs.at[1:].set(final_covs)
        _, _, log_marginal = final_carry

    return filtered_means, filtered_covs, log_marginal


def kalman_smoother(
    observations: jnp.ndarray,  # Shape: (T, d_obs)
    initial_mean: jnp.ndarray,  # Shape: (d_state,)
    initial_cov: jnp.ndarray,  # Shape: (d_state, d_state)
    A: jnp.ndarray,  # Shape: (d_state, d_state)
    Q: jnp.ndarray,  # Shape: (d_state, d_state)
    C: jnp.ndarray,  # Shape: (d_obs, d_state)
    R: jnp.ndarray,  # Shape: (d_obs, d_obs)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kalman smoothing for linear Gaussian state space model.

    Computes the smoothed state distributions p(x_t | y_{1:T}) using
    the Rauch-Tung-Striebel smoother.

    Args:
        observations: Observed sequence (T, d_obs)
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)

    Returns:
        smoothed_means: Smoothed state means (T, d_state)
        smoothed_covs: Smoothed state covariances (T, d_state, d_state)
    """
    T, d_obs = observations.shape
    d_state = len(initial_mean)

    # First run forward Kalman filter
    filtered_means, filtered_covs, _ = kalman_filter(
        observations, initial_mean, initial_cov, A, Q, C, R
    )

    # Initialize smoother with final filtered values
    smoothed_means = jnp.zeros((T, d_state))
    smoothed_covs = jnp.zeros((T, d_state, d_state))

    smoothed_means = smoothed_means.at[-1].set(filtered_means[-1])
    smoothed_covs = smoothed_covs.at[-1].set(filtered_covs[-1])

    def scan_step(carry, t):
        """Scan step for backward smoothing."""
        next_smoothed_mean, next_smoothed_cov = carry

        # Predicted values for time t+1
        predicted_mean = A @ filtered_means[t]
        predicted_cov = A @ filtered_covs[t] @ A.T + Q

        # Smoother gain
        smoother_gain = filtered_covs[t] @ A.T @ jnp.linalg.inv(predicted_cov)

        # Smoothed estimates for time t
        smoothed_mean = filtered_means[t] + smoother_gain @ (
            next_smoothed_mean - predicted_mean
        )
        smoothed_cov = (
            filtered_covs[t]
            + smoother_gain @ (next_smoothed_cov - predicted_cov) @ smoother_gain.T
        )

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    if T > 1:
        # Run backward smoothing for t = T-2, ..., 0
        time_indices = jnp.arange(T - 2, -1, -1)
        init_carry = (smoothed_means[-1], smoothed_covs[-1])
        _, step_results = jax.lax.scan(scan_step, init_carry, time_indices)
        step_means, step_covs = step_results

        # Update smoothed arrays (reverse order since we went backwards)
        smoothed_means = smoothed_means.at[:-1].set(step_means[::-1])
        smoothed_covs = smoothed_covs.at[:-1].set(step_covs[::-1])

    return smoothed_means, smoothed_covs


def sample_linear_gaussian_dataset(
    initial_mean: jnp.ndarray,
    initial_cov: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
    T: Const[int],
    K: Const[int] = const(1),  # Number of sequences to generate
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Sample a dataset from the linear Gaussian state space model using kernel function.

    Args:
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)
        T: Number of time steps
        K: Number of sequences to generate

    Returns:
        Tuple of (true_states, observations, constraints)
        - true_states: Shape (K, T, d_state) if K > 1, else (T, d_state)
        - observations: Shape (K, T, d_obs) if K > 1, else (T, d_obs)
    """

    def sample_single_sequence():
        def scan_step(carry, _):
            trace = linear_gaussian.simulate(*carry)
            # Extract the new carry from the trace return value
            new_carry = trace.get_retval()
            return new_carry, trace

        # Initial carry: dummy state, time=0, and all parameters
        dummy_state = jnp.zeros_like(initial_mean)  # Match state dimension
        init_carry = (dummy_state, jnp.array(0), initial_mean, initial_cov, A, Q, C, R)

        # Use jax.lax.scan to generate T timesteps
        final_carry, traces = jax.lax.scan(scan_step, init_carry, None, length=T.value)

        # Extract states and observations from traces using tree operations
        all_choices = jax.tree.map(get_choices, traces)
        states = all_choices["state"]  # Shape: (T, d_state)
        observations = all_choices["obs"]  # Shape: (T, d_obs)

        return states, observations

    if K.value == 1:
        # Single sequence
        states, observations = sample_single_sequence()
        constraints = {"obs": observations}
    else:
        # Multiple sequences using modular_vmap
        vectorized_sample = modular_vmap(
            sample_single_sequence, in_axes=(), axis_size=K.value
        )
        # Generate K sequences
        all_states, all_observations = vectorized_sample()
        states = all_states  # Shape: (K, T, d_state)
        observations = all_observations  # Shape: (K, T, d_obs)
        constraints = {"obs": observations}

    return states, observations, constraints


# =============================================================================
# INFERENCE TESTING API
# =============================================================================


def discrete_hmm_test_dataset(
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
    T: Const[int],
) -> dict:
    """
    Generate test dataset for discrete HMM with standardized format.

    Args:
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)
        T: Number of time steps

    Returns:
        Dictionary with keys:
        - "z": latent state sequence (T,)
        - "obs": observation sequence (T,)
    """
    states, observations, _ = sample_hmm_dataset(
        initial_probs, transition_matrix, emission_matrix, T
    )
    return {"z": states, "obs": observations}


def discrete_hmm_exact_log_marginal(
    observations: jnp.ndarray,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute exact log marginal likelihood for discrete HMM.

    Args:
        observations: Observed sequence (T,)
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)

    Returns:
        Log marginal likelihood of observations
    """
    _, log_marginal = forward_filter(
        observations, initial_probs, transition_matrix, emission_matrix
    )
    return log_marginal


def linear_gaussian_test_dataset(
    initial_mean: jnp.ndarray,
    initial_cov: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
    T: Const[int],
) -> dict:
    """
    Generate test dataset for linear Gaussian SSM with standardized format.

    Args:
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)
        T: Number of time steps

    Returns:
        Dictionary with keys:
        - "z": latent state sequence (T, d_state)
        - "obs": observation sequence (T, d_obs)
    """
    states, observations, _ = sample_linear_gaussian_dataset(
        initial_mean, initial_cov, A, Q, C, R, T
    )
    return {"z": states, "obs": observations}


def linear_gaussian_exact_log_marginal(
    observations: jnp.ndarray,
    initial_mean: jnp.ndarray,
    initial_cov: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute exact log marginal likelihood for linear Gaussian SSM.

    Args:
        observations: Observed sequence (T, d_obs)
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)

    Returns:
        Log marginal likelihood of observations
    """
    _, _, log_marginal = kalman_filter(
        observations, initial_mean, initial_cov, A, Q, C, R
    )
    return log_marginal


def discrete_hmm_inference_problem(
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
    T: Const[int],
) -> tuple[dict, jnp.ndarray]:
    """
    Generate inference problem for discrete HMM: dataset + exact log marginal.

    This is a convenience function that combines dataset generation and exact
    inference to create complete inference problems for testing approximate methods.

    Args:
        initial_probs: Initial state distribution (K,)
        transition_matrix: Transition probabilities (K, K)
        emission_matrix: Emission probabilities (K, M)
        T: Number of time steps

    Returns:
        Tuple of (dataset, exact_log_marginal) where:
        - dataset: Dictionary with keys "z" (latent) and "obs" (observed)
        - exact_log_marginal: Log marginal likelihood of observations
    """
    dataset = discrete_hmm_test_dataset(
        initial_probs, transition_matrix, emission_matrix, T
    )
    log_marginal = discrete_hmm_exact_log_marginal(
        dataset["obs"], initial_probs, transition_matrix, emission_matrix
    )
    return dataset, log_marginal


def linear_gaussian_inference_problem(
    initial_mean: jnp.ndarray,
    initial_cov: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
    T: Const[int],
) -> tuple[dict, jnp.ndarray]:
    """
    Generate inference problem for linear Gaussian SSM: dataset + exact log marginal.

    This is a convenience function that combines dataset generation and exact
    inference to create complete inference problems for testing approximate methods.

    Args:
        initial_mean: Initial state mean (d_state,)
        initial_cov: Initial state covariance (d_state, d_state)
        A: State transition matrix (d_state, d_state)
        Q: Process noise covariance (d_state, d_state)
        C: Observation matrix (d_obs, d_state)
        R: Observation noise covariance (d_obs, d_obs)
        T: Number of time steps

    Returns:
        Tuple of (dataset, exact_log_marginal) where:
        - dataset: Dictionary with keys "z" (latent) and "obs" (observed)
        - exact_log_marginal: Log marginal likelihood of observations
    """
    dataset = linear_gaussian_test_dataset(initial_mean, initial_cov, A, Q, C, R, T)
    log_marginal = linear_gaussian_exact_log_marginal(
        dataset["obs"], initial_mean, initial_cov, A, Q, C, R
    )
    return dataset, log_marginal
