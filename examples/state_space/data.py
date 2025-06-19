"""
Data generation utilities for rejuvenation SMC case study.

This module provides synthetic data generation for:
- Discrete Hidden Markov Models (HMMs)
- Linear Gaussian State Space Models
- Directed trajectories for 2D visualization
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import Tuple, NamedTuple

from genjax.distributions import categorical, multivariate_normal
from genjax.pjax import seed


class HMMData(NamedTuple):
    """Container for HMM synthetic data."""

    true_states: jnp.ndarray  # Shape: (T,)
    observations: jnp.ndarray  # Shape: (T,)
    initial_probs: jnp.ndarray  # Shape: (n_states,)
    transition_matrix: jnp.ndarray  # Shape: (n_states, n_states)
    emission_matrix: jnp.ndarray  # Shape: (n_states, n_obs)


class LinearGaussianData(NamedTuple):
    """Container for Linear Gaussian SSM synthetic data."""

    true_states: jnp.ndarray  # Shape: (T, d_state)
    observations: jnp.ndarray  # Shape: (T, d_obs)
    initial_mean: jnp.ndarray  # Shape: (d_state,)
    initial_cov: jnp.ndarray  # Shape: (d_state, d_state)
    A: jnp.ndarray  # Dynamics matrix: (d_state, d_state)
    Q: jnp.ndarray  # Process noise: (d_state, d_state)
    C: jnp.ndarray  # Observation matrix: (d_obs, d_state)
    R: jnp.ndarray  # Observation noise: (d_obs, d_obs)


def generate_hmm_parameters(
    n_states: int = 3,
    n_obs: int = 5,
    seed_val: int = 42,
    sticky: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate random HMM parameters.

    Args:
        n_states: Number of hidden states
        n_obs: Number of observation symbols
        seed_val: Random seed for reproducibility
        sticky: If True, create sticky dynamics (high self-transition probability)

    Returns:
        Tuple of (initial_probs, transition_matrix, emission_matrix)
    """
    key = jax.random.PRNGKey(seed_val)
    key1, key2, key3 = jax.random.split(key, 3)

    # Initial distribution
    initial_probs = jax.nn.softmax(jax.random.normal(key1, (n_states,)))

    # Transition matrix
    if sticky:
        # Create sticky dynamics with high self-transition probability
        transition_matrix = jnp.eye(n_states) * 0.8
        off_diagonal = (
            jax.nn.softmax(jax.random.normal(key2, (n_states, n_states)), axis=1) * 0.2
        )
        transition_matrix = transition_matrix + off_diagonal * (1 - jnp.eye(n_states))
        # Normalize rows
        transition_matrix = transition_matrix / transition_matrix.sum(
            axis=1, keepdims=True
        )
    else:
        # Random transition matrix
        transition_matrix = jax.nn.softmax(
            jax.random.normal(key2, (n_states, n_states)), axis=1
        )

    # Emission matrix
    emission_matrix = jax.nn.softmax(jax.random.normal(key3, (n_states, n_obs)), axis=1)

    return initial_probs, transition_matrix, emission_matrix


def generate_hmm_data(
    key: jax.random.PRNGKey,
    T: int,
    initial_probs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    emission_matrix: jnp.ndarray,
) -> HMMData:
    """
    Generate synthetic HMM data.

    Args:
        key: JAX random key
        T: Number of time steps
        initial_probs: Initial state distribution
        transition_matrix: State transition probabilities
        emission_matrix: Emission probabilities

    Returns:
        HMMData with true states and observations
    """
    initial_probs.shape[0]

    def sample_trajectory():
        states = []
        observations = []

        # Sample initial state
        key_init, key_obs = jrand.split(key)
        state = categorical.sample(jnp.log(initial_probs))
        states.append(state)

        # Sample initial observation
        obs = categorical.sample(jnp.log(emission_matrix[state]))
        observations.append(obs)

        # Generate trajectory
        key_rest = key_obs
        for t in range(1, T):
            key_rest, key_state, key_obs = jrand.split(key_rest, 3)

            # Sample next state
            state = categorical.sample(jnp.log(transition_matrix[states[-1]]))
            states.append(state)

            # Sample observation
            obs = categorical.sample(jnp.log(emission_matrix[state]))
            observations.append(obs)

        return jnp.array(states), jnp.array(observations)

    true_states, observations = seed(sample_trajectory)(key)

    return HMMData(
        true_states=true_states,
        observations=observations,
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
    )


def generate_linear_gaussian_parameters(
    d_state: int = 2,
    d_obs: int = 2,
    seed_val: int = 42,
    stable: bool = True,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Generate stable linear Gaussian SSM parameters.

    Args:
        d_state: State dimension
        d_obs: Observation dimension
        seed_val: Random seed
        stable: If True, ensure dynamics are stable

    Returns:
        Tuple of (initial_mean, initial_cov, A, Q, C, R)
    """
    key = jax.random.PRNGKey(seed_val)
    key1, key2, key3 = jax.random.split(key, 3)

    # Initial distribution
    initial_mean = jax.random.normal(key1, (d_state,))
    initial_cov = jnp.eye(d_state) * 0.5

    # Dynamics
    A_raw = jax.random.normal(key2, (d_state, d_state))
    if stable:
        # Scale to ensure eigenvalues < 1
        A = A_raw / (1.1 * jnp.max(jnp.abs(jnp.linalg.eigvals(A_raw))))
    else:
        A = A_raw
    Q = jnp.eye(d_state) * 0.1  # Process noise

    # Observation model
    C = jax.random.normal(key3, (d_obs, d_state))
    R = jnp.eye(d_obs) * 0.05  # Reduced observation noise

    return initial_mean, initial_cov, A, Q, C, R


def generate_directed_linear_gaussian_parameters(
    d_state: int = 2,
    d_obs: int = 2,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Create linear Gaussian SSM parameters for directed trajectory from lower left to upper right.

    Args:
        d_state: State dimension (should be 2 for 2D visualization)
        d_obs: Observation dimension

    Returns:
        Tuple of (initial_mean, initial_cov, A, Q, C, R)
    """
    # Initial distribution - start in lower left
    initial_mean = jnp.array([-3.0, -3.0])  # Lower left corner
    initial_cov = jnp.eye(d_state) * 0.1  # Small initial uncertainty

    # Dynamics - slight damping to allow drift to dominate
    A = jnp.array(
        [
            [0.9, 0.05],  # x component
            [0.05, 0.9],  # y component
        ]
    )

    # Small process noise
    Q = jnp.eye(d_state) * 0.1

    # Observation model - observe state directly with noise
    C = jnp.eye(d_obs)  # Direct observation
    R = jnp.eye(d_obs) * 0.05  # Reduced observation noise

    return initial_mean, initial_cov, A, Q, C, R


def generate_linear_gaussian_data(
    key: jax.random.PRNGKey,
    T: int,
    initial_mean: jnp.ndarray,
    initial_cov: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
    drift: jnp.ndarray = None,
) -> LinearGaussianData:
    """
    Generate synthetic linear Gaussian SSM data.

    Args:
        key: JAX random key
        T: Number of time steps
        initial_mean: Initial state mean
        initial_cov: Initial state covariance
        A: Dynamics matrix
        Q: Process noise covariance
        C: Observation matrix
        R: Observation noise covariance
        drift: Optional drift vector to add at each timestep

    Returns:
        LinearGaussianData with true states and observations
    """
    initial_mean.shape[0]

    def sample_trajectory():
        states = []
        observations = []

        # Sample initial state
        key_init, key_obs = jrand.split(key)
        x0 = multivariate_normal.sample(initial_mean, initial_cov, sample_shape=())
        states.append(x0)

        # Sample initial observation
        y0 = multivariate_normal.sample(C @ x0, R, sample_shape=())
        observations.append(y0)

        # Generate trajectory
        key_rest = key_obs
        for t in range(1, T):
            key_rest, key_state, key_obs = jrand.split(key_rest, 3)

            # Compute next state mean
            mean_t = A @ states[-1]
            if drift is not None:
                mean_t = mean_t + drift

            # Sample next state
            xt = multivariate_normal.sample(mean_t, Q, sample_shape=())
            states.append(xt)

            # Sample observation
            yt = multivariate_normal.sample(C @ xt, R, sample_shape=())
            observations.append(yt)

        return jnp.array(states), jnp.array(observations)

    true_states, observations = seed(sample_trajectory)(key)

    return LinearGaussianData(
        true_states=true_states,
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        A=A,
        Q=Q,
        C=C,
        R=R,
    )


def generate_directed_trajectory_data(
    key: jax.random.PRNGKey,
    T: int = 16,
    d_state: int = 2,
    d_obs: int = 2,
) -> LinearGaussianData:
    """
    Generate synthetic data with directed trajectory from lower left to upper right.

    This is a convenience function that combines parameter generation and data generation
    for the specific case of 2D visualization with directed motion.

    Args:
        key: JAX random key
        T: Number of time steps (default 16 for 4x4 grid)
        d_state: State dimension (should be 2)
        d_obs: Observation dimension

    Returns:
        LinearGaussianData with trajectory moving from lower left to upper right
    """
    # Get directed parameters
    initial_mean, initial_cov, A, Q, C, R = (
        generate_directed_linear_gaussian_parameters(d_state, d_obs)
    )

    # Define drift toward upper right
    drift = jnp.array([0.25, 0.25])  # Constant drift

    # Generate data with drift
    return generate_linear_gaussian_data(
        key, T, initial_mean, initial_cov, A, Q, C, R, drift
    )


def generate_challenging_trajectory_data(
    key: jax.random.PRNGKey,
    T: int = 16,
    d_state: int = 2,
    d_obs: int = 2,
    drift_magnitude: float = 0.4,  # > 1 sigma of process noise (sigma=sqrt(0.1)≈0.32)
    obs_noise_scale: float = 3.0,  # Increase observation noise significantly
) -> LinearGaussianData:
    """
    Generate challenging synthetic data for testing inference algorithm performance.

    This creates a more difficult inference problem by:
    1. Increasing observation noise significantly
    2. Making true state dynamics move distances > 1 sigma of model process noise

    Args:
        key: JAX random key
        T: Number of time steps
        d_state: State dimension (should be 2)
        d_obs: Observation dimension
        drift_magnitude: Drift per timestep (should be > sqrt(Q diagonal) ≈ 0.32)
        obs_noise_scale: Multiplier for observation noise

    Returns:
        LinearGaussianData with challenging dynamics and noisy observations
    """
    # Start with directed parameters as base
    initial_mean, initial_cov, A, Q, C, R = (
        generate_directed_linear_gaussian_parameters(d_state, d_obs)
    )

    # Increase observation noise significantly to make inference harder
    R_challenging = R * obs_noise_scale  # Much higher observation noise

    # Create large drift that's > 1 sigma of process noise
    # Process noise std = sqrt(0.1) ≈ 0.32, so drift_magnitude=0.4 is > 1 sigma
    drift = jnp.array([drift_magnitude, drift_magnitude])

    # Generate data with challenging parameters
    return generate_linear_gaussian_data(
        key, T, initial_mean, initial_cov, A, Q, C, R_challenging, drift
    )


def generate_extreme_trajectory_data(
    key: jax.random.PRNGKey,
    T: int = 16,
    d_state: int = 2,
    d_obs: int = 2,
    drift_magnitude: float = 0.6,  # 2x sigma of process noise
    obs_noise_scale: float = 5.0,  # Very high observation noise
    nonlinear_drift: bool = True,  # Add nonlinear component to drift
) -> LinearGaussianData:
    """
    Generate extremely challenging synthetic data to stress-test inference algorithms.

    This creates an even more difficult inference problem by:
    1. Very high observation noise (5x original)
    2. Large drift (2x sigma of process noise)
    3. Optional nonlinear drift pattern

    Args:
        key: JAX random key
        T: Number of time steps
        d_state: State dimension (should be 2)
        d_obs: Observation dimension
        drift_magnitude: Base drift per timestep
        obs_noise_scale: Multiplier for observation noise
        nonlinear_drift: Add time-varying nonlinear drift

    Returns:
        LinearGaussianData with extremely challenging dynamics
    """
    # Start with directed parameters as base
    initial_mean, initial_cov, A, Q, C, R = (
        generate_directed_linear_gaussian_parameters(d_state, d_obs)
    )

    # Very high observation noise
    R_extreme = R * obs_noise_scale

    # Base drift - much larger than process noise
    base_drift = jnp.array([drift_magnitude, drift_magnitude])

    if nonlinear_drift:
        # Create time-varying trajectory with nonlinear component
        def sample_trajectory_nonlinear():
            states = []
            observations = []

            # Sample initial state
            key_init, key_obs = jrand.split(key)
            x0 = multivariate_normal.sample(initial_mean, initial_cov, sample_shape=())
            states.append(x0)

            # Sample initial observation
            y0 = multivariate_normal.sample(C @ x0, R_extreme, sample_shape=())
            observations.append(y0)

            # Generate trajectory with time-varying drift
            key_rest = key_obs
            for t in range(1, T):
                key_rest, key_state, key_obs = jrand.split(key_rest, 3)

                # Time-varying drift (spiral + linear)
                time_factor = t / T
                spiral_component = jnp.array(
                    [
                        0.2 * jnp.sin(2 * jnp.pi * time_factor),  # Circular motion
                        0.2 * jnp.cos(2 * jnp.pi * time_factor),
                    ]
                )
                total_drift = base_drift + spiral_component

                # Compute next state mean
                mean_t = A @ states[-1] + total_drift

                # Sample next state
                xt = multivariate_normal.sample(mean_t, Q, sample_shape=())
                states.append(xt)

                # Sample observation
                yt = multivariate_normal.sample(C @ xt, R_extreme, sample_shape=())
                observations.append(yt)

            return jnp.array(states), jnp.array(observations)

        from genjax.pjax import seed

        true_states, observations = seed(sample_trajectory_nonlinear)(key)

        return LinearGaussianData(
            true_states=true_states,
            observations=observations,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            A=A,
            Q=Q,
            C=C,
            R=R_extreme,
        )
    else:
        # Simple large linear drift
        return generate_linear_gaussian_data(
            key, T, initial_mean, initial_cov, A, Q, C, R_extreme, base_drift
        )


# Convenience functions for standard experiments
def generate_standard_hmm_data(
    key: jax.random.PRNGKey,
    T: int = 20,
    n_states: int = 3,
    n_obs: int = 5,
) -> HMMData:
    """Generate standard HMM data for experiments."""
    initial_probs, transition_matrix, emission_matrix = generate_hmm_parameters(
        n_states, n_obs, seed_val=42, sticky=True
    )
    return generate_hmm_data(key, T, initial_probs, transition_matrix, emission_matrix)


def generate_standard_linear_gaussian_data(
    key: jax.random.PRNGKey,
    T: int = 20,
    d_state: int = 2,
    d_obs: int = 2,
) -> LinearGaussianData:
    """Generate standard linear Gaussian data for experiments."""
    initial_mean, initial_cov, A, Q, C, R = generate_linear_gaussian_parameters(
        d_state, d_obs, seed_val=42, stable=True
    )
    return generate_linear_gaussian_data(key, T, initial_mean, initial_cov, A, Q, C, R)
