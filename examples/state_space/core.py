"""
Core models and inference for rejuvenation SMC case study.

This module demonstrates rejuvenation SMC on discrete HMMs and linear Gaussian models,
comparing against exact inference baselines using the library's SMC primitives.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List
import jax.random as jrand

from genjax.core import gen, const, sel
from genjax.distributions import categorical, multivariate_normal
from genjax.inference import rejuvenation_smc, mh, mala
from genjax.extras.state_space import (
    discrete_hmm,
    forward_filter,
    sample_hmm_dataset,
    linear_gaussian,
    kalman_filter,
)
from genjax.pjax import seed


# =============================================================================
# DISCRETE HMM EXPERIMENTS
# =============================================================================


def create_hmm_params(n_states: int = 3, n_obs: int = 5, seed: int = 42):
    """Create random HMM parameters for testing."""
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)

    # Initial distribution
    initial_probs = jax.nn.softmax(jax.random.normal(key1, (n_states,)))

    # Transition matrix (row-stochastic)
    transition_matrix = jax.nn.softmax(
        jax.random.normal(key2, (n_states, n_states)), axis=1
    )

    # Emission matrix (row-stochastic)
    emission_matrix = jax.nn.softmax(jax.random.normal(key3, (n_states, n_obs)), axis=1)

    return initial_probs, transition_matrix, emission_matrix


def run_discrete_hmm_experiment(
    key: jax.random.PRNGKey,
    T: int = 20,
    n_particles_list: List[int] = [100, 500, 1000, 2000],
    n_states: int = 3,
    n_obs: int = 5,
) -> Dict[str, Any]:
    """
    Run discrete HMM experiment comparing SMC to exact inference.

    Args:
        key: JAX random key
        T: Number of time steps
        n_particles_list: List of particle counts to test
        n_states: Number of hidden states
        n_obs: Number of observation symbols

    Returns:
        Dictionary with results and exact baseline
    """
    # Generate HMM parameters
    initial_probs, transition_matrix, emission_matrix = create_hmm_params(
        n_states, n_obs
    )

    # Generate test data
    key1, key2 = jrand.split(key)

    # Create closure for sample_hmm_dataset that captures T as static
    def sample_dataset():
        return sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, const(T)
        )

    true_states, observations, constraints = seed(sample_dataset)(key1)

    # Compute exact log marginal likelihood
    _, exact_log_marginal = forward_filter(
        observations, initial_probs, transition_matrix, emission_matrix
    )

    # Simple proposal for HMM states (uniform)
    @gen
    def hmm_proposal(
        prev_state, time_index, initial_probs, transition_matrix, emission_matrix
    ):
        """Uniform proposal over states."""
        n_states = initial_probs.shape[0]
        uniform_logits = jnp.log(jnp.ones(n_states) / n_states)
        return categorical(uniform_logits) @ "state"

    # MCMC kernel for rejuvenation
    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    # Prepare observations for rejuvenation_smc
    obs_sequence = {"obs": observations}

    # Initial arguments for discrete_hmm
    initial_args = (
        jnp.array(0),  # prev_state (unused at t=0)
        jnp.array(0),  # time_index
        initial_probs,
        transition_matrix,
        emission_matrix,
    )

    # Run SMC with different particle counts
    results = []
    for n_particles in n_particles_list:
        key2, subkey = jrand.split(key2)

        # Run rejuvenation SMC with default 1 rejuvenation move
        smc_result = seed(rejuvenation_smc)(
            subkey,
            discrete_hmm,
            hmm_proposal,
            const(mcmc_kernel),
            obs_sequence,
            initial_args,
            const(n_particles),
        )

        log_marginal = smc_result.log_marginal_likelihood()
        error = jnp.abs(log_marginal - exact_log_marginal)

        results.append(
            {
                "n_particles": n_particles,
                "log_marginal": float(log_marginal),
                "error": float(error),
                "relative_error": float(error / jnp.abs(exact_log_marginal)),
                "ess": float(smc_result.effective_sample_size()),
            }
        )

    return {
        "model": "discrete_hmm",
        "T": T,
        "n_states": n_states,
        "n_obs": n_obs,
        "exact_log_marginal": float(exact_log_marginal),
        "results": results,
    }


# =============================================================================
# PARTICLE COLLECTION FOR VISUALIZATION
# =============================================================================


def run_linear_gaussian_with_particles(
    key: jax.random.PRNGKey,
    T: int = 16,  # Use 16 for (4,4) grid
    n_particles: int = 100,
    d_state: int = 2,  # 2D state for visualization
    d_obs: int = 2,
    use_directed: bool = True,  # Use directed trajectory
) -> Dict[str, Any]:
    """
    Run linear Gaussian experiment with particle collection for visualization.

    Args:
        key: JAX random key
        T: Number of time steps (16 for 4x4 grid)
        n_particles: Number of particles
        d_state: State dimension (2 for 2D visualization)
        d_obs: Observation dimension
        use_directed: If True, use directed lower-left to upper-right trajectory

    Returns:
        Dictionary with particles at all timesteps
    """
    # Import data generation utilities
    from .data import (
        generate_directed_trajectory_data,
        generate_linear_gaussian_data,
        generate_linear_gaussian_parameters,
    )

    # Generate data
    key1, key2 = jrand.split(key)

    if use_directed:
        # Generate directed trajectory data
        data = generate_directed_trajectory_data(key1, T, d_state, d_obs)
        # Use the standard linear_gaussian model - data already has drift baked in
        model = linear_gaussian
    else:
        # Generate standard data
        initial_mean, initial_cov, A, Q, C, R = generate_linear_gaussian_parameters(
            d_state, d_obs
        )
        data = generate_linear_gaussian_data(
            key1, T, initial_mean, initial_cov, A, Q, C, R
        )
        model = linear_gaussian

    # Extract components
    true_states = data.true_states
    observations = data.observations
    initial_mean = data.initial_mean
    initial_cov = data.initial_cov
    A = data.A
    Q = data.Q
    C = data.C
    R = data.R

    # Standard proposal for linear Gaussian (uses prior as proposal)
    @gen
    def lg_proposal(prev_state, time_index, initial_mean, initial_cov, A, Q, C, R):
        """Prior proposal for linear Gaussian."""
        # Use JAX conditional instead of Python if
        mean = jnp.where(time_index == 0, initial_mean, A @ prev_state)
        cov = jnp.where(time_index == 0, initial_cov, Q)
        return multivariate_normal(mean, cov) @ "state"

    # MCMC kernel using MH for simplicity
    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    # Prepare observations for rejuvenation_smc
    obs_sequence = {"obs": observations}

    # Initial arguments for linear_gaussian
    initial_args = (
        jnp.zeros(d_state),  # prev_state (unused at t=0)
        jnp.array(0),  # time_index
        initial_mean,
        initial_cov,
        A,
        Q,
        C,
        R,
    )

    # Run rejuvenation SMC with all particles
    all_particles = seed(rejuvenation_smc)(
        key2,
        model,  # Use the selected model (with or without drift)
        lg_proposal,
        const(mcmc_kernel),
        obs_sequence,
        initial_args,
        const(n_particles),
        const(True),  # return_all_particles - Get all timesteps!
    )

    return {
        "model": "linear_gaussian_2d",
        "T": T,
        "d_state": d_state,
        "d_obs": d_obs,
        "n_particles": n_particles,
        "true_states": true_states,
        "observations": observations,
        "all_particles": all_particles,
        "params": {
            "initial_mean": initial_mean,
            "initial_cov": initial_cov,
            "A": A,
            "Q": Q,
            "C": C,
            "R": R,
        },
    }


# =============================================================================
# LINEAR GAUSSIAN EXPERIMENTS
# =============================================================================


def create_linear_gaussian_params(d_state: int = 2, d_obs: int = 2, seed: int = 42):
    """Create stable linear Gaussian SSM parameters."""
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)

    # Initial distribution
    initial_mean = jax.random.normal(key1, (d_state,))
    initial_cov = jnp.eye(d_state) * 0.5

    # Dynamics (ensure stability)
    A_raw = jax.random.normal(key2, (d_state, d_state))
    # Scale to ensure eigenvalues < 1
    A = A_raw / (1.1 * jnp.max(jnp.abs(jnp.linalg.eigvals(A_raw))))
    Q = jnp.eye(d_state) * 0.1  # Process noise

    # Observation model
    C = jax.random.normal(key3, (d_obs, d_state))
    R = jnp.eye(d_obs) * 0.05  # Reduced observation noise

    return initial_mean, initial_cov, A, Q, C, R


def run_rejuvenation_comparison(
    key: jax.random.PRNGKey,
    T: int = 16,  # For frames 0, 4, 8, 12
    n_particles: int = 500,
    d_state: int = 2,
    d_obs: int = 2,
    step_size: float = 0.1,  # For MALA
    include_particle_comparison: bool = False,
    challenging: bool = False,  # Use challenging data generation
    extreme: bool = False,  # Use extreme data generation
) -> Dict[str, Any]:
    """
    Run comparison of different rejuvenation strategies.

    Args:
        key: JAX random key
        T: Number of time steps
        n_particles: Number of particles
        d_state: State dimension
        d_obs: Observation dimension
        step_size: Step size for MALA
        include_particle_comparison: Include different particle counts
        challenging: Use challenging data (higher noise, larger drift)
        extreme: Use extreme data (very high noise, nonlinear drift)

    Returns:
        Dictionary with results for each strategy
    """
    from .data import (
        generate_directed_trajectory_data,
        generate_challenging_trajectory_data,
        generate_extreme_trajectory_data,
    )

    # Generate data
    key1, key2 = jrand.split(key)
    if extreme:
        data = generate_extreme_trajectory_data(key1, T, d_state, d_obs)
    elif challenging:
        data = generate_challenging_trajectory_data(key1, T, d_state, d_obs)
    else:
        data = generate_directed_trajectory_data(key1, T, d_state, d_obs)

    # Extract components
    true_states = data.true_states
    observations = data.observations
    initial_mean = data.initial_mean
    initial_cov = data.initial_cov
    A = data.A
    Q = data.Q
    C = data.C
    R = data.R

    # Standard proposal
    @gen
    def lg_proposal(prev_state, time_index, initial_mean, initial_cov, A, Q, C, R):
        """Prior proposal for linear Gaussian."""
        mean = jnp.where(time_index == 0, initial_mean, A @ prev_state)
        cov = jnp.where(time_index == 0, initial_cov, Q)
        return multivariate_normal(mean, cov) @ "state"

    # MH kernel
    def mh_kernel(trace):
        return mh(trace, sel("state"))

    # MALA kernel
    def mala_kernel(trace):
        return mala(trace, sel("state"), step_size)

    # Prepare for rejuvenation_smc
    obs_sequence = {"obs": observations}
    initial_args = (
        jnp.zeros(d_state),
        jnp.array(0),
        initial_mean,
        initial_cov,
        A,
        Q,
        C,
        R,
    )

    results = {}

    # Run each strategy
    strategies = [
        ("mh_1", mh_kernel, 1),
        ("mh_5", mh_kernel, 5),
        ("mala_1", mala_kernel, 1),
        ("mala_5", mala_kernel, 5),
    ]

    # If including particle comparison, also run with different particle counts
    if include_particle_comparison:
        particle_counts = [100, 500, 1000]
        base_strategies = strategies
        strategies = []
        for n_p in particle_counts:
            for name, kernel, n_moves in base_strategies:
                strategies.append((f"{name}_n{n_p}", kernel, n_moves, n_p))
    else:
        # Add default particle count to strategies
        strategies = [
            (name, kernel, n_moves, n_particles) for name, kernel, n_moves in strategies
        ]

    for name, kernel, n_moves, n_p in strategies:
        key2, subkey = jrand.split(key2)

        all_particles = seed(rejuvenation_smc)(
            subkey,
            linear_gaussian,
            lg_proposal,
            const(kernel),
            obs_sequence,
            initial_args,
            const(n_p),
            const(True),  # return_all_particles
            const(n_moves),  # n_rejuvenation_moves
        )

        results[name] = {
            "all_particles": all_particles,
            "true_states": true_states,
            "observations": observations,
            "n_particles": n_p,
            "n_moves": n_moves,
            "kernel_type": "mh" if "mh" in name else "mala",
        }

    return results


def run_linear_gaussian_experiment(
    key: jax.random.PRNGKey,
    T: int = 20,
    n_particles_list: List[int] = [100, 500, 1000, 2000],
    d_state: int = 2,
    d_obs: int = 2,
) -> Dict[str, Any]:
    """
    Run linear Gaussian experiment comparing SMC to exact inference.

    Args:
        key: JAX random key
        T: Number of time steps
        n_particles_list: List of particle counts to test
        d_state: State dimension
        d_obs: Observation dimension

    Returns:
        Dictionary with results and exact baseline
    """
    # Generate model parameters
    initial_mean, initial_cov, A, Q, C, R = create_linear_gaussian_params(
        d_state, d_obs
    )

    # Generate synthetic data
    key1, key2 = jrand.split(key)

    # Sample trajectory
    def sample_trajectory():
        states = []
        observations = []

        # Initial state
        x0 = multivariate_normal.sample(initial_mean, initial_cov, sample_shape=())
        states.append(x0)
        y0 = multivariate_normal.sample(C @ x0, R, sample_shape=())
        observations.append(y0)

        # Generate trajectory
        for t in range(1, T):
            xt = multivariate_normal.sample(A @ states[t - 1], Q, sample_shape=())
            states.append(xt)
            yt = multivariate_normal.sample(C @ xt, R, sample_shape=())
            observations.append(yt)

        return jnp.array(states), jnp.array(observations)

    true_states, observations = seed(sample_trajectory)(key1)

    # Compute exact log marginal using Kalman filter
    filtered_means, filtered_covs, log_marginals = kalman_filter(
        observations, initial_mean, initial_cov, A, Q, C, R
    )
    exact_log_marginal = jnp.sum(log_marginals)

    # Proposal for linear Gaussian (uses prior as proposal)
    @gen
    def lg_proposal(prev_state, time_index, initial_mean, initial_cov, A, Q, C, R):
        """Prior proposal for linear Gaussian."""
        # Use JAX conditional instead of Python if
        mean = jnp.where(time_index == 0, initial_mean, A @ prev_state)
        cov = jnp.where(time_index == 0, initial_cov, Q)
        return multivariate_normal(mean, cov) @ "state"

    # MCMC kernel using MH for simplicity
    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    # Prepare observations for rejuvenation_smc
    obs_sequence = {"obs": observations}

    # Initial arguments for linear_gaussian
    initial_args = (
        jnp.zeros(d_state),  # prev_state (unused at t=0)
        jnp.array(0),  # time_index
        initial_mean,
        initial_cov,
        A,
        Q,
        C,
        R,
    )

    # Run SMC with different particle counts
    results = []
    for n_particles in n_particles_list:
        key2, subkey = jrand.split(key2)

        # Run rejuvenation SMC
        smc_result = seed(rejuvenation_smc)(
            subkey,
            linear_gaussian,
            lg_proposal,
            const(mcmc_kernel),
            obs_sequence,
            initial_args,
            const(n_particles),
        )

        log_marginal = smc_result.log_marginal_likelihood()
        error = jnp.abs(log_marginal - exact_log_marginal)

        results.append(
            {
                "n_particles": n_particles,
                "log_marginal": float(log_marginal),
                "error": float(error),
                "relative_error": float(error / jnp.abs(exact_log_marginal)),
                "ess": float(smc_result.effective_sample_size()),
            }
        )

    return {
        "model": "linear_gaussian",
        "T": T,
        "d_state": d_state,
        "d_obs": d_obs,
        "exact_log_marginal": float(exact_log_marginal),
        "results": results,
    }
