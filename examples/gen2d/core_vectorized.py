"""Vectorized core model using modular_vmap for gen2d case study."""

import jax
import jax.numpy as jnp
from typing import Tuple

from genjax import gen, const, Const, sel
from genjax.core import Trace
from genjax.distributions import multivariate_normal
from genjax.inference import rejuvenation_smc, mh
from genjax.pjax import seed

# Type aliases for clarity
State = Tuple[jnp.ndarray, jnp.ndarray]  # (positions, velocities)


@gen
def single_component_dynamics(
    k: int,
    prev_pos: jnp.ndarray,  # (2,)
    prev_vel: jnp.ndarray,  # (2,)
    is_initial: bool,
    dt: float,
    process_noise: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample dynamics for a single component.

    Args:
        k: Component index
        prev_pos: Previous position (2D)
        prev_vel: Previous velocity (2D)
        is_initial: Whether this is the initial timestep
        dt: Time step size
        process_noise: Process noise standard deviation

    Returns:
        (position, velocity) for this component
    """
    # Initial conditions
    init_pos_mean = jnp.array([20.0 + k * 20.0, 20.0 + k * 20.0])  # Spread components
    init_vel_mean = jnp.zeros(2)

    # Transition conditions
    trans_pos_mean = prev_pos + prev_vel * dt
    trans_vel_mean = prev_vel

    # Select based on time index using JAX
    pos_mean = jax.lax.select(is_initial, init_pos_mean, trans_pos_mean)
    vel_mean = jax.lax.select(is_initial, init_vel_mean, trans_vel_mean)

    # Covariances
    pos_cov = jax.lax.select(
        is_initial, 10.0 * jnp.eye(2), process_noise**2 * jnp.eye(2)
    )
    vel_cov = jax.lax.select(
        is_initial, 0.5 * jnp.eye(2), process_noise**2 * jnp.eye(2)
    )

    # Sample
    position = multivariate_normal(pos_mean, pos_cov) @ f"position_{k}"
    velocity = multivariate_normal(vel_mean, vel_cov) @ f"velocity_{k}"

    return position, velocity


@gen
def vectorized_gen2d_model(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
    obs_std_const: Const[float],
):
    """
    Vectorized version of gen2d model using gen_fn.vmap.

    This avoids Python for loops by vectorizing component updates.
    """
    # Extract values from Const wrappers
    K = n_components_const.value
    dt_val = dt_const.value
    process_noise_val = process_noise_const.value
    obs_std_val = obs_std_const.value

    # Use JAX control flow for initial vs transition
    is_initial = time_index == 0

    # Extract previous state (handle None case for initial)
    if prev_state is None:
        prev_positions = jnp.zeros((K, 2))
        prev_velocities = jnp.zeros((K, 2))
    else:
        prev_positions, prev_velocities = prev_state

    # Try vmap again - vectorize over all K components
    vectorized_dynamics = single_component_dynamics.vmap(
        in_axes=(
            0,
            0,
            0,
            None,
            None,
            None,
        )  # vectorize over component index, positions, velocities
    )

    # Prepare inputs for vectorization
    component_indices = jnp.arange(K)  # [0, 1, 2, ..., K-1]

    # Apply vectorized dynamics
    positions, velocities = (
        vectorized_dynamics(
            component_indices,  # (K,) - component indices
            prev_positions,  # (K, 2) - previous positions
            prev_velocities,  # (K, 2) - previous velocities
            is_initial,  # scalar - broadcast to all components
            dt_val,  # scalar - broadcast to all components
            process_noise_val,  # scalar - broadcast to all components
        )
        @ "all_components"
    )

    # Simple observation model - observe mean position with noise
    mean_pos = jnp.mean(positions, axis=0)
    _ = multivariate_normal(mean_pos, obs_std_val**2 * jnp.eye(2)) @ "obs"

    # Return state for next timestep (feedback loop)
    new_state = (positions, velocities)
    new_time_index = time_index + 1
    return (
        new_state,
        new_time_index,
        n_components_const,
        dt_const,
        process_noise_const,
        obs_std_const,
    )


@gen
def vectorized_proposal(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
    obs_std_const: Const[float],
):
    """Vectorized proposal function matching the model."""
    # Extract values
    K = n_components_const.value
    dt_val = dt_const.value
    process_noise_val = process_noise_const.value

    is_initial = time_index == 0

    # Extract previous state
    if prev_state is None:
        prev_positions = jnp.zeros((K, 2))
        prev_velocities = jnp.zeros((K, 2))
    else:
        prev_positions, prev_velocities = prev_state

    # Define proposal for single component (without observations)
    @gen
    def single_component_proposal(
        k: int,
        prev_pos: jnp.ndarray,
        prev_vel: jnp.ndarray,
        is_initial: bool,
        dt: float,
        process_noise: float,
    ):
        """Proposal for single component dynamics."""
        # Same logic as dynamics but without the @ addressing
        init_pos_mean = jnp.array([20.0 + k * 20.0, 20.0 + k * 20.0])
        init_vel_mean = jnp.zeros(2)

        trans_pos_mean = prev_pos + prev_vel * dt
        trans_vel_mean = prev_vel

        pos_mean = jax.lax.select(is_initial, init_pos_mean, trans_pos_mean)
        vel_mean = jax.lax.select(is_initial, init_vel_mean, trans_vel_mean)

        pos_cov = jax.lax.select(
            is_initial, 10.0 * jnp.eye(2), process_noise**2 * jnp.eye(2)
        )
        vel_cov = jax.lax.select(
            is_initial, 0.5 * jnp.eye(2), process_noise**2 * jnp.eye(2)
        )

        _ = multivariate_normal(pos_mean, pos_cov) @ f"position_{k}"
        _ = multivariate_normal(vel_mean, vel_cov) @ f"velocity_{k}"

    # Use vmap for proposal too
    vectorized_proposal = single_component_proposal.vmap(
        in_axes=(0, 0, 0, None, None, None)
    )

    component_indices = jnp.arange(K)

    _ = (
        vectorized_proposal(
            component_indices,
            prev_positions,
            prev_velocities,
            is_initial,
            dt_val,
            process_noise_val,
        )
        @ "all_proposals"
    )


def create_vectorized_mcmc_kernel(n_components: int):
    """Create MCMC kernel that updates all components at once."""

    def vectorized_kernel(trace: Trace) -> Trace:
        """Update all positions and velocities."""
        # Update all positions at once
        position_selection = sel()
        for k in range(n_components):
            position_selection = position_selection | sel(f"position_{k}")
        trace = mh(trace, position_selection)

        # Update all velocities at once
        velocity_selection = sel()
        for k in range(n_components):
            velocity_selection = velocity_selection | sel(f"velocity_{k}")
        trace = mh(trace, velocity_selection)

        return trace

    return vectorized_kernel


def run_vectorized_gen2d_inference(
    observations: jnp.ndarray,  # (T, max_pixels, 2) padded observations
    observation_counts: jnp.ndarray,  # (T,) number of valid pixels per frame
    n_components: int = 5,
    n_particles: int = 100,
    dt: float = 0.1,
    process_noise: float = 0.5,
    obs_std: float = 2.0,
    n_rejuvenation_moves: int = 2,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
):
    """
    Run SMC inference using vectorized model.

    This version uses modular_vmap to handle component loops efficiently.
    """
    # Create MCMC kernel
    mcmc_kernel = create_vectorized_mcmc_kernel(n_components)

    # Prepare static parameters as Const objects
    n_components_const = const(n_components)
    dt_const = const(dt)
    process_noise_const = const(process_noise)
    obs_std_const = const(obs_std)

    # Prepare initial state
    initial_positions = jnp.zeros((n_components, 2))
    initial_velocities = jnp.zeros((n_components, 2))
    initial_state = (initial_positions, initial_velocities)

    # Simple observation structure
    T = len(observation_counts)
    dummy_obs = jnp.zeros((T, 2))  # Dummy observations
    simple_obs_dict = {"obs": dummy_obs}

    # Initial arguments
    initial_args = (
        initial_state,
        jnp.array(0),
        n_components_const,
        dt_const,
        process_noise_const,
        obs_std_const,
    )

    # Run SMC with vectorized model
    particles = seed(rejuvenation_smc)(
        key,
        vectorized_gen2d_model,
        vectorized_proposal,
        const(mcmc_kernel),
        simple_obs_dict,
        initial_args,
        const(n_particles),
        const(True),  # return_all_particles
        const(n_rejuvenation_moves),
    )

    return particles
