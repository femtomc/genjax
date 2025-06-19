"""Try using repeat pattern instead of manual vmap."""

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
def component_dynamics(
    prev_pos: jnp.ndarray,  # (2,)
    prev_vel: jnp.ndarray,  # (2,)
    is_initial: bool,
    dt: float,
    process_noise: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample dynamics for a single component - without component index."""
    # Initial conditions - will be different for each component due to different prev_pos
    init_pos_mean = prev_pos + jnp.array([10.0, 10.0])  # Different offset per component
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

    # Sample - use generic addressing since we'll use repeat
    position = multivariate_normal(pos_mean, pos_cov) @ "position"
    velocity = multivariate_normal(vel_mean, vel_cov) @ "velocity"

    return position, velocity


@gen
def repeat_gen2d_model(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
    obs_std_const: Const[float],
):
    """
    Gen2d model using repeat pattern instead of manual vmap.
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

    # Try using repeat pattern - this should create K independent samples
    repeated_dynamics = component_dynamics.repeat(n_components_const.value)

    # Call with vectorized arguments
    component_results = (
        repeated_dynamics(
            prev_positions,  # (K, 2) - previous positions
            prev_velocities,  # (K, 2) - previous velocities
            is_initial,  # scalar - broadcast to all components
            dt_val,  # scalar - broadcast to all components
            process_noise_val,  # scalar - broadcast to all components
        )
        @ "dynamics"
    )

    # Extract positions and velocities from the result
    positions, velocities = component_results

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
def repeat_proposal(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
    obs_std_const: Const[float],
):
    """Proposal function using repeat pattern."""
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

    # Define proposal for single component (same structure as model)
    @gen
    def component_proposal(
        prev_pos: jnp.ndarray,
        prev_vel: jnp.ndarray,
        is_initial: bool,
        dt: float,
        process_noise: float,
    ):
        """Proposal for single component dynamics."""
        # Same logic as dynamics
        init_pos_mean = prev_pos + jnp.array([10.0, 10.0])
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

        _ = multivariate_normal(pos_mean, pos_cov) @ "position"
        _ = multivariate_normal(vel_mean, vel_cov) @ "velocity"

    # Use repeat for proposal too
    repeated_proposal = component_proposal.repeat(n_components_const.value)

    _ = (
        repeated_proposal(
            prev_positions, prev_velocities, is_initial, dt_val, process_noise_val
        )
        @ "dynamics"
    )


def create_repeat_mcmc_kernel(n_components: int):
    """Create MCMC kernel for repeat model."""

    def repeat_kernel(trace: Trace) -> Trace:
        """Update all positions and velocities."""
        # For repeat pattern, addresses are indexed automatically
        position_selection = sel("position")
        velocity_selection = sel("velocity")

        trace = mh(trace, position_selection)
        trace = mh(trace, velocity_selection)

        return trace

    return repeat_kernel


def run_repeat_gen2d_inference(
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
    Run SMC inference using repeat pattern.
    """
    # Create MCMC kernel
    mcmc_kernel = create_repeat_mcmc_kernel(n_components)

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

    # Run SMC with repeat model
    particles = seed(rejuvenation_smc)(
        key,
        repeat_gen2d_model,
        repeat_proposal,
        const(mcmc_kernel),
        simple_obs_dict,
        initial_args,
        const(n_particles),
        const(True),  # return_all_particles
        const(n_rejuvenation_moves),
    )

    return particles
