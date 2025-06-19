"""Test minimal SMC setup to isolate the vmap issue."""

import jax
import jax.numpy as jnp
from typing import Tuple

from genjax import gen, const, Const
from genjax.distributions import multivariate_normal
from genjax.inference import init
from genjax.pjax import seed


@gen
def single_component_dynamics(
    k: int,
    prev_pos: jnp.ndarray,  # (2,)
    prev_vel: jnp.ndarray,  # (2,)
    is_initial: bool,
    dt: float,
    process_noise: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample dynamics for a single component."""
    # Initial conditions
    init_pos_mean = jnp.array([20.0 + k * 20.0, 20.0 + k * 20.0])
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
def minimal_vmap_model(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
    obs_std_const: Const[float],
):
    """Minimal model using vmap internally."""
    K = n_components_const.value
    dt_val = dt_const.value
    process_noise_val = process_noise_const.value
    obs_std_val = obs_std_const.value

    is_initial = time_index == 0

    if prev_state is None:
        prev_positions = jnp.zeros((K, 2))
        prev_velocities = jnp.zeros((K, 2))
    else:
        prev_positions, prev_velocities = prev_state

    # Create vmap
    vectorized_dynamics = single_component_dynamics.vmap(
        in_axes=(0, 0, 0, None, None, None)
    )

    # Call vmap
    positions, velocities = (
        vectorized_dynamics(
            jnp.arange(K),  # (K,) - component indices
            prev_positions,  # (K, 2) - previous positions
            prev_velocities,  # (K, 2) - previous velocities
            is_initial,  # scalar - broadcast to all components
            dt_val,  # scalar - broadcast to all components
            process_noise_val,  # scalar - broadcast to all components
        )
        @ "all_components"
    )

    # Simple observation
    mean_pos = jnp.mean(positions, axis=0)
    _ = multivariate_normal(mean_pos, obs_std_val**2 * jnp.eye(2)) @ "obs"

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


def test_smc_init():
    """Test SMC init with minimal vmap model."""
    print("=== Testing SMC init with vmap model ===\n")

    # Prepare model arguments
    initial_positions = jnp.zeros((2, 2))
    initial_velocities = jnp.zeros((2, 2))
    initial_state = (initial_positions, initial_velocities)

    initial_args = (
        initial_state,
        jnp.array(0),
        const(2),  # K
        const(0.1),  # dt
        const(0.5),  # process_noise
        const(2.0),  # obs_std
    )

    # Test the model directly first
    print("Testing model directly...")
    try:
        trace = minimal_vmap_model.simulate(*initial_args)
        print("✅ Direct model call succeeded!")
        print(f"Choices: {list(trace.get_choices().keys())}")
    except Exception as e:
        print(f"❌ Direct model call failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test with constraints
    print("\nTesting model with constraints...")
    try:
        constraints = {"obs": jnp.array([10.0, 10.0])}
        trace, weight = minimal_vmap_model.generate(constraints, *initial_args)
        print("✅ Model with constraints succeeded!")
        print(f"Weight: {weight}")
    except Exception as e:
        print(f"❌ Model with constraints failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test SMC init
    print("\nTesting SMC init...")
    try:
        constraints = {"obs": jnp.array([10.0, 10.0])}
        particles = seed(init)(
            jax.random.PRNGKey(42),
            minimal_vmap_model,
            initial_args,
            const(10),  # n_particles
            constraints,
        )
        print("✅ SMC init succeeded!")
        print(f"Log ML: {particles.log_marginal_likelihood()}")
        print(f"ESS: {particles.effective_sample_size()}")

    except Exception as e:
        print(f"❌ SMC init failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_smc_init()
