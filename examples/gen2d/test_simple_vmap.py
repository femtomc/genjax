"""Test simple vmap case to isolate the issue."""

import jax
import jax.numpy as jnp
from typing import Tuple

from genjax import gen
from genjax.distributions import multivariate_normal


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


def test_vmap_with_exact_signature():
    """Test vmap with exact function signature."""
    print("Testing vmap with our exact function signature...")

    # Parameters
    K = 3
    dt_val = 0.1
    process_noise_val = 0.5
    is_initial = True

    # Prepare inputs
    component_indices = jnp.arange(K)  # [0, 1, 2]
    prev_positions = jnp.zeros((K, 2))  # (3, 2)
    prev_velocities = jnp.zeros((K, 2))  # (3, 2)

    print("Function signature has 6 parameters:")
    print(f"  k: component_indices shape {component_indices.shape}")
    print(f"  prev_pos: prev_positions shape {prev_positions.shape}")
    print(f"  prev_vel: prev_velocities shape {prev_velocities.shape}")
    print(f"  is_initial: {is_initial} (scalar)")
    print(f"  dt: {dt_val} (scalar)")
    print(f"  process_noise: {process_noise_val} (scalar)")

    try:
        # Test single call first
        print("\\nTesting single component call...")
        single_result = single_component_dynamics.simulate(
            0,
            prev_positions[0],
            prev_velocities[0],
            is_initial,
            dt_val,
            process_noise_val,
        )
        print(f"Single call works. Choices: {list(single_result.get_choices().keys())}")

        # Now test vmap
        print("\\nTesting vmap...")
        vectorized_dynamics = single_component_dynamics.vmap(
            in_axes=(0, 0, 0, None, None, None)  # 6 parameters -> 6 in_axes entries
        )

        print("Vmap created successfully, now calling...")
        result = vectorized_dynamics.simulate(
            component_indices,  # (K,)
            prev_positions,  # (K, 2)
            prev_velocities,  # (K, 2)
            is_initial,  # scalar
            dt_val,  # scalar
            process_noise_val,  # scalar
        )

        print("✅ Vmap succeeded!")
        print(f"Result type: {type(result)}")
        print(f"Choices keys: {list(result.get_choices().keys())}")
        print(f"Return value shape: {jnp.array(result.get_retval()).shape}")

    except Exception as e:
        print(f"❌ Vmap failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_vmap_with_exact_signature()
