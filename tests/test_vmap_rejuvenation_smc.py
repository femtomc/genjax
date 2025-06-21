"""
Test Vmap integration with rejuvenation_smc.

This test isolates the issue where Vmap combinators fail when used inside
model functions called by rejuvenation_smc, due to argument count mismatches
during generate vs regenerate phases.
"""

import jax.numpy as jnp
import jax.random as jrand

from genjax import gen, normal, Vmap, Const, seed
from genjax.inference import rejuvenation_smc, mh
from genjax.core import sel, const


@gen
def simple_sensor_ray(distance: float, ray_idx: int):
    """Simple sensor model for a single ray."""
    return normal(distance, 0.1) @ "measurement"


@gen
def model_with_vmap(prev_state):
    """Model function that uses Vmap internally (like localization)."""
    # Simple state transition
    new_state = normal(prev_state + 0.1, 0.2) @ "state"

    # Vectorized sensor observations (like LIDAR)
    true_distances = jnp.array([1.0, 2.0, 3.0])  # 3 rays
    ray_indices = jnp.arange(3)

    # This is where the issue occurs
    vectorized_sensor = Vmap(
        simple_sensor_ray,
        in_axes=Const((0, 0)),  # distance=0, ray_idx=0
        axis_size=Const(3),
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    # Apply vectorized sensor - this fails during rejuvenation_smc
    vectorized_sensor(true_distances, ray_indices) @ "sensors"

    return new_state


@gen
def simple_proposal(constraints, old_choices, prev_state):
    """Simple proposal function."""
    return normal(prev_state, 0.1) @ "state"


def test_vmap_with_rejuvenation_smc_basic():
    """Test that Vmap works with rejuvenation_smc - basic case.

    This test verifies that the Vmap issue has been fixed and Vmap now works
    correctly with rejuvenation_smc during MCMC regeneration.
    """
    key = jrand.key(42)

    # Create synthetic observations
    # Shape: (T, n_sensors) = (2, 3)
    observations = jnp.array(
        [
            [1.1, 2.1, 3.1],  # Timestep 1
            [1.2, 2.2, 3.2],  # Timestep 2
        ]
    )

    # Observation structure for rejuvenation_smc
    obs_sequence = {"sensors": {"measurement": observations}}

    # Initial arguments
    initial_state = 0.0
    initial_args = (initial_state,)

    # MCMC kernel
    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    # This should now work with the Vmap fix
    result = seed(rejuvenation_smc)(
        key,
        model_with_vmap,
        simple_proposal,
        const(mcmc_kernel),
        obs_sequence,
        initial_args,
        const(10),  # n_particles
    )

    # Verify the results
    assert result.log_marginal_likelihood() is not None
    assert result.effective_sample_size() > 0

    # Check final particles
    final_traces = result.traces
    choices = final_traces.get_choices()
    assert "state" in choices
    assert choices["state"].shape == (10,)  # n_particles


@gen
def model_without_vmap(prev_state):
    """Control model without Vmap (should work)."""
    # Simple state transition
    new_state = normal(prev_state + 0.1, 0.2) @ "state"

    # Manual sensor observations (no Vmap)
    normal(1.0, 0.1) @ "sensor_0"
    normal(2.0, 0.1) @ "sensor_1"
    normal(3.0, 0.1) @ "sensor_2"

    return new_state


def test_rejuvenation_smc_without_vmap():
    """Control test - rejuvenation_smc without Vmap should work."""
    key = jrand.key(42)

    # Create synthetic observations for manual sensors
    obs_sequence = {
        "sensor_0": jnp.array([1.1, 1.2]),
        "sensor_1": jnp.array([2.1, 2.2]),
        "sensor_2": jnp.array([3.1, 3.2]),
    }

    # Initial arguments
    initial_state = 0.0
    initial_args = (initial_state,)

    # MCMC kernel
    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    # This should work fine
    result = seed(rejuvenation_smc)(
        key,
        model_without_vmap,
        simple_proposal,
        const(mcmc_kernel),
        obs_sequence,
        initial_args,
        const(10),  # n_particles
    )

    # Basic checks
    assert result.log_marginal_likelihood() is not None
    assert result.effective_sample_size() > 0


@gen
def debug_sensor_ray(*args):
    """Debug version that prints arguments received."""
    print(
        f"debug_sensor_ray received {len(args)} arguments: {[type(arg) for arg in args]}"
    )

    # Extract distance regardless of argument pattern
    if len(args) >= 1:
        distance = args[0]
    else:
        distance = 1.0

    return normal(distance, 0.1) @ "measurement"


@gen
def model_with_debug_vmap(prev_state):
    """Model with debug Vmap to understand argument patterns."""
    new_state = normal(prev_state + 0.1, 0.2) @ "state"

    true_distances = jnp.array([1.0, 2.0])  # Simplified to 2 rays
    ray_indices = jnp.arange(2)

    # Try different in_axes configurations
    vectorized_sensor = Vmap(
        debug_sensor_ray,
        in_axes=Const((0, 0)),  # Start with basic case
        axis_size=Const(2),
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    vectorized_sensor(true_distances, ray_indices) @ "sensors"
    return new_state


def test_debug_vmap_arguments():
    """Debug test to understand what arguments Vmap receives."""
    key = jrand.key(42)

    # Just test the model directly first
    initial_state = 0.0
    seeded_model = seed(model_with_debug_vmap.simulate)

    print("=== Testing model directly ===")
    trace = seeded_model(key, initial_state)
    result = trace.get_retval()
    print(f"Direct model result: {result}")

    # Then test with rejuvenation_smc to see the difference
    print("\n=== Testing with rejuvenation_smc ===")

    observations = jnp.array(
        [
            [1.1, 2.1],  # Timestep 1
        ]
    )
    obs_sequence = {"sensors": {"measurement": observations}}
    initial_args = (initial_state,)

    def mcmc_kernel(trace):
        return mh(trace, sel("state"))

    try:
        result = seed(rejuvenation_smc)(
            key,
            model_with_debug_vmap,
            simple_proposal,
            const(mcmc_kernel),
            obs_sequence,
            initial_args,
            const(5),
        )
        print("rejuvenation_smc succeeded!")
    except Exception as e:
        print(f"rejuvenation_smc failed: {e}")


@gen
def lidar_sensor_ray(distance: float, ray_idx: int):
    """LIDAR sensor model for a single ray (like in localization)."""
    return normal(distance, 0.8) @ "distance"


@gen
def localization_like_model(prev_pose, world_size):
    """Model similar to localization case study."""
    # Simple 2D pose transition
    x = normal(prev_pose[0] + 0.1, 0.3) @ "x"
    y = normal(prev_pose[1] + 0.1, 0.3) @ "y"

    # Clip to world boundaries
    x = jnp.clip(x, 0.1, world_size - 0.1)
    y = jnp.clip(y, 0.1, world_size - 0.1)

    current_pose = jnp.array([x, y])

    # Simulate LIDAR measurements (8 rays like in localization)
    true_distances = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ray_indices = jnp.arange(8)

    # This is the problematic Vmap usage from localization
    vectorized_lidar = Vmap(
        lidar_sensor_ray,
        in_axes=Const((0, 0)),
        axis_size=Const(8),
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    # This fails during rejuvenation_smc
    vectorized_lidar(true_distances, ray_indices) @ "lidar"

    return current_pose, world_size


@gen
def localization_proposal(constraints, old_choices, prev_pose, world_size):
    """Proposal for localization-like model."""
    x = normal(prev_pose[0], 0.5) @ "x"
    y = normal(prev_pose[1], 0.5) @ "y"

    x = jnp.clip(x, 0.1, world_size - 0.1)
    y = jnp.clip(y, 0.1, world_size - 0.1)

    return jnp.array([x, y]), world_size


def test_localization_vmap_issue():
    """Test the specific Vmap issue from localization case study.

    This verifies that the LIDAR sensor observations using Vmap now work
    correctly with rejuvenation_smc after the fix.
    """
    key = jrand.key(42)
    world_size = 10.0

    # LIDAR observations: (T, n_rays) = (2, 8)
    lidar_observations = jnp.array(
        [
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],  # Timestep 1
            [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],  # Timestep 2
        ]
    )

    obs_sequence = {"lidar": {"distance": lidar_observations}}

    # Initial pose and world
    initial_pose = jnp.array([5.0, 5.0])
    initial_args = (initial_pose, world_size)

    def mcmc_kernel(trace):
        return mh(trace, sel("x") | sel("y"))

    # This should now work with the Vmap fix
    result = seed(rejuvenation_smc)(
        key,
        localization_like_model,
        localization_proposal,
        const(mcmc_kernel),
        obs_sequence,
        initial_args,
        const(10),
    )

    # Verify the results
    assert result.log_marginal_likelihood() is not None
    assert result.effective_sample_size() > 0

    # Check final particles have the right structure
    final_traces = result.traces
    choices = final_traces.get_choices()
    assert "x" in choices
    assert "y" in choices
    assert choices["x"].shape == (10,)  # n_particles
    assert choices["y"].shape == (10,)  # n_particles


if __name__ == "__main__":
    # Run debug test to understand the issue
    test_debug_vmap_arguments()

    # Run the specific tests
    print("\n=== Testing localization-like Vmap issue ===")
    try:
        # Run the test manually without pytest
        key = jrand.key(42)
        world_size = 10.0

        lidar_observations = jnp.array(
            [
                [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],  # Timestep 1
                [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],  # Timestep 2
            ]
        )

        obs_sequence = {"lidar": {"distance": lidar_observations}}
        initial_pose = jnp.array([5.0, 5.0])
        initial_args = (initial_pose, world_size)

        def mcmc_kernel(trace):
            return mh(trace, sel("x") | sel("y"))

        result = seed(rejuvenation_smc)(
            key,
            localization_like_model,
            localization_proposal,
            const(mcmc_kernel),
            obs_sequence,
            initial_args,
            const(10),
        )
        print("✗ Test failed - expected ValueError but got success")

    except ValueError as e:
        if "vmap in_axes must be" in str(e):
            print("✓ Test passed (correctly caught the expected Vmap error)")
        else:
            print(f"✗ Test failed - wrong ValueError: {e}")
    except Exception as e:
        print(f"✗ Test failed unexpectedly: {e}")
