"""Simple test to debug SMC functionality."""

import jax
import jax.numpy as jnp
from genjax import gen, const, Const
from genjax.distributions import multivariate_normal
from genjax.inference import rejuvenation_smc, mh
from genjax.pjax import seed
from genjax import sel


@gen
def simple_model(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
):
    """Very simple tracking model with just positions."""
    K = n_components_const.value
    dt_val = dt_const.value
    process_noise_val = process_noise_const.value

    is_initial = time_index == 0

    # Just track positions (no velocities for simplicity)
    positions = []
    for k in range(K):
        # Use JAX control flow to avoid TracerBoolConversionError
        init_mean = jnp.array([0.0, 0.0])
        init_cov = 10.0 * jnp.eye(2)

        trans_mean = prev_state[k] if prev_state is not None else init_mean
        trans_cov = process_noise_val**2 * jnp.eye(2)

        # Select based on time index using JAX
        mean = jax.lax.select(is_initial, init_mean, trans_mean)
        cov = jax.lax.select(is_initial, init_cov, trans_cov)

        pos = multivariate_normal(mean, cov) @ f"position_{k}"
        positions.append(pos)

    positions = jnp.stack(positions)

    # Simple observation model - just observe first position with noise
    _ = multivariate_normal(positions[0], 1.0 * jnp.eye(2)) @ "obs"

    # Return state for next timestep
    new_time_index = time_index + 1
    return positions, new_time_index, n_components_const, dt_const, process_noise_const


@gen
def simple_proposal(
    prev_state,
    time_index,
    n_components_const: Const[int],
    dt_const: Const[float],
    process_noise_const: Const[float],
):
    """Simple proposal matching the model."""
    K = n_components_const.value
    dt_val = dt_const.value
    process_noise_val = process_noise_const.value

    is_initial = time_index == 0

    for k in range(K):
        # Use JAX control flow
        init_mean = jnp.array([0.0, 0.0])
        init_cov = 10.0 * jnp.eye(2)

        trans_mean = prev_state[k] if prev_state is not None else init_mean
        trans_cov = process_noise_val**2 * jnp.eye(2)

        mean = jax.lax.select(is_initial, init_mean, trans_mean)
        cov = jax.lax.select(is_initial, init_cov, trans_cov)

        _ = multivariate_normal(mean, cov) @ f"position_{k}"


def test_simple_smc():
    """Test basic SMC functionality."""
    print("Testing simple SMC...")

    # Simple parameters
    n_components = 2
    n_particles = 5
    n_timesteps = 3
    dt = 0.1
    process_noise = 0.5

    # Create dummy observations
    key = jax.random.PRNGKey(42)
    obs_key, smc_key = jax.random.split(key)

    # Generate some dummy observation data (time series for each observable)
    obs_data = {}
    dummy_obs = jax.random.normal(obs_key, (n_timesteps, 2))
    obs_data["obs"] = dummy_obs  # (T, 2) array

    # Prepare constants
    n_components_const = const(n_components)
    dt_const = const(dt)
    process_noise_const = const(process_noise)

    # Initial state
    initial_positions = jnp.zeros((n_components, 2))
    initial_args = (
        initial_positions,
        jnp.array(0),
        n_components_const,
        dt_const,
        process_noise_const,
    )

    # Simple kernel - just update positions
    def simple_kernel(trace):
        # Update one at a time to keep it simple
        for k in range(n_components):
            trace = mh(trace, sel(f"position_{k}"))
        return trace

    print("Running SMC...")

    # Test just the model compilation first
    try:
        test_result = seed(simple_model)(
            jax.random.PRNGKey(0),
            initial_positions,
            jnp.array(0),
            n_components_const,
            dt_const,
            process_noise_const,
        )
        print(f"Model compiled successfully. Result type: {type(test_result)}")
        print(f"Result value: {test_result}")

        # Test with simulate method instead (simulate doesn't take key first)
        test_trace = simple_model.simulate(
            initial_positions,
            jnp.array(0),
            n_components_const,
            dt_const,
            process_noise_const,
        )
        print(f"Simulate successful. Trace type: {type(test_trace)}")
        choices = test_trace.get_choices()
        print(
            f"Choices: {list(choices.keys()) if hasattr(choices, 'keys') else choices}"
        )
        print(f"Return value: {test_trace.get_retval()}")
        print(f"Score: {test_trace.get_score()}")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Now test SMC
    try:
        particles = seed(rejuvenation_smc)(
            smc_key,
            simple_model,
            simple_proposal,
            const(simple_kernel),
            obs_data,
            initial_args,
            const(n_particles),
            const(True),  # return_all_particles
            const(1),  # n_rejuvenation_moves
        )

        print("SMC completed successfully!")
        print(f"Particles type: {type(particles)}")

        # Access log marginal likelihood
        try:
            log_ml = particles.log_marginal_likelihood()
            print(f"Log marginal likelihood: {log_ml}")
        except Exception as e:
            print(f"Could not get log marginal likelihood: {e}")

        # Access effective sample size
        try:
            ess = particles.effective_sample_size()
            print(f"Effective sample size: {ess}")
        except Exception as e:
            print(f"Could not get effective sample size: {e}")

        print(
            f"SMC succeeded with {n_particles} particles and {n_timesteps} timesteps!"
        )

    except Exception as e:
        print(f"SMC failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_simple_smc()
