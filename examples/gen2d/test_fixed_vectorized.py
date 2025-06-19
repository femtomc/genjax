"""Test fixed vectorized implementation."""

import jax
import jax.numpy as jnp
import time

from core_vectorized_fixed import run_vectorized_gen2d_inference
from data import generate_tracking_data


def test_fixed_vectorized():
    """Test the fixed vectorized implementation."""
    print("Testing fixed vectorized implementation...")

    # Test parameters
    n_components = 3  # Test with 3 components
    n_particles = 20
    n_frames = 4

    # Generate test data
    key = jax.random.PRNGKey(42)
    data_key, inference_key = jax.random.split(key)

    grids, observations, counts = generate_tracking_data(
        pattern="oscillators",
        grid_size=32,
        n_steps=n_frames,
        max_pixels=100,
        key=data_key,
    )

    print(f"Generated {n_frames + 1} frames with {jnp.sum(counts)} total active pixels")

    # Test fixed vectorized implementation
    print(f"\\n--- Testing Fixed Vectorized Implementation (K={n_components}) ---")
    start_time = time.time()
    try:
        vectorized_particles = run_vectorized_gen2d_inference(
            observations=observations,
            observation_counts=counts,
            n_components=n_components,
            n_particles=n_particles,
            dt=0.1,
            process_noise=0.5,
            obs_std=2.0,
            n_rejuvenation_moves=1,
            key=inference_key,
        )
        vectorized_time = time.time() - start_time
        vectorized_log_ml = vectorized_particles.log_marginal_likelihood()
        vectorized_ess = vectorized_particles.effective_sample_size()

        print(f"✅ Fixed Vectorized: {vectorized_time:.2f}s")
        print(f"   Log ML: {vectorized_log_ml}")
        print(f"   ESS: {vectorized_ess}")

    except Exception as e:
        print(f"❌ Fixed Vectorized failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test different component sizes
    print("\\n--- Testing Scalability ---")
    for K in [2, 3, 4, 5]:
        start_time = time.time()
        try:
            test_particles = run_vectorized_gen2d_inference(
                observations=observations,
                observation_counts=counts,
                n_components=K,
                n_particles=10,  # Fewer particles for speed
                dt=0.1,
                process_noise=0.5,
                obs_std=2.0,
                n_rejuvenation_moves=1,
                key=jax.random.split(inference_key, K)[0],
            )
            test_time = time.time() - start_time
            test_log_ml = test_particles.log_marginal_likelihood()

            print(f"K={K}: {test_time:.2f}s, Log ML: {test_log_ml}")

        except Exception as e:
            print(f"K={K}: Failed - {e}")

    print("\\n✅ Fixed vectorized tests completed!")


if __name__ == "__main__":
    test_fixed_vectorized()
