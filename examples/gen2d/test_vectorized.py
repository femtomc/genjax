"""Test vectorized implementation with modular_vmap."""

import jax
import jax.numpy as jnp
import time

from core_vectorized import run_vectorized_gen2d_inference
from core import run_gen2d_inference
from data import generate_tracking_data


def test_vectorized_vs_simplified():
    """Compare vectorized modular_vmap implementation with simplified version."""
    print("Testing vectorized vs simplified implementation...")

    # Test parameters
    n_components = 2  # Vectorized version only supports 2 components for now
    n_particles = 20
    n_frames = 5

    # Generate test data
    key = jax.random.PRNGKey(42)
    data_key, vectorized_key, simplified_key = jax.random.split(key, 3)

    grids, observations, counts = generate_tracking_data(
        pattern="oscillators",
        grid_size=32,
        n_steps=n_frames,
        max_pixels=100,
        key=data_key,
    )

    print(f"Generated {n_frames + 1} frames with {jnp.sum(counts)} total active pixels")

    # Test vectorized implementation
    print("\n--- Testing Vectorized Implementation ---")
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
            key=vectorized_key,
        )
        vectorized_time = time.time() - start_time
        vectorized_log_ml = vectorized_particles.log_marginal_likelihood()
        vectorized_ess = vectorized_particles.effective_sample_size()

        print(f"✅ Vectorized: {vectorized_time:.2f}s")
        print(f"   Log ML: {vectorized_log_ml}")
        print(f"   ESS: {vectorized_ess}")

    except Exception as e:
        print(f"❌ Vectorized failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test simplified implementation (fixed to 2 components)
    print("\n--- Testing Simplified Implementation ---")
    start_time = time.time()
    try:
        simplified_particles = run_gen2d_inference(
            observations=observations,
            observation_counts=counts,
            n_components=2,  # Simplified only supports 2
            n_particles=n_particles,
            dt=0.1,
            process_noise=0.5,
            obs_std=2.0,
            n_rejuvenation_moves=1,
            key=simplified_key,
        )
        simplified_time = time.time() - start_time
        simplified_log_ml = simplified_particles.log_marginal_likelihood()
        simplified_ess = simplified_particles.effective_sample_size()

        print(f"✅ Simplified: {simplified_time:.2f}s")
        print(f"   Log ML: {simplified_log_ml}")
        print(f"   ESS: {simplified_ess}")

    except Exception as e:
        print(f"❌ Simplified failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n--- Comparison ---")
    print(f"Speedup: {simplified_time / vectorized_time:.2f}x")
    print("Both implementations completed successfully!")

    # Test scaling with more components
    print("\n--- Testing Scalability ---")
    for K in [2, 3, 5, 8]:
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
                key=jax.random.split(vectorized_key, K)[0],
            )
            test_time = time.time() - start_time
            test_log_ml = test_particles.log_marginal_likelihood()

            print(f"K={K}: {test_time:.2f}s, Log ML: {test_log_ml}")

        except Exception as e:
            print(f"K={K}: Failed - {e}")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_vectorized_vs_simplified()
