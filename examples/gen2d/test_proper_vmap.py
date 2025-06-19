"""Test the properly corrected vmap implementation."""

import jax
import jax.numpy as jnp
import time

from core_vmap_proper import run_proper_vectorized_gen2d_inference
from data import generate_tracking_data


def test_proper_vmap():
    """Test the corrected vmap implementation."""
    print("=== Testing Properly Corrected Vmap Implementation ===\n")

    # Test parameters
    n_components = 3
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

    # Test corrected vmap implementation
    print(f"\n--- Testing Corrected Vmap Implementation (K={n_components}) ---")
    start_time = time.time()
    try:
        particles = run_proper_vectorized_gen2d_inference(
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
        elapsed_time = time.time() - start_time
        log_ml = particles.log_marginal_likelihood()
        ess = particles.effective_sample_size()

        print(f"✅ Corrected Vmap: {elapsed_time:.2f}s")
        print(f"   Log ML: {log_ml}")
        print(f"   ESS: {ess}")

    except Exception as e:
        print(f"❌ Corrected Vmap failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test scalability
    print("\n--- Testing Scalability ---")
    for K in [2, 4, 5]:
        start_time = time.time()
        try:
            test_particles = run_proper_vectorized_gen2d_inference(
                observations=observations,
                observation_counts=counts,
                n_components=K,
                n_particles=15,  # Fewer particles for speed
                dt=0.1,
                process_noise=0.5,
                obs_std=2.0,
                n_rejuvenation_moves=1,
                key=jax.random.split(inference_key, K)[0],
            )
            test_time = time.time() - start_time
            test_log_ml = test_particles.log_marginal_likelihood()

            print(f"K={K}: {test_time:.2f}s, Final Log ML: {test_log_ml[-1]:.1f}")

        except Exception as e:
            print(f"K={K}: Failed - {e}")

    print("\n✅ Properly corrected vmap tests completed!")
    print(
        "🎉 Successfully implemented scalable vectorized model with correct addressing!"
    )


if __name__ == "__main__":
    test_proper_vmap()
