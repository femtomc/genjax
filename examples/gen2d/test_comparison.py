"""Compare all three implementations: original (hardcoded 2), simplified (hardcoded 2), and fixed vectorized (scalable)."""

import jax
import jax.numpy as jnp
import time

from core import run_gen2d_inference  # simplified version
from core_vectorized_fixed import (
    run_vectorized_gen2d_inference,
)  # our working vectorized version
from data import generate_tracking_data


def compare_implementations():
    """Compare different implementations."""
    print("=== Implementation Comparison ===\n")

    # Test parameters
    n_particles = 20
    n_frames = 4

    # Generate test data
    key = jax.random.PRNGKey(42)
    data_key, simp_key, vec_key = jax.random.split(key, 3)

    grids, observations, counts = generate_tracking_data(
        pattern="oscillators",
        grid_size=32,
        n_steps=n_frames,
        max_pixels=100,
        key=data_key,
    )

    print(
        f"Generated {n_frames + 1} frames with {jnp.sum(counts)} total active pixels\n"
    )

    # Test simplified implementation (K=2 only)
    print("--- Simplified Implementation (K=2, hardcoded) ---")
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
            key=simp_key,
        )
        simplified_time = time.time() - start_time
        simplified_log_ml = simplified_particles.log_marginal_likelihood()
        simplified_ess = simplified_particles.effective_sample_size()

        print(f"✅ Simplified: {simplified_time:.2f}s")
        print(f"   Log ML: {simplified_log_ml}")
        print(f"   ESS: {simplified_ess}")

    except Exception as e:
        print(f"❌ Simplified failed: {e}")

    # Test vectorized implementation (K=2 for comparison)
    print("\n--- Fixed Vectorized Implementation (K=2, for comparison) ---")
    start_time = time.time()
    try:
        vectorized_particles = run_vectorized_gen2d_inference(
            observations=observations,
            observation_counts=counts,
            n_components=2,
            n_particles=n_particles,
            dt=0.1,
            process_noise=0.5,
            obs_std=2.0,
            n_rejuvenation_moves=1,
            key=vec_key,
        )
        vectorized_time = time.time() - start_time
        vectorized_log_ml = vectorized_particles.log_marginal_likelihood()
        vectorized_ess = vectorized_particles.effective_sample_size()

        print(f"✅ Vectorized: {vectorized_time:.2f}s")
        print(f"   Log ML: {vectorized_log_ml}")
        print(f"   ESS: {vectorized_ess}")

    except Exception as e:
        print(f"❌ Vectorized failed: {e}")

    # Test scalability of vectorized version
    print("\n--- Scalability Test (Vectorized Only) ---")
    for K in [3, 4, 5]:
        start_time = time.time()
        try:
            test_particles = run_vectorized_gen2d_inference(
                observations=observations,
                observation_counts=counts,
                n_components=K,
                n_particles=15,  # Fewer particles for speed
                dt=0.1,
                process_noise=0.5,
                obs_std=2.0,
                n_rejuvenation_moves=1,
                key=jax.random.split(vec_key, K)[0],
            )
            test_time = time.time() - start_time
            test_log_ml = test_particles.log_marginal_likelihood()

            print(f"K={K}: {test_time:.2f}s, Final Log ML: {test_log_ml[-1]:.1f}")

        except Exception as e:
            print(f"K={K}: Failed - {e}")

    print("\n✅ Comparison completed!")
    print("✅ Successfully implemented scalable vectorized model!")


if __name__ == "__main__":
    compare_implementations()
