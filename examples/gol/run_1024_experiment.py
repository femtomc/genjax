#!/usr/bin/env python3
"""Run the 1024x1024 GOL experiment and report timing and accuracy."""

import time
import jax
import jax.numpy as jnp
import jax.random as jrand

# Import GOL modules
import core
from data import get_wizards_logo

def run_experiment():
    """Run the 1024x1024 GOL experiment with 500 Gibbs iterations."""
    print("Setting up 1024x1024 Game of Life experiment...")
    
    # Parameters
    size = 1024
    chain_length = 500
    flip_prob = 0.03
    seed = 42
    
    # Get the 1024x1024 wizards pattern
    print(f"Loading wizards logo at {size}x{size} resolution...")
    # Since the original is 512x512, this will upscale
    target = jax.image.resize(get_wizards_logo(), (size, size), method="nearest")
    target = jnp.where(target > 0.5, 1, 0)  # Re-binarize
    
    print(f"Target pattern shape: {target.shape}")
    print(f"Active cells: {jnp.sum(target)}")
    print(f"Grid size: {size}x{size} = {size*size} cells")
    print(f"State space size: 2^{size*size}")
    
    # Create sampler
    sampler = core.GibbsSampler(target, flip_prob)
    
    # Time the experiment
    print(f"\nRunning {chain_length} Gibbs iterations...")
    key = jrand.key(seed)
    
    start_time = time.time()
    run_summary = core.run_sampler_and_get_summary(
        key, sampler, chain_length, n_steps_per_summary_frame=1
    )
    elapsed_time = time.time() - start_time
    
    # Get accuracy metrics
    final_pred_post = run_summary.predictive_posterior_scores[-1]
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(target)
    total_bits = target.size
    accuracy = (1 - final_n_bit_flips / total_bits) * 100
    
    # Report results
    print(f"\n{'='*50}")
    print(f"EXPERIMENT RESULTS - {size}x{size} Grid")
    print(f"{'='*50}")
    print(f"Chain length: {chain_length} iterations")
    print(f"Flip probability: {flip_prob}")
    print(f"Runtime: {elapsed_time:.2f} seconds")
    print(f"Time per iteration: {elapsed_time/chain_length:.3f} seconds")
    print(f"Final predictive posterior: {final_pred_post:.6f}")
    print(f"Reconstruction errors: {final_n_bit_flips} bits out of {total_bits}")
    print(f"Reconstruction accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    return elapsed_time, accuracy

if __name__ == "__main__":
    # Check JAX device
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    
    runtime, accuracy = run_experiment()