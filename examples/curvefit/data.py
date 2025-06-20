"""
Common data generation for curvefit case study.

This module provides standardized test datasets for benchmarking across
GenJAX and NumPyro implementations. All frameworks use the same
underlying data to ensure fair comparisons.
"""

import jax.numpy as jnp
import jax.random as jrand


def sinfn(x, freq, offset):
    """
    Sine function with frequency and offset parameters.

    Args:
        x: Input locations
        freq: Frequency parameter
        offset: Offset parameter

    Returns:
        Sine wave values
    """
    return jnp.sin(2.0 * jnp.pi * freq * x + offset)


def generate_test_dataset(
    key=None, n_points=20, true_freq=0.3, true_offset=1.5, noise_std=0.3, seed=42
):
    """
    Generate a standardized test dataset for all framework implementations.

    This function creates synthetic data from a simple sine curve with Gaussian noise.

    Args:
        key: JAX random key (if None, uses seed)
        n_points: Number of data points
        true_freq: True frequency parameter
        true_offset: True phase offset parameter
        noise_std: Standard deviation of observation noise (default 0.3)
        seed: Random seed (used if key is None)

    Returns:
        Dictionary with:
            - xs: Input locations (JAX array)
            - ys: Observed values (JAX array)
            - true_params: True parameters dict
            - clean_ys: Noise-free deterministic values
    """
    if key is None:
        key = jrand.key(seed)

    key1, key2 = jrand.split(key, 2)

    # Generate input locations - use finer spacing for smoother curves
    xs = jnp.linspace(0, n_points / 4, n_points, dtype=jnp.float32)

    # Generate clean deterministic values
    clean_ys = sinfn(xs, true_freq, true_offset)

    # Add observation noise
    noise = jrand.normal(key1, shape=(n_points,)) * noise_std
    ys = clean_ys + noise

    # Package results
    result = {
        "xs": xs,
        "ys": ys,
        "true_params": {
            "freq": true_freq,
            "offset": true_offset,
            "noise_std": noise_std,
        },
        "clean_ys": clean_ys,
    }

    return result


def print_dataset_summary(data_dict, name="Dataset"):
    """
    Print a summary of the dataset characteristics.

    Args:
        data_dict: Dictionary from generate_test_dataset
        name: Name for the dataset
    """
    print(f"\n=== {name} Summary ===")
    print(f"  Number of points: {len(data_dict['xs'])}")
    print(f"  True frequency: {data_dict['true_params']['freq']:.3f}")
    print(f"  True offset: {data_dict['true_params']['offset']:.3f}")
    print(f"  Noise std: {data_dict['true_params']['noise_std']:.3f}")
    print(f"  Y range: [{data_dict['ys'].min():.3f}, {data_dict['ys'].max():.3f}]")
