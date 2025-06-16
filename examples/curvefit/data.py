"""
Common data generation for curvefit case study.

This module provides standardized test datasets for benchmarking across
GenJAX, Pyro, and NumPyro implementations. All frameworks use the same
underlying data to ensure fair comparisons.
"""

import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import torch


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


def convert_to_torch(data_dict):
    """
    Convert JAX arrays to PyTorch tensors for Pyro implementation.

    Args:
        data_dict: Dictionary from generate_test_dataset

    Returns:
        Dictionary with PyTorch tensors
    """
    torch_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, jnp.ndarray):
            torch_dict[key] = torch.from_numpy(np.array(value)).float()
        elif isinstance(value, dict):
            torch_dict[key] = convert_to_torch(value)
        else:
            torch_dict[key] = value

    return torch_dict


def convert_to_numpy(data_dict):
    """
    Convert JAX arrays to NumPy arrays for general plotting.

    Args:
        data_dict: Dictionary from generate_test_dataset

    Returns:
        Dictionary with NumPy arrays
    """
    numpy_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, jnp.ndarray):
            numpy_dict[key] = np.array(value)
        elif isinstance(value, dict):
            numpy_dict[key] = convert_to_numpy(value)
        else:
            numpy_dict[key] = value

    return numpy_dict


def get_standard_datasets():
    """
    Generate standard test datasets for benchmarking.

    Returns:
        Dictionary with different test scenarios
    """
    datasets = {}

    # Small dataset (20 points) - nice sine curve
    datasets["small"] = generate_test_dataset(
        seed=42, n_points=20, true_freq=0.3, true_offset=1.5
    )

    # Medium dataset (50 points) - longer range
    datasets["medium"] = generate_test_dataset(
        seed=123, n_points=50, true_freq=0.2, true_offset=2.0
    )

    # Large dataset (100 points) - full cycles
    datasets["large"] = generate_test_dataset(
        seed=456, n_points=100, true_freq=0.15, true_offset=0.8
    )

    # Higher frequency sine curve
    datasets["high_freq"] = generate_test_dataset(
        seed=789, n_points=30, true_freq=0.5, true_offset=1.2
    )

    # Lower noise sine curve
    datasets["low_noise"] = generate_test_dataset(
        seed=101, n_points=30, true_freq=0.4, true_offset=1.8, noise_std=0.15
    )

    return datasets


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


# Example usage and testing
if __name__ == "__main__":
    print("=== Common Dataset Generation ===")

    # Generate the default test dataset
    data = generate_test_dataset(seed=42, n_points=20)
    print_dataset_summary(data, "Default Test Dataset")

    # Show framework-specific conversions
    torch_data = convert_to_torch(data)
    numpy_data = convert_to_numpy(data)

    print("\nFramework conversions:")
    print(f"  JAX xs shape: {data['xs'].shape}, dtype: {data['xs'].dtype}")
    print(
        f"  PyTorch xs shape: {torch_data['xs'].shape}, dtype: {torch_data['xs'].dtype}"
    )
    print(
        f"  NumPy xs shape: {numpy_data['xs'].shape}, dtype: {numpy_data['xs'].dtype}"
    )

    # Generate standard datasets
    print("\n=== Standard Test Datasets ===")
    datasets = get_standard_datasets()
    for name, dataset in datasets.items():
        print_dataset_summary(dataset, name.title())
