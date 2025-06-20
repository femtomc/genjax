"""
Data export/import utilities for curvefit experiments.

Enables separation of expensive inference computations from plotting,
allowing for rapid visualization iteration without recomputation.
"""

import os
import csv
import json
import numpy as np
import jax.numpy as jnp
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


def _convert_types(obj):
    """Convert JAX/NumPy arrays to Python types for JSON serialization."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_types(v) for v in obj]
    else:
        return obj


def create_experiment_directory(
    base_dir: str = "data", experiment_name: Optional[str] = None
) -> str:
    """Create timestamped experiment directory."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"curvefit_experiment_{timestamp}"

    exp_dir = os.path.join("examples/curvefit", base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_metadata(data_dir: str, config: Dict[str, Any]):
    """Save experiment configuration and metadata."""
    metadata_path = os.path.join(data_dir, "experiment_metadata.json")

    # Convert any JAX arrays to regular Python types
    config = _convert_types(config)

    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def save_dataset(data_dir: str, xs, ys, true_freq: float, true_offset: float):
    """Save the dataset used for inference."""
    dataset_path = os.path.join(data_dir, "dataset.csv")
    with open(dataset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "true_freq", "true_offset"])
        # Write true parameters in first row
        writer.writerow(["", "", true_freq, true_offset])
        # Write data points
        for x, y in zip(xs, ys):
            writer.writerow([float(x), float(y), "", ""])
    print(f"Saved dataset: {dataset_path}")


def save_inference_results(
    data_dir: str,
    method_name: str,
    samples: Dict[str, jnp.ndarray],
    weights: Optional[jnp.ndarray] = None,
    timing_stats: Optional[Tuple[float, float]] = None,
    additional_stats: Optional[Dict[str, Any]] = None,
):
    """Save inference results for a specific method."""
    method_dir = os.path.join(data_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # Save samples
    samples_path = os.path.join(method_dir, "samples.csv")
    with open(samples_path, "w", newline="") as f:
        writer = csv.writer(f)
        # For curve fitting, we expect freq and offset parameters
        writer.writerow(["sample_id", "freq", "offset", "weight"])

        freq_samples = samples.get("freq", samples.get("curve", {}).get("freq", []))
        off_samples = samples.get("off", samples.get("curve", {}).get("off", []))

        if weights is None:
            weights = jnp.ones(len(freq_samples))

        for i, (freq, off, w) in enumerate(zip(freq_samples, off_samples, weights)):
            writer.writerow([i, float(freq), float(off), float(w)])

    # Save timing statistics if provided
    if timing_stats is not None:
        timing_path = os.path.join(method_dir, "timing.csv")
        with open(timing_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mean_time_ms", "std_time_ms"])
            # Convert to milliseconds for consistency
            writer.writerow(
                [float(timing_stats[0] * 1000), float(timing_stats[1] * 1000)]
            )

    # Save additional statistics if provided
    if additional_stats is not None:
        stats_path = os.path.join(method_dir, "statistics.json")
        # Convert JAX arrays to Python types
        stats = _convert_types(additional_stats)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"Saved {method_name} results: {method_dir}")


def save_benchmark_summary(data_dir: str, results: Dict[str, Dict[str, Any]]):
    """Save a summary of all benchmark results."""
    summary_path = os.path.join(data_dir, "benchmark_summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "mean_time_ms",
                "std_time_ms",
                "freq_mean",
                "freq_std",
                "offset_mean",
                "offset_std",
                "acceptance_rate",
            ]
        )

        for method_name, result in results.items():
            timing = result.get("timing_stats", (0, 0))
            samples = result.get("samples", {})
            stats = result.get("additional_stats", {})

            # Extract frequency and offset samples
            freq_samples = samples.get(
                "freq", samples.get("curve", {}).get("freq", jnp.array([0]))
            )
            off_samples = samples.get(
                "off", samples.get("curve", {}).get("off", jnp.array([0]))
            )

            writer.writerow(
                [
                    method_name,
                    float(timing[0] * 1000),  # Convert to ms
                    float(timing[1] * 1000),
                    float(jnp.mean(freq_samples)),
                    float(jnp.std(freq_samples)),
                    float(jnp.mean(off_samples)),
                    float(jnp.std(off_samples)),
                    stats.get("acceptance_rate", ""),
                ]
            )

    print(f"Saved benchmark summary: {summary_path}")


def load_experiment_metadata(data_dir: str) -> Dict[str, Any]:
    """Load experiment metadata."""
    metadata_path = os.path.join(data_dir, "experiment_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_dataset(data_dir: str) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """Load dataset from saved experiment."""
    dataset_path = os.path.join(data_dir, "dataset.csv")

    xs, ys = [], []
    true_freq, true_offset = None, None

    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == 0:  # First data row has true parameters
                true_freq = float(row["true_freq"])
                true_offset = float(row["true_offset"])
            else:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))

    return jnp.array(xs), jnp.array(ys), true_freq, true_offset


def load_inference_results(data_dir: str, method_name: str) -> Dict[str, Any]:
    """Load inference results for a specific method."""
    method_dir = os.path.join(data_dir, method_name)

    # Load samples
    samples_path = os.path.join(method_dir, "samples.csv")
    freq_samples, off_samples, weights = [], [], []

    with open(samples_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_samples.append(float(row["freq"]))
            off_samples.append(float(row["offset"]))
            weights.append(float(row["weight"]))

    samples = {
        "freq": jnp.array(freq_samples),
        "off": jnp.array(off_samples),
        "curve": {"freq": jnp.array(freq_samples), "off": jnp.array(off_samples)},
    }

    # Load timing if available
    timing_stats = None
    timing_path = os.path.join(method_dir, "timing.csv")
    if os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert back to seconds
                timing_stats = (
                    float(row["mean_time_ms"]) / 1000,
                    float(row["std_time_ms"]) / 1000,
                )
                break

    # Load additional statistics if available
    additional_stats = {}
    stats_path = os.path.join(method_dir, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            additional_stats = json.load(f)

    return {
        "samples": samples,
        "weights": jnp.array(weights),
        "timing_stats": timing_stats,
        "additional_stats": additional_stats,
    }


def load_benchmark_results(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all benchmark results from an experiment directory."""
    metadata = load_experiment_metadata(data_dir)
    xs, ys, true_freq, true_offset = load_dataset(data_dir)

    results = {
        "metadata": metadata,
        "dataset": {
            "xs": xs,
            "ys": ys,
            "true_freq": true_freq,
            "true_offset": true_offset,
        },
        "methods": {},
    }

    # Load results for each method
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in [".", ".."]:
            # Check if it's a method directory (has samples.csv)
            if os.path.exists(os.path.join(item_path, "samples.csv")):
                results["methods"][item] = load_inference_results(data_dir, item)

    return results


def get_latest_experiment(base_dir: str = "data") -> Optional[str]:
    """Get the most recent experiment directory."""
    experiments = list_experiments(base_dir)
    if experiments:
        # Assuming experiments are named with timestamps, latest is last when sorted
        return experiments[-1]
    return None
