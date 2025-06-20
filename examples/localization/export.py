"""
Data export utilities for localization experiments.

Handles saving experimental results to CSV files for later analysis and plotting.
"""

import os
import csv
import json
import jax.numpy as jnp
from typing import Dict, Any, List
from .core import Pose


def save_experiment_metadata(data_dir: str, config: Dict[str, Any]):
    """Save experiment configuration metadata."""
    metadata_path = os.path.join(data_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def save_ground_truth_data(
    data_dir: str, true_poses: List[Pose], observations: List, controls=None
):
    """Save ground truth trajectory and observations."""
    # Save poses
    poses_path = os.path.join(data_dir, "ground_truth_poses.csv")
    with open(poses_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "x", "y", "theta"])
        for t, pose in enumerate(true_poses):
            writer.writerow([t, float(pose.x), float(pose.y), float(pose.theta)])

    # Save observations
    obs_path = os.path.join(data_dir, "ground_truth_observations.csv")
    with open(obs_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Handle vectorized observations (multiple LIDAR rays)
        if hasattr(observations[0], "__len__") and len(observations[0]) > 1:
            n_rays = len(observations[0])
            writer.writerow(["timestep"] + [f"ray_{i}" for i in range(n_rays)])
            for t, obs in enumerate(observations):
                writer.writerow([t] + [float(x) for x in obs])
        else:
            writer.writerow(["timestep", "observation"])
            for t, obs in enumerate(observations):
                writer.writerow([t, float(obs)])

    print(f"Saved ground truth: {poses_path}, {obs_path}")


def save_smc_results(data_dir: str, method_name: str, result: Dict[str, Any]):
    """Save SMC method results including particles, weights, and diagnostics."""
    method_dir = os.path.join(data_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    particle_history = result["particle_history"]
    weight_history = result["weight_history"]
    diagnostic_weights = result["diagnostic_weights"]
    timing_stats = result["timing_stats"]

    # Save timing data
    timing_path = os.path.join(method_dir, "timing.csv")
    with open(timing_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_time_sec", "std_time_sec"])
        writer.writerow([float(timing_stats[0]), float(timing_stats[1])])

    # Save particle trajectories (one file per timestep)
    particles_dir = os.path.join(method_dir, "particles")
    os.makedirs(particles_dir, exist_ok=True)

    for t, (particles, weights) in enumerate(zip(particle_history, weight_history)):
        particles_path = os.path.join(particles_dir, f"timestep_{t:03d}.csv")
        with open(particles_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["particle_id", "x", "y", "theta", "weight"])
            for i, (particle, weight) in enumerate(zip(particles, weights)):
                writer.writerow(
                    [
                        i,
                        float(particle.x),
                        float(particle.y),
                        float(particle.theta),
                        float(weight),
                    ]
                )

    # Save diagnostic weights (all timesteps in one file)
    if diagnostic_weights is not None and hasattr(diagnostic_weights, "shape"):
        diag_path = os.path.join(method_dir, "diagnostic_weights.csv")
        with open(diag_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header: timestep, particle_0, particle_1, ..., particle_N
            T, N = diagnostic_weights.shape
            writer.writerow(["timestep"] + [f"particle_{i}" for i in range(N)])
            for t in range(T):
                row = [t] + [float(diagnostic_weights[t, i]) for i in range(N)]
                writer.writerow(row)

    print(f"Saved {method_name} results: {method_dir}")


def save_benchmark_results(
    data_dir: str, benchmark_results: Dict[str, Any], config: Dict[str, Any]
):
    """Save complete benchmark results for all SMC methods."""
    # Save experiment metadata
    save_experiment_metadata(data_dir, config)

    # Save results for each method
    for method_name, result in benchmark_results.items():
        save_smc_results(data_dir, method_name, result)

    # Save summary comparison
    summary_path = os.path.join(data_dir, "benchmark_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "mean_time_sec", "std_time_sec", "n_particles", "n_timesteps"]
        )
        for method_name, result in benchmark_results.items():
            timing_stats = result["timing_stats"]
            n_particles = len(result["particle_history"][0])
            n_timesteps = len(result["particle_history"])
            writer.writerow(
                [
                    method_name,
                    float(timing_stats[0]),
                    float(timing_stats[1]),
                    n_particles,
                    n_timesteps,
                ]
            )

    print(f"Saved benchmark summary: {summary_path}")


def load_experiment_metadata(data_dir: str) -> Dict[str, Any]:
    """Load experiment configuration metadata."""
    metadata_path = os.path.join(data_dir, "experiment_metadata.json")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_ground_truth_data(data_dir: str):
    """Load ground truth trajectory and observations."""
    # Load poses
    poses_path = os.path.join(data_dir, "ground_truth_poses.csv")
    poses = []
    with open(poses_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pose = Pose(x=float(row["x"]), y=float(row["y"]), theta=float(row["theta"]))
            poses.append(pose)

    # Load observations
    obs_path = os.path.join(data_dir, "ground_truth_observations.csv")
    observations = []
    with open(obs_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ray_0" in row:  # Multi-ray observations
                obs = [
                    float(row[f"ray_{i}"]) for i in range(len(row) - 1)
                ]  # -1 for timestep column
            else:  # Single observation
                obs = float(row["observation"])
            observations.append(obs)

    return poses, observations


def load_smc_results(data_dir: str, method_name: str) -> Dict[str, Any]:
    """Load SMC method results from saved files."""
    method_dir = os.path.join(data_dir, method_name)

    # Load timing data
    timing_path = os.path.join(method_dir, "timing.csv")
    with open(timing_path, "r") as f:
        reader = csv.DictReader(f)
        timing_row = next(reader)
        timing_stats = (
            float(timing_row["mean_time_sec"]),
            float(timing_row["std_time_sec"]),
        )

    # Load particle data
    particles_dir = os.path.join(method_dir, "particles")
    particle_files = sorted(
        [f for f in os.listdir(particles_dir) if f.startswith("timestep_")]
    )

    particle_history = []
    weight_history = []

    for filename in particle_files:
        filepath = os.path.join(particles_dir, filename)
        particles = []
        weights = []

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                particle = Pose(
                    x=float(row["x"]), y=float(row["y"]), theta=float(row["theta"])
                )
                particles.append(particle)
                weights.append(float(row["weight"]))

        particle_history.append(particles)
        weight_history.append(jnp.array(weights))

    # Load diagnostic weights
    diag_path = os.path.join(method_dir, "diagnostic_weights.csv")
    diagnostic_weights = None
    if os.path.exists(diag_path):
        with open(diag_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                T = len(rows)
                N = len(rows[0]) - 1  # -1 for timestep column
                diagnostic_weights = jnp.zeros((T, N))
                for row in rows:
                    t = int(row["timestep"])
                    for i in range(N):
                        diagnostic_weights = diagnostic_weights.at[t, i].set(
                            float(row[f"particle_{i}"])
                        )

    return {
        "timing_stats": timing_stats,
        "particle_history": particle_history,
        "weight_history": weight_history,
        "diagnostic_weights": diagnostic_weights,
    }


def load_benchmark_results(data_dir: str) -> Dict[str, Any]:
    """Load complete benchmark results for all SMC methods."""
    # Load summary to get method names
    summary_path = os.path.join(data_dir, "benchmark_summary.csv")
    method_names = []
    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_names.append(row["method"])

    # Load results for each method
    results = {}
    for method_name in method_names:
        results[method_name] = load_smc_results(data_dir, method_name)

    return results
