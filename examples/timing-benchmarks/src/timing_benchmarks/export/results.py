"""Export and import utilities for benchmark results.

This module handles saving benchmark results to disk and loading them back
for visualization and analysis. Uses a structured directory format with
CSV and JSON files for portability.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def create_experiment_directory(
    base_dir: str = None,
    experiment_name: Optional[str] = None
) -> Path:
    """Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments (defaults to timing-benchmarks/data)
        experiment_name: Optional experiment name (defaults to timestamp)
        
    Returns:
        Path to created experiment directory
    """
    if base_dir is None:
        # Default to timing-benchmarks/data directory
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        base_dir = os.path.join(module_dir, "data")
    
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"benchmark_{timestamp}"
    
    exp_path = base_path / experiment_name
    exp_path.mkdir(exist_ok=True)
    
    return exp_path


def save_metadata(
    exp_dir: Path,
    config: Dict[str, Any],
    description: str = ""
):
    """Save experiment metadata.
    
    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary
        description: Optional experiment description
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "config": config,
    }
    
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_benchmark_times(
    exp_dir: Path,
    framework: str,
    method: str,
    times: list
):
    """Save timing data as CSV.
    
    Args:
        exp_dir: Experiment directory
        framework: Framework name
        method: Method identifier
        times: List of execution times
    """
    framework_dir = exp_dir / framework
    framework_dir.mkdir(exist_ok=True)
    
    timing_file = framework_dir / f"{method}_times.csv"
    with open(timing_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "time_seconds"])
        for i, time in enumerate(times):
            writer.writerow([i, time])


def save_benchmark_samples(
    exp_dir: Path,
    framework: str,
    method: str,
    samples: Dict[str, np.ndarray]
):
    """Save parameter samples.
    
    Args:
        exp_dir: Experiment directory
        framework: Framework name
        method: Method identifier
        samples: Dictionary of parameter samples
    """
    framework_dir = exp_dir / framework
    framework_dir.mkdir(exist_ok=True)
    
    samples_dir = framework_dir / f"{method}_samples"
    samples_dir.mkdir(exist_ok=True)
    
    for param, values in samples.items():
        param_file = samples_dir / f"{param}.csv"
        # Convert to numpy array if it's a JAX array
        values_np = np.array(values)
        np.savetxt(param_file, values_np, delimiter=",")


def save_timing_results(
    exp_dir: Path,
    framework: str,
    method: str,
    results: Dict[str, Any]
):
    """Save timing results for a specific framework and method.
    
    Args:
        exp_dir: Experiment directory
        framework: Framework name (e.g., "genjax", "gen.jl")
        method: Method name (e.g., "is", "hmc")
        results: Results dictionary from timing functions
    """
    # Create framework directory
    framework_dir = exp_dir / framework
    framework_dir.mkdir(exist_ok=True)
    
    # Save summary statistics
    summary_file = framework_dir / f"{method}_summary.json"
    summary = {
        "framework": framework,
        "method": method,
        "mean_time": results.get("mean_time"),
        "std_time": results.get("std_time"),
        "n_points": results.get("n_points"),
        "n_particles": results.get("n_particles"),
        "n_samples": results.get("n_samples"),
        "n_warmup": results.get("n_warmup"),
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed timing data
    if "times" in results and len(results["times"]) > 0:
        save_benchmark_times(exp_dir, framework, method, results["times"])
    
    # Save samples if available
    if "samples" in results:
        save_benchmark_samples(exp_dir, framework, method, results["samples"])
    
    # Save log weights for IS
    if "log_weights" in results:
        weights_file = framework_dir / f"{method}_log_weights.csv"
        weights_np = np.array(results["log_weights"])
        np.savetxt(weights_file, weights_np, delimiter=",")


def save_is_comparison_summary(
    exp_dir: Path,
    comparison_results: Dict[str, Dict[str, Any]]
):
    """Save importance sampling comparison summary.
    
    Args:
        exp_dir: Experiment directory
        comparison_results: Nested results by config and framework
    """
    summary_data = []
    
    # Flatten nested results
    for config_key, framework_results in comparison_results.items():
        for framework, results in framework_results.items():
            row = {
                "config": config_key,
                "framework": framework,
                "mean_time": results.get("mean_time"),
                "std_time": results.get("std_time"),
                "n_points": results.get("n_points"),
                "n_particles": results.get("n_particles"),
            }
            summary_data.append(row)
    
    # Save as CSV
    summary_file = exp_dir / "is_comparison_summary.csv"
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)


def save_hmc_comparison_summary(
    exp_dir: Path,
    comparison_results: Dict[str, Any]
):
    """Save HMC comparison summary.
    
    Args:
        exp_dir: Experiment directory
        comparison_results: Results by framework
    """
    summary_data = []
    
    for framework, results in comparison_results.items():
        row = {
            "framework": framework,
            "mean_time": results.get("mean_time"),
            "std_time": results.get("std_time"),
            "n_points": results.get("n_points"),
            "n_samples": results.get("n_samples"),
            "n_warmup": results.get("n_warmup"),
        }
        summary_data.append(row)
    
    # Save as CSV
    summary_file = exp_dir / "hmc_comparison_summary.csv"
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)


def save_benchmark_results(
    results: Dict[str, Any],
    experiment_name: Optional[str] = None,
    base_dir: str = None,
    description: str = ""
) -> Path:
    """Save complete benchmark results.
    
    Args:
        results: Complete results dictionary
        experiment_name: Optional experiment name
        base_dir: Base directory for experiments
        description: Experiment description
        
    Returns:
        Path to experiment directory
    """
    # Create experiment directory
    exp_dir = create_experiment_directory(base_dir, experiment_name)
    
    # Save metadata
    config = results.get("config", {})
    save_metadata(exp_dir, config, description)
    
    # Save IS comparison results
    if "is_comparison" in results:
        for config_key, framework_results in results["is_comparison"].items():
            for framework, framework_data in framework_results.items():
                save_timing_results(
                    exp_dir,
                    framework,
                    f"is_{config_key}",
                    framework_data
                )
        
        save_is_comparison_summary(exp_dir, results["is_comparison"])
    
    # Save HMC comparison results
    if "hmc_comparison" in results:
        for framework, framework_data in results["hmc_comparison"].items():
            save_timing_results(exp_dir, framework, "hmc", framework_data)
        
        save_hmc_comparison_summary(exp_dir, results["hmc_comparison"])
    
    print(f"Results saved to: {exp_dir}")
    return exp_dir


def load_benchmark_results(exp_dir: str) -> Dict[str, Any]:
    """Load benchmark results from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Dictionary with loaded results
    """
    exp_path = Path(exp_dir)
    
    # Load metadata
    with open(exp_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    results = {
        "metadata": metadata,
        "exp_dir": str(exp_path),
    }
    
    # Load IS comparison summary
    is_summary_file = exp_path / "is_comparison_summary.csv"
    if is_summary_file.exists():
        results["is_summary"] = pd.read_csv(is_summary_file)
    
    # Load HMC comparison summary
    hmc_summary_file = exp_path / "hmc_comparison_summary.csv"
    if hmc_summary_file.exists():
        results["hmc_summary"] = pd.read_csv(hmc_summary_file)
    
    # Load detailed timing data
    timing_data = {}
    for framework_dir in exp_path.iterdir():
        if framework_dir.is_dir() and framework_dir.name in ["genjax", "gen.jl", "numpyro", "pyro"]:
            framework = framework_dir.name
            timing_data[framework] = {}
            
            # Load all CSV files in framework directory
            for csv_file in framework_dir.glob("*_times.csv"):
                method_config = csv_file.stem.replace("_times", "")
                df = pd.read_csv(csv_file)
                timing_data[framework][method_config] = df["time_seconds"].values
    
    results["timing_data"] = timing_data
    
    return results


def get_latest_experiment(base_dir: str = None) -> Optional[Path]:
    """Get the most recent experiment directory.
    
    Args:
        base_dir: Base directory for experiments (defaults to timing-benchmarks/data)
        
    Returns:
        Path to latest experiment or None
    """
    if base_dir is None:
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        base_dir = os.path.join(module_dir, "data")
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Find all benchmark directories
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("benchmark_")]
    
    if not exp_dirs:
        return None
    
    # Sort by modification time
    latest = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    return latest