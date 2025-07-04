"""Combine results from different framework benchmarks and run comparisons.

This module provides functionality to combine results from multiple benchmark
runs and orchestrate comparison benchmarks across frameworks.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from ..export.results import load_benchmark_results
from ..visualization.plots import create_all_figures
from ..data.generation import PolynomialDataset
from ..benchmarks.genjax import (
    genjax_polynomial_is_timing,
    genjax_polynomial_hmc_timing,
)

# Optional imports for other frameworks
try:
    from ..benchmarks.pyro import (
        pyro_polynomial_is_timing,
        pyro_polynomial_hmc_timing,
    )
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False

# Import Gen.jl interface if available
try:
    from ..julia_interface import GenJLBenchmark
    HAS_GEN_JL = True
except ImportError:
    HAS_GEN_JL = False


def combine_benchmark_results(result_dirs: List[str]) -> Dict[str, Any]:
    """Combine results from multiple benchmark runs.
    
    Args:
        result_dirs: List of experiment directories to combine
        
    Returns:
        Combined results dictionary
    """
    combined = {
        "is_summary": [],
        "hmc_summary": [],
        "metadata": {}
    }
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            results = load_benchmark_results(dir_path)
            
            # Combine IS summaries
            if "is_summary" in results:
                combined["is_summary"].append(results["is_summary"])
            
            # Combine HMC summaries
            if "hmc_summary" in results:
                combined["hmc_summary"].append(results["hmc_summary"])
            
            # Store metadata by directory
            combined["metadata"][dir_path] = results.get("metadata", {})
    
    # Concatenate dataframes
    if combined["is_summary"]:
        combined["is_summary"] = pd.concat(combined["is_summary"], ignore_index=True)
    
    if combined["hmc_summary"]:
        combined["hmc_summary"] = pd.concat(combined["hmc_summary"], ignore_index=True)
    
    return combined


### Gen.jl Timing Functions (moved from core.py) ###

def gen_polynomial_is_timing(
    dataset: PolynomialDataset,
    n_particles: int,
    repeats: int = 100,
    setup_julia: bool = False
) -> Dict[str, Any]:
    """Time Gen.jl importance sampling on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_particles: Number of importance sampling particles
        repeats: Number of timing repetitions
        setup_julia: Whether to run Julia setup first
        
    Returns:
        Dictionary with timing results
    """
    if not HAS_GEN_JL:
        return {
            "framework": "gen.jl",
            "method": "is",
            "n_particles": n_particles,
            "n_points": dataset.n_points,
            "times": [],
            "mean_time": float('nan'),
            "std_time": float('nan'),
            "error": "Gen.jl interface not available"
        }
    
    # Create Gen.jl interface
    gen_jl = GenJLBenchmark()
    
    if setup_julia:
        gen_jl.setup_julia_environment()
    
    # Convert to numpy for Julia interface
    import jax.numpy as jnp
    xs_np = jnp.array(dataset.xs)
    ys_np = jnp.array(dataset.ys)
    
    # Run benchmark
    result = gen_jl.run_polynomial_is(xs_np, ys_np, n_particles, repeats=repeats)
    
    # Add dataset info
    result["n_points"] = dataset.n_points
    
    return result


def gen_polynomial_hmc_timing(
    dataset: PolynomialDataset,
    n_samples: int,
    n_warmup: int = 500,
    repeats: int = 100,
    setup_julia: bool = False
) -> Dict[str, Any]:
    """Time Gen.jl HMC on polynomial regression.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        setup_julia: Whether to run Julia setup first
        
    Returns:
        Dictionary with timing results
    """
    if not HAS_GEN_JL:
        return {
            "framework": "gen.jl",
            "method": "hmc",
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "n_points": dataset.n_points,
            "times": [],
            "mean_time": float('nan'),
            "std_time": float('nan'),
            "error": "Gen.jl interface not available"
        }
    
    # Create Gen.jl interface
    gen_jl = GenJLBenchmark()
    
    if setup_julia:
        gen_jl.setup_julia_environment()
    
    # Convert to numpy for Julia interface
    import jax.numpy as jnp
    xs_np = jnp.array(dataset.xs)
    ys_np = jnp.array(dataset.ys)
    
    # Run benchmark
    result = gen_jl.run_polynomial_hmc(
        xs_np, ys_np, n_samples, n_warmup=n_warmup, repeats=repeats
    )
    
    # Add dataset info
    result["n_points"] = dataset.n_points
    
    return result


### Benchmark Suite Functions ###

def run_polynomial_is_comparison(
    dataset: PolynomialDataset,
    n_particles_list: List[int] = [100, 1000, 10000],
    repeats: int = 100,
    frameworks: List[str] = ["genjax", "gen.jl"],
) -> Dict[str, Dict[str, Any]]:
    """Run importance sampling comparison across frameworks.
    
    Args:
        dataset: Polynomial dataset
        n_particles_list: List of particle counts to test
        repeats: Number of timing repetitions
        frameworks: List of frameworks to benchmark
        
    Returns:
        Nested dictionary with results for each framework and particle count
    """
    results = {}
    
    for n_particles in n_particles_list:
        key = f"n{n_particles}"
        results[key] = {}
        
        if "genjax" in frameworks:
            print(f"Running GenJAX IS with {n_particles} particles...")
            results[key]["genjax"] = genjax_polynomial_is_timing(
                dataset, n_particles, repeats=repeats
            )
        
        if "gen.jl" in frameworks:
            print(f"Running Gen.jl IS with {n_particles} particles...")
            results[key]["gen.jl"] = gen_polynomial_is_timing(
                dataset, n_particles, repeats=repeats
            )
        
        if "pyro" in frameworks and HAS_PYRO:
            print(f"Running Pyro IS with {n_particles} particles...")
            results[key]["pyro"] = pyro_polynomial_is_timing(
                dataset, n_particles, repeats=repeats
            )
    
    return results


def run_polynomial_hmc_comparison(
    dataset: PolynomialDataset,
    n_samples: int = 1000,
    n_warmup: int = 500,
    repeats: int = 100,
    frameworks: List[str] = ["genjax", "gen.jl"],
) -> Dict[str, Any]:
    """Run HMC comparison across frameworks.
    
    Args:
        dataset: Polynomial dataset
        n_samples: Number of HMC samples
        n_warmup: Number of warmup samples
        repeats: Number of timing repetitions
        frameworks: List of frameworks to benchmark
        
    Returns:
        Dictionary with results for each framework
    """
    results = {}
    
    if "genjax" in frameworks:
        print(f"Running GenJAX HMC with {n_samples} samples...")
        results["genjax"] = genjax_polynomial_hmc_timing(
            dataset, n_samples, n_warmup=n_warmup, repeats=repeats
        )
    
    if "gen.jl" in frameworks:
        print(f"Running Gen.jl HMC with {n_samples} samples...")
        results["gen.jl"] = gen_polynomial_hmc_timing(
            dataset, n_samples, n_warmup=n_warmup, repeats=repeats
        )
    
    if "pyro" in frameworks and HAS_PYRO:
        print(f"Running Pyro HMC with {n_samples} samples...")
        results["pyro"] = pyro_polynomial_hmc_timing(
            dataset, n_samples, n_warmup=n_warmup, repeats=repeats
        )
    
    return results