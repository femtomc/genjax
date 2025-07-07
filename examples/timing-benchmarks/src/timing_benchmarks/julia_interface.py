"""Julia interface for Gen.jl benchmarks.

This module provides a Python interface to run Gen.jl benchmarks
through Julia's PyJulia package.
"""

import numpy as np
from typing import Dict, Any, Optional
import subprocess
import os
import json


class GenJLBenchmark:
    """Interface to Gen.jl benchmarks through Julia."""
    
    def __init__(self):
        self.julia_dir = self._find_julia_dir()
        self.julia_available = self._check_julia()
    
    def _find_julia_dir(self) -> str:
        """Find the Julia directory."""
        # Get the module directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to timing-benchmarks directory
        timing_benchmarks_dir = os.path.dirname(os.path.dirname(module_dir))
        julia_dir = os.path.join(timing_benchmarks_dir, "julia")
        return julia_dir
    
    def _check_julia(self) -> bool:
        """Check if Julia is available."""
        try:
            result = subprocess.run(
                ["julia", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def setup_julia_environment(self):
        """Set up Julia environment and packages."""
        if not self.julia_available:
            print("Julia not found. Please install Julia.")
            return
        
        print("Setting up Julia environment...")
        
        # Activate and instantiate the Julia project
        julia_cmd = f"""
        using Pkg
        Pkg.activate("{self.julia_dir}")
        Pkg.instantiate()
        """
        
        subprocess.run(
            ["julia", "-e", julia_cmd],
            cwd=self.julia_dir
        )
    
    def run_polynomial_is(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_particles: int,
        repeats: int = 100
    ) -> Dict[str, Any]:
        """Run polynomial regression importance sampling in Gen.jl.
        
        Args:
            xs: X values
            ys: Y values
            n_particles: Number of particles
            repeats: Number of timing repetitions
            
        Returns:
            Timing results dictionary
        """
        if not self.julia_available:
            return {
                "framework": "gen.jl",
                "method": "is",
                "n_particles": n_particles,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": "Julia not available"
            }
        
        # Save data to temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
            # Write header
            f.write("x,y\n")
            # Write data
            for x, y in zip(xs, ys):
                f.write(f"{x},{y}\n")
            data_file = f.name
        
        # Julia script to run
        julia_script = f"""
        using Pkg
        Pkg.activate("{self.julia_dir}")
        
        include("{self.julia_dir}/src/TimingBenchmarks.jl")
        using .TimingBenchmarks
        using CSV
        using DataFrames
        
        # Load data
        df = CSV.read("{data_file}", DataFrame)
        xs = Float64.(df.x)
        ys = Float64.(df.y)
        
        # Create polynomial data structure
        poly_data = PolynomialData(xs, ys, Dict("a" => 1.0, "b" => -2.0, "c" => 3.0), 0.05, length(xs))
        
        # Run benchmark
        result = run_polynomial_is_benchmark(poly_data, {n_particles}; repeats={repeats})
        
        # Save results
        using JSON
        open("{data_file}.json", "w") do f
            JSON.print(f, result)
        end
        """
        
        # Run Julia script
        result = subprocess.run(
            ["julia", "-e", julia_script],
            cwd=self.julia_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Julia error: {result.stderr}")
            return {
                "framework": "gen.jl",
                "method": "is",
                "n_particles": n_particles,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": f"Julia execution failed: {result.stderr}"
            }
        
        # Load results
        try:
            with open(f"{data_file}.json", "r") as f:
                julia_results = json.load(f)
            
            # Clean up temporary files
            os.unlink(data_file)
            os.unlink(f"{data_file}.json")
            
            return julia_results
            
        except Exception as e:
            return {
                "framework": "gen.jl",
                "method": "is",
                "n_particles": n_particles,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": f"Failed to load results: {str(e)}"
            }
    
    def run_polynomial_hmc(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_samples: int,
        n_warmup: int = 500,
        repeats: int = 100
    ) -> Dict[str, Any]:
        """Run polynomial regression HMC in Gen.jl.
        
        Args:
            xs: X values
            ys: Y values
            n_samples: Number of samples
            n_warmup: Number of warmup samples
            repeats: Number of timing repetitions
            
        Returns:
            Timing results dictionary
        """
        if not self.julia_available:
            return {
                "framework": "gen.jl",
                "method": "hmc",
                "n_samples": n_samples,
                "n_warmup": n_warmup,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": "Julia not available"
            }
        
        # Save data to temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
            # Write header
            f.write("x,y\n")
            # Write data
            for x, y in zip(xs, ys):
                f.write(f"{x},{y}\n")
            data_file = f.name
        
        # Julia script to run
        julia_script = f"""
        using Pkg
        Pkg.activate("{self.julia_dir}")
        
        include("{self.julia_dir}/src/TimingBenchmarks.jl")
        using .TimingBenchmarks
        using CSV
        using DataFrames
        
        # Load data
        df = CSV.read("{data_file}", DataFrame)
        xs = Float64.(df.x)
        ys = Float64.(df.y)
        
        # Create polynomial data structure
        poly_data = PolynomialData(xs, ys, Dict("a" => 1.0, "b" => -2.0, "c" => 3.0), 0.05, length(xs))
        
        # Run benchmark
        result = run_polynomial_hmc_benchmark(poly_data, {n_samples}; n_warmup={n_warmup}, repeats={repeats})
        
        # Save results
        using JSON
        open("{data_file}.json", "w") do f
            JSON.print(f, result)
        end
        """
        
        # Run Julia script
        result = subprocess.run(
            ["julia", "-e", julia_script],
            cwd=self.julia_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Julia error: {result.stderr}")
            return {
                "framework": "gen.jl",
                "method": "hmc",
                "n_samples": n_samples,
                "n_warmup": n_warmup,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": f"Julia execution failed: {result.stderr}"
            }
        
        # Load results
        try:
            with open(f"{data_file}.json", "r") as f:
                julia_results = json.load(f)
            
            # Clean up temporary files
            os.unlink(data_file)
            os.unlink(f"{data_file}.json")
            
            return julia_results
            
        except Exception as e:
            return {
                "framework": "gen.jl",
                "method": "hmc",
                "n_samples": n_samples,
                "n_warmup": n_warmup,
                "times": [],
                "mean_time": np.nan,
                "std_time": np.nan,
                "error": f"Failed to load results: {str(e)}"
            }