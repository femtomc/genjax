#!/usr/bin/env python
"""Run Gen.jl HMC benchmarks."""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from timing_benchmarks.data.generation import generate_polynomial_data

# Generate dataset
print("Generating polynomial dataset...")
dataset = generate_polynomial_data(n_points=50, seed=42)

# Save dataset to CSV for Julia
data_file = Path("julia/data/temp_dataset.csv")
data_file.parent.mkdir(exist_ok=True)
with open(data_file, "w") as f:
    f.write("x,y\n")
    for x, y in zip(dataset.xs, dataset.ys):
        f.write(f"{x},{y}\n")

# Chain lengths to benchmark
chain_lengths = [100, 500, 1000]

# Create output directory
output_dir = Path("data/genjl")
output_dir.mkdir(parents=True, exist_ok=True)

# Julia script to run HMC benchmarks
julia_script = """
using Pkg
Pkg.activate("julia")

include("julia/src/TimingBenchmarks.jl")
using .TimingBenchmarks
using CSV, DataFrames, JSON

# Load data
df = CSV.read("julia/data/temp_dataset.csv", DataFrame)
xs = Float64.(df.x)
ys = Float64.(df.y)

# Create polynomial data struct
data = PolynomialData(xs, ys, Dict("a" => 0.0, "b" => 0.0, "c" => 0.0), 0.05, length(xs))

# Run benchmarks for different chain lengths
chain_lengths = [100, 500, 1000]
for n_samples in chain_lengths
    println("Running HMC with $n_samples samples...")
    
    result = run_polynomial_hmc_benchmark(
        data, n_samples;
        n_warmup=500,
        repeats=10,
        step_size=0.01,
        n_leapfrog=20
    )
    
    # Save result
    output_file = "data/genjl/hmc_n$(n_samples).json"
    open(output_file, "w") do f
        JSON.print(f, result, 2)
    end
    
    println("✓ Gen.jl HMC (n=$n_samples): $(result["mean_time"])s ± $(result["std_time"])s")
    println("  Saved to: $output_file")
end
"""

# Write Julia script to temporary file
julia_script_file = Path("run_hmc_benchmark.jl")
with open(julia_script_file, "w") as f:
    f.write(julia_script)

print("\n" + "="*60)
print("Running HMC benchmarks for Gen.jl")
print("="*60)

try:
    # Run Julia script
    result = subprocess.run(
        ["julia", "--project=julia", str(julia_script_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running Julia: {result.stderr}")
    else:
        print(result.stdout)
        
except Exception as e:
    print(f"Failed to run Julia: {e}")

finally:
    # Cleanup
    if julia_script_file.exists():
        julia_script_file.unlink()
    if data_file.exists():
        data_file.unlink()

print("\n" + "="*60)
print("Gen.jl HMC benchmarking complete!")
print("="*60)