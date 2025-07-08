#!/usr/bin/env python
"""Test Gen.jl HMC implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'timing_benchmarks', 'curvefit-benchmarks'))
from genjl import genjl_polynomial_hmc_timing, GenJLBenchmark

# Check if Julia is available
gen_jl = GenJLBenchmark()
if not gen_jl.julia_available:
    print("Julia not available. Please install Julia to run Gen.jl benchmarks.")
    sys.exit(1)

# Generate test dataset
print("Generating test dataset...")
dataset = generate_polynomial_data(n_points=20, seed=42)

# Test HMC with small parameters
print("Running Gen.jl HMC test...")
result = genjl_polynomial_hmc_timing(
    dataset=dataset,
    n_samples=100,
    n_warmup=50,
    repeats=2,
    step_size=0.01,
    n_leapfrog=20
)

print(f"\nTest Results:")
print(f"Framework: {result.get('framework', 'N/A')}")
print(f"Method: {result.get('method', 'N/A')}")
print(f"Mean time: {result.get('mean_time', 'N/A'):.3f}s")
print(f"Std time: {result.get('std_time', 'N/A'):.3f}s")
print(f"N samples: {result.get('n_samples', 'N/A')}")
print(f"N warmup: {result.get('n_warmup', 'N/A')}")

if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print("Test completed successfully!")