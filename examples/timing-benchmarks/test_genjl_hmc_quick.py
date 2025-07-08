#!/usr/bin/env python
"""Quick test of Gen.jl HMC functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data

# Test Gen.jl
print("Testing Gen.jl HMC setup...")
print("-" * 60)

# Small dataset for quick test
dataset = generate_polynomial_data(n_points=10)

try:
    # Import and test
    import sys
    sys.path.append('src/timing_benchmarks/curvefit-benchmarks')
    from genjl import genjl_polynomial_hmc_timing
    
    print("Running Gen.jl HMC test (10 samples, 2 repeats)...")
    result = genjl_polynomial_hmc_timing(
        dataset, 
        n_samples=10, 
        n_warmup=10, 
        repeats=2,
        step_size=0.01,
        n_leapfrog=20
    )
    
    print(f"✓ Gen.jl HMC successful!")
    print(f"  Mean time: {result['mean_time']:.3f}s")
    print(f"  Framework: {result['framework']}")
    print(f"  Method: {result['method']}")
    print(f"  Samples shape: a={len(result['samples']['a'])}")
    
except Exception as e:
    print(f"✗ Gen.jl HMC failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")