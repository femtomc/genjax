#!/usr/bin/env python
"""Quick test of HMC implementations."""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from timing_benchmarks.data.generation import generate_polynomial_data
# Note the hyphen in curvefit-benchmarks
sys.path.insert(0, str(Path(__file__).parent / "src" / "timing_benchmarks" / "curvefit-benchmarks"))
import genjax as genjax_module
import numpyro as numpyro_module

def main():
    # Generate small test dataset
    print("Generating test dataset...")
    dataset = generate_polynomial_data(n_points=20, seed=42)
    
    # Test GenJAX HMC
    print("\nTesting GenJAX HMC...")
    try:
        result = genjax_module.genjax_polynomial_hmc_timing(
            dataset,
            n_samples=100,
            n_warmup=50,
            repeats=2,  # Just 2 repeats for quick test
            step_size=0.01,
            n_leapfrog=10,
        )
        print(f"✓ GenJAX HMC: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        print(f"  Samples shape: a={result['samples']['a'].shape}")
    except Exception as e:
        print(f"✗ GenJAX HMC failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test NumPyro HMC
    print("\nTesting NumPyro HMC...")
    try:
        result = numpyro_module.numpyro_polynomial_hmc_timing(
            dataset,
            n_samples=100,
            n_warmup=50,
            repeats=2,
            step_size=0.01,
        )
        print(f"✓ NumPyro HMC: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        print(f"  Samples shape: a={result['samples']['a'].shape}")
    except Exception as e:
        print(f"✗ NumPyro HMC failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()