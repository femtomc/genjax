#!/usr/bin/env python
"""Test HMC benchmarks."""
import sys
sys.path.insert(0, 'src')

from timing_benchmarks.data.generation import generate_polynomial_data

# Generate test data
dataset = generate_polynomial_data(n_points=20, seed=42)
print(f"Generated dataset with {dataset.n_points} points")

# Import and test GenJAX HMC
sys.modules.pop('genjax', None)  # Clear any cached genjax module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "genjax_benchmarks", 
    "src/timing_benchmarks/curvefit-benchmarks/genjax.py"
)
genjax_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genjax_benchmarks)

print("\nTesting GenJAX HMC...")
try:
    result = genjax_benchmarks.genjax_polynomial_hmc_timing(
        dataset,
        n_samples=100,
        n_warmup=50,
        repeats=2,
        step_size=0.01,
        n_leapfrog=10,
    )
    print(f"✓ GenJAX HMC: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
except Exception as e:
    print(f"✗ GenJAX HMC failed: {e}")
    import traceback
    traceback.print_exc()

# Import and test NumPyro HMC
# First set up the parent package structure
import timing_benchmarks
import timing_benchmarks.curvefit_benchmarks
sys.modules['timing_benchmarks'] = timing_benchmarks
sys.modules['timing_benchmarks.curvefit_benchmarks'] = timing_benchmarks.curvefit_benchmarks

spec = importlib.util.spec_from_file_location(
    "timing_benchmarks.curvefit_benchmarks.numpyro",
    "src/timing_benchmarks/curvefit-benchmarks/numpyro.py"
)
numpyro_benchmarks = importlib.util.module_from_spec(spec)
sys.modules["timing_benchmarks.curvefit_benchmarks.numpyro"] = numpyro_benchmarks
spec.loader.exec_module(numpyro_benchmarks)

print("\nTesting NumPyro HMC...")
try:
    result = numpyro_benchmarks.numpyro_polynomial_hmc_timing(
        dataset,
        n_samples=100,
        n_warmup=50,
        repeats=2,
        step_size=0.01,
    )
    print(f"✓ NumPyro HMC: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
except Exception as e:
    print(f"✗ NumPyro HMC failed: {e}")
    import traceback
    traceback.print_exc()