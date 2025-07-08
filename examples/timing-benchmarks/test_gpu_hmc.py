#!/usr/bin/env python
"""Test GPU HMC for all frameworks."""

import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # Force JAX to use GPU

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test JAX GPU
print("="*60)
print("Testing JAX GPU availability:")
import jax
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("="*60)

# Test PyTorch GPU
print("\nTesting PyTorch GPU availability:")
try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not available")
print("="*60)

# Quick test for each framework
from timing_benchmarks.data.generation import generate_polynomial_data

# Generate small test dataset
dataset = generate_polynomial_data(n_points=10)

print("\nRunning quick HMC tests on GPU:")
print("-"*60)

# Test GenJAX HMC
try:
    from timing_benchmarks.curvefit_benchmarks.genjax import genjax_polynomial_hmc_timing
    print("Testing GenJAX HMC...")
    result = genjax_polynomial_hmc_timing(dataset, n_samples=10, n_warmup=10, repeats=2)
    print(f"✓ GenJAX HMC: {result['mean_time']:.3f}s")
except Exception as e:
    print(f"✗ GenJAX HMC failed: {e}")

# Test NumPyro HMC
try:
    from timing_benchmarks.curvefit_benchmarks.numpyro import numpyro_polynomial_hmc_timing
    print("\nTesting NumPyro HMC...")
    result = numpyro_polynomial_hmc_timing(dataset, n_samples=10, n_warmup=10, repeats=2)
    print(f"✓ NumPyro HMC: {result['mean_time']:.3f}s")
except Exception as e:
    print(f"✗ NumPyro HMC failed: {e}")

# Test Handcoded JAX HMC
try:
    from timing_benchmarks.curvefit_benchmarks.handcoded_tfp import handcoded_tfp_polynomial_hmc_timing
    print("\nTesting Handcoded JAX HMC...")
    result = handcoded_tfp_polynomial_hmc_timing(dataset, n_samples=10, n_warmup=10, repeats=2)
    print(f"✓ Handcoded JAX HMC: {result['mean_time']:.3f}s")
except Exception as e:
    print(f"✗ Handcoded JAX HMC failed: {e}")

# Test Handcoded PyTorch HMC
try:
    from timing_benchmarks.curvefit_benchmarks.handcoded_torch import handcoded_torch_polynomial_hmc_timing
    print("\nTesting Handcoded PyTorch HMC...")
    result = handcoded_torch_polynomial_hmc_timing(dataset, n_samples=10, n_warmup=10, repeats=2, device="cuda")
    print(f"✓ Handcoded PyTorch HMC: {result['mean_time']:.3f}s")
except Exception as e:
    print(f"✗ Handcoded PyTorch HMC failed: {e}")

# Test Pyro HMC
try:
    from timing_benchmarks.curvefit_benchmarks.pyro import pyro_polynomial_hmc_timing
    print("\nTesting Pyro HMC...")
    result = pyro_polynomial_hmc_timing(dataset, n_samples=10, n_warmup=10, repeats=2, device="cuda")
    print(f"✓ Pyro HMC: {result['mean_time']:.3f}s")
except Exception as e:
    print(f"✗ Pyro HMC failed: {e}")

print("\n" + "="*60)
print("GPU HMC test complete!")