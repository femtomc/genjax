#!/usr/bin/env python
"""Simple test of HMC implementations."""

import os
os.chdir('/home/femtomc/genjax-popl-2026/genjax/examples/timing-benchmarks')

# Now run the benchmarks directly
import subprocess
import sys

# Test GenJAX HMC
print("Testing GenJAX HMC...")
result = subprocess.run([
    sys.executable, "-m", 
    "timing_benchmarks.curvefit-benchmarks.genjax",
    "--method", "hmc",
    "--n-samples", "100",
    "--n-warmup", "50",
    "--repeats", "2",
    "--n-points", "20"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

# Test NumPyro HMC
print("\n\nTesting NumPyro HMC...")
result = subprocess.run([
    sys.executable, "-m",
    "timing_benchmarks.curvefit-benchmarks.numpyro", 
    "--method", "hmc",
    "--n-samples", "100",
    "--n-warmup", "50", 
    "--repeats", "2",
    "--n-points", "20"
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)