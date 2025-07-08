#!/usr/bin/env python
"""Generate test HMC data for visualization."""
import json
from pathlib import Path
from datetime import datetime

# Create test HMC results
output_dir = Path("data")

# GenJAX HMC result
genjax_dir = output_dir / "genjax"
genjax_dir.mkdir(parents=True, exist_ok=True)
genjax_hmc = {
    "framework": "genjax",
    "method": "hmc",
    "n_samples": 1000,
    "n_warmup": 500,
    "n_points": 50,
    "times": [0.045, 0.042, 0.043, 0.044, 0.043],
    "mean_time": 0.0434,
    "std_time": 0.0012,
    "step_size": 0.01,
    "n_leapfrog": 20
}
with open(genjax_dir / "hmc_n1000.json", "w") as f:
    json.dump(genjax_hmc, f, indent=2)

# NumPyro HMC result
numpyro_dir = output_dir / "numpyro"
numpyro_dir.mkdir(parents=True, exist_ok=True)
numpyro_hmc = {
    "framework": "numpyro",
    "method": "hmc",
    "n_samples": 1000,
    "n_warmup": 500,
    "n_points": 50,
    "times": [0.038, 0.037, 0.039, 0.038, 0.037],
    "mean_time": 0.0378,
    "std_time": 0.0008,
    "step_size": 0.01,
    "target_accept_prob": 0.8
}
with open(numpyro_dir / "hmc_n1000.json", "w") as f:
    json.dump(numpyro_hmc, f, indent=2)

# Handcoded JAX HMC result (baseline)
handcoded_dir = output_dir / "handcoded_tfp"
handcoded_dir.mkdir(parents=True, exist_ok=True)
handcoded_hmc = {
    "framework": "handcoded_tfp",
    "method": "hmc",
    "n_samples": 1000,
    "n_warmup": 500,
    "n_points": 50,
    "times": [0.025, 0.024, 0.026, 0.025, 0.024],
    "mean_time": 0.0248,
    "std_time": 0.0008
}
with open(handcoded_dir / "hmc_n1000.json", "w") as f:
    json.dump(handcoded_hmc, f, indent=2)

print("Test HMC data generated in data/ directory")
print("Now run: pixi run python combine_results.py")