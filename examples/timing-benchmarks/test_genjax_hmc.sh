#!/bin/bash
cd /home/femtomc/genjax-popl-2026/genjax/examples/timing-benchmarks

echo "Testing GenJAX HMC..."
pixi run python -c "
import sys
sys.path.insert(0, 'src')
from timing_benchmarks.data.generation import generate_polynomial_data

# Load genjax benchmarks module
import importlib.util
spec = importlib.util.spec_from_file_location('genjax_bench', 'src/timing_benchmarks/curvefit-benchmarks/genjax.py')
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)

# Test
dataset = generate_polynomial_data(n_points=20, seed=42)
result = gb.genjax_polynomial_hmc_timing(dataset, n_samples=100, n_warmup=50, repeats=2)
print(f'GenJAX HMC: {result[\"mean_time\"]:.3f}s ± {result[\"std_time\"]:.3f}s')
"

echo -e "\nTesting NumPyro HMC..."
pixi run python -c "
import sys
sys.path.insert(0, 'src')
from timing_benchmarks.data.generation import generate_polynomial_data

# Load numpyro benchmarks module
import importlib.util
spec = importlib.util.spec_from_file_location('numpyro_bench', 'src/timing_benchmarks/curvefit-benchmarks/numpyro.py')
nb = importlib.util.module_from_spec(spec)

# Hack to make relative imports work
import types
timing_benchmarks = types.ModuleType('timing_benchmarks')
timing_benchmarks.data = types.ModuleType('data')
sys.modules['timing_benchmarks'] = timing_benchmarks
sys.modules['timing_benchmarks.data'] = timing_benchmarks.data
timing_benchmarks.data.generation = sys.modules['__main__']

spec.loader.exec_module(nb)

# Test
dataset = generate_polynomial_data(n_points=20, seed=42)
result = nb.numpyro_polynomial_hmc_timing(dataset, n_samples=100, n_warmup=50, repeats=2)
print(f'NumPyro HMC: {result[\"mean_time\"]:.3f}s ± {result[\"std_time\"]:.3f}s')
"