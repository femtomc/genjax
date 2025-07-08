#!/usr/bin/env python
"""Debug why Pyro is missing from IS plot."""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_framework_results(data_dir, framework):
    """Load all results for a given framework."""
    # Check if we should look in curvefit subdirectory
    if Path(data_dir).name != "curvefit":
        framework_dir = Path(data_dir) / "curvefit" / framework
    else:
        framework_dir = Path(data_dir) / framework
        
    if not framework_dir.exists():
        # Try old location as fallback
        framework_dir = Path(data_dir) / framework
        if not framework_dir.exists():
            print(f"Warning: No results found for {framework}")
            return None
    
    results = {}
    
    # Load IS results
    for n_particles in [100, 1000, 5000, 10000, 100000]:
        result_file = framework_dir / f"is_n{n_particles}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                results[f"is_n{n_particles}"] = json.load(f)
                print(f"Loaded IS n={n_particles} for {framework}")
    
    return results

# Load results
frameworks = ["genjax", "numpyro", "handcoded_tfp", "pyro", "genjl", "genjl_dynamic", "genjl_optimized", "handcoded_torch"]
all_results = {}
for framework in frameworks:
    results = load_framework_results("data", framework)
    if results:
        all_results[framework] = results

# Create dataframe
rows = []
for framework, framework_results in all_results.items():
    if not framework_results:
        continue
        
    for key, result in framework_results.items():
        if key.startswith("is_n"):
            n_particles = int(key.split("_n")[1])
            rows.append({
                'framework': framework,
                'method': 'IS',
                'n_particles': n_particles,
                'mean_time': result.get('mean_time', np.nan),
                'std_time': result.get('std_time', np.nan),
                'n_points': result.get('n_points', 50)
            })

df = pd.DataFrame(rows)

# Filter for IS results
is_df = df[df['method'] == 'IS']
is_df = is_df[is_df['n_particles'].isin([1000, 5000, 10000])]

print("\nIS DataFrame:")
print(is_df[['framework', 'n_particles', 'mean_time']].sort_values(['n_particles', 'framework']))

print("\nFrameworks in IS data:", sorted(is_df['framework'].unique()))