#!/usr/bin/env python
"""Analyze GPU benchmark results."""
import pandas as pd

# Load results
df = pd.read_csv('data/benchmark_20250706_122408/is_comparison_summary.csv')

print('GPU Benchmark Results - Importance Sampling')
print('=' * 80)
print('GenJAX vs NumPyro vs Handcoded JAX on NVIDIA RTX 4090')
print('Dataset: 50 data points, polynomial regression')
print('=' * 80)

for n_particles in [100, 1000, 10000, 100000]:
    print(f'\nN = {n_particles:,} particles:')
    subset = df[df['n_particles'] == n_particles]
    
    # Get NumPyro baseline
    numpyro_time = subset[subset['framework'] == 'numpyro']['mean_time'].values[0]
    
    # Show results for each framework
    for _, row in subset.iterrows():
        speedup_vs_numpyro = numpyro_time / row['mean_time']
        print(f'  {row["framework"]:15} {row["mean_time"]*1000:8.3f} ms    '
              f'(speedup vs NumPyro: {speedup_vs_numpyro:6.1f}x)')

# Calculate overall speedups
print('\n' + '=' * 80)
print('Summary - Average Speedups vs NumPyro:')
print('=' * 80)

for framework in ['genjax', 'handcoded_jax']:
    speedups = []
    for n_particles in [100, 1000, 10000, 100000]:
        subset = df[df['n_particles'] == n_particles]
        numpyro_time = subset[subset['framework'] == 'numpyro']['mean_time'].values[0]
        framework_time = subset[subset['framework'] == framework]['mean_time'].values[0]
        speedups.append(numpyro_time / framework_time)
    
    avg_speedup = sum(speedups) / len(speedups)
    print(f'{framework:15} Average speedup: {avg_speedup:6.1f}x')