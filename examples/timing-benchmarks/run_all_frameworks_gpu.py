#!/usr/bin/env python
"""Run comprehensive GPU benchmarks across all frameworks."""
import subprocess
import sys
import os

# Add src to Python path
sys.path.insert(0, 'src')

from timing_benchmarks.data import generate_polynomial_data
from timing_benchmarks.analysis import run_polynomial_is_comparison
from timing_benchmarks.export import save_benchmark_results
from timing_benchmarks.julia_interface import GenJLBenchmark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Julia environment first
print("Setting up Julia environment...")
gen_jl = GenJLBenchmark()
if gen_jl.julia_available:
    gen_jl.setup_julia_environment()
    print("Julia environment ready.")
else:
    print("WARNING: Julia not available, Gen.jl benchmarks will be skipped.")

# Generate dataset
print("\nGenerating dataset...")
dataset = generate_polynomial_data(n_points=50, seed=42)

# Frameworks to benchmark
frameworks = ["genjax", "numpyro", "handcoded_jax"]
if gen_jl.julia_available:
    frameworks.append("gen.jl")

# Run benchmarks for each framework separately
n_particles_list = [100, 1000, 10000, 100000]
repeats = 20  # Reduced for speed

print(f"\nRunning benchmarks with frameworks: {frameworks}")
print(f"Particle counts: {n_particles_list}")
print(f"Repeats: {repeats}")

# Collect all results
all_results = {}

# Run GenJAX, NumPyro, and handcoded JAX on GPU
print("\n" + "="*60)
print("Running GPU benchmarks (GenJAX, NumPyro, Handcoded JAX)...")
print("="*60)
gpu_frameworks = ["genjax", "numpyro", "handcoded_jax"]
gpu_results = run_polynomial_is_comparison(
    dataset,
    n_particles_list=n_particles_list,
    repeats=repeats,
    frameworks=gpu_frameworks
)
all_results.update(gpu_results)

# Run Gen.jl benchmarks (CPU)
if "gen.jl" in frameworks:
    print("\n" + "="*60)
    print("Running Gen.jl benchmarks...")
    print("="*60)
    genjl_results = run_polynomial_is_comparison(
        dataset,
        n_particles_list=n_particles_list,
        repeats=repeats,
        frameworks=["gen.jl"]
    )
    all_results.update(genjl_results)

# Check if we need to run Pyro in separate environment
try:
    import pyro
    pyro_available = True
    print("\n" + "="*60)
    print("Running Pyro benchmarks...")
    print("="*60)
    pyro_results = run_polynomial_is_comparison(
        dataset,
        n_particles_list=n_particles_list,
        repeats=repeats,
        frameworks=["pyro"]
    )
    all_results.update(pyro_results)
except ImportError:
    pyro_available = False
    print("\nPyro not available in current environment.")
    print("To run Pyro benchmarks, use: pixi run -e pyro python run_all_frameworks_gpu.py")

# Save raw results
config = {
    "benchmark": "polynomial_is_all_frameworks",
    "data_size": dataset.n_points,
    "n_particles_list": n_particles_list,
    "repeats": repeats,
    "frameworks": frameworks,
    "gpu_enabled": True
}

save_results = {
    "config": config,
    "is_comparison": all_results
}

exp_dir = save_benchmark_results(
    save_results,
    description="All frameworks IS benchmark comparison"
)

# Create summary dataframe
summary_data = []
for particle_key, particle_results in all_results.items():
    n_particles = int(particle_key[1:])  # Remove 'n' prefix
    for framework, result in particle_results.items():
        if result and 'mean_time' in result and not np.isnan(result['mean_time']):
            summary_data.append({
                'framework': framework,
                'n_particles': n_particles,
                'mean_time': result['mean_time'],
                'std_time': result['std_time'],
                'n_points': result['n_points']
            })

df = pd.DataFrame(summary_data)

# Print summary table
print("\n" + "="*80)
print("SUMMARY - All Frameworks GPU Benchmark Results")
print("="*80)
print(f"Dataset: {dataset.n_points} points, polynomial regression")
print(f"Hardware: NVIDIA RTX 4090 (GPU) + CPU for Gen.jl")
print("="*80)

# Create a nice table
for n_particles in n_particles_list:
    print(f"\nN = {n_particles:,} particles:")
    subset = df[df['n_particles'] == n_particles].sort_values('mean_time')
    
    # Get fastest time for speedup calculation
    fastest_time = subset['mean_time'].min()
    
    print(f"{'Framework':<20} {'Time (ms)':<15} {'Speedup':<15} {'Relative to'}")
    print("-" * 65)
    
    for _, row in subset.iterrows():
        time_ms = row['mean_time'] * 1000
        speedup = row['mean_time'] / fastest_time
        print(f"{row['framework']:<20} {time_ms:<15.3f} {speedup:<15.1f}x  (vs fastest)")

# Save summary table to CSV
summary_path = os.path.join(exp_dir, "all_frameworks_summary.csv")
df.to_csv(summary_path, index=False)
print(f"\nSummary saved to: {summary_path}")

# Create visualization
print("\nGenerating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Colors for each framework
colors = {
    'genjax': '#0173B2',
    'numpyro': '#029E73', 
    'handcoded_jax': '#CC3311',
    'gen.jl': '#DE8F05',
    'pyro': '#EE7733'
}

# Plot 1: Runtime vs Particles (log-log)
for framework in df['framework'].unique():
    framework_data = df[df['framework'] == framework]
    times_ms = framework_data['mean_time'].values * 1000
    particles = framework_data['n_particles'].values
    
    # Sort by particles for proper line plotting
    sort_idx = np.argsort(particles)
    particles = particles[sort_idx]
    times_ms = times_ms[sort_idx]
    
    ax1.loglog(particles, times_ms, 'o-', 
               label=framework.replace('_', ' ').title(), 
               color=colors.get(framework, 'gray'),
               linewidth=2.5, markersize=10, alpha=0.9)

ax1.set_xlabel('Number of Particles', fontsize=16, fontweight='bold')
ax1.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
ax1.set_title('Runtime Scaling Comparison', fontsize=18, fontweight='bold')
ax1.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, which='both')
ax1.tick_params(labelsize=12)

# Plot 2: Speedup comparison (bar chart)
# Calculate speedups relative to slowest framework for each particle count
speedup_data = []
for n_particles in n_particles_list:
    subset = df[df['n_particles'] == n_particles]
    slowest_time = subset['mean_time'].max()
    
    for framework in subset['framework'].unique():
        framework_time = subset[subset['framework'] == framework]['mean_time'].values[0]
        speedup = slowest_time / framework_time
        speedup_data.append({
            'framework': framework,
            'n_particles': n_particles,
            'speedup': speedup
        })

speedup_df = pd.DataFrame(speedup_data)

# Create grouped bar chart
x = np.arange(len(n_particles_list))
width = 0.15
multiplier = 0

for framework in df['framework'].unique():
    offset = width * multiplier
    framework_speedups = []
    for n_particles in n_particles_list:
        speedup = speedup_df[(speedup_df['framework'] == framework) & 
                           (speedup_df['n_particles'] == n_particles)]['speedup'].values
        framework_speedups.append(speedup[0] if len(speedup) > 0 else 0)
    
    bars = ax2.bar(x + offset, framework_speedups, width, 
                    label=framework.replace('_', ' ').title(),
                    color=colors.get(framework, 'gray'), alpha=0.9)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.0f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    multiplier += 1

ax2.set_xlabel('Number of Particles', fontsize=16, fontweight='bold')
ax2.set_ylabel('Speedup (vs Slowest)', fontsize=16, fontweight='bold')
ax2.set_title('Relative Performance', fontsize=18, fontweight='bold')
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels([f'{n:,}' for n in n_particles_list])
ax2.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')
ax2.tick_params(labelsize=12)

plt.suptitle('Comprehensive Framework Comparison - GPU + CPU Benchmarks', 
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figures
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/all_frameworks_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figs/all_frameworks_comparison.png', dpi=300, bbox_inches='tight')
print("Plots saved to figs/all_frameworks_comparison.{pdf,png}")

print("\nBenchmark complete!")