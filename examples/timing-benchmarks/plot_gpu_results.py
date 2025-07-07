#!/usr/bin/env python
"""Plot GPU benchmark results."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('data/benchmark_20250706_122408/is_comparison_summary.csv')

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Extract data for plotting
particles = [100, 1000, 10000, 100000]
frameworks = ['genjax', 'numpyro', 'handcoded_jax']
colors = {'genjax': '#0173B2', 'numpyro': '#029E73', 'handcoded_jax': '#CC3311'}

# Plot 1: Runtime vs Particles (log-log)
for framework in frameworks:
    framework_data = df[df['framework'] == framework]
    times = [framework_data[framework_data['n_particles'] == n]['mean_time'].values[0] * 1000 
             for n in particles]
    ax1.loglog(particles, times, 'o-', label=framework.replace('_', ' ').title(), 
               color=colors[framework], linewidth=2, markersize=8)

ax1.set_xlabel('Number of Particles', fontsize=14)
ax1.set_ylabel('Runtime (ms)', fontsize=14)
ax1.set_title('GPU Performance: Runtime vs Particles', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup vs NumPyro
speedups = {framework: [] for framework in ['genjax', 'handcoded_jax']}
for n in particles:
    numpyro_time = df[(df['framework'] == 'numpyro') & (df['n_particles'] == n)]['mean_time'].values[0]
    for framework in ['genjax', 'handcoded_jax']:
        framework_time = df[(df['framework'] == framework) & (df['n_particles'] == n)]['mean_time'].values[0]
        speedups[framework].append(numpyro_time / framework_time)

x = np.arange(len(particles))
width = 0.35

bars1 = ax2.bar(x - width/2, speedups['genjax'], width, 
                 label='GenJAX', color=colors['genjax'])
bars2 = ax2.bar(x + width/2, speedups['handcoded_jax'], width, 
                 label='Handcoded JAX', color=colors['handcoded_jax'])

ax2.set_xlabel('Number of Particles', fontsize=14)
ax2.set_ylabel('Speedup vs NumPyro', fontsize=14)
ax2.set_title('Speedup Comparison (Higher is Better)', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{n:,}' for n in particles])
ax2.set_yscale('log')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

plt.suptitle('GPU Benchmark Results - NVIDIA RTX 4090', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('figs/gpu_benchmark_results.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figs/gpu_benchmark_results.png', dpi=300, bbox_inches='tight')
print("Figures saved to figs/gpu_benchmark_results.{pdf,png}")