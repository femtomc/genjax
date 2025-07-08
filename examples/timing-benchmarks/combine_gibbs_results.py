#!/usr/bin/env python
"""Combine and plot Gibbs sampler benchmark results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_gibbs_results(data_dir):
    """Load all Gibbs sampler results from data directories."""
    results = {}
    
    # Framework directories
    framework_dirs = {
        'genjax': 'genjax_gibbs_handcoded',
        'handcoded_jax': 'handcoded_jax_gibbs',
        'handcoded_torch': 'handcoded_torch_gibbs'
    }
    
    for framework, dirname in framework_dirs.items():
        framework_dir = Path(data_dir) / dirname
        if not framework_dir.exists():
            print(f"Warning: No results found for {framework} at {framework_dir}")
            continue
            
        framework_results = []
        
        # Load all JSON files in the directory
        for json_file in framework_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                result = json.load(f)
                # Extract framework name from result or filename
                if 'framework' not in result:
                    if 'genjax' in str(json_file):
                        result['framework'] = 'genjax'
                    elif 'jax' in str(json_file):
                        result['framework'] = 'handcoded_jax'
                    elif 'torch' in str(json_file):
                        result['framework'] = 'handcoded_torch'
                framework_results.append(result)
        
        if framework_results:
            results[framework] = framework_results
    
    return results

def create_gibbs_plots(results, output_dir):
    """Create Gibbs sampler comparison plots."""
    # Set style to match all_frameworks_comparison
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Convert results to DataFrame
    rows = []
    for framework, framework_results in results.items():
        for result in framework_results:
            rows.append({
                'framework': result.get('framework', framework),
                'grid_size': result['grid_size'],
                'n_sweeps': result.get('n_sweeps', 10),
                'mean_time': result['mean_time'],
                'std_time': result['std_time']
            })
    
    df = pd.DataFrame(rows)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    # Colors for each framework (matching all_frameworks_comparison)
    colors = {
        'genjax': '#0173B2',           # Blue
        'genjax_optimized': '#0173B2', # Blue
        'handcoded_jax': '#029E73',    # Green
        'handcoded_jax_optimized': '#029E73',  # Green
        'handcoded_torch': '#EE3377',  # Pink/Red
        'handcoded_torch_optimized': '#EE3377'  # Pink/Red
    }
    
    # Display names
    display_names = {
        'genjax': 'GenJAX',
        'genjax_optimized': 'GenJAX',
        'handcoded_jax': 'Handcoded JAX',
        'handcoded_jax_optimized': 'Handcoded JAX',
        'handcoded_torch': 'Handcoded PyTorch',
        'handcoded_torch_optimized': 'Handcoded PyTorch'
    }
    
    # Plot 1: Runtime vs Grid Size
    for framework in df['framework'].unique():
        framework_data = df[df['framework'] == framework].sort_values('grid_size')
        if len(framework_data) > 0:
            grid_sizes = framework_data['grid_size'].values
            mean_times = framework_data['mean_time'].values * 1000  # Convert to ms
            std_times = framework_data['std_time'].values * 1000
            
            # Make GenJAX more prominent
            if 'genjax' in framework:
                linewidth = 4.5  # Much thicker line for GenJAX
                alpha = 1.0      # Full opacity
            else:
                linewidth = 2.0  # Thinner line for others
                alpha = 0.8      # Slightly transparent
            
            ax1.loglog(grid_sizes, mean_times,
                      linestyle='-', label=display_names.get(framework, framework),
                      color=colors.get(framework, 'gray'),
                      linewidth=linewidth, alpha=alpha, marker='o', markersize=8)
            
            # Add shaded region for ±1 std
            ax1.fill_between(grid_sizes, 
                           mean_times - std_times,
                           mean_times + std_times,
                           color=colors.get(framework, 'gray'),
                           alpha=0.2)
    
    ax1.set_xlabel('Grid Size', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
    ax1.set_title('Gibbs Sampler Runtime Scaling', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=12)
    ax1.set_xticks([16, 32, 64])
    ax1.set_xticklabels(['16', '32', '64'])
    
    # Plot 2: Bar chart at largest grid size
    max_grid_size = df['grid_size'].max()
    bar_data = df[df['grid_size'] == max_grid_size].copy()
    
    if len(bar_data) > 0:
        # Sort by mean time for consistent ordering
        bar_data = bar_data.sort_values('mean_time')
        
        x = np.arange(len(bar_data))
        frameworks = bar_data['framework'].values
        mean_times = bar_data['mean_time'].values * 1000
        
        bars = ax2.bar(x, mean_times, 
                       color=[colors.get(f, 'gray') for f in frameworks],
                       alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (framework, time) in enumerate(zip(frameworks, mean_times)):
            if time < 10:
                label_text = f'{time:.1f}ms'
            else:
                label_text = f'{int(time)}ms'
            ax2.annotate(label_text,
                        xy=(i, time),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
        ax2.set_xlabel('Framework', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
        ax2.set_title(f'Runtime at {max_grid_size}×{max_grid_size} Grid', fontsize=18, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([display_names.get(f, f) for f in frameworks], 
                           rotation=15, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(labelsize=12)
        
        # Add "Smaller is better" text
        ax2.text(0.02, 0.98, 'Smaller is better', 
                transform=ax2.transAxes, 
                ha='left', va='top',
                fontsize=12, style='italic', alpha=0.7)
    
    # Add single legend below both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
              bbox_to_anchor=(0.5, -0.12), frameon=True, fancybox=True, 
              shadow=True, fontsize=16, prop={'weight': 'bold', 'size': 16})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save figures
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'gibbs_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'gibbs_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir}/gibbs_comparison.{{pdf,png}}")
    plt.close()

def main():
    # Load results
    data_dir = Path("data")
    results = load_gibbs_results(data_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("GIBBS SAMPLER BENCHMARK RESULTS")
    print("="*80)
    
    for framework, framework_results in results.items():
        print(f"\n{framework}:")
        for result in framework_results:
            grid_size = result['grid_size']
            mean_time = result['mean_time'] * 1000
            std_time = result['std_time'] * 1000
            print(f"  Grid {grid_size}x{grid_size}: {mean_time:.2f} ± {std_time:.2f} ms")
    
    # Create plots
    output_dir = Path("figs")
    create_gibbs_plots(results, output_dir)

if __name__ == "__main__":
    main()