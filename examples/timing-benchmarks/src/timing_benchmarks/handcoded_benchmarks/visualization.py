"""Visualization for handcoded benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
from datetime import datetime


def create_benchmark_comparison_plot(results: Dict[str, List[Dict]], 
                                   benchmark_name: str,
                                   x_param: str,
                                   x_label: str,
                                   output_prefix: str = 'benchmark_comparison'):
    """Create comparison plots similar to all_frameworks_comparison."""
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Create figure - compact height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    # Colors for each framework
    colors = {
        'genjax': '#0173B2',           # Blue
        'handcoded_jax': '#CC3311',    # Red  
        'handcoded_torch': '#56B4E9'   # Light Blue
    }
    
    # Display names
    display_names = {
        'genjax': 'Ours',
        'handcoded_jax': 'Handcoded JAX',
        'handcoded_torch': 'Handcoded PyTorch'
    }
    
    # Extract data for plotting
    plot_data = {}
    for framework, framework_results in results.items():
        if framework_results:
            x_values = [r[x_param] for r in framework_results]
            mean_times = [r['mean_time'] * 1000 for r in framework_results]  # Convert to ms
            std_times = [r['std_time'] * 1000 for r in framework_results]
            plot_data[framework] = (x_values, mean_times, std_times)
    
    # Plot 1: Runtime scaling comparison
    ax1.set_title('Runtime Scaling Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel(x_label, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    for framework, (x_vals, mean_times, std_times) in plot_data.items():
        # Use dotted line for GenJAX, solid for others
        linestyle = ':' if framework == 'genjax' else '-'
        
        # Make Ours and Handcoded JAX more prominent
        if framework in ['genjax', 'handcoded_jax']:
            linewidth = 3.5
            alpha = 1.0
        else:
            linewidth = 1.2
            alpha = 0.7
        
        # Plot mean line
        ax1.loglog(x_vals, mean_times,
                  linestyle=linestyle, label=display_names.get(framework, framework),
                  color=colors.get(framework, 'gray'),
                  linewidth=linewidth, alpha=alpha)
        
        # Add shaded region for ±1 std
        mean_times = np.array(mean_times)
        std_times = np.array(std_times)
        ax1.fill_between(x_vals, 
                       mean_times - std_times,
                       mean_times + std_times,
                       color=colors.get(framework, 'gray'),
                       alpha=0.2)
    
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    
    # Plot 2: Relative performance vs handcoded baseline
    ax2.set_title('Relative Performance vs Handcoded Baseline', fontsize=16, fontweight='bold')
    ax2.set_xlabel(x_label, fontsize=16, fontweight='bold')
    ax2.set_ylabel('Scale Factor', fontsize=16, fontweight='bold')
    ax2.set_yscale('log')
    
    # Get handcoded baseline times
    if 'handcoded_jax' in plot_data:
        baseline_x, baseline_times, _ = plot_data['handcoded_jax']
        
        # Create x positions for bars
        x_positions = np.arange(len(baseline_x))
        
        # Calculate scale factors for each x value
        for idx, x_val in enumerate(baseline_x):
            baseline_time = baseline_times[idx]
            
            # Get data for this x value
            frameworks_at_x = []
            for framework in ['genjax', 'handcoded_torch']:  # Exclude handcoded_jax
                if framework in plot_data:
                    fx, ft, _ = plot_data[framework]
                    if x_val in fx:
                        fidx = fx.index(x_val)
                        scale_factor = ft[fidx] / baseline_time
                        frameworks_at_x.append((framework, scale_factor))
            
            # Sort by scale factor (fastest first)
            frameworks_at_x.sort(key=lambda x: x[1])
            
            # Plot bars
            n_frameworks = len(frameworks_at_x)
            if n_frameworks > 0:
                width = 0.8 / n_frameworks
                
                for i, (framework, scale_factor) in enumerate(frameworks_at_x):
                    offset = (i - n_frameworks/2 + 0.5) * width
                    x_pos = x_positions[idx] + offset
                    
                    # Label only on first occurrence
                    label = display_names.get(framework, framework) if idx == 0 else None
                    
                    # Use hatching for GenJAX
                    hatch = '///' if framework == 'genjax' else None
                    
                    bar = ax2.bar(x_pos, scale_factor, width,
                                 label=label,
                                 color=colors.get(framework, 'gray'), 
                                 alpha=0.9, edgecolor='black', linewidth=0.5,
                                 hatch=hatch)
                    
                    # Add value label
                    if scale_factor < 10:
                        label_text = f'{scale_factor:.1f}×'
                    else:
                        label_text = f'{int(scale_factor)}×'
                    ax2.text(x_pos, scale_factor, label_text,
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
    
    # Add "Smaller is better" text
    ax2.text(0.02, 0.98, 'Smaller is better', 
            transform=ax2.transAxes, 
            ha='left', va='top',
            fontsize=12, style='italic', alpha=0.7)
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(x) for x in baseline_x])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(1, 1e3)
    ax2.set_yticks([1, 10, 100, 1000])
    ax2.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
    
    # Add single legend below both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
              bbox_to_anchor=(0.5, -0.12), frameon=True, fancybox=True, 
              shadow=True, fontsize=16, prop={'weight': 'bold', 'size': 16})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save figures
    for ext in ['pdf', 'png']:
        filename = f'{output_prefix}_{benchmark_name}.{ext}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.close()


def save_benchmark_results_csv(results: Dict[str, List[Dict]], 
                             benchmark_name: str,
                             output_prefix: str = 'benchmark_results'):
    """Save benchmark results to CSV for analysis."""
    rows = []
    
    for framework, framework_results in results.items():
        for result in framework_results:
            row = {
                'framework': framework,
                'benchmark': benchmark_name,
                **result
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_prefix}_{benchmark_name}_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Print summary
    print(f"\nSummary for {benchmark_name}:")
    print("-" * 50)
    
    # Group by parameter and show results
    param_col = 'grid_size' if benchmark_name == 'gol' else 'n_data'
    
    for param_val in sorted(df[param_col].unique()):
        subset = df[df[param_col] == param_val].sort_values('mean_time')
        print(f"\n{param_col.replace('_', ' ').title()}: {param_val}")
        print(f"{'Framework':<20} {'Mean Time (ms)':<15} {'Std (ms)':<10}")
        print("-" * 45)
        
        for _, row in subset.iterrows():
            framework_name = row['framework'].replace('_', ' ').title()
            if row['framework'] == 'genjax':
                framework_name = 'Ours'
            print(f"{framework_name:<20} {row['mean_time']*1000:<15.3f} {row['std_time']*1000:<10.3f}")
    
    return df