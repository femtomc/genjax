#!/usr/bin/env python
"""Combine handcoded benchmark results and generate data scaling plots."""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse


def load_framework_results(data_dir, framework):
    """Load all results for a given framework."""
    framework_dir = Path(data_dir) / framework
    if not framework_dir.exists():
        print(f"Warning: No results found for {framework}")
        return None
    
    results = {"gmm": []}
    
    # Load GMM results
    for result_file in sorted(framework_dir.glob("gmm_n*.json")):
        with open(result_file, "r") as f:
            results["gmm"].append(json.load(f))
    
    return results


def create_data_scaling_plot(benchmark_type, all_results, output_dir):
    """Create a data scaling plot matching all_frameworks_comparison aesthetics."""
    # Use the same style as all_frameworks_comparison
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with compact height like all_frameworks_comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    # Colors matching all_frameworks_comparison
    colors = {
        'genjax_handcoded': '#0173B2',      # Blue (same as genjax)
        'handcoded_jax': '#CC3311',         # Red (same as handcoded_tfp)
        'handcoded_torch': '#56B4E9'        # Light Blue
    }
    
    # Display names matching style
    display_names = {
        'genjax_handcoded': 'Ours',  # Matching "Ours" label
        'handcoded_jax': 'Handcoded JAX',
        'handcoded_torch': 'Handcoded PyTorch'
    }
    
    # Framework order for consistent plotting
    framework_order = ['genjax_handcoded', 'handcoded_jax', 'handcoded_torch']
    
    # Prepare data for plotting
    plot_data = {}
    for framework in framework_order:
        if framework in all_results:
            results = all_results[framework]
            if results and results.get(benchmark_type):
                data = results[benchmark_type]
                
                x_values = [r["n_data"] for r in data]
                
                mean_times = [r["mean_time"] * 1000 for r in data]  # Convert to ms
                std_times = [r["std_time"] * 1000 for r in data]
                
                # Sort by x values
                sorted_indices = np.argsort(x_values)
                x_values = np.array(x_values)[sorted_indices]
                mean_times = np.array(mean_times)[sorted_indices]
                std_times = np.array(std_times)[sorted_indices]
                
                plot_data[framework] = {
                    'x': x_values,
                    'mean': mean_times,
                    'std': std_times
                }
    
    # Left subplot: Runtime scaling (log-log)
    for framework in framework_order:
        if framework in plot_data:
            data = plot_data[framework]
            
            # Make GenJAX more prominent (matching all_frameworks_comparison)
            if framework == 'genjax_handcoded':
                linewidth = 4.5  # Much thicker line for GenJAX
                alpha = 1.0      # Full opacity
            else:
                linewidth = 2.0  # Thinner line for others
                alpha = 0.8      # Slightly transparent
            
            # Plot solid line
            ax1.loglog(data['x'], data['mean'],
                      linestyle='-', label=display_names[framework],
                      color=colors[framework],
                      linewidth=linewidth, alpha=alpha)
            
            # Add shaded region for ±1 std
            ax1.fill_between(data['x'], 
                           data['mean'] - data['std'],
                           data['mean'] + data['std'],
                           color=colors[framework],
                           alpha=0.2)
    
    # Styling for left plot (matching all_frameworks_comparison)
    ax1.set_xlabel('Number of Observations', fontsize=16, fontweight='bold')
    title = 'GMM Runtime Scaling Comparison'
    
    ax1.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
    ax1.set_title(title, fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=12)
    
    # Right subplot: Relative performance bars
    # Calculate slowdown relative to handcoded JAX
    slowdown_data = []
    
    # Get unique x values
    all_x_values = set()
    for data in plot_data.values():
        all_x_values.update(data['x'])
    x_values_list = sorted(list(all_x_values))
    
    # Calculate slowdowns
    for x_val in x_values_list:
        # Find baseline (handcoded JAX) time
        baseline_time = None
        if 'handcoded_jax' in plot_data:
            idx = np.where(plot_data['handcoded_jax']['x'] == x_val)[0]
            if len(idx) > 0:
                baseline_time = plot_data['handcoded_jax']['mean'][idx[0]]
        
        if baseline_time is not None:
            for framework in framework_order:
                if framework in plot_data and framework != 'handcoded_jax':
                    idx = np.where(plot_data[framework]['x'] == x_val)[0]
                    if len(idx) > 0:
                        mean_time = plot_data[framework]['mean'][idx[0]]
                        slowdown = mean_time / baseline_time
                        slowdown_data.append({
                            'framework': framework,
                            'x_value': x_val,
                            'slowdown': slowdown
                        })
    
    if slowdown_data:
        slowdown_df = pd.DataFrame(slowdown_data)
        
        # Use largest x values for the bar plot (matching all_frameworks_comparison)
        max_x_values = sorted(slowdown_df['x_value'].unique())[-3:]  # Last 3 values
        x = np.arange(len(max_x_values))
        
        # Process each x value
        for idx, x_val in enumerate(max_x_values):
            # Get data for this x value
            x_data = slowdown_df[slowdown_df['x_value'] == x_val]
            
            # Sort by slowdown (fastest first)
            x_data = x_data.sort_values('slowdown')
            
            # Calculate bar width
            n_frameworks = len(x_data)
            if n_frameworks == 0:
                continue
            width = 0.8 / n_frameworks
            
            # Plot bars
            for i, (_, row) in enumerate(x_data.iterrows()):
                framework = row['framework']
                slowdown = row['slowdown']
                
                # Position bar
                offset = (i - n_frameworks/2 + 0.5) * width
                x_pos = x[idx] + offset
                
                # Only label the first occurrence
                label = display_names[framework] if idx == 0 else None
                
                # Simple bars (matching all_frameworks_comparison)
                bar = ax2.bar(x_pos, slowdown, width,
                             label=label,
                             color=colors[framework], 
                             alpha=0.9, edgecolor='black', linewidth=0.5)
                
                # Add value label on bar
                if slowdown > 0:
                    # Format label based on value
                    if slowdown < 10:
                        label_text = f'{slowdown:.1f}×'
                    else:
                        label_text = f'{int(slowdown)}×'
                    ax2.annotate(label_text,
                                xy=(x_pos, slowdown),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10, fontweight='bold')
        
        # Styling for right plot (matching all_frameworks_comparison)
        ax2.set_xlabel('Number of Observations', fontsize=16, fontweight='bold')
        x_labels = [f'{int(v/1000)}K' if v >= 1000 else str(int(v)) for v in max_x_values]
        
        ax2.set_ylabel('Scale Factor', fontsize=16, fontweight='bold')
        ax2.set_title('Relative Performance vs Handcoded Baseline', fontsize=18, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        ax2.tick_params(labelsize=12)
        
        # Add "Smaller is better" text - top left
        ax2.text(0.02, 0.98, 'Smaller is better', 
                transform=ax2.transAxes, 
                ha='left', va='top',
                fontsize=12, style='italic', alpha=0.7)
        
        # Set y-axis limits and ticks (matching all_frameworks_comparison)
        if 'handcoded_torch' in plot_data:
            # PyTorch is much slower for GMM
            ax2.set_ylim(0.5, 1e6)
            ax2.set_yticks([1, 100, 10000, 1000000])
            ax2.set_yticklabels(['$10^0$', '$10^2$', '$10^4$', '$10^6$'])
        else:
            # Normal range
            ax2.set_ylim(0.5, 10)
            ax2.set_yticks([1, 2, 5, 10])
            ax2.set_yticklabels(['1', '2', '5', '10'])
    
    # Add single legend below both subplots (matching all_frameworks_comparison)
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
              bbox_to_anchor=(0.5, -0.12), frameon=True, fancybox=True, 
              shadow=True, fontsize=16, prop={'weight': 'bold', 'size': 16})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save figure
    output_path = Path(output_dir) / f'{benchmark_type}_all_frameworks_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def print_scaling_analysis(benchmark_type, all_results):
    """Print scaling analysis for the benchmark."""
    print(f"\n{'='*60}")
    print(f"{benchmark_type.upper()} Scaling Analysis")
    print(f"{'='*60}")
    
    for framework, results in all_results.items():
        if results and results.get(benchmark_type):
            data = results[benchmark_type]
            
            x_values = [r["n_data"] for r in data]
            x_name = "data points"
            
            times = [r["mean_time"] for r in data]
            
            if len(times) >= 2:
                # Calculate scaling exponent
                log_times = np.log(times)
                log_sizes = np.log(x_values)
                alpha, intercept = np.polyfit(log_sizes, log_times, 1)
                
                print(f"\n{framework}:")
                print(f"  Scaling exponent: {alpha:.2f} (time ~ n^{alpha:.2f})")
                print(f"  Range: {x_name} {min(x_values)} to {max(x_values)}")
                print(f"  Time range: {min(times)*1000:.2f}ms to {max(times)*1000:.2f}ms")
                
                # Calculate relative performance vs handcoded JAX
                if framework != 'handcoded_jax' and 'handcoded_jax' in all_results:
                    jax_data = all_results['handcoded_jax'].get(benchmark_type, [])
                    if jax_data:
                        jax_times = {r["n_data"]: r["mean_time"] for r in jax_data}
                        
                        speedups = []
                        for r in data:
                            key = r["n_data"]
                            if key in jax_times:
                                speedup = r["mean_time"] / jax_times[key]
                                speedups.append(speedup)
                        
                        if speedups:
                            avg_speedup = np.mean(speedups)
                            print(f"  Average relative to handcoded JAX: {avg_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Generate data scaling plots for handcoded benchmarks")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir", default="figs", help="Output directory for plots"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results from all frameworks
    all_results = {}
    frameworks = ["genjax_handcoded", "handcoded_jax", "handcoded_torch"]
    
    for framework in frameworks:
        results = load_framework_results(args.data_dir, framework)
        if results:
            all_results[framework] = results
    
    if not all_results:
        print("No results found!")
        return
    
    # Create data scaling plots with all_frameworks_comparison aesthetics
    print("Creating GMM data scaling plots...")
    
    # GMM data scaling plot  
    create_data_scaling_plot("gmm", all_results, args.output_dir)
    print_scaling_analysis("gmm", all_results)
    
    print(f"\n{'='*60}")
    print(f"Data scaling plots saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()