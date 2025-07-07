#!/usr/bin/env python
"""Combine benchmark results from all frameworks and generate plots and tables."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse

def load_framework_results(data_dir, framework):
    """Load all results for a given framework."""
    framework_dir = Path(data_dir) / framework
    if not framework_dir.exists():
        print(f"Warning: No results found for {framework}")
        return None
    
    results = {}
    
    # Load IS results
    for n_particles in [100, 1000, 10000, 100000]:
        result_file = framework_dir / f"is_n{n_particles}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                results[f"is_n{n_particles}"] = json.load(f)
    
    # Load HMC results if available
    hmc_file = framework_dir / "hmc_n1000.json"
    if hmc_file.exists():
        with open(hmc_file, "r") as f:
            results["hmc"] = json.load(f)
    
    return results


def create_summary_table(all_results):
    """Create a summary dataframe from all results."""
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
            elif key == "hmc":
                rows.append({
                    'framework': framework,
                    'method': 'HMC',
                    'n_samples': result.get('n_samples', 1000),
                    'mean_time': result.get('mean_time', np.nan),
                    'std_time': result.get('std_time', np.nan),
                    'n_points': result.get('n_points', 50)
                })
    
    return pd.DataFrame(rows)


def create_plots(df, output_dir):
    """Create comparison plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter IS results
    is_df = df[df['method'] == 'IS'].copy()
    
    if len(is_df) == 0:
        print("No IS results to plot")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors for each framework
    colors = {
        'genjax': '#0173B2',             # Blue
        'numpyro': '#029E73',            # Green
        'handcoded_tfp': '#CC3311',      # Red
        'genjl': '#DE8F05',              # Orange
        'pyro': '#EE7733'                # Light Orange
    }
    
    # Display names for frameworks
    display_names = {
        'genjax': 'GenJAX',
        'numpyro': 'NumPyro',
        'handcoded_tfp': 'Handcoded JAX',
        'genjl': 'Gen.jl',
        'pyro': 'Pyro'
    }
    
    # Plot 1: Runtime vs Particles (log-log) with variance shading
    for framework in is_df['framework'].unique():
        framework_data = is_df[is_df['framework'] == framework].sort_values('n_particles')
        if len(framework_data) > 0:
            n_particles = framework_data['n_particles'].values
            mean_times = framework_data['mean_time'].values * 1000
            std_times = framework_data['std_time'].values * 1000
            
            # Plot mean line
            ax1.loglog(n_particles, mean_times,
                      'o-', label=display_names.get(framework, framework),
                      color=colors.get(framework, 'gray'),
                      linewidth=2.5, markersize=10, alpha=0.9)
            
            # Add shaded region for ±1 std
            ax1.fill_between(n_particles, 
                           mean_times - std_times,
                           mean_times + std_times,
                           color=colors.get(framework, 'gray'),
                           alpha=0.2)
    
    ax1.set_xlabel('Number of Particles', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=16, fontweight='bold')
    ax1.set_title('Runtime Scaling Comparison', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=12)
    
    # Plot 2: Times slower than handcoded baseline
    # Calculate slowdown relative to handcoded JAX for each particle count
    slowdown_data = []
    for n_particles in is_df['n_particles'].unique():
        subset = is_df[is_df['n_particles'] == n_particles]
        if len(subset) > 0:
            # Find handcoded baseline time
            handcoded_time = subset[subset['framework'] == 'handcoded_tfp']['mean_time'].values
            if len(handcoded_time) > 0:
                baseline_time = handcoded_time[0]
                
                for _, row in subset.iterrows():
                    if not np.isnan(row['mean_time']):
                        slowdown = row['mean_time'] / baseline_time
                        slowdown_data.append({
                            'framework': row['framework'],
                            'n_particles': n_particles,
                            'slowdown': slowdown
                        })
    
    if slowdown_data:
        slowdown_df = pd.DataFrame(slowdown_data)
        
        # Create grouped bar chart
        frameworks = [f for f in ['genjax', 'numpyro', 'handcoded_tfp', 'genjl', 'pyro'] 
                     if f in slowdown_df['framework'].unique()]
        n_particles_list = sorted(slowdown_df['n_particles'].unique())
        x = np.arange(len(n_particles_list))
        width = 0.8 / len(frameworks)
        
        for i, framework in enumerate(frameworks):
            framework_slowdowns = []
            for n_particles in n_particles_list:
                slowdown = slowdown_df[(slowdown_df['framework'] == framework) & 
                                     (slowdown_df['n_particles'] == n_particles)]['slowdown'].values
                framework_slowdowns.append(slowdown[0] if len(slowdown) > 0 else None)
            
            # Skip if no valid data
            valid_indices = [j for j, val in enumerate(framework_slowdowns) if val is not None]
            if not valid_indices:
                continue
                
            offset = (i - len(frameworks)/2 + 0.5) * width
            x_positions = x[valid_indices] + offset
            y_values = [framework_slowdowns[j] for j in valid_indices]
            
            bars = ax2.bar(x_positions, y_values, width,
                          label=display_names.get(framework, framework),
                          color=colors.get(framework, 'gray'), alpha=0.9)
            
            # Add value labels on bars
            for bar, val in zip(bars, y_values):
                height = bar.get_height()
                if height > 0:
                    # Format label based on value
                    if height < 10:
                        label = f'{height:.1f}×'
                    else:
                        label = f'{height:.0f}×'
                    ax2.annotate(label,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10, fontweight='bold')
        
        # Add horizontal line at y=1 for baseline
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(0.02, 1.02, 'Handcoded baseline', transform=ax2.get_yaxis_transform(), 
                fontsize=10, alpha=0.7)
        
        ax2.set_xlabel('Number of Particles', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Times Slower than Handcoded JAX', fontsize=16, fontweight='bold')
        ax2.set_title('Relative Performance vs Handcoded Baseline', fontsize=18, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{n:,}' for n in n_particles_list])
        ax2.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        ax2.tick_params(labelsize=12)
        
        # Set y-axis limits to show range better
        ax2.set_ylim(0.5, max(slowdown_df['slowdown'].max() * 1.5, 100))
    
    plt.suptitle('Comprehensive Framework Comparison - GPU Benchmarks', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figures
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'all_frameworks_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'all_frameworks_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir}/all_frameworks_comparison.{{pdf,png}}")
    plt.close()


def create_latex_table(df, output_file):
    """Create a LaTeX table from the results."""
    is_df = df[df['method'] == 'IS'].copy()
    
    if len(is_df) == 0:
        print("No IS results for table")
        return
    
    # Create pivot table
    pivot = is_df.pivot(index='framework', columns='n_particles', values='mean_time')
    
    # Convert to milliseconds
    pivot = pivot * 1000
    
    # Calculate speedups relative to slowest
    speedup_pivot = pd.DataFrame(index=pivot.index, columns=pivot.columns)
    for col in pivot.columns:
        max_time = pivot[col].max()
        speedup_pivot[col] = max_time / pivot[col]
    
    # Create LaTeX table
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GPU Benchmark Results: Runtime (ms) and Speedup}\n")
        f.write("\\label{tab:gpu-benchmarks}\n")
        f.write("\\begin{tabular}{l" + "r" * len(pivot.columns) + "}\n")
        f.write("\\toprule\n")
        f.write("Framework & " + " & ".join([f"N={n:,}" for n in pivot.columns]) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Get display names
        display_names_local = {
            'genjax': 'GenJAX',
            'numpyro': 'NumPyro',
            'handcoded_tfp': 'Handcoded JAX',
            'genjl': 'Gen.jl',
            'pyro': 'Pyro'
        }
        
        for framework in pivot.index:
            row = display_names_local.get(framework, framework)
            for n in pivot.columns:
                time = pivot.loc[framework, n]
                speedup = speedup_pivot.loc[framework, n]
                if not np.isnan(time):
                    row += f" & {time:.2f} ({speedup:.1f}×)"
                else:
                    row += " & --"
            row += " \\\\\n"
            f.write(row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Combine benchmark results")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing framework results")
    parser.add_argument("--output-dir", type=str, default="figs",
                        help="Output directory for plots")
    parser.add_argument("--frameworks", nargs="+",
                        default=["genjax", "numpyro", "handcoded_tfp", "pyro", "genjl"],
                        help="Frameworks to include")
    
    args = parser.parse_args()
    
    # Load results from all frameworks
    all_results = {}
    for framework in args.frameworks:
        results = load_framework_results(args.data_dir, framework)
        if results:
            all_results[framework] = results
    
    if not all_results:
        print("No results found!")
        return
    
    # Create summary dataframe
    df = create_summary_table(all_results)
    
    # Save CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    csv_file = output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Summary saved to {csv_file}")
    
    # Create plots
    create_plots(df, output_dir)
    
    # Create LaTeX table
    tex_file = output_dir / "benchmark_table.tex"
    create_latex_table(df, tex_file)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - All Frameworks GPU Benchmark Results")
    print("="*80)
    
    is_df = df[df['method'] == 'IS']
    for n_particles in sorted(is_df['n_particles'].unique()):
        print(f"\nN = {n_particles:,} particles:")
        subset = is_df[is_df['n_particles'] == n_particles].sort_values('mean_time')
        
        if len(subset) > 0:
            fastest_time = subset['mean_time'].min()
            
            # Get display names
            display_names_local = {
                'genjax': 'GenJAX',
                'numpyro': 'NumPyro',
                'handcoded_tfp': 'Handcoded JAX',
                'genjl': 'Gen.jl',
                'pyro': 'Pyro'
            }
            
            print(f"{'Framework':<20} {'Time (ms)':<15} {'Speedup':<15}")
            print("-" * 50)
            
            for _, row in subset.iterrows():
                if not np.isnan(row['mean_time']):
                    time_ms = row['mean_time'] * 1000
                    speedup = row['mean_time'] / fastest_time
                    framework_name = display_names_local.get(row['framework'], row['framework'])
                    print(f"{framework_name:<20} {time_ms:<15.3f} {speedup:<15.1f}x")


if __name__ == "__main__":
    main()