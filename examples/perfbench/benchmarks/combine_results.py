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
    base = Path(data_dir)
    candidates = []
    if base.name != "curvefit":
        candidates.append(base / "curvefit" / framework)
        if framework == "genjl":
            candidates.append(base / "curvefit" / "genjl_dynamic")
    else:
        candidates.append(base / framework)
        if framework == "genjl":
            candidates.append(base / "genjl_dynamic")
    candidates.append(base / framework)
    if framework == "genjl":
        candidates.append(base / "genjl_dynamic")
    candidates = [cand for cand in candidates if cand.exists()]
    if not candidates:
        print(f"Warning: No results found for {framework}")
        return None
    
    results = {}
    base = Path(data_dir)

    def maybe_load(path: Path, key: str):
        if key not in results and path.exists():
            with open(path, "r") as f:
                results[key] = json.load(f)
            return True
        return False

    for framework_dir in candidates:
        for n_particles in [100, 1000, 5000, 10000, 100000]:
            key = f"is_n{n_particles}"
            maybe_load(framework_dir / f"is_n{n_particles}.json", key)

        for n_samples in [100, 500, 1000, 5000, 10000]:
            key = f"hmc_n{n_samples}"
            if maybe_load(framework_dir / f"hmc_n{n_samples}.json", key):
                print(f"Loaded HMC results from {framework_dir / f'hmc_n{n_samples}.json'}")
            else:
                top_level = base / framework / f"hmc_n{n_samples}.json"
                if maybe_load(top_level, key):
                    print(f"Loaded HMC results from {top_level}")
    
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
            elif key.startswith("hmc_n"):
                n_samples = int(key.split("_n")[1])
                rows.append({
                    'framework': framework,
                    'method': 'HMC',
                    'n_samples': n_samples,
                    'mean_time': result.get('mean_time', np.nan),
                    'std_time': result.get('std_time', np.nan),
                    'n_points': result.get('n_points', 50)
                })
    
    return pd.DataFrame(rows)


def create_hmc_comparison_plot(results_df, output_dir):
    """Create HMC comparison plot across frameworks with varying chain lengths - matching IS plot style."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.legend_handler
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    # Filter for HMC results only
    hmc_df = results_df[results_df['method'] == 'HMC'].copy()
    
    if "n_samples" not in hmc_df.columns:
        print("No HMC results found")
        return
    
    # Filter out 5000 samples
    hmc_df = hmc_df[hmc_df['n_samples'] != 5000]
    
    if len(hmc_df) == 0:
        print("No HMC results found")
        return
    
    # Setup styling to match IS plot
    sns.set_style("white")
    plt.rcParams.update({"font.size": 18})
    
    # Create figure - single plot (reduced height to match IS)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Framework colors matching the IS plot
    colors = {
        'genjax': 'deepskyblue',
        'handcoded_jax': 'gold',
        'numpyro': 'coral',
        'pyro': 'mediumseagreen',
        'genjl': 'darkblue',  # Use same color as genjl_dynamic in IS plot
        'genjl_optimized': '#CC77AA',
        'genjl_dynamic': 'darkblue',
        'handcoded_torch': 'darkslategray'
    }
    
    # Display names for frameworks
    display_names = {
        'genjax': 'Ours',
        'numpyro': 'NumPyro',
        'handcoded_jax': 'Handcoded JAX',
        'genjl': 'Gen.jl',
        'genjl_optimized': 'Gen.jl (static+Map)',
        'genjl_dynamic': 'Gen.jl',
        'pyro': 'Pyro',
        'handcoded_torch': 'Handcoded PyTorch'
    }
    
    # Get unique chain lengths
    chain_lengths = sorted(hmc_df['n_samples'].unique())
    x = np.arange(len(chain_lengths))
    
    # Get timing data for all frameworks
    timing_data = []
    baseline_times = {}  # Store handcoded baseline times
    
    for n_samples in chain_lengths:
        subset = hmc_df[hmc_df['n_samples'] == n_samples]
        if len(subset) > 0:
            # Find handcoded baseline time
            handcoded_time = subset[subset['framework'] == 'handcoded_jax']['mean_time'].values
            if len(handcoded_time) > 0:
                baseline_times[n_samples] = handcoded_time[0] * 1000  # Convert to ms
                
            for _, row in subset.iterrows():
                if not np.isnan(row['mean_time']):
                    timing_data.append({
                        'framework': row['framework'],
                        'n_samples': n_samples,
                        'time_ms': row['mean_time'] * 1000  # Convert to ms
                    })
    
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        
        # Create grouped bar chart (similar to IS plot)
        # Sort bars by performance (fastest to slowest) for each chain length
        n_samples_list = sorted(timing_df['n_samples'].unique())
        x = np.arange(len(n_samples_list))
        
        # Process each chain length separately to sort by performance
        for idx, n_samples in enumerate(n_samples_list):
            # Get data for this chain length
            chain_data = timing_df[timing_df['n_samples'] == n_samples]
            
            # Sort by time (fastest first)
            chain_data = chain_data.sort_values('time_ms')
            
            # Don't filter out handcoded baseline - we'll show it as a bar
            # chain_data = chain_data[chain_data['framework'] != 'handcoded_jax']
            
            # Calculate bar width based on number of frameworks
            n_frameworks = len(chain_data)
            if n_frameworks == 0:
                continue
            width = 0.8 / n_frameworks
            
            # Plot bars for this chain length
            for i, (_, row) in enumerate(chain_data.iterrows()):
                framework = row['framework']
                time_ms = row['time_ms']
                
                # Position bar
                offset = (i - n_frameworks/2 + 0.5) * width
                x_pos = x[idx] + offset
                
                # Only label the first occurrence of each framework
                label = display_names.get(framework, framework) if idx == 0 else None
                
                # Simple bars for all frameworks
                bar = ax.bar(x_pos, time_ms, width,
                             label=label,
                             color=colors.get(framework, 'gray'), 
                             alpha=0.9, edgecolor='black', linewidth=0.5)
                
                # Add gold star at bottom of GenJAX bars
                if framework == 'genjax':
                    # Position star near x-axis (at y=10 on log scale)
                    star_y = 10
                    ax.scatter(x_pos, star_y, marker='*', s=200, 
                               color='gold', edgecolor='darkgoldenrod', 
                               linewidth=1.5, zorder=10)
                
                # Add scale factor label on bar (skip for handcoded baseline itself)
                if time_ms > 0 and framework != 'handcoded_jax':
                    if n_samples in baseline_times:
                        scale_factor = time_ms / baseline_times[n_samples]
                        # Format label based on value
                        if scale_factor < 10:
                            label_text = f'{scale_factor:.1f}×'
                        else:
                            label_text = f'{int(scale_factor)}×'
                        ax.annotate(label_text,
                                    xy=(x_pos, time_ms),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=10, fontweight='bold')
                    else:
                        # Debug: print if baseline is missing
                        print(f"Warning: No baseline time for n_samples={n_samples}")
        
        # Add horizontal dotted line at handcoded baseline level for visual guide
        if baseline_times:
            # Use the first (smallest) baseline time for the horizontal line
            first_baseline = baseline_times[min(baseline_times.keys())]
            ax.axhline(y=first_baseline, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
            
            # Add annotation with the baseline value
            ax.annotate(f'{first_baseline:.2f} ms', 
                       xy=(0.01, first_baseline),
                       xycoords=('axes fraction', 'data'),
                       xytext=(0, 2),
                       textcoords='offset points',
                       ha='left', va='bottom',
                       fontsize=10, 
                       color='gray',
                       alpha=0.8)
        
        ax.set_xlabel('(HMC) Chain Length', fontsize=14, fontweight='bold')
        ax.set_ylabel('Wall clock time (ms)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(n):,}' for n in n_samples_list])
        ax.grid(False)  # No grid, matching IS style
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        
        # Add "Smaller is better" text - top left
        ax.text(0.02, 0.98, 'Smaller is better', 
                transform=ax.transAxes, 
                ha='left', va='top',
                fontsize=12, style='italic', alpha=0.7)
        
        # Set y-axis limits for milliseconds scale
        ax.set_ylim(1, 1000000)  # From 10^0 to 10^6 ms
        ax.set_yticks([1, 100, 10000, 1000000])
        ax.set_yticklabels(['$10^{0}$', '$10^{2}$', '$10^{4}$', '$10^{6}$'])
        
        # Remove axis frame to match IS style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    
    # Add legend below the plot with bold font (matching IS plot)
    # Create legend handles for all frameworks that appear in the data
    all_frameworks = sorted(timing_df['framework'].unique())
    framework_order = ['genjax', 'handcoded_jax', 'numpyro', 'pyro', 'genjl', 'handcoded_torch']
    
    # Create handles and labels for frameworks in preferred order
    handles = []
    labels = []
    for fw in framework_order:
        if fw in all_frameworks:
            color = colors.get(fw, 'gray')
            handle = plt.Rectangle((0,0),1,1, fc=color, alpha=0.9, edgecolor='black', linewidth=0.5)
            handles.append(handle)
            labels.append(display_names.get(fw, fw))
        
        # Create custom legend with star for GenJAX (matching IS plot)
        custom_handles = []
        custom_labels = []
        
        for i, (handle, label) in enumerate(zip(handles, labels)):
            if label == 'Ours':
                # Create a combined legend entry with bar and star
                bar_patch = mpatches.Rectangle((0, 0), 1, 1, 
                                             facecolor=colors['genjax'], 
                                             edgecolor='black', 
                                             linewidth=0.5,
                                             alpha=0.9)
                star_marker = mlines.Line2D([], [], color='gold', marker='*', 
                                          markersize=10, markeredgecolor='darkgoldenrod',
                                          markeredgewidth=1, linestyle='None')
                custom_handles.append((bar_patch, star_marker))
                custom_labels.append(label)
            else:
                custom_handles.append(handle)
                custom_labels.append(label)
        
        # Single row legend with smaller font
        n_items = len(custom_labels)
        
        fig.legend(custom_handles, custom_labels, loc='lower center', ncol=n_items, 
                  bbox_to_anchor=(0.5, -0.10), frameon=True, fancybox=True, 
                  shadow=True, fontsize=12, prop={'size': 12},
                  handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0)})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Reduced space for legend
    
    # Save figures with explicit naming
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'benchmark_timings_hmc_all_frameworks.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'benchmark_timings_hmc_all_frameworks.png', dpi=300, bbox_inches='tight')
    print(f"HMC benchmark plots saved to {output_dir}/benchmark_timings_hmc_all_frameworks.{{pdf,png}}")
    plt.close()


def create_plots(df, output_dir):
    """Create comparison plots."""
    # Use same theme as faircoin figure
    import seaborn as sns
    sns.set_style("white")
    plt.rcParams.update({"font.size": 18})
    import matplotlib.legend_handler
    
    # Filter IS results for specific particle counts (guard if column absent)
    if 'method' not in df.columns or 'IS' not in df['method'].unique() or 'n_particles' not in df.columns:
        print("No IS results to plot")
        return

    is_df = df[df['method'] == 'IS'].copy()
    is_df = is_df[is_df['n_particles'].isin([1000, 5000, 10000])]
    
    if len(is_df) == 0:
        print("No IS results to plot")
        return
    
    # Create figure - single plot (reduced height)
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 3.5))
    fig.patch.set_facecolor('white')  # Ensure white background
    ax2.set_facecolor('white')  # Ensure axes background is white
    
    # Colors for each framework - matching faircoin color scheme with distinctive palette
    colors = {
        'genjax': 'deepskyblue',         # Same as faircoin
        'handcoded_jax': 'gold',         # Same as faircoin (handcoded baseline)
        'numpyro': 'coral',              # Same as faircoin
        'pyro': 'mediumseagreen',        # Distinctive green
        'genjl': '#AA3377',              # Purple/Red (keeping original)
        'genjl_optimized': '#CC77AA',    # Lighter Purple (static+Map)
        'genjl_dynamic': 'darkblue',     # Dark blue as requested
        'handcoded_torch': 'darkslategray'  # Distinctive dark gray-blue
    }
    
    # Display names for frameworks
    display_names = {
        'genjax': 'Ours',
        'numpyro': 'NumPyro',
        'handcoded_jax': 'Handcoded JAX',
        'genjl': 'Gen.jl',
        'genjl_optimized': 'Gen.jl (static+Map)',
        'genjl_dynamic': 'Gen.jl',
        'pyro': 'Pyro',
        'handcoded_torch': 'Handcoded PyTorch'
    }
    
    # Get timing data for all frameworks
    timing_data = []
    baseline_times = {}  # Store handcoded baseline times
    
    for n_particles in is_df['n_particles'].unique():
        subset = is_df[is_df['n_particles'] == n_particles]
        if len(subset) > 0:
            # Find handcoded baseline time
            handcoded_time = subset[subset['framework'] == 'handcoded_jax']['mean_time'].values
            if len(handcoded_time) > 0:
                baseline_times[n_particles] = handcoded_time[0] * 1000  # Convert to ms
                
            for _, row in subset.iterrows():
                if not np.isnan(row['mean_time']):
                    timing_data.append({
                        'framework': row['framework'],
                        'n_particles': n_particles,
                        'time_ms': row['mean_time'] * 1000  # Convert to ms
                    })
    
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        
        # Create grouped bar chart (exclude handcoded_jax as it's the baseline)
        # Sort bars by performance (fastest to slowest) for each particle count
        n_particles_list = sorted(timing_df['n_particles'].unique())
        x = np.arange(len(n_particles_list))
        
        # Process each particle count separately to sort by performance
        for idx, n_particles in enumerate(n_particles_list):
            # Get data for this particle count
            particle_data = timing_df[timing_df['n_particles'] == n_particles]
            
            # Sort by time (fastest first)
            particle_data = particle_data.sort_values('time_ms')
            
            # Don't filter out handcoded baseline - we'll show it as a bar
            # particle_data = particle_data[particle_data['framework'] != 'handcoded_jax']
            
            # Calculate bar width based on number of frameworks
            n_frameworks = len(particle_data)
            if n_frameworks == 0:
                continue
            width = 0.8 / n_frameworks
            
            
            # Plot bars for this particle count
            for i, (_, row) in enumerate(particle_data.iterrows()):
                framework = row['framework']
                time_ms = row['time_ms']
                
                # Position bar
                offset = (i - n_frameworks/2 + 0.5) * width
                x_pos = x[idx] + offset
                
                # Only label the first occurrence of each framework
                label = display_names.get(framework, framework) if idx == 0 else None
                
                # Simple bars for all frameworks
                bar = ax2.bar(x_pos, time_ms, width,
                             label=label,
                             color=colors.get(framework, 'gray'), 
                             alpha=0.9, edgecolor='black', linewidth=0.5)
                
                # Add gold star at bottom of GenJAX bars
                if framework == 'genjax':
                    ax2.scatter(x_pos, 0.018, marker='*', s=120, 
                               color='gold', edgecolor='darkgoldenrod', 
                               linewidth=1, zorder=10)
                
                # Add scale factor label on bar (skip for handcoded baseline itself)
                if time_ms > 0 and n_particles in baseline_times and framework != 'handcoded_jax':
                    scale_factor = time_ms / baseline_times[n_particles]
                    # Format label based on value
                    if scale_factor < 10:
                        label_text = f'{scale_factor:.1f}×'
                    else:
                        label_text = f'{int(scale_factor)}×'
                    ax2.annotate(label_text,
                                xy=(x_pos, time_ms),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10, fontweight='bold')
        
        # Add horizontal dotted line at handcoded baseline level for visual guide
        if baseline_times:
            # Use the first (smallest) baseline time for the horizontal line
            first_baseline = baseline_times[min(baseline_times.keys())]
            ax2.axhline(y=first_baseline, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
            
            # Add annotation with the baseline value
            ax2.annotate(f'{first_baseline:.2f} ms', 
                        xy=(0.01, first_baseline),
                        xycoords=('axes fraction', 'data'),
                        xytext=(0, 2),
                        textcoords='offset points',
                        ha='left', va='bottom',
                        fontsize=10, 
                        color='gray',
                        alpha=0.8)
        
        ax2.set_xlabel('(Importance Sampling) Number of Particles', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Wall clock time (ms)', fontsize=14, fontweight='bold')
        # Remove title for paper integration
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{int(n):,}' for n in n_particles_list])
        # Remove individual legend - will add single legend at bottom
        ax2.grid(False)  # No grid, matching faircoin style
        ax2.set_yscale('log')
        ax2.tick_params(labelsize=12)
        
        
        # Add "Smaller is better" text - top left
        ax2.text(0.02, 0.98, 'Smaller is better', 
                transform=ax2.transAxes, 
                ha='left', va='top',
                fontsize=12, style='italic', alpha=0.7)
        
        # Set y-axis limits for milliseconds scale
        ax2.set_ylim(0.01, 10000)  # From 0.01ms to 10000ms
        ax2.set_yticks([0.01, 1, 100, 10000])
        ax2.set_yticklabels(['$10^{-2}$', '$10^{0}$', '$10^{2}$', '$10^{4}$'])
        
        # Remove axis frame to match faircoin style
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(True)  # Keep x-axis line
        ax2.spines['left'].set_visible(True)  # Keep y-axis line for log scale
    
    # Remove the big title at the top
    
    # Add single legend below the plot with bold font
    # Get handles and labels from the bar plot
    handles, labels = ax2.get_legend_handles_labels()
    
    # Sort to match the framework order we want
    framework_order = ['genjax', 'handcoded_jax', 'numpyro', 'pyro', 'genjl_dynamic', 'handcoded_torch']
    sorted_handles_labels = []
    for fw in framework_order:
        display_name = display_names.get(fw, fw)
        if display_name in labels:
            idx = labels.index(display_name)
            sorted_handles_labels.append((handles[idx], labels[idx]))
    
    if sorted_handles_labels:
        handles, labels = zip(*sorted_handles_labels)
    
    # Create custom legend with star for GenJAX
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    custom_handles = []
    custom_labels = []
    
    for i, (handle, label) in enumerate(zip(handles, labels)):
        if label == 'Ours':
            # Create a combined legend entry with bar and star
            bar_patch = mpatches.Rectangle((0, 0), 1, 1, 
                                         facecolor=colors['genjax'], 
                                         edgecolor='black', 
                                         linewidth=0.5,
                                         alpha=0.9)
            star_marker = mlines.Line2D([], [], color='gold', marker='*', 
                                      markersize=10, markeredgecolor='darkgoldenrod',
                                      markeredgewidth=1, linestyle='None')
            custom_handles.append((bar_patch, star_marker))
            custom_labels.append(label)
        else:
            custom_handles.append(handle)
            custom_labels.append(label)
    
    # Single row legend with smaller font
    n_items = len(custom_labels)
    
    fig.legend(custom_handles, custom_labels, loc='lower center', ncol=n_items, 
              bbox_to_anchor=(0.5, -0.10), frameon=True, fancybox=True, 
              shadow=True, fontsize=12, prop={'size': 12},
              handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0)})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Reduced space for legend
    
    # Save figures with explicit naming
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'benchmark_timings_is_all_frameworks.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'benchmark_timings_is_all_frameworks.png', dpi=300, bbox_inches='tight')
    print(f"IS benchmark plots saved to {output_dir}/benchmark_timings_is_all_frameworks.{{pdf,png}}")
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
            'genjax': 'Ours',
            'numpyro': 'NumPyro',
            'handcoded_jax': 'Handcoded JAX',
            'genjl': 'Gen.jl',
            'genjl_optimized': 'Gen.jl (optimized)',
            'genjl_dynamic': 'Gen.jl',
            'pyro': 'Pyro',
            'handcoded_torch': 'Handcoded PyTorch'
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
                        default=["genjax", "numpyro", "handcoded_jax", "pyro", "genjl", "genjl_dynamic", "handcoded_torch"],
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
    
    # Create HMC comparison plot if HMC data exists
    create_hmc_comparison_plot(df, output_dir)
    
    # Create LaTeX table
    tex_file = output_dir / "benchmark_table.tex"
    create_latex_table(df, tex_file)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - All Frameworks Benchmark Results")
    print("="*80)
    
    is_df = df[df['method'] == 'IS'] if 'method' in df.columns else pd.DataFrame()
    if 'n_particles' in is_df.columns and not is_df.empty:
        for n_particles in sorted(is_df['n_particles'].unique()):
            print(f"\nN = {n_particles:,} particles:")
            subset = is_df[is_df['n_particles'] == n_particles].sort_values('mean_time')
            
            if len(subset) > 0:
                fastest_time = subset['mean_time'].min()
                
                # Get display names
                display_names_local = {
                    'genjax': 'Ours',
                    'numpyro': 'NumPyro',
                    'handcoded_jax': 'Handcoded JAX',
                    'genjl': 'Gen.jl',
                    'genjl_optimized': 'Gen.jl (optimized)',
                    'genjl_dynamic': 'Gen.jl',
                    'pyro': 'Pyro',
                    'handcoded_torch': 'Handcoded PyTorch'
                }
                
                print(f"{'Framework':<20} {'Time (ms)':<15} {'Speedup':<15}")
                print("-" * 50)
                
                for _, row in subset.iterrows():
                    if not np.isnan(row['mean_time']):
                        time_ms = row['mean_time'] * 1000
                        speedup = row['mean_time'] / fastest_time
                        framework_name = display_names_local.get(row['framework'], row['framework'])
                        print(f"{framework_name:<20} {time_ms:<15.3f} {speedup:<15.1f}x")
    else:
        print("\nNo IS results available for summary.")


if __name__ == "__main__":
    main()
