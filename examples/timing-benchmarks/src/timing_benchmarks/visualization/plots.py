"""Visualization functions for timing benchmark results.

This module creates publication-quality figures comparing performance
across frameworks and configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import os

# Import shared visualization utilities
# Add path to genjax/examples directory to import shared utils
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(examples_dir)
from viz import (
    setup_publication_fonts,
    FIGURE_SIZES,
    get_method_color,
    apply_grid_style,
    apply_standard_ticks,
    save_publication_figure,
    PRIMARY_COLORS,
    MARKER_SPECS,
    LINE_SPECS,
    ALPHA_VALUES,
)

# Import export utilities
from ..export.results import load_benchmark_results


# Color mapping for frameworks
FRAMEWORK_COLORS = {
    "genjax": PRIMARY_COLORS["genjax_is"],      # GenJAX blue
    "gen.jl": PRIMARY_COLORS["genjax_hmc"],     # Gen.jl orange  
    "numpyro": PRIMARY_COLORS["numpyro_hmc"],   # NumPyro green
    "pyro": PRIMARY_COLORS["pyro_vi"],          # Pyro red
    "stan": "#956CB4",                          # Stan purple
    "handcoded": "#8C613C",                     # Handcoded JAX brown
}


def create_is_comparison_plot(
    results: Dict[str, Any],
    save_dir: str = None,
    show_error_bars: bool = True
) -> None:
    """Plot importance sampling timing comparison across frameworks.
    
    Creates a horizontal grouped bar chart comparing execution times for different
    particle counts across frameworks.
    
    Args:
        results: Loaded benchmark results
        save_dir: Directory to save figures (defaults to timing-benchmarks/figs)
        show_error_bars: Whether to show standard deviation error bars
    """
    if save_dir is None:
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(module_dir, "figs")
    
    setup_publication_fonts()
    
    # Extract IS summary data
    if "is_summary" not in results:
        print("No IS comparison data found")
        return
    
    df = results["is_summary"]
    
    # Create figure with horizontal layout
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for plotting
    frameworks = df["framework"].unique()
    configs = df["config"].unique()
    
    # Set up bar positions
    y = np.arange(len(configs))
    height = 0.8 / len(frameworks)
    
    # Plot horizontal bars for each framework
    for i, framework in enumerate(frameworks):
        framework_data = df[df["framework"] == framework]
        means = []
        stds = []
        
        for config in configs:
            config_data = framework_data[framework_data["config"] == config]
            if not config_data.empty:
                means.append(config_data["mean_time"].values[0])
                stds.append(config_data["std_time"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(frameworks)/2 + 0.5) * height
        bars = ax.barh(
            y + offset,
            means,
            height,
            label=framework.upper(),
            color=FRAMEWORK_COLORS.get(framework, "gray"),
            xerr=stds if show_error_bars else None,
            capsize=5,
            alpha=ALPHA_VALUES["bar_charts"]
        )
    
    # Customize plot
    ax.set_ylabel("Number of Particles", fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_yticks(y)
    
    # Extract particle counts from config names (e.g., "n1000" -> "1000")
    particle_labels = [config.replace("n", "") for config in configs]
    ax.set_yticklabels(particle_labels)
    
    # Apply standard styling
    apply_grid_style(ax, style="x_only")  # Only x-axis grid for horizontal bars
    apply_standard_ticks(ax)
    ax.legend(loc="upper right", fontsize=16)
    
    # Log scale for x-axis if range is large
    if df["mean_time"].max() / df["mean_time"].min() > 10:
        ax.set_xscale("log")
        ax.set_xlabel("Time (seconds, log scale)", fontweight="bold")
    
    # Save figure
    save_publication_figure(
        fig,
        f"{save_dir}/timing_benchmarks_is_comparison.pdf"
    )
    # Also save as PNG for viewing
    fig.savefig(
        f"{save_dir}/timing_benchmarks_is_comparison.png",
        dpi=300, bbox_inches="tight", format="png"
    )


def create_hmc_comparison_plot(
    results: Dict[str, Any],
    save_dir: str = None,
    show_error_bars: bool = True
) -> None:
    """Plot HMC timing comparison across frameworks.
    
    Creates a horizontal bar chart comparing HMC execution times.
    
    Args:
        results: Loaded benchmark results
        save_dir: Directory to save figures
        show_error_bars: Whether to show standard deviation error bars
    """
    if save_dir is None:
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(module_dir, "figs")
    
    setup_publication_fonts()
    
    # Extract HMC summary data
    if "hmc_summary" not in results:
        print("No HMC comparison data found")
        return
    
    df = results["hmc_summary"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Prepare data for plotting
    frameworks = df["framework"].values
    means = df["mean_time"].values
    stds = df["std_time"].values
    
    # Create horizontal bars
    y_pos = np.arange(len(frameworks))
    bars = ax.barh(
        y_pos,
        means,
        xerr=stds if show_error_bars else None,
        capsize=5,
        alpha=0.8
    )
    
    # Color bars by framework
    for i, (bar, framework) in enumerate(zip(bars, frameworks)):
        bar.set_color(FRAMEWORK_COLORS.get(framework, "gray"))
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.upper() for f in frameworks])
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    
    # Apply standard styling
    apply_grid_style(ax)
    apply_standard_ticks(ax)
    
    # Save figure
    save_publication_figure(
        fig,
        f"{save_dir}/timing_benchmarks_hmc_comparison.pdf"
    )
    # Also save as PNG for viewing
    fig.savefig(
        f"{save_dir}/timing_benchmarks_hmc_comparison.png",
        dpi=300, bbox_inches="tight", format="png"
    )


def create_scaling_analysis_plot(
    results: Dict[str, Any],
    save_dir: str = None,
    log_scale: bool = True
) -> None:
    """Plot scaling analysis showing runtime vs problem size.
    
    Creates a line plot showing how runtime scales with data points
    for different frameworks and methods.
    
    Args:
        results: Loaded benchmark results with scaling data
        save_dir: Directory to save figures
        log_scale: Whether to use log-log scale
    """
    setup_publication_fonts()
    
    # This requires multiple runs with different data sizes
    # For now, create a placeholder that can be filled with real data
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_large"])
    
    # Example scaling data (would come from multiple benchmark runs)
    data_sizes = [10, 50, 100, 500, 1000]
    
    # Placeholder for actual scaling results
    # In practice, this would be extracted from multiple benchmark runs
    
    ax.set_xlabel("Number of Data Points", fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontweight="bold")
    
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Data Points (log scale)", fontweight="bold")
        ax.set_ylabel("Time (seconds, log scale)", fontweight="bold")
    
    apply_grid_style(ax)
    ax.legend(loc="upper left", fontsize=16)
    
    save_publication_figure(
        fig,
        f"{save_dir}/timing_benchmarks_scaling_analysis.pdf"
    )
    # Also save as PNG for viewing
    fig.savefig(
        f"{save_dir}/timing_benchmarks_scaling_analysis.png",
        dpi=300, bbox_inches="tight", format="png"
    )


def create_speedup_ratios_plot(
    results: Dict[str, Any],
    baseline_framework: str = "gen.jl",
    save_dir: str = None
) -> None:
    """Plot speedup ratios relative to a baseline framework.
    
    Shows how much faster/slower each framework is compared to baseline.
    
    Args:
        results: Loaded benchmark results
        baseline_framework: Framework to use as baseline (1.0x)
        save_dir: Directory to save figures
    """
    if save_dir is None:
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(module_dir, "figs")
    
    setup_publication_fonts()
    
    # Extract IS summary data for speedup calculation
    if "is_summary" not in results:
        print("No comparison data found for speedup analysis")
        return
    
    df = results["is_summary"]
    
    # Check if we have multiple frameworks for comparison
    unique_frameworks = df["framework"].unique()
    if len(unique_frameworks) == 1:
        print(f"Only one framework found ({unique_frameworks[0]}), skipping speedup ratios")
        return
    
    # Create figure with subplots for different particle counts
    configs = df["config"].unique()
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(4 * n_configs, 5), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for idx, (ax, config) in enumerate(zip(axes, configs)):
        config_data = df[df["config"] == config]
        
        # Get baseline time
        baseline_data = config_data[config_data["framework"] == baseline_framework]
        if baseline_data.empty:
            continue
        
        baseline_time = baseline_data["mean_time"].values[0]
        
        # Calculate speedups
        frameworks = []
        speedups = []
        
        for _, row in config_data.iterrows():
            frameworks.append(row["framework"])
            speedups.append(baseline_time / row["mean_time"])
        
        # Create bar chart
        y_pos = np.arange(len(frameworks))
        bars = ax.barh(y_pos, speedups, alpha=0.8)
        
        # Color bars
        for i, (bar, framework) in enumerate(zip(bars, frameworks)):
            bar.set_color(FRAMEWORK_COLORS.get(framework, "gray"))
        
        # Add vertical line at 1.0
        ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
        
        # Customize
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels([f.upper() for f in frameworks])
        ax.set_xlabel("Speedup vs " + baseline_framework.upper(), fontweight="bold")
        
        # Extract particle count for title
        particles = config.replace("n", "")
        ax.set_title(f"{particles} Particles", fontsize=18)
        
        apply_grid_style(ax)
    
    plt.tight_layout()
    save_publication_figure(
        fig,
        f"{save_dir}/timing_benchmarks_speedup_ratios.pdf"
    )
    # Also save as PNG for viewing
    fig.savefig(
        f"{save_dir}/timing_benchmarks_speedup_ratios.png",
        dpi=300, bbox_inches="tight", format="png"
    )


def create_method_comparison_grid(
    results: Dict[str, Any],
    save_dir: str = None
) -> None:
    """Create a grid comparing IS and HMC performance.
    
    2x2 grid showing:
    - IS timing comparison
    - HMC timing comparison
    - IS effective sample size (if available)
    - HMC diagnostics (if available)
    
    Args:
        results: Loaded benchmark results
        save_dir: Directory to save figures
    """
    if save_dir is None:
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(module_dir, "figs")
    
    setup_publication_fonts()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIGURE_SIZES["framework_comparison"]
    )
    
    # Top left: IS timing (horizontal bars)
    if "is_summary" in results:
        df_is = results["is_summary"]
        # Simplified bar plot for the grid
        frameworks = df_is["framework"].unique()
        
        # Average across all particle counts
        avg_times = df_is.groupby("framework")["mean_time"].mean()
        
        bars = ax1.barh(
            range(len(avg_times)),
            avg_times.values,
            color=[FRAMEWORK_COLORS.get(f, "gray") for f in avg_times.index],
            alpha=ALPHA_VALUES["bar_charts"]
        )
        ax1.set_yticks(range(len(avg_times)))
        ax1.set_yticklabels([f.upper() for f in avg_times.index])
        ax1.set_xlabel("Avg Time (s)", fontweight="bold")
        ax1.set_title("Importance Sampling", fontsize=18)
        apply_grid_style(ax1, style="x_only")
        apply_standard_ticks(ax1)
    
    # Top right: HMC timing (horizontal bars)
    if "hmc_summary" in results:
        df_hmc = results["hmc_summary"]
        frameworks = df_hmc["framework"].values
        times = df_hmc["mean_time"].values
        
        bars = ax2.barh(
            range(len(frameworks)),
            times,
            color=[FRAMEWORK_COLORS.get(f, "gray") for f in frameworks],
            alpha=ALPHA_VALUES["bar_charts"]
        )
        ax2.set_yticks(range(len(frameworks)))
        ax2.set_yticklabels([f.upper() for f in frameworks])
        ax2.set_xlabel("Time (s)", fontweight="bold")
        ax2.set_title("HMC", fontsize=18)
        apply_grid_style(ax2, style="x_only")
        apply_standard_ticks(ax2)
    
    # Bottom panels: Placeholder for quality metrics
    ax3.text(0.5, 0.5, "IS Quality Metrics\n(ESS, Log ML)",
             ha="center", va="center", fontsize=16, alpha=0.5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    
    ax4.text(0.5, 0.5, "HMC Diagnostics\n(Accept Rate, R-hat)",
             ha="center", va="center", fontsize=16, alpha=0.5)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    
    plt.tight_layout()
    save_publication_figure(
        fig,
        f"{save_dir}/timing_benchmarks_method_comparison_grid.pdf"
    )
    # Also save as PNG for viewing
    fig.savefig(
        f"{save_dir}/timing_benchmarks_method_comparison_grid.png",
        dpi=300, bbox_inches="tight", format="png"
    )


def create_all_figures(
    exp_dir: str,
    save_dir: str = None
) -> None:
    """Create all benchmark figures from experiment directory.
    
    Args:
        exp_dir: Path to experiment directory with results
        save_dir: Directory to save figures (defaults to timing-benchmarks/figs)
    """
    if save_dir is None:
        # Default to timing-benchmarks/figs directory
        import os
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(module_dir, "figs")
    
    # Load results
    results = load_benchmark_results(exp_dir)
    
    # Create output directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Generate all figures
    print("Creating IS timing comparison...")
    create_is_comparison_plot(results, save_dir)
    
    print("Creating HMC timing comparison...")
    create_hmc_comparison_plot(results, save_dir)
    
    print("Creating speedup ratios...")
    create_speedup_ratios_plot(results, save_dir)
    
    print("Creating method comparison grid...")
    create_method_comparison_grid(results, save_dir)
    
    print(f"All figures saved to {save_dir}/")


# Figure specification for documentation
FIGURE_SPECIFICATIONS = {
    "timing_benchmarks_is_comparison.pdf": {
        "description": "Grouped bar chart comparing IS performance across frameworks",
        "dimensions": FIGURE_SIZES["framework_comparison"],
        "content": "Execution time vs particle count for each framework"
    },
    "timing_benchmarks_hmc_comparison.pdf": {
        "description": "Horizontal bar chart comparing HMC performance",
        "dimensions": FIGURE_SIZES["single_medium"],
        "content": "HMC execution time for each framework"
    },
    "timing_benchmarks_speedup_ratios.pdf": {
        "description": "Speedup ratios relative to baseline framework",
        "dimensions": "Variable based on configs",
        "content": "Relative performance (speedup factor) for each configuration"
    },
    "timing_benchmarks_method_comparison_grid.pdf": {
        "description": "2x2 grid comparing IS and HMC methods",
        "dimensions": FIGURE_SIZES["framework_comparison"],
        "content": "Overview of timing and quality metrics"
    },
    "timing_benchmarks_scaling_analysis.pdf": {
        "description": "Log-log plot of runtime vs problem size",
        "dimensions": FIGURE_SIZES["single_large"],
        "content": "Scaling behavior for different frameworks"
    }
}