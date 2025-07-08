"""Main script to run handcoded benchmarks."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.timing_benchmarks.handcoded_benchmarks.gol_benchmarks import run_gol_benchmarks
from src.timing_benchmarks.handcoded_benchmarks.gmm_benchmarks import run_gmm_benchmarks
from src.timing_benchmarks.handcoded_benchmarks.visualization import (
    create_benchmark_comparison_plot, save_benchmark_results_csv
)

# Try to import PyTorch benchmarks (may not be available in all environments)
try:
    from src.timing_benchmarks.handcoded_benchmarks.gol_torch_benchmarks import run_torch_gol_benchmarks
    from src.timing_benchmarks.handcoded_benchmarks.gmm_torch_benchmarks import run_torch_gmm_benchmarks
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available in this environment. Skipping PyTorch benchmarks.")


def main():
    parser = argparse.ArgumentParser(description="Run handcoded benchmarks")
    parser.add_argument("--benchmark", choices=["gol", "gmm", "all"], default="all",
                       help="Which benchmark to run")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                       help="Device to run PyTorch benchmarks on")
    parser.add_argument("--repeats", type=int, default=100,
                       help="Number of timing repetitions")
    parser.add_argument("--output-dir", default="figs",
                       help="Output directory for plots")
    
    # GOL specific arguments
    parser.add_argument("--grid-sizes", nargs="+", type=int, 
                       default=[10, 50, 100, 200],
                       help="Grid sizes for Game of Life")
    parser.add_argument("--gol-steps", type=int, default=10,
                       help="Number of GOL steps per timing")
    parser.add_argument("--flip-prob", type=float, default=0.03,
                       help="Probability of rule violation in GOL")
    
    # GMM specific arguments
    parser.add_argument("--data-sizes", nargs="+", type=int,
                       default=[100, 1000, 10000, 100000],
                       help="Data sizes for GMM")
    parser.add_argument("--gmm-steps", type=int, default=10,
                       help="Number of GMM inference steps per timing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # Run benchmarks
    if args.benchmark in ["gol", "all"]:
        print("="*60)
        print("Running Game of Life Benchmarks")
        print("="*60)
        
        # Run JAX benchmarks
        gol_results = run_gol_benchmarks(
            grid_sizes=args.grid_sizes,
            n_steps=args.gol_steps,
            flip_prob=args.flip_prob,
            repeats=args.repeats
        )
        
        # Run PyTorch benchmarks if available
        if TORCH_AVAILABLE and args.device:
            torch_gol_results = run_torch_gol_benchmarks(
                grid_sizes=args.grid_sizes,
                n_steps=args.gol_steps,
                flip_prob=args.flip_prob,
                repeats=args.repeats,
                device=args.device
            )
            # Merge results
            gol_results.update(torch_gol_results)
        
        # Create visualization
        create_benchmark_comparison_plot(
            gol_results,
            benchmark_name="gol",
            x_param="grid_size",
            x_label="Grid Size",
            output_prefix="handcoded_comparison"
        )
        
        # Save CSV
        save_benchmark_results_csv(
            gol_results,
            benchmark_name="gol",
            output_prefix="handcoded_results"
        )
    
    if args.benchmark in ["gmm", "all"]:
        print("\n" + "="*60)
        print("Running Gaussian Mixture Model Benchmarks")
        print("="*60)
        
        # Run JAX benchmarks
        gmm_results = run_gmm_benchmarks(
            data_sizes=args.data_sizes,
            n_steps=args.gmm_steps,
            repeats=args.repeats
        )
        
        # Run PyTorch benchmarks if available
        if TORCH_AVAILABLE and args.device:
            torch_gmm_results = run_torch_gmm_benchmarks(
                data_sizes=args.data_sizes,
                n_steps=args.gmm_steps,
                repeats=args.repeats,
                device=args.device
            )
            # Merge results
            gmm_results.update(torch_gmm_results)
        
        # Create visualization
        create_benchmark_comparison_plot(
            gmm_results,
            benchmark_name="gmm",
            x_param="n_data",
            x_label="Number of Data Points",
            output_prefix="handcoded_comparison"
        )
        
        # Save CSV
        save_benchmark_results_csv(
            gmm_results,
            benchmark_name="gmm",
            output_prefix="handcoded_results"
        )
    
    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()