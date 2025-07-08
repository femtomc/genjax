"""Handcoded PyTorch GMM benchmarks."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .gmm_torch_benchmarks import time_torch_gmm


def main():
    parser = argparse.ArgumentParser(description="Run handcoded PyTorch GMM benchmarks")
    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"],
        help="Device to run PyTorch benchmarks on"
    )
    
    # GMM arguments
    parser.add_argument(
        "--data-sizes", nargs="+", type=int,
        default=[1000, 10000, 50000, 100000],
        help="Data sizes for GMM"
    )
    parser.add_argument(
        "--gmm-steps", type=int, default=10,
        help="Number of GMM inference steps per timing"
    )
    
    parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of timing repetitions"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/handcoded_torch",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run GMM benchmarks
    print(f"Running handcoded PyTorch GMM benchmarks on {args.device.upper()}...")
    gmm_results = []
    for n_data in args.data_sizes:
        print(f"  Data size: {n_data:,}")
        result = time_torch_gmm(n_data, args.gmm_steps, args.repeats, args.device)
        gmm_results.append(result)
        
        # Save individual result
        result_file = output_dir / f"gmm_n{n_data}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
    
    results["gmm"] = gmm_results
    
    # Save summary
    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()