#!/usr/bin/env python
"""Run HMC benchmarks with multiple chain lengths for all frameworks."""

import argparse
import json
import sys
import time
from pathlib import Path
import os

# Configure JAX to use GPU if available
# Try CUDA first, fall back to CPU if not available
try:
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Prevent memory preallocation
except:
    pass

import jax
import jax.numpy as jnp
import importlib.util

# Check JAX GPU availability
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import PolynomialDataset from the correct location
from timing_benchmarks.data.generation import PolynomialDataset


def load_module(framework_name):
    """Dynamically load a framework module."""
    module_path = Path(__file__).parent / "src" / "timing_benchmarks" / "curvefit-benchmarks" / f"{framework_name}.py"
    if not module_path.exists():
        print(f"Warning: Module {module_path} not found")
        return None
    
    spec = importlib.util.spec_from_file_location(f"timing_benchmarks.curvefit_benchmarks.{framework_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"timing_benchmarks.curvefit_benchmarks.{framework_name}"] = module
    spec.loader.exec_module(module)
    return module


# Import the data generation function too
from timing_benchmarks.data.generation import generate_polynomial_data


def run_framework_hmc(framework, n_samples, dataset, repeats=100, n_warmup=500, **kwargs):
    """Run HMC benchmark for a specific framework."""
    # Load the framework module
    module = load_module(framework)
    if module is None:
        return None
    
    # Get the HMC timing function
    hmc_fn_name = f"{framework}_polynomial_hmc_timing"
    if not hasattr(module, hmc_fn_name):
        print(f"Warning: {hmc_fn_name} not found in {framework} module")
        return None
    
    hmc_timing_fn = getattr(module, hmc_fn_name)
    
    # Prepare framework-specific kwargs
    if framework == "numpyro":
        # NumPyro now uses fixed n_leapfrog like other frameworks
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "n_leapfrog": kwargs.get("n_leapfrog", 20)
        }
    elif framework == "genjax":
        # GenJAX uses step_size and n_leapfrog
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "n_leapfrog": kwargs.get("n_leapfrog", 20)
        }
    elif framework == "pyro":
        # Pyro uses device parameter
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "num_steps": kwargs.get("n_leapfrog", 20),
            "device": kwargs.get("device", "cpu")
        }
    elif framework == "handcoded_torch":
        # PyTorch uses device parameter
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "n_leapfrog": kwargs.get("n_leapfrog", 20),
            "device": kwargs.get("device", "cpu")
        }
    elif framework == "genjl":
        # Gen.jl uses step_size and n_leapfrog
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "n_leapfrog": kwargs.get("n_leapfrog", 20)
        }
    else:
        # Default: pass step_size and n_leapfrog
        framework_kwargs = {
            "step_size": kwargs.get("step_size", 0.01),
            "n_leapfrog": kwargs.get("n_leapfrog", 20)
        }
    
    # Run the timing
    try:
        results = hmc_timing_fn(
            dataset=dataset,
            n_samples=n_samples,
            n_warmup=n_warmup,
            repeats=repeats,
            **framework_kwargs
        )
        return results
    except Exception as e:
        print(f"Error running {framework} HMC: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run HMC benchmarks with varying chain lengths")
    parser.add_argument("--frameworks", nargs="+", 
                       default=["genjax", "numpyro", "handcoded_tfp", "handcoded_torch", "pyro", "genjl"],
                       help="Frameworks to benchmark")
    parser.add_argument("--chain-lengths", nargs="+", type=int,
                       default=[100, 500, 1000, 5000],
                       help="HMC chain lengths to test")
    parser.add_argument("--n-points", type=int, default=50,
                       help="Number of data points")
    parser.add_argument("--repeats", type=int, default=100,
                       help="Number of timing repetitions")
    parser.add_argument("--n-warmup", type=int, default=500,
                       help="Number of HMC warmup steps")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                       help="Output directory for results")
    parser.add_argument("--step-size", type=float, default=0.01,
                       help="HMC step size")
    parser.add_argument("--n-leapfrog", type=int, default=20,
                       help="Number of leapfrog steps (GenJAX)")
    parser.add_argument("--target-accept-prob", type=float, default=0.8,
                       help="Target acceptance probability (NumPyro)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for PyTorch/Pyro (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Print GPU status
    print("\n" + "="*60)
    print("GPU Configuration:")
    print("="*60)
    print(f"Device setting: {args.device}")
    
    # Check PyTorch GPU if needed
    if args.device == "cuda" and any(f in args.frameworks for f in ["pyro", "handcoded_torch"]):
        try:
            import torch
            if torch.cuda.is_available():
                print(f"PyTorch CUDA: Available ({torch.cuda.get_device_name(0)})")
            else:
                print("PyTorch CUDA: Not available - will fall back to CPU")
        except ImportError:
            print("PyTorch: Not installed")
    
    # Generate dataset
    print(f"\nGenerating polynomial dataset with {args.n_points} points...")
    dataset = generate_polynomial_data(n_points=args.n_points)
    
    # Run benchmarks for each framework and chain length
    for framework in args.frameworks:
        print(f"\n{'='*60}")
        print(f"Running HMC benchmarks for {framework}")
        print(f"{'='*60}")
        
        framework_dir = args.output_dir / framework
        framework_dir.mkdir(parents=True, exist_ok=True)
        
        for n_samples in args.chain_lengths:
            print(f"\nChain length: {n_samples}")
            print("-" * 40)
            
            # Run benchmark
            results = run_framework_hmc(
                framework=framework,
                n_samples=n_samples,
                dataset=dataset,
                repeats=args.repeats,
                n_warmup=args.n_warmup,
                step_size=args.step_size,
                n_leapfrog=args.n_leapfrog,
                target_accept_prob=args.target_accept_prob,
                device=args.device
            )
            
            if results is not None:
                # Clean results for JSON serialization
                clean_results = {
                    k: v for k, v in results.items() 
                    if k not in ["samples", "log_weights"]
                }
                
                # Convert numpy/JAX arrays to Python types
                if "times" in clean_results:
                    clean_results["times"] = [float(t) for t in clean_results["times"]]
                if "mean_time" in clean_results:
                    clean_results["mean_time"] = float(clean_results["mean_time"])
                if "std_time" in clean_results:
                    clean_results["std_time"] = float(clean_results["std_time"])
                
                # Save results
                output_file = framework_dir / f"hmc_n{n_samples}.json"
                with open(output_file, "w") as f:
                    json.dump(clean_results, f, indent=2)
                
                print(f"✓ {framework} HMC (n={n_samples}): {results['mean_time']:.3f}s ± {results['std_time']:.3f}s")
                print(f"  Saved to: {output_file}")
            else:
                print(f"✗ {framework} HMC (n={n_samples}): Failed")
    
    print("\n" + "="*60)
    print("HMC benchmarking complete!")
    print("="*60)
    print(f"\nRun the following to generate comparison plots:")
    print(f"python combine_results.py --frameworks {' '.join(args.frameworks)}")


if __name__ == "__main__":
    main()