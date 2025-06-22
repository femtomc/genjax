"""
GenJAX Curvefit Case Study - Simplified Main Entry Point

Supports three modes:
- quick: Fast demonstration with basic visualizations
- full: Complete analysis with all visualizations
- benchmark: Framework comparison (IS 1000 vs HMC methods)
"""

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GenJAX Curvefit Case Study - Bayesian Sine Wave Parameter Estimation"
    )

    parser.add_argument(
        "mode",
        choices=["quick", "full", "benchmark", "generative", "vectorization"],
        nargs="?",
        default="quick",
        help="Analysis mode: quick (fast viz), full (complete), benchmark (compare frameworks), generative (programming figure), vectorization (patterns figure)",
    )

    # Analysis parameters
    parser.add_argument(
        "--n-points", type=int, default=10, help="Number of data points (default: 10)"
    )
    parser.add_argument(
        "--n-samples-is",
        type=int,
        default=1000,
        help="Number of importance sampling particles (default: 1000)",
    )
    parser.add_argument(
        "--n-samples-hmc",
        type=int,
        default=1000,
        help="Number of HMC samples (default: 1000)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=500,
        help="Number of HMC warmup samples (default: 500)",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=20,
        help="Timing repetitions (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    return parser.parse_args()


def run_quick_mode(args):
    """Run quick demonstration mode."""
    from examples.curvefit.figs import (
        save_onepoint_trace_viz,
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_inference_viz,
    )

    print("=== Quick Mode: Basic Visualizations ===")

    print("\n1. Generating trace visualizations...")
    save_onepoint_trace_viz()
    save_multipoint_trace_viz()
    save_four_multipoint_trace_vizs()

    print("\n2. Generating inference visualization...")
    save_inference_viz(seed=args.seed)

    print("\n✓ Quick mode complete!")
    print("Generated figures in examples/curvefit/figs/")


def run_full_mode(args):
    """Run full analysis mode."""
    from examples.curvefit.figs import (
        save_onepoint_trace_viz,
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_inference_viz,
        save_inference_scaling_viz,
        save_genjax_posterior_comparison,
        save_framework_comparison_figure,
        save_log_density_viz,
        save_parameter_posterior_methods_comparison,
    )

    print("=== Full Mode: Complete Analysis ===")

    print("\n1. Generating trace visualizations...")
    save_onepoint_trace_viz()
    save_multipoint_trace_viz()
    save_four_multipoint_trace_vizs()

    print("\n2. Generating inference scaling analysis...")
    save_inference_scaling_viz()

    print("\n3. Generating inference visualization...")
    save_inference_viz(seed=args.seed)

    print("\n4. Generating GenJAX posterior comparison...")
    save_genjax_posterior_comparison(
        n_points=args.n_points,
        n_samples_is=args.n_samples_is,
        n_samples_hmc=args.n_samples_hmc,
        n_warmup=args.n_warmup,
        seed=args.seed,
        timing_repeats=args.timing_repeats,
    )

    print("\n5. Generating framework comparison...")
    print(
        f"   Parameters: {args.n_points} points, IS {args.n_samples_is} particles, HMC {args.n_samples_hmc} samples"
    )

    save_framework_comparison_figure(
        n_points=args.n_points,
        n_samples_is=args.n_samples_is,
        n_samples_hmc=args.n_samples_hmc,
        n_warmup=args.n_warmup,
        seed=args.seed,
        timing_repeats=args.timing_repeats,
    )

    print("\n6. Generating density visualizations...")
    save_log_density_viz()

    print("\n7. Generating parameter posterior methods comparison...")
    save_parameter_posterior_methods_comparison(seed=args.seed)

    print("\n✓ Full mode complete!")
    print("Generated figures in examples/curvefit/figs/")


def run_benchmark_mode(args):
    """Run benchmark comparison mode."""
    from examples.curvefit.figs import save_framework_comparison_figure

    print("=== Benchmark Mode: Framework Comparison ===")
    print("Parameters:")
    print(f"  - Data points: {args.n_points}")
    print(f"  - IS particles: {args.n_samples_is}")
    print(f"  - HMC samples: {args.n_samples_hmc}")
    print(f"  - HMC warmup: {args.n_warmup}")
    print(f"  - Timing repeats: {args.timing_repeats}")
    print(f"  - Random seed: {args.seed}")

    results = save_framework_comparison_figure(
        n_points=args.n_points,
        n_samples_is=args.n_samples_is,
        n_samples_hmc=args.n_samples_hmc,
        n_warmup=args.n_warmup,
        seed=args.seed,
        timing_repeats=args.timing_repeats,
    )

    print("\n=== Benchmark Summary ===")
    for method_key, result in results.items():
        mean_time = result["timing"][0] * 1000
        std_time = result["timing"][1] * 1000
        print(f"{result['method']}: {mean_time:.1f} ± {std_time:.1f} ms")
        if "accept_rate" in result:
            print(f"  Accept rate: {result['accept_rate']:.3f}")

    print("\n✓ Benchmark complete!")
    print("Generated comparison figure in examples/curvefit/figs/")


def run_generative_mode(args):
    """Run generative programming figure mode."""
    from examples.curvefit.figs import save_programming_with_generative_functions_figure

    print("=== Generative Mode: Programming with Generative Functions Figure ===")

    print("\nGenerating programming with generative functions figure...")
    save_programming_with_generative_functions_figure()

    print("\n✓ Generative mode complete!")
    print("Generated figure in examples/curvefit/figs/")


def run_vectorization_mode(args):
    """Run vectorization patterns figure mode."""
    from examples.curvefit.figs import save_vectorization_patterns_figure

    print("=== Vectorization Mode: Two Natural Vectorization Patterns Figure ===")

    print("\nGenerating vectorization patterns figure...")
    save_vectorization_patterns_figure()

    print("\n✓ Vectorization mode complete!")
    print("Generated figure in examples/curvefit/figs/")


def main():
    """Main entry point."""
    args = parse_args()

    print("\n🚀 GenJAX Curvefit Case Study")
    print(f"Mode: {args.mode}")

    if args.mode == "quick":
        run_quick_mode(args)
    elif args.mode == "full":
        run_full_mode(args)
    elif args.mode == "benchmark":
        run_benchmark_mode(args)
    elif args.mode == "generative":
        run_generative_mode(args)
    elif args.mode == "vectorization":
        run_vectorization_mode(args)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
