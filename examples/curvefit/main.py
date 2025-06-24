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
        choices=["quick", "full", "benchmark", "generative", "vectorization", "outlier", "is-only", "scaling"],
        nargs="?",
        default="quick",
        help="Analysis mode: quick (fast viz), full (complete), benchmark (compare frameworks), generative (programming figure), vectorization (patterns figure), outlier (generative conditionals), is-only (IS comparison only), scaling (inference scaling analysis)",
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
    
    # Outlier model parameters
    parser.add_argument(
        "--outlier-rate", type=float, default=0.2, help="Prior outlier probability (default: 0.2)"
    )
    parser.add_argument(
        "--outlier-mean", type=float, default=0.0, help="Outlier distribution mean (default: 0.0)"
    )
    parser.add_argument(
        "--outlier-std", type=float, default=5.0, help="Outlier distribution std dev (default: 5.0)"
    )
    parser.add_argument(
        "--outlier-comprehensive", action="store_true", 
        help="Run comprehensive outlier analysis with all figures"
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
        save_individual_method_parameter_density,
        save_is_comparison_parameter_density,
        save_is_single_resample_comparison,
        save_is_only_timing_comparison,
        save_is_only_parameter_density,
        create_all_legends,
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

    print("\n7. Generating individual method parameter density figures...")
    save_individual_method_parameter_density(
        n_points=args.n_points,
        n_samples=args.n_samples_is,
        seed=args.seed
    )

    print("\n8. Generating IS comparison parameter density figures...")
    save_is_comparison_parameter_density(
        n_points=args.n_points,
        seed=args.seed
    )

    print("\n9. Generating IS single particle resampling comparison...")
    save_is_single_resample_comparison(
        n_points=args.n_points,
        seed=args.seed,
        n_trials=1000
    )

    print("\n10. Generating IS-only timing comparison...")
    save_is_only_timing_comparison(
        n_points=args.n_points,
        seed=args.seed,
        timing_repeats=args.timing_repeats
    )

    print("\n11. Generating IS-only parameter density figures...")
    save_is_only_parameter_density(
        n_points=args.n_points,
        seed=args.seed
    )

    print("\n12. Creating all legends...")
    create_all_legends()

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
    print("=== Generative Mode: Programming with Generative Functions Figure ===")
    print("\n⚠️  This mode has been removed during cleanup.")
    print("The 'programming with generative functions' plotting code was removed as requested.")


def run_vectorization_mode(args):
    """Run vectorization patterns figure mode."""
    print("=== Vectorization Mode: Two Natural Vectorization Patterns Figure ===")
    print("\n⚠️  This mode has been removed during cleanup.")
    print("The 'vectorization patterns' plotting code was removed as requested.")


def run_outlier_mode(args):
    """Run outlier model experiments for generative conditionals validation."""
    from examples.curvefit.figs import (
        # Outlier-specific visualizations
        save_outlier_conditional_demo,
        save_outlier_trace_viz,
        save_outlier_inference_viz_beta,
        save_outlier_posterior_comparison_beta,
        save_outlier_data_viz,
        save_outlier_inference_comparison,
        save_outlier_indicators_viz,
        save_outlier_method_comparison,
        save_outlier_framework_comparison,
        save_outlier_parameter_posterior_histogram,
        save_outlier_scaling_study,
        save_outlier_rate_sensitivity,
    )
    
    print("=== Outlier Mode: Demonstrating Robust Curve Fitting with Generative Conditionals ===")
    print("Model features:")
    print("  - GenJAX Cond combinator for mixture modeling")
    print("  - Automatic outlier detection")
    print("  - Robustness comparison vs standard model")
    print("\nParameters:")
    print(f"  - Data points: {args.n_points}")
    print(f"  - Outlier rate: {args.outlier_rate}")
    print(f"  - IS particles: {args.n_samples_is}")
    print(f"  - Random seed: {args.seed}")
    
    # Generate the main demo figure
    save_outlier_conditional_demo(
        n_points=args.n_points,
        outlier_rate=args.outlier_rate,
        seed=args.seed,
        n_samples_is=args.n_samples_is,
    )
    
    print("\n✓ Outlier mode complete!")
    print("Generated figure demonstrates:")
    print("  - GenJAX's Cond combinator for natural mixture modeling")
    print("  - Improved robustness with automatic outlier detection")
    print("  - Zero-cost abstraction (compiles to efficient jnp.where)")
    print("  - Seamless integration with all inference algorithms")


def run_is_only_mode(args):
    """Run IS-only comparison mode."""
    from examples.curvefit.figs import (
        save_is_only_timing_comparison,
        save_is_only_parameter_density,
    )

    print("=== IS-Only Mode: Importance Sampling Comparison ===")
    print("Comparing IS methods with N=5, N=1000, and N=5000 particles")
    print(f"Parameters:")
    print(f"  - Data points: {args.n_points}")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Timing repeats: {args.timing_repeats}")

    print("\n1. Generating IS-only timing comparison...")
    save_is_only_timing_comparison(
        n_points=args.n_points,
        seed=args.seed,
        timing_repeats=args.timing_repeats,
    )

    print("\n2. Generating IS-only parameter density comparison...")
    save_is_only_parameter_density(
        n_points=args.n_points,
        seed=args.seed,
    )

    print("\n✓ IS-only mode complete!")
    print("Generated figures in examples/curvefit/figs/")


def run_scaling_mode(args):
    """Run inference scaling analysis mode."""
    from examples.curvefit.figs import save_inference_scaling_viz

    print("=== Scaling Mode: Inference Scaling Analysis ===")
    print("Analyzing performance scaling with different particle counts")
    print("This demonstrates GPU vectorization benefits")
    
    print("\nGenerating inference scaling visualization...")
    save_inference_scaling_viz()
    
    print("\n✓ Scaling mode complete!")
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
    elif args.mode == "outlier":
        run_outlier_mode(args)
    elif args.mode == "is-only":
        run_is_only_mode(args)
    elif args.mode == "scaling":
        run_scaling_mode(args)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
