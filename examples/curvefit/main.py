import argparse
from figs import (
    save_onepoint_trace_viz,
    save_multipoint_trace_viz,
    save_four_multipoint_trace_vizs,
    save_inference_viz,
    save_inference_scaling_viz,
    save_comprehensive_benchmark_figure,
    save_genjax_scaling_benchmark,
    save_genjax_posterior_comparison,
)


def main():
    """Main CLI for curvefit case study."""
    parser = argparse.ArgumentParser(
        description="GenJAX Curvefit Case Study - Bayesian Sine Wave Parameter Estimation"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "traces",
            "inference",
            "scaling",
            "benchmark",
            "genjax-scaling",
            "posterior-comparison",
        ],
        default="all",
        help="Which figures to generate (default: all)",
    )

    # Benchmark-specific parameters
    parser.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="Number of data points for benchmark (default: 20)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples per method (default: 1000)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=500,
        help="Number of warmup samples for MCMC (default: 500)",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=30,
        help="Number of timing repetitions (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=== GenJAX Curvefit Case Study ===")
    print(f"Mode: {args.mode}")

    if args.mode in ["all", "traces"]:
        print("\nGenerating trace visualizations...")
        save_onepoint_trace_viz()
        save_multipoint_trace_viz()
        save_four_multipoint_trace_vizs()

    if args.mode in ["all", "inference"]:
        print("\nGenerating inference visualizations...")
        save_inference_viz()

    if args.mode in ["all", "scaling"]:
        print("\nGenerating scaling analysis...")
        save_inference_scaling_viz()

    if args.mode in ["all", "benchmark"]:
        print("\nRunning comprehensive benchmark...")
        print(
            f"Parameters: {args.n_points} points, {args.n_samples} samples, "
            f"{args.n_warmup} warmup, {args.timing_repeats} timing repeats"
        )

        try:
            benchmark_results, posterior_samples = save_comprehensive_benchmark_figure(
                n_points=args.n_points,
                n_samples=args.n_samples,
                n_warmup=args.n_warmup,
                seed=args.seed,
                timing_repeats=args.timing_repeats,
            )
            print("Benchmark completed successfully!")

        except ImportError as e:
            print(f"Warning: Some frameworks not available - {e}")
            print("Install numpyro and/or pyro-ppl for full framework comparison")
        except Exception as e:
            print(f"Benchmark failed: {e}")
            print("Continuing with other figures...")

    if args.mode in ["all", "genjax-scaling"]:
        print("\nRunning GenJAX scaling analysis...")
        try:
            save_genjax_scaling_benchmark(
                n_points=args.n_points,
                timing_repeats=args.timing_repeats,
                seed=args.seed,
            )
            print("GenJAX scaling analysis completed successfully!")
        except Exception as e:
            print(f"GenJAX scaling analysis failed: {e}")
            print("Continuing with other figures...")

    if args.mode in ["all", "posterior-comparison"]:
        print("\nRunning GenJAX posterior comparison...")
        try:
            save_genjax_posterior_comparison(
                n_points=args.n_points,
                timing_repeats=args.timing_repeats,
                seed=args.seed,
            )
            print("GenJAX posterior comparison completed successfully!")
        except Exception as e:
            print(f"GenJAX posterior comparison failed: {e}")
            print("Continuing with other figures...")

    print("\n=== Curvefit case study complete! ===")


if __name__ == "__main__":
    main()
