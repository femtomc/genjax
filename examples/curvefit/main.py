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
        choices=["quick", "full", "benchmark", "traces"],
        nargs="?",
        default="quick",
        help="Analysis mode: quick (fast viz), full (complete), benchmark (compare frameworks), traces (only trace figures)",
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
    import jax.random as jrand
    from examples.curvefit.figs import (
        # save_onepoint_trace_viz,  # UNUSED: generates 010_onepoint_trace.pdf
        save_four_separate_onepoint_traces,
        # save_four_onepoint_trace_densities,  # UNUSED: generates 051-054_onepoint_trace_density.pdf
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_four_separate_batched_multipoint_traces,
        save_four_batched_multipoint_trace_densities,
        save_inference_viz,
    )

    print("=== Quick Mode: Basic Visualizations ===")
    
    # Initialize the master key from CLI seed
    key = jrand.key(args.seed)
    
    # Split keys for each visualization function
    keys = jrand.split(key, 5)  # Reduced to 5 functions that need keys

    print("\n1. Generating trace visualizations...")
    # save_onepoint_trace_viz(keys[0])  # UNUSED
    save_four_separate_onepoint_traces(keys[0])
    # save_four_onepoint_trace_densities(keys[2])  # UNUSED
    save_multipoint_trace_viz(keys[1])
    save_four_multipoint_trace_vizs(keys[2])
    save_four_separate_batched_multipoint_traces(keys[3])
    save_four_batched_multipoint_trace_densities(keys[4])

    print("\n2. Generating inference visualization...")
    save_inference_viz(seed=args.seed)

    print("\n✓ Quick mode complete!")
    print("Generated figures in examples/curvefit/figs/")


def run_full_mode(args):
    """Run full analysis mode."""
    from examples.curvefit.figs import (
        # save_onepoint_trace_viz,  # UNUSED: generates 010_onepoint_trace.pdf
        save_four_separate_onepoint_traces,
        # save_four_onepoint_trace_densities,  # UNUSED: generates 051-054_onepoint_trace_density.pdf
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_four_separate_batched_multipoint_traces,
        save_four_batched_multipoint_trace_densities,
        save_inference_viz,
        save_inference_scaling_viz,
        # save_genjax_posterior_comparison,  # UNUSED: generates 090_genjax_posterior_comparison.pdf
        # save_framework_comparison_figure,  # UNUSED: generates 080_framework_comparison_n{n}.pdf
        # save_log_density_viz,  # UNUSED: generates 070_log_density.pdf
        # save_parameter_posterior_methods_comparison,  # UNUSED: generates 100_parameter_posterior_3d_comparison.pdf
    )

    print("=== Full Mode: Complete Analysis ===")
    
    # Initialize the master key from CLI seed
    import jax.random as jrand
    key = jrand.key(args.seed)
    
    # Split keys for each visualization function
    keys = jrand.split(key, 5)  # Reduced to 5 functions that need keys

    print("\n1. Generating trace visualizations...")
    # save_onepoint_trace_viz(keys[0])  # UNUSED
    save_four_separate_onepoint_traces(keys[0])
    # save_four_onepoint_trace_densities(keys[2])  # UNUSED
    save_multipoint_trace_viz(keys[1])
    save_four_multipoint_trace_vizs(keys[2])
    save_four_separate_batched_multipoint_traces(keys[3])
    save_four_batched_multipoint_trace_densities(keys[4])

    print("\n2. Generating inference scaling analysis...")
    save_inference_scaling_viz()

    print("\n3. Generating inference visualization...")
    save_inference_viz(seed=args.seed)

    # UNUSED: Commented out sections that generate figures not used in paper
    # print("\n4. Generating GenJAX posterior comparison...")
    # save_genjax_posterior_comparison(...)
    
    # print("\n5. Generating framework comparison...")
    # save_framework_comparison_figure(...)
    
    # print("\n6. Generating density visualizations...")
    # save_log_density_viz()
    
    # print("\n7. Generating parameter posterior methods comparison...")
    # save_parameter_posterior_methods_comparison(seed=args.seed)

    print("\n✓ Full mode complete!")
    print("Generated figures in examples/curvefit/figs/")


def run_benchmark_mode(args):
    """Run benchmark comparison mode."""
    # UNUSED: Both functions generate figures not used in paper
    # from examples.curvefit.figs import (
    #     save_framework_comparison_figure,  # UNUSED: generates 080_framework_comparison_n{n}.pdf
    #     save_parameter_posterior_methods_comparison,  # UNUSED: generates 100_parameter_posterior_3d_comparison.pdf
    # )

    print("=== Benchmark Mode: Framework Comparison ===")
    print("NOTE: Benchmark mode currently disabled - generates unused figures")
    print("The following figures are not used in the paper:")
    print("  - 080_framework_comparison_n{n}.pdf")
    print("  - 100_parameter_posterior_3d_comparison.pdf")
    
    # UNUSED: Commented out benchmark execution
    # print("Parameters:")
    # print(f"  - Data points: {args.n_points}")
    # print(f"  - IS particles: {args.n_samples_is}")
    # print(f"  - HMC samples: {args.n_samples_hmc}")
    # print(f"  - HMC warmup: {args.n_warmup}")
    # print(f"  - Timing repeats: {args.timing_repeats}")
    # print(f"  - Random seed: {args.seed}")

    # results = save_framework_comparison_figure(
    #     n_points=args.n_points,
    #     n_samples_is=args.n_samples_is,
    #     n_samples_hmc=args.n_samples_hmc,
    #     n_warmup=args.n_warmup,
    #     seed=args.seed,
    #     timing_repeats=args.timing_repeats,
    # )

    # print("\n=== Benchmark Summary ===")
    # for method_key, result in results.items():
    #     mean_time = result["timing"][0] * 1000
    #     std_time = result["timing"][1] * 1000
    #     print(f"{result['method']}: {mean_time:.1f} ± {std_time:.1f} ms")
    #     if "accept_rate" in result:
    #         print(f"  Accept rate: {result['accept_rate']:.3f}")

    # print("\n✓ Benchmark complete!")
    # print("Generated comparison figure in examples/curvefit/figs/")
    
    # print("\n8. Generating 3D parameter posterior comparison...")
    # save_parameter_posterior_methods_comparison(
    #     n_points=args.n_points,
    #     n_samples_is=5000,  # Fixed for 3D comparison
    #     n_samples_hmc=args.n_samples_hmc,
    #     n_warmup=args.n_warmup,
    #     seed=args.seed,
    # )
    
    # # Also generate the density sphere version
    # try:
    #     from examples.curvefit.figs_3d_sphere import save_parameter_posterior_3d_sphere
    #     print("\n9. Generating 3D density sphere visualization...")
    #     save_parameter_posterior_3d_sphere(
    #         n_points=args.n_points,
    #         n_samples_is=5000,
    #         n_samples_hmc=args.n_samples_hmc,
    #         n_warmup=args.n_warmup,
    #         seed=args.seed,
    #     )
    # except Exception as e:
    #     print(f"  Density sphere visualization failed: {e}")
    
    # # Generate the voxel version
    # try:
    #     from examples.curvefit.figs_3d_voxels import save_parameter_posterior_3d_voxels
    #     print("\n10. Generating 3D voxel visualization...")
    #     save_parameter_posterior_3d_voxels(
    #         n_points=args.n_points,
    #         n_samples_is=5000,
    #         n_samples_hmc=args.n_samples_hmc,
    #         n_warmup=args.n_warmup,
    #         seed=args.seed,
    #     )
    # except Exception as e:
    #     print(f"  Voxel visualization failed: {e}")
    
    # # Generate the density surface comparison
    # try:
    #     from examples.curvefit.figs_density_3d import save_genjax_density_comparison
    #     print("\n11. Generating GenJAX density surface comparison...")
    #     save_genjax_density_comparison(
    #         n_points=args.n_points,
    #         n_samples_is=5000,
    #         n_samples_hmc_single=1500,
    #         n_samples_hmc_multi=1500,
    #         n_warmup=args.n_warmup,
    #         seed=args.seed,
    #         timing_repeats=args.timing_repeats,
    #     )
    # except Exception as e:
    #     print(f"  Density surface comparison failed: {e}")
    
    # NOTE: Overview IS figures are no longer used in the paper
    # Commenting out to avoid generating unused figures
    # # Generate overview IS figures
    # try:
    #     from examples.curvefit.figs_overview_is import generate_overview_figures
    #     print("\n12. Generating overview IS comparison figures...")
    #     generate_overview_figures(
    #         n_points=args.n_points,
    #         n_samples_50=50,
    #         n_samples_5000=5000,
    #         seed=args.seed,
    #         timing_repeats=args.timing_repeats,
    #     )
    # except Exception as e:
    #     print(f"  Overview figures failed: {e}")




def run_traces_mode(args):
    """Run traces-only mode - generates all trace figures without inference."""
    import jax.random as jrand
    from examples.curvefit.figs import (
        # save_onepoint_trace_viz,  # UNUSED: generates 010_onepoint_trace.pdf
        save_four_separate_onepoint_traces,
        # save_four_onepoint_trace_densities,  # UNUSED: generates 051-054_onepoint_trace_density.pdf
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_four_separate_batched_multipoint_traces,
        save_four_batched_multipoint_trace_densities,
    )

    print("=== Traces Mode: Generate All Trace Figures ===")
    print("This mode generates trace visualizations without running inference.")
    
    # Initialize the master key from CLI seed
    key = jrand.key(args.seed)
    
    # Split keys for each visualization function
    keys = jrand.split(key, 5)  # Reduced to 5 functions that need keys

    print("\n1. Generating onepoint trace visualizations...")
    # save_onepoint_trace_viz(keys[0])  # UNUSED
    save_four_separate_onepoint_traces(keys[0])
    
    # UNUSED: Onepoint trace density visualizations
    # print("\n2. Generating onepoint trace density visualizations...")
    # save_four_onepoint_trace_densities(keys[2])
    
    print("\n2. Generating multipoint trace visualizations...")
    save_multipoint_trace_viz(keys[1])
    save_four_multipoint_trace_vizs(keys[2])
    
    print("\n3. Generating batched multipoint trace visualizations...")
    save_four_separate_batched_multipoint_traces(keys[3])
    
    print("\n4. Generating batched multipoint trace density visualizations...")
    save_four_batched_multipoint_trace_densities(keys[4])

    print("\n✓ Traces mode complete!")
    print("Generated trace figures in examples/curvefit/figs/")
    print("\nFigures generated (used in paper):")
    print("  - 011-013_onepoint_trace.pdf")
    print("  - 020_multipoint_trace.pdf")
    print("  - 030_four_multipoint_traces.pdf")
    print("  - 031-034_batched_multipoint_trace.pdf")
    print("  - 040-043_batched_multipoint_trace_density.pdf")


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
    elif args.mode == "traces":
        run_traces_mode(args)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
