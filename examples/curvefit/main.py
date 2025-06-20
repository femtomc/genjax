import argparse
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GenJAX Curvefit Case Study - Bayesian Sine Wave Parameter Estimation"
    )

    # Add subcommands for generate-data and plot-figures
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    gen_parser = subparsers.add_parser(
        "generate-data", help="Generate experimental data"
    )
    gen_parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "traces",
            "inference",
            "scaling",
            "benchmark",
            "genjax-scaling",
            "posterior-comparison",
        ],
        default=["traces", "inference", "posterior-comparison"],
        help="Which experiments to run (default: traces inference posterior-comparison)",
    )
    gen_parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name (default: timestamped)",
    )
    gen_parser.add_argument(
        "--n-points", type=int, default=20, help="Number of data points (default: 20)"
    )
    gen_parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples per method (default: 1000)",
    )
    gen_parser.add_argument(
        "--n-warmup",
        type=int,
        default=500,
        help="Number of warmup samples for MCMC (default: 500)",
    )
    gen_parser.add_argument(
        "--timing-repeats",
        type=int,
        default=10,
        help="Number of timing repetitions (default: 10)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    gen_parser.add_argument(
        "--export-data",
        action="store_true",
        default=True,
        help="Export data to CSV (default: True)",
    )

    # Plot figures command
    plot_parser = subparsers.add_parser(
        "plot-figures", help="Plot figures from saved data"
    )
    plot_parser.add_argument(
        "--experiment-name", type=str, help="Experiment to plot (default: most recent)"
    )
    plot_parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "traces",
            "inference",
            "scaling",
            "benchmark",
            "genjax-scaling",
            "posterior-comparison",
        ],
        help="Which figures to plot (default: all available)",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default="figs",
        help="Output directory for figures (default: figs)",
    )

    # Legacy mode support (backwards compatibility)
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "traces",
            "inference",
            "scaling",
            "benchmark",
            "genjax-only",
            "genjax-fast",
            "genjax-scaling",
            "posterior-comparison",
        ],
        help="Legacy mode selection (use generate-data or plot-figures instead)",
    )

    # Legacy parameters (for backwards compatibility)
    parser.add_argument(
        "--n-points", type=int, default=20, help="Number of data points"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-warmup", type=int, default=500, help="MCMC warmup samples")
    parser.add_argument(
        "--timing-repeats", type=int, default=30, help="Timing repetitions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def generate_data(args):
    """Generate experimental data and optionally export."""
    import jax.random as jrand
    from examples.curvefit.export import (
        create_experiment_directory,
        save_experiment_metadata,
        save_dataset,
        save_inference_results,
        save_benchmark_summary,
    )
    from examples.curvefit.data import generate_test_dataset
    from examples.curvefit.core import (
        infer_latents,
        hmc_infer_latents,
        numpyro_run_hmc_inference_jit,
    )
    from genjax import const
    from examples.utils import timing

    print("=== GenJAX Curvefit Data Generation ===")
    print(f"Modes: {args.modes}")
    print(f"Parameters: {args.n_points} points, {args.n_samples} samples")

    # Create experiment directory
    exp_dir = None
    if args.export_data:
        exp_dir = create_experiment_directory(experiment_name=args.experiment_name)
        print(f"Experiment directory: {exp_dir}")

    # Generate dataset
    key = jrand.key(args.seed)
    data = generate_test_dataset(
        key=key, n_points=args.n_points, true_freq=0.3, true_offset=1.5, noise_std=0.3
    )
    xs, ys = data["xs"], data["ys"]
    true_freq = data["true_params"]["freq"]
    true_offset = data["true_params"]["offset"]

    # Save experiment metadata and dataset
    if exp_dir:
        config = {
            "modes": args.modes,
            "n_points": args.n_points,
            "n_samples": args.n_samples,
            "n_warmup": args.n_warmup,
            "timing_repeats": args.timing_repeats,
            "seed": args.seed,
            "true_freq": true_freq,
            "true_offset": true_offset,
            "timestamp": datetime.now().isoformat(),
        }
        save_experiment_metadata(exp_dir, config)
        save_dataset(exp_dir, xs, ys, true_freq, true_offset)

    results = {}

    # Run posterior comparison if requested
    if "posterior-comparison" in args.modes:
        print("\nRunning posterior comparison experiments...")

        # GenJAX Importance Sampling
        print("  - GenJAX Importance Sampling...")
        key, subkey = jrand.split(key)

        def run_is():
            from genjax import seed

            return seed(infer_latents)(subkey, xs, ys, const(args.n_samples))

        is_times, (is_mean, is_std) = timing(
            run_is, repeats=args.timing_repeats, inner_repeats=1
        )
        is_samples, is_weights = run_is()

        is_result = {
            "samples": {
                "curve": {
                    "freq": is_samples.get_choices()["curve"]["freq"],
                    "off": is_samples.get_choices()["curve"]["off"],
                }
            },
            "weights": is_weights,
            "timing_stats": (is_mean, is_std),
        }
        results["genjax_is"] = is_result

        # GenJAX HMC
        print("  - GenJAX HMC...")
        key, subkey = jrand.split(key)

        def run_hmc():
            from genjax import seed

            return seed(hmc_infer_latents)(
                subkey,
                xs,
                ys,
                const(args.n_samples),
                const(args.n_warmup),
                step_size=const(0.01),
                n_steps=const(20),
            )

        hmc_times, (hmc_mean, hmc_std) = timing(
            run_hmc, repeats=args.timing_repeats, inner_repeats=1
        )
        hmc_traces, hmc_diagnostics = run_hmc()

        # Extract samples from traces
        hmc_choices = hmc_traces.get_choices()

        hmc_result = {
            "samples": {
                "curve": {
                    "freq": hmc_choices["curve"]["freq"],
                    "off": hmc_choices["curve"]["off"],
                }
            },
            "timing_stats": (hmc_mean, hmc_std),
            "additional_stats": {
                "acceptance_rate": hmc_diagnostics.get("acceptance_rate", 0.0)
            },
        }
        results["genjax_hmc"] = hmc_result

        # NumPyro HMC (if available)
        try:
            print("  - NumPyro HMC...")
            key, subkey = jrand.split(key)

            def run_numpyro():
                return numpyro_run_hmc_inference_jit(
                    subkey,
                    xs,
                    ys,
                    num_samples=args.n_samples,
                    num_warmup=args.n_warmup,
                    step_size=0.01,
                    target_accept_prob=0.8,
                )

            numpyro_times, (numpyro_mean, numpyro_std) = timing(
                run_numpyro, repeats=args.timing_repeats, inner_repeats=1
            )
            numpyro_result_obj = run_numpyro()

            numpyro_result = {
                "samples": {
                    "freq": numpyro_result_obj.samples["freq"],
                    "off": numpyro_result_obj.samples["off"],
                },
                "timing_stats": (numpyro_mean, numpyro_std),
                "additional_stats": {
                    "acceptance_rate": 0.0  # NumPyro doesn't report this easily
                },
            }
            results["numpyro_hmc"] = numpyro_result

        except Exception as e:
            print(f"  - NumPyro not available: {e}")

    # Save all results
    if exp_dir and results:
        for method_name, result in results.items():
            save_inference_results(
                exp_dir,
                method_name,
                result["samples"],
                result.get("weights"),
                result.get("timing_stats"),
                result.get("additional_stats"),
            )
        save_benchmark_summary(exp_dir, results)

    print(f"\nData generation complete! Experiment: {args.experiment_name}")
    return exp_dir


def plot_figures(args):
    """Plot figures from saved experimental data."""
    from examples.curvefit.export import get_latest_experiment, load_benchmark_results
    from examples.curvefit.figs import (
        save_onepoint_trace_viz,
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_inference_viz,
        save_genjax_posterior_comparison_from_data,
    )

    print("=== GenJAX Curvefit Figure Generation ===")

    # Get experiment to plot
    if args.experiment_name is None:
        args.experiment_name = get_latest_experiment()
        if args.experiment_name is None:
            print("No experiments found. Run 'generate-data' first.")
            return

    print(f"Plotting from experiment: {args.experiment_name}")

    # Load experiment data
    exp_dir = f"examples/curvefit/data/{args.experiment_name}"
    data = load_benchmark_results(exp_dir)

    # Determine which modes to plot
    if args.modes is None:
        # Plot all modes that have data
        available_modes = []
        if data["methods"]:
            if "genjax_is" in data["methods"] and "genjax_hmc" in data["methods"]:
                available_modes.append("posterior-comparison")
        # Always include basic plots
        available_modes.extend(["traces", "inference"])
        args.modes = available_modes

    print(f"Modes to plot: {args.modes}")

    # Plot trace visualizations
    if "traces" in args.modes:
        print("\nGenerating trace visualizations...")
        save_onepoint_trace_viz()
        save_multipoint_trace_viz()
        save_four_multipoint_trace_vizs()

    # Plot inference visualization
    if "inference" in args.modes:
        print("\nGenerating inference visualization...")
        save_inference_viz()

    # Plot posterior comparison from saved data
    if "posterior-comparison" in args.modes and data["methods"]:
        print("\nGenerating posterior comparison from saved data...")
        save_genjax_posterior_comparison_from_data(data, output_dir=args.output_dir)

    print("\nFigure generation complete!")


def legacy_mode(args):
    """Handle legacy mode for backwards compatibility."""
    from examples.curvefit.figs import (
        save_onepoint_trace_viz,
        save_multipoint_trace_viz,
        save_four_multipoint_trace_vizs,
        save_inference_viz,
        save_inference_scaling_viz,
        save_comprehensive_benchmark_figure,
        save_genjax_scaling_benchmark,
        save_genjax_posterior_comparison,
    )

    print("=== GenJAX Curvefit Case Study (Legacy Mode) ===")
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
            print("Install numpyro for full framework comparison")
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

    if args.mode == "genjax-only":
        print("\n=== GenJAX-Only Mode ===")
        print("Running all GenJAX demonstrations without framework comparisons...")

        print("\nGenerating trace visualizations...")
        save_onepoint_trace_viz()
        save_multipoint_trace_viz()
        save_four_multipoint_trace_vizs()

        print("\nGenerating inference visualizations...")
        save_inference_viz()

        print("\nGenerating light scaling analysis...")
        print(
            "Skipping heavy scaling analysis (save_inference_scaling_viz) - use 'scaling' mode for full analysis"
        )

        print("\nRunning light GenJAX scaling analysis...")
        try:
            save_genjax_scaling_benchmark(
                n_points=min(10, args.n_points),  # Reduce data points
                timing_repeats=max(
                    1, args.timing_repeats // 5
                ),  # Reduce repeats significantly
                seed=args.seed,
            )
            print("GenJAX scaling analysis completed successfully!")
        except Exception as e:
            print(f"GenJAX scaling analysis failed: {e}")
            print("Continuing...")

    if args.mode == "genjax-fast":
        print("\n=== GenJAX-Fast Mode ===")
        print("Running fast GenJAX demonstrations (no heavy scaling analysis)...")

        print("\nGenerating trace visualizations...")
        save_onepoint_trace_viz()
        save_multipoint_trace_viz()
        save_four_multipoint_trace_vizs()

        print("\nGenerating inference visualizations...")
        save_inference_viz()

        print(
            "Skipping scaling analyses for speed - use 'genjax-only' for light scaling or 'scaling'/'genjax-scaling' for full analysis"
        )

    print("\n=== Curvefit case study complete! ===")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle new commands
    if args.command == "generate-data":
        generate_data(args)
    elif args.command == "plot-figures":
        plot_figures(args)
    # Handle legacy mode
    elif args.mode:
        legacy_mode(args)
    else:
        # Default to showing help
        print(
            "Usage: python -m examples.curvefit.main {generate-data|plot-figures} [options]"
        )
        print(
            "       python -m examples.curvefit.main --mode {all|traces|...} [options]  (legacy)"
        )
        print("\nRun with -h for detailed help.")


if __name__ == "__main__":
    main()
