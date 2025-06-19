#!/usr/bin/env python3
"""
Main script for state space model case study.

Demonstrates Sequential Monte Carlo with rejuvenation on discrete HMMs
and linear Gaussian state space models.
"""

import argparse
import jax.random as jrand
import json
import os

from examples.state_space.core import (
    run_discrete_hmm_experiment,
    run_linear_gaussian_experiment,
)
from examples.state_space.figs import create_all_figures


def main():
    parser = argparse.ArgumentParser(description="State space model case study")

    # Figure generation options
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Generate convergence comparison plot",
    )
    parser.add_argument(
        "--log-marginal",
        action="store_true",
        help="Generate log marginal comparison plot",
    )
    parser.add_argument(
        "--ess", action="store_true", help="Generate ESS comparison plot"
    )
    parser.add_argument("--summary", action="store_true", help="Generate summary table")
    parser.add_argument(
        "--particles-2d",
        action="store_true",
        help="Generate 2D particle evolution visualization",
    )
    parser.add_argument(
        "--rejuvenation-comparison",
        action="store_true",
        help="Compare different rejuvenation strategies (MH vs MALA, K=1 vs K=5)",
    )
    parser.add_argument(
        "--challenging",
        action="store_true",
        help="Use challenging scenario (3x observation noise, drift > 1σ)",
    )
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Use extreme scenario (5x observation noise, nonlinear drift)",
    )
    parser.add_argument(
        "--difficulty-comparison",
        action="store_true",
        help="Compare algorithm performance across difficulty levels",
    )

    # Experiment parameters
    parser.add_argument(
        "--time-steps", type=int, default=20, help="Number of time steps (default: 20)"
    )
    parser.add_argument(
        "--particles",
        type=str,
        default="100,500,1000,2000",
        help="Comma-separated list of particle counts (default: 100,500,1000,2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # If no specific figure requested, generate convergence by default
    if not any(
        [
            args.all,
            args.convergence,
            args.log_marginal,
            args.ess,
            args.summary,
            args.particles_2d,
            args.rejuvenation_comparison,
            args.difficulty_comparison,
        ]
    ):
        args.convergence = True

    # Parse particle counts
    n_particles_list = [int(x) for x in args.particles.split(",")]

    # Run experiments
    print("=" * 60)
    print("Running State Space Model Experiments")
    print("=" * 60)
    print(f"Time steps: {args.time_steps}")
    print(f"Particle counts: {n_particles_list}")
    print(f"Random seed: {args.seed}")
    print()

    # Set up random keys
    key = jrand.PRNGKey(args.seed)
    key_hmm, key_lg, key_2d, key_rejuv = jrand.split(key, 4)

    # Handle 2D particle visualization separately
    if args.particles_2d or args.all:
        print("Running 2D particle evolution experiment...")
        from examples.state_space.core import run_linear_gaussian_with_particles
        from examples.state_space.figs import (
            plot_2d_particle_evolution,
            plot_particle_density_evolution,
        )

        # Run with 16 timesteps for (4,4) grid
        particle_data = run_linear_gaussian_with_particles(
            key_2d, T=16, n_particles=200, d_state=2, d_obs=2
        )

        # Generate particle visualizations
        figs_dir = os.path.join(os.path.dirname(__file__), "figs")
        os.makedirs(figs_dir, exist_ok=True)
        plot_2d_particle_evolution(
            particle_data, scenario_name="standard", model_name="linear_gaussian_2d"
        )
        print(f"✓ Created {os.path.join('figs', 'particle_evolution_2d.pdf')}")

        plot_particle_density_evolution(
            particle_data, save_path=os.path.join(figs_dir, "particle_density.pdf")
        )
        print(f"✓ Created {os.path.join('figs', 'particle_density.pdf')}")

        # If only particles-2d was requested, exit early
        if args.particles_2d and not args.all and not args.rejuvenation_comparison:
            print("\n" + "=" * 60)
            print("✓ Particle visualization completed successfully!")
            print("=" * 60)
            return

    # Handle rejuvenation comparison and difficulty comparison
    if args.rejuvenation_comparison or args.difficulty_comparison or args.all:
        print("Running rejuvenation strategy comparisons...")
        from examples.state_space.core import run_rejuvenation_comparison
        from examples.state_space.figs import (
            plot_rejuvenation_comparison,
            plot_difficulty_comparison,
        )

        # Determine scenarios to run
        scenarios_to_run = []
        if args.challenging:
            scenarios_to_run.append(("challenging", True, False))
        elif args.extreme:
            scenarios_to_run.append(("extreme", False, True))
        else:
            scenarios_to_run.append(("standard", False, False))

        # If running difficulty comparison, need all three scenarios
        if args.difficulty_comparison or args.all:
            scenarios_to_run = [
                ("standard", False, False),
                ("challenging", True, False),
                ("extreme", False, True),
            ]

        # Generate data for each scenario
        scenario_results = {}
        for scenario_name, challenging, extreme in scenarios_to_run:
            print(f"   Running {scenario_name} scenario...")
            key_rejuv, subkey = jrand.split(key_rejuv)

            rejuv_results = run_rejuvenation_comparison(
                subkey,
                T=16,  # For frames 0, 4, 8, 12
                n_particles=500,
                d_state=2,
                d_obs=2,
                step_size=0.08,  # Optimized step size
                challenging=challenging,
                extreme=extreme,
            )
            scenario_results[scenario_name] = rejuv_results

        # Generate figures
        figs_dir = os.path.join(os.path.dirname(__file__), "figs")
        os.makedirs(figs_dir, exist_ok=True)

        # Individual scenario comparisons
        if args.rejuvenation_comparison or args.all:
            for scenario_name, results in scenario_results.items():
                print(f"   Generating {scenario_name} rejuvenation comparison...")
                plot_rejuvenation_comparison(
                    results,
                    frames=[0, 4, 8, 12],
                    scenario_name=scenario_name,
                    model_name="linear_gaussian_2d",
                )

        # Difficulty comparison
        if args.difficulty_comparison or args.all:
            if len(scenario_results) >= 3:
                print("   Generating difficulty comparisons...")
                strategies = ["mh_1", "mh_5", "mala_1", "mala_5"]
                for strategy in strategies:
                    plot_difficulty_comparison(
                        scenario_results["standard"],
                        scenario_results["challenging"],
                        scenario_results["extreme"],
                        frames=[0, 4, 8, 12],
                        strategy=strategy,
                        model_name="linear_gaussian_2d",
                    )
            else:
                print("   Difficulty comparison requires all three scenarios")

        print("✓ Rejuvenation analysis completed!")

        # If only rejuvenation/difficulty comparison was requested, exit early
        if (
            args.rejuvenation_comparison or args.difficulty_comparison
        ) and not args.all:
            print("\n" + "=" * 60)
            print("✓ Rejuvenation analysis completed successfully!")
            print("=" * 60)
            return

    # Run standard experiments
    print("Running discrete HMM experiment...")
    hmm_results = run_discrete_hmm_experiment(
        key_hmm, T=args.time_steps, n_particles_list=n_particles_list
    )
    print(f"✓ Exact log marginal: {hmm_results['exact_log_marginal']:.4f}")

    # Run linear Gaussian experiment
    print("\nRunning linear Gaussian experiment...")
    lg_results = run_linear_gaussian_experiment(
        key_lg, T=args.time_steps, n_particles_list=n_particles_list
    )
    print(f"✓ Exact log marginal: {lg_results['exact_log_marginal']:.4f}")

    # Save results
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    results_path = os.path.join(figs_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"hmm": hmm_results, "linear_gaussian": lg_results}, f, indent=2)
    print(f"\n✓ Saved results to {os.path.join('figs', 'results.json')}")

    # Generate figures
    print("\nGenerating figures...")

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")

    if args.all:
        create_all_figures(hmm_results, lg_results, figs_dir)
    else:
        from examples.state_space.figs import (
            plot_convergence_comparison,
            plot_log_marginal_comparison,
            plot_ess_comparison,
            plot_summary_table,
        )

        if args.convergence:
            plot_convergence_comparison(
                hmm_results,
                lg_results,
                save_path=os.path.join(figs_dir, "convergence_comparison.pdf"),
            )
            print(f"✓ Created {os.path.join('figs', 'convergence_comparison.pdf')}")

        if args.log_marginal:
            plot_log_marginal_comparison(
                hmm_results,
                lg_results,
                save_path=os.path.join(figs_dir, "log_marginal_comparison.pdf"),
            )
            print(f"✓ Created {os.path.join('figs', 'log_marginal_comparison.pdf')}")

        if args.ess:
            plot_ess_comparison(
                hmm_results,
                lg_results,
                save_path=os.path.join(figs_dir, "ess_comparison.pdf"),
            )
            print(f"✓ Created {os.path.join('figs', 'ess_comparison.pdf')}")

        if args.summary:
            plot_summary_table(
                hmm_results,
                lg_results,
                save_path=os.path.join(figs_dir, "summary_table.pdf"),
            )
            print(f"✓ Created {os.path.join('figs', 'summary_table.pdf')}")

    print("\n" + "=" * 60)
    print("✓ All experiments completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
