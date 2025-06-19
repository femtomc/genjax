"""
Main script for testing and demonstrating Programmable MCTS.

This script shows how MCTS can work with probabilistic game models
implemented as GenJAX generative functions.
"""

import jax.numpy as jnp
import jax.random as jrand
import argparse

from .core import (
    MCTS,
    create_empty_board,
    print_board,
    demonstrate_probabilistic_mcts,
    create_tic_tac_toe_model,
)
from .visualizations import create_all_visualizations
from .exact_solver import validate_solver, benchmark_solver_performance
from .quick_demo import demo_mcts_vs_exact


def test_basic_mcts(key: jnp.ndarray, num_simulations: int = 100) -> None:
    """Test basic MCTS functionality with probabilistic model."""
    print(f"=== Testing Probabilistic MCTS with {num_simulations} simulations ===")

    # Use the probabilistic model
    demonstrate_probabilistic_mcts(key, num_simulations)
    print()


def test_different_positions(key: jnp.ndarray, num_simulations: int = 50) -> None:
    """Test MCTS on different board positions."""
    print("=== Testing Different Positions ===")

    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    # Test position 1: Empty board
    state1 = create_empty_board()
    print("Position 1: Empty board")
    print_board(state1)

    key1, key = jrand.split(key)
    action1, root1 = mcts.search(key1, state1, num_simulations)
    print(f"MCTS chooses: {action1} (visits: {root1.visits})")

    # Test position 2: Mid-game
    state2 = create_empty_board()
    state2 = state2.at[1, 1].set(1)  # X in center
    state2 = state2.at[0, 0].set(-1)  # O in corner

    print("\nPosition 2: Mid-game (O to move)")
    print_board(state2)

    key2, key = jrand.split(key)
    action2, root2 = mcts.search(key2, state2, num_simulations)
    print(f"MCTS chooses: {action2} (visits: {root2.visits})")

    print()


def test_model_uncertainty(key: jnp.ndarray, num_simulations: int = 50) -> None:
    """Demonstrate how model uncertainty affects MCTS decisions."""
    print("=== Testing Model Uncertainty ===")
    print("The probabilistic model adds noise to observations.")
    print("This simulates uncertainty about game dynamics.\n")

    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    # Same position, multiple runs
    state = create_empty_board()
    state = state.at[1, 1].set(1)  # X in center

    print("Testing same position multiple times:")
    print_board(state)

    keys = jrand.split(key, 3)
    for i in range(3):
        action, root = mcts.search(keys[i], state, num_simulations)
        print(f"Run {i + 1}: MCTS chooses {action} (visits: {root.visits})")

    print()


def test_exact_solver() -> None:
    """Test and benchmark the exact minimax solver."""
    print("=== Testing Exact Tic-Tac-Toe Solver ===")

    # Validate solver
    print("Validating exact solver against known results...")
    if validate_solver():
        print("✅ Exact solver validation passed!")
    else:
        print("❌ Exact solver validation failed!")
        return

    # Benchmark performance
    print("\nBenchmarking solver performance...")
    perf_results = benchmark_solver_performance()

    print(f"Empty board analysis: {perf_results['empty_board']['time_ms']:.2f}ms")
    print(f"  Optimal action: {perf_results['empty_board']['optimal_action']}")
    print(f"  Available actions: {perf_results['empty_board']['num_actions']}")

    print(f"Mid-game analysis: {perf_results['mid_game']['time_ms']:.2f}ms")
    print(f"  Optimal action: {perf_results['mid_game']['optimal_action']}")
    print(f"  Available actions: {perf_results['mid_game']['num_actions']}")

    print(f"Game tree analysis: {perf_results['full_tree']['time_ms']:.2f}ms")
    print(
        f"  Game-theoretic result: {perf_results['full_tree']['game_theoretic_value']}"
    )

    print("\nPerfect play sequence from empty board:")
    for i, action in enumerate(perf_results["full_tree"]["perfect_sequence"]):
        player = "X" if i % 2 == 0 else "O"
        print(f"  Move {i + 1}: {player} plays {action}")

    print()


def main():
    """Main entry point with command line options."""
    parser = argparse.ArgumentParser(
        description="Test Programmable MCTS with Probabilistic Models"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "basic",
            "positions",
            "uncertainty",
            "visualizations",
            "exact",
            "demo",
            "all",
        ],
        default="all",
        help="Test mode to run",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/programmable_mcts/figs",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    key = jrand.PRNGKey(args.seed)

    print("Programmable MCTS with Probabilistic Models")
    print("===========================================")
    print("MCTS is now parametric over GenJAX generative functions!")
    print()

    if args.mode in ["basic", "all"]:
        key, subkey = jrand.split(key)
        test_basic_mcts(subkey, args.simulations)

    if args.mode in ["positions", "all"]:
        key, subkey = jrand.split(key)
        test_different_positions(subkey, args.simulations)

    if args.mode in ["uncertainty", "all"]:
        key, subkey = jrand.split(key)
        test_model_uncertainty(subkey, args.simulations)

    if args.mode in ["exact"]:
        test_exact_solver()

    if args.mode in ["demo", "all"]:
        key, subkey = jrand.split(key)
        demo_mcts_vs_exact(subkey, args.output_dir)

    if args.mode in ["visualizations"]:
        key, subkey = jrand.split(key)
        print("=== Creating MCTS Visualizations ===")
        create_all_visualizations(subkey, args.output_dir)

    print("Testing complete!")
    print("\nNext steps for Programmable MCTS:")
    print("1. Learn better models from game experience")
    print("2. Use inference to update model parameters")
    print("3. Handle more complex games and state spaces")


if __name__ == "__main__":
    main()
