"""
Visualization utilities for Programmable MCTS.

This module provides tools to visualize and understand how the MCTS algorithm
works with probabilistic game models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .core import (
    MCTS,
    MCTSNode,
    create_tic_tac_toe_model,
    create_empty_board,
    get_legal_actions_ttt,
)
from .exact_solver import (
    get_all_action_values,
    get_optimal_action,
    classify_action_optimality,
)
from .theoretical_solver import (
    get_theoretical_action_values,
    get_theoretical_optimal_action,
    validate_theoretical_solver,
)


def visualize_board_with_heatmap(
    state: jnp.ndarray,
    action_stats: Dict[Tuple[int, int], Tuple[int, float]],
    title: str = "MCTS Action Evaluation",
    save_path: Optional[str] = None,
) -> None:
    """Visualize Tic-Tac-Toe board with MCTS action evaluation heatmap.

    Args:
        state: Current board state (3x3 array)
        action_stats: Dict mapping (row, col) -> (visits, avg_reward)
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Current board state
    ax1.set_title("Current Board", fontsize=14, fontweight="bold")
    draw_board(ax1, state)

    # Right plot: Action evaluation heatmap
    ax2.set_title("MCTS Action Evaluation", fontsize=14, fontweight="bold")

    # Create heatmap data
    visits_grid = np.zeros((3, 3))
    reward_grid = np.zeros((3, 3))

    for (row, col), (visits, avg_reward) in action_stats.items():
        visits_grid[row, col] = visits
        reward_grid[row, col] = avg_reward

    # Use visits as heatmap intensity, reward as color
    heatmap_data = visits_grid * reward_grid

    # Create heatmap
    im = ax2.imshow(
        heatmap_data, cmap="RdYlGn", aspect="equal", vmin=0, vmax=heatmap_data.max()
    )

    # Add text annotations
    for i in range(3):
        for j in range(3):
            if (i, j) in action_stats:
                visits, avg_reward = action_stats[(i, j)]
                text = f"V:{visits}\nR:{avg_reward:.2f}"
                ax2.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white"
                    if heatmap_data[i, j] > heatmap_data.max() / 2
                    else "black",
                )

    # Format heatmap
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(["0", "1", "2"])
    ax2.set_yticklabels(["0", "1", "2"])
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Visits × Avg Reward", rotation=270, labelpad=15)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def draw_board(ax, state: jnp.ndarray) -> None:
    """Draw a Tic-Tac-Toe board on given axes."""
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")

    # Draw grid lines
    for i in range(4):
        ax.axhline(i - 0.5, color="black", linewidth=2)
        ax.axvline(i - 0.5, color="black", linewidth=2)

    # Draw X's and O's
    symbols = {0: "", 1: "X", -1: "O"}
    colors = {0: "black", 1: "blue", -1: "red"}

    for i in range(3):
        for j in range(3):
            symbol = symbols[int(state[i, j])]
            if symbol:
                ax.text(
                    j,
                    2 - i,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=24,
                    fontweight="bold",
                    color=colors[int(state[i, j])],
                )

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["2", "1", "0"])  # Flip for display
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def visualize_search_tree(
    root: MCTSNode,
    max_depth: int = 2,
    title: str = "MCTS Search Tree",
    save_path: Optional[str] = None,
) -> None:
    """Visualize the MCTS search tree structure.

    Args:
        root: Root node of the search tree
        max_depth: Maximum depth to visualize
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect nodes up to max_depth
    nodes_by_depth = {0: [root]}

    for depth in range(max_depth):
        if depth not in nodes_by_depth:
            break
        nodes_by_depth[depth + 1] = []
        for node in nodes_by_depth[depth]:
            nodes_by_depth[depth + 1].extend(node.children.values())

    # Calculate positions
    node_positions = {}
    colors = []
    sizes = []
    labels = []

    for depth, nodes in nodes_by_depth.items():
        if not nodes:
            continue

        y = max_depth - depth
        x_positions = np.linspace(0, 10, len(nodes)) if len(nodes) > 1 else [5]

        for i, node in enumerate(nodes):
            x = x_positions[i]
            node_positions[id(node)] = (x, y)

            # Node color based on average reward
            reward = node.average_reward
            colors.append(reward)

            # Node size based on visits
            sizes.append(max(50, node.visits * 5))

            # Node label
            if depth == 0:
                labels.append(f"Root\nV:{node.visits}")
            else:
                action_str = str(node.action_taken) if node.action_taken else "?"
                labels.append(f"{action_str}\nV:{node.visits}\nR:{reward:.2f}")

    # Draw nodes
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=colors,
        s=sizes,
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
    )

    # Draw edges
    for depth in range(max_depth):
        if depth not in nodes_by_depth or depth + 1 not in nodes_by_depth:
            continue

        for parent in nodes_by_depth[depth]:
            parent_pos = node_positions[id(parent)]
            for child in parent.children.values():
                if id(child) in node_positions:
                    child_pos = node_positions[id(child)]
                    ax.plot(
                        [parent_pos[0], child_pos[0]],
                        [parent_pos[1], child_pos[1]],
                        "k-",
                        alpha=0.5,
                        linewidth=1,
                    )

    # Add labels
    for i, (pos, label) in enumerate(zip(node_positions.values(), labels)):
        ax.annotate(
            label,
            pos,
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Format plot
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, max_depth + 0.5)
    ax.set_xlabel("Tree Width")
    ax.set_ylabel("Tree Depth")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Average Reward", rotation=270, labelpad=15)

    # Legend for node sizes
    legend_sizes = [50, 100, 200]
    legend_labels = ["Low visits", "Medium visits", "High visits"]
    legend_handles = [
        plt.scatter([], [], s=s, c="gray", alpha=0.7, edgecolors="black")
        for s in legend_sizes
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        title="Node Visits",
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved search tree to {save_path}")

    plt.show()


def compare_mcts_runs(
    key: jnp.ndarray,
    state: jnp.ndarray,
    num_runs: int = 5,
    num_simulations: int = 50,
    title: str = "MCTS Consistency Analysis",
    save_path: Optional[str] = None,
) -> None:
    """Compare multiple MCTS runs to show consistency and uncertainty.

    Args:
        key: Random key for reproducibility
        state: Board state to analyze
        num_runs: Number of MCTS runs to compare
        num_simulations: Simulations per run
        title: Plot title
        save_path: Optional path to save figure
    """
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    # Run MCTS multiple times
    keys = jrand.split(key, num_runs)
    run_results = []

    for i in range(num_runs):
        action, root = mcts.search(keys[i], state, num_simulations)

        # Extract action statistics
        action_stats = {}
        for act, child in root.children.items():
            action_stats[act] = (child.visits, child.average_reward)

        run_results.append(
            {
                "run": i + 1,
                "best_action": action,
                "action_stats": action_stats,
                "root_visits": root.visits,
            }
        )

    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    # Top: Board state
    ax_board = plt.subplot(2, 3, (1, 2))
    ax_board.set_title("Board Position", fontsize=14, fontweight="bold")
    draw_board(ax_board, state)

    # Middle: Action frequency across runs
    ax_freq = plt.subplot(2, 3, 3)
    action_counts = {}
    for result in run_results:
        action = result["best_action"]
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1

    if action_counts:
        actions, counts = zip(*action_counts.items())
        action_labels = [f"({r},{c})" for r, c in actions]
        ax_freq.bar(range(len(actions)), counts, alpha=0.7)
        ax_freq.set_xticks(range(len(actions)))
        ax_freq.set_xticklabels(action_labels, rotation=45)
        ax_freq.set_ylabel("Frequency")
        ax_freq.set_title(f"Best Action Frequency\n({num_runs} runs)")

    # Bottom: Detailed results for each run
    for i, result in enumerate(run_results[:3]):  # Show first 3 runs
        ax = plt.subplot(2, 3, 4 + i)

        # Create simple heatmap for this run
        visits_grid = np.zeros((3, 3))
        for (row, col), (visits, _) in result["action_stats"].items():
            visits_grid[row, col] = visits

        im = ax.imshow(visits_grid, cmap="Blues", aspect="equal")

        # Add text annotations
        for (row, col), (visits, avg_reward) in result["action_stats"].items():
            ax.text(
                col,
                row,
                f"V:{visits}\nR:{avg_reward:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Mark best action
        best_action = result["best_action"]
        if best_action:
            row, col = best_action
            rect = patches.Rectangle(
                (col - 0.4, row - 0.4),
                0.8,
                0.8,
                linewidth=3,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(f"Run {result['run']}: {best_action}", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_yticklabels(["0", "1", "2"])

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to {save_path}")

    plt.show()


def demonstrate_mcts_progression(
    key: jnp.ndarray,
    state: jnp.ndarray,
    simulation_counts: List[int] = [10, 25, 50, 100, 200],
    title: str = "MCTS Convergence Analysis",
    save_path: Optional[str] = None,
) -> None:
    """Show how MCTS decisions change with more simulations.

    Args:
        key: Random key for reproducibility
        state: Board state to analyze
        simulation_counts: List of simulation counts to test
        title: Plot title
        save_path: Optional path to save figure
    """
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Board state in first subplot
    axes[0].set_title("Board Position", fontsize=12, fontweight="bold")
    draw_board(axes[0], state)

    # Test different simulation counts
    for i, num_sims in enumerate(simulation_counts):
        if i + 1 >= len(axes):
            break

        ax = axes[i + 1]

        # Run MCTS
        action, root = mcts.search(key, state, num_sims)

        # Create heatmap
        visits_grid = np.zeros((3, 3))
        reward_grid = np.zeros((3, 3))

        for (row, col), child in root.children.items():
            visits_grid[row, col] = child.visits
            reward_grid[row, col] = child.average_reward

        # Combined metric: visits weighted by reward
        heatmap_data = visits_grid * (reward_grid + 1) / 2  # Normalize rewards to [0,1]

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="equal")

        # Add annotations
        for (row, col), child in root.children.items():
            ax.text(
                col,
                row,
                f"V:{child.visits}\nR:{child.average_reward:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Mark best action
        if action:
            row, col = action
            rect = patches.Rectangle(
                (col - 0.4, row - 0.4),
                0.8,
                0.8,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(f"{num_sims} simulations\nBest: {action}", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_yticklabels(["0", "1", "2"])

    # Hide unused subplots
    for i in range(len(simulation_counts) + 1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved progression analysis to {save_path}")

    plt.show()


def visualize_mcts_vs_exact(
    state: jnp.ndarray,
    key: jnp.ndarray,
    num_simulations: int = 100,
    title: str = "MCTS vs Exact Solution Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Compare MCTS performance against exact minimax solution.

    Args:
        state: Board state to analyze
        key: Random key for MCTS
        num_simulations: Number of MCTS simulations
        title: Plot title
        save_path: Optional path to save figure
    """
    # Run MCTS
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)
    mcts_action, root = mcts.search(key, state, num_simulations)

    # Get exact solution
    exact_action_values = get_all_action_values(state)
    exact_optimal_action = get_optimal_action(state)

    # Classify MCTS performance
    if mcts_action:
        classification, mcts_value, optimal_value = classify_action_optimality(
            mcts_action, state
        )
    else:
        classification, mcts_value, optimal_value = "no_action", 0.0, 0.0

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Board state
    ax_board = axes[0, 0]
    ax_board.set_title("Board Position", fontsize=12, fontweight="bold")
    draw_board(ax_board, state)

    # Top middle: MCTS evaluation
    ax_mcts = axes[0, 1]
    ax_mcts.set_title(
        f"MCTS Evaluation\n({num_simulations} simulations)",
        fontsize=12,
        fontweight="bold",
    )

    # Create MCTS heatmap
    mcts_visits = np.zeros((3, 3))
    mcts_rewards = np.zeros((3, 3))

    for (row, col), child in root.children.items():
        mcts_visits[row, col] = child.visits
        mcts_rewards[row, col] = child.average_reward

    # Normalize MCTS rewards to match exact values range
    if mcts_rewards.max() > 0:
        mcts_rewards_normalized = (mcts_rewards - mcts_rewards.min()) / (
            mcts_rewards.max() - mcts_rewards.min()
        )
        mcts_rewards_normalized = mcts_rewards_normalized * 2 - 1  # Scale to [-1, 1]
    else:
        mcts_rewards_normalized = mcts_rewards

    im_mcts = ax_mcts.imshow(
        mcts_rewards_normalized, cmap="RdYlGn", aspect="equal", vmin=-1, vmax=1
    )

    # Add MCTS annotations
    for (row, col), child in root.children.items():
        text = f"V:{child.visits}\nR:{child.average_reward:.2f}"
        ax_mcts.text(col, row, text, ha="center", va="center", fontsize=8)

    # Mark MCTS choice
    if mcts_action:
        row, col = mcts_action
        rect = patches.Rectangle(
            (col - 0.4, row - 0.4),
            0.8,
            0.8,
            linewidth=3,
            edgecolor="blue",
            facecolor="none",
        )
        ax_mcts.add_patch(rect)

    ax_mcts.set_xticks(range(3))
    ax_mcts.set_yticks(range(3))
    ax_mcts.set_xticklabels(["0", "1", "2"])
    ax_mcts.set_yticklabels(["0", "1", "2"])

    # Top right: Exact solution
    ax_exact = axes[0, 2]
    ax_exact.set_title("Exact Minimax Solution", fontsize=12, fontweight="bold")

    # Create exact solution heatmap
    exact_grid = np.full((3, 3), np.nan)
    for (row, col), value in exact_action_values.items():
        exact_grid[row, col] = value

    im_exact = ax_exact.imshow(
        exact_grid, cmap="RdYlGn", aspect="equal", vmin=-1, vmax=1
    )

    # Add exact value annotations
    for (row, col), value in exact_action_values.items():
        ax_exact.text(
            col,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Mark optimal action
    if exact_optimal_action:
        row, col = exact_optimal_action
        rect = patches.Rectangle(
            (col - 0.4, row - 0.4),
            0.8,
            0.8,
            linewidth=3,
            edgecolor="gold",
            facecolor="none",
        )
        ax_exact.add_patch(rect)

    ax_exact.set_xticks(range(3))
    ax_exact.set_yticks(range(3))
    ax_exact.set_xticklabels(["0", "1", "2"])
    ax_exact.set_yticklabels(["0", "1", "2"])

    # Bottom row: Analysis
    ax_comparison = axes[1, 0]
    ax_comparison.set_title("Action Comparison", fontsize=12, fontweight="bold")

    # Create comparison table
    comparison_data = []
    all_actions = set(exact_action_values.keys()) | set(root.children.keys())

    for action in sorted(all_actions):
        row_data = {
            "Action": f"({action[0]},{action[1]})",
            "MCTS Visits": root.children[action].visits
            if action in root.children
            else 0,
            "MCTS Reward": root.children[action].average_reward
            if action in root.children
            else 0.0,
            "Exact Value": exact_action_values.get(action, 0.0),
        }
        comparison_data.append(row_data)

    # Create table
    table_text = []
    headers = ["Action", "MCTS V", "MCTS R", "Exact V"]
    table_text.append(headers)

    for data in comparison_data[:6]:  # Show top 6 actions
        row = [
            data["Action"],
            str(data["MCTS Visits"]),
            f"{data['MCTS Reward']:.2f}",
            f"{data['Exact Value']:.1f}",
        ]
        table_text.append(row)

    ax_comparison.axis("tight")
    ax_comparison.axis("off")
    table = ax_comparison.table(
        cellText=table_text[1:], colLabels=table_text[0], cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Bottom middle: Performance metrics
    ax_metrics = axes[1, 1]
    ax_metrics.set_title("Performance Analysis", fontsize=12, fontweight="bold")

    metrics_text = [
        f"MCTS Choice: {mcts_action}",
        f"Optimal Choice: {exact_optimal_action}",
        f"Classification: {classification.upper()}",
        f"MCTS Value: {mcts_value:.2f}",
        f"Optimal Value: {optimal_value:.2f}",
        f"Value Difference: {abs(optimal_value - mcts_value):.2f}",
        f"Total Simulations: {num_simulations}",
        f"Root Visits: {root.visits}",
    ]

    ax_metrics.axis("off")
    for i, text in enumerate(metrics_text):
        color = (
            "green"
            if classification == "optimal"
            else "orange"
            if classification == "suboptimal"
            else "red"
        )
        if "Classification" in text:
            ax_metrics.text(
                0.1,
                0.9 - i * 0.1,
                text,
                fontsize=11,
                fontweight="bold",
                color=color,
                transform=ax_metrics.transAxes,
            )
        else:
            ax_metrics.text(
                0.1, 0.9 - i * 0.1, text, fontsize=10, transform=ax_metrics.transAxes
            )

    # Bottom right: Value correlation plot
    ax_corr = axes[1, 2]
    ax_corr.set_title("MCTS vs Exact Values", fontsize=12, fontweight="bold")

    # Collect data for correlation plot
    mcts_values = []
    exact_values = []
    action_labels = []

    for action in exact_action_values.keys():
        if action in root.children:
            mcts_values.append(root.children[action].average_reward)
            exact_values.append(exact_action_values[action])
            action_labels.append(f"({action[0]},{action[1]})")

    if mcts_values and exact_values:
        ax_corr.scatter(exact_values, mcts_values, alpha=0.7, s=100)

        # Add labels
        for i, label in enumerate(action_labels):
            ax_corr.annotate(
                label,
                (exact_values[i], mcts_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Add diagonal line (perfect correlation)
        min_val = min(min(exact_values), min(mcts_values))
        max_val = max(max(exact_values), max(mcts_values))
        ax_corr.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.5,
            label="Perfect correlation",
        )

        ax_corr.set_xlabel("Exact Minimax Value")
        ax_corr.set_ylabel("MCTS Average Reward")
        ax_corr.legend()
        ax_corr.grid(True, alpha=0.3)

    # Add colorbars
    fig.colorbar(im_mcts, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im_exact, ax=axes[0, 2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved MCTS vs Exact comparison to {save_path}")

    plt.show()


def visualize_mcts_vs_theoretical(
    state: jnp.ndarray,
    key: jnp.ndarray,
    num_simulations: int = 100,
    title: str = "MCTS vs Theoretical Optimal",
    save_path: Optional[str] = None,
) -> None:
    """Compare MCTS performance against theoretical optimal play.

    Args:
        state: Board state to analyze
        key: Random key for MCTS
        num_simulations: Number of MCTS simulations
        title: Plot title
        save_path: Optional path to save figure
    """
    # Run MCTS
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)
    mcts_action, root = mcts.search(key, state, num_simulations)

    # Get theoretical solution
    theoretical_action_values = get_theoretical_action_values(state)
    theoretical_optimal_action = get_theoretical_optimal_action(state)

    # Classify MCTS performance
    if mcts_action and mcts_action in theoretical_action_values:
        mcts_value = theoretical_action_values[mcts_action]
        optimal_value = (
            max(theoretical_action_values.values())
            if theoretical_action_values
            else 0.0
        )
        value_diff = abs(optimal_value - mcts_value)

        if value_diff < 0.05:
            classification = "optimal"
        elif value_diff < 0.2:
            classification = "good"
        else:
            classification = "suboptimal"
    else:
        classification = "unknown"
        mcts_value = 0.0
        optimal_value = 0.0

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Board state
    ax_board = axes[0, 0]
    ax_board.set_title("Board Position", fontsize=12, fontweight="bold")
    draw_board(ax_board, state)

    # Top middle: MCTS evaluation
    ax_mcts = axes[0, 1]
    ax_mcts.set_title(
        f"MCTS Evaluation\n({num_simulations} simulations)",
        fontsize=12,
        fontweight="bold",
    )

    # Create MCTS heatmap
    mcts_visits = np.zeros((3, 3))
    mcts_rewards = np.zeros((3, 3))

    for (row, col), child in root.children.items():
        mcts_visits[row, col] = child.visits
        mcts_rewards[row, col] = child.average_reward

    im_mcts = ax_mcts.imshow(
        mcts_rewards, cmap="RdYlGn", aspect="equal", vmin=-1, vmax=1
    )

    # Add MCTS annotations
    for (row, col), child in root.children.items():
        text = f"V:{child.visits}\nR:{child.average_reward:.2f}"
        ax_mcts.text(col, row, text, ha="center", va="center", fontsize=8)

    # Mark MCTS choice
    if mcts_action:
        row, col = mcts_action
        rect = patches.Rectangle(
            (col - 0.4, row - 0.4),
            0.8,
            0.8,
            linewidth=3,
            edgecolor="blue",
            facecolor="none",
        )
        ax_mcts.add_patch(rect)

    ax_mcts.set_xticks(range(3))
    ax_mcts.set_yticks(range(3))
    ax_mcts.set_xticklabels(["0", "1", "2"])
    ax_mcts.set_yticklabels(["0", "1", "2"])

    # Top right: Theoretical solution
    ax_theoretical = axes[0, 2]
    ax_theoretical.set_title("Theoretical Optimal", fontsize=12, fontweight="bold")

    # Create theoretical solution heatmap
    theoretical_grid = np.full((3, 3), np.nan)
    for (row, col), value in theoretical_action_values.items():
        theoretical_grid[row, col] = value

    im_theoretical = ax_theoretical.imshow(
        theoretical_grid, cmap="RdYlGn", aspect="equal", vmin=-1, vmax=1
    )

    # Add theoretical value annotations
    for (row, col), value in theoretical_action_values.items():
        color = "white" if abs(value) > 0.5 else "black"
        ax_theoretical.text(
            col,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    # Mark optimal action
    if theoretical_optimal_action:
        row, col = theoretical_optimal_action
        rect = patches.Rectangle(
            (col - 0.4, row - 0.4),
            0.8,
            0.8,
            linewidth=3,
            edgecolor="gold",
            facecolor="none",
        )
        ax_theoretical.add_patch(rect)

    ax_theoretical.set_xticks(range(3))
    ax_theoretical.set_yticks(range(3))
    ax_theoretical.set_xticklabels(["0", "1", "2"])
    ax_theoretical.set_yticklabels(["0", "1", "2"])

    # Bottom row: Analysis
    ax_comparison = axes[1, 0]
    ax_comparison.set_title("Action Comparison", fontsize=12, fontweight="bold")

    # Create comparison table
    comparison_data = []
    all_actions = set(theoretical_action_values.keys()) | set(root.children.keys())

    for action in sorted(all_actions):
        mcts_visits = root.children[action].visits if action in root.children else 0
        mcts_reward = (
            root.children[action].average_reward if action in root.children else 0.0
        )
        theoretical_value = theoretical_action_values.get(action, 0.0)

        comparison_data.append(
            [
                f"({action[0]},{action[1]})",
                str(mcts_visits),
                f"{mcts_reward:.2f}",
                f"{theoretical_value:.1f}",
            ]
        )

    headers = ["Action", "MCTS V", "MCTS R", "Theory V"]

    ax_comparison.axis("tight")
    ax_comparison.axis("off")
    table = ax_comparison.table(
        cellText=comparison_data, colLabels=headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Bottom middle: Performance metrics
    ax_metrics = axes[1, 1]
    ax_metrics.set_title("Performance Analysis", fontsize=12, fontweight="bold")

    metrics_text = [
        f"MCTS Choice: {mcts_action}",
        f"Optimal Choice: {theoretical_optimal_action}",
        f"Classification: {classification.upper()}",
        f"MCTS Value: {mcts_value:.2f}",
        f"Optimal Value: {optimal_value:.2f}",
        f"Value Difference: {abs(optimal_value - mcts_value):.2f}",
        f"Total Simulations: {num_simulations}",
        f"Root Visits: {root.visits}",
    ]

    ax_metrics.axis("off")
    for i, text in enumerate(metrics_text):
        color = (
            "green"
            if classification == "optimal"
            else "orange"
            if classification == "good"
            else "red"
        )
        if "Classification" in text:
            ax_metrics.text(
                0.1,
                0.9 - i * 0.1,
                text,
                fontsize=11,
                fontweight="bold",
                color=color,
                transform=ax_metrics.transAxes,
            )
        else:
            ax_metrics.text(
                0.1, 0.9 - i * 0.1, text, fontsize=10, transform=ax_metrics.transAxes
            )

    # Bottom right: Key insights
    ax_insights = axes[1, 2]
    ax_insights.set_title("Key Insights", fontsize=12, fontweight="bold")

    insights = [
        "• Corners/center are optimal on empty board",
        "• Edges have slight disadvantage (-0.2)",
        "• MCTS learns these preferences from simulation",
        "• More simulations → better decisions",
        "• Model uncertainty creates exploration",
    ]

    ax_insights.axis("off")
    for i, insight in enumerate(insights):
        ax_insights.text(
            0.05, 0.9 - i * 0.15, insight, fontsize=10, transform=ax_insights.transAxes
        )

    # Add colorbars
    fig.colorbar(im_mcts, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im_theoretical, ax=axes[0, 2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved MCTS vs Theoretical comparison to {save_path}")

    plt.show()


def benchmark_comparison_analysis(
    key: jnp.ndarray,
    simulation_counts: List[int] = [10, 25, 50, 100, 200, 500],
    num_positions: int = 5,
    title: str = "MCTS Accuracy vs Simulation Count",
    save_path: Optional[str] = None,
) -> None:
    """Analyze MCTS accuracy vs exact solution across different simulation counts.

    Args:
        key: Random key
        simulation_counts: List of simulation counts to test
        num_positions: Number of random positions to test
        title: Plot title
        save_path: Optional path to save figure
    """
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    # Generate test positions
    test_positions = []
    keys = jrand.split(key, num_positions)

    for i in range(num_positions):
        # Create random mid-game positions
        state = create_empty_board()
        num_moves = jrand.randint(keys[i], (), 1, 6)  # 1-5 moves

        pos_key = keys[i]
        for move in range(num_moves):
            legal_actions = get_legal_actions_ttt(state)
            if not legal_actions:
                break

            pos_key, action_key = jrand.split(pos_key)
            action_idx = jrand.randint(action_key, (), 0, len(legal_actions))
            row, col = legal_actions[action_idx]

            # Alternate between X and O
            player = 1 if move % 2 == 0 else -1
            state = state.at[row, col].set(player)

        test_positions.append(state)

    # Test MCTS accuracy for each simulation count
    results = []

    for num_sims in simulation_counts:
        position_results = []

        for pos_idx, state in enumerate(test_positions):
            pos_key = jrand.split(key, num_positions + 1)[pos_idx]

            # Run MCTS
            mcts_action, root = mcts.search(pos_key, state, num_sims)

            # Get exact solution
            exact_action_values = get_all_action_values(state)
            exact_optimal_action = get_optimal_action(state)

            if mcts_action and exact_action_values:
                classification, mcts_value, optimal_value = classify_action_optimality(
                    mcts_action, state
                )
                value_difference = abs(optimal_value - mcts_value)
                is_optimal = classification == "optimal"

                position_results.append(
                    {
                        "position": pos_idx,
                        "mcts_action": mcts_action,
                        "optimal_action": exact_optimal_action,
                        "classification": classification,
                        "value_difference": value_difference,
                        "is_optimal": is_optimal,
                        "mcts_value": mcts_value,
                        "optimal_value": optimal_value,
                    }
                )

        # Aggregate results for this simulation count
        if position_results:
            accuracy = sum(r["is_optimal"] for r in position_results) / len(
                position_results
            )
            avg_value_diff = sum(r["value_difference"] for r in position_results) / len(
                position_results
            )

            results.append(
                {
                    "num_simulations": num_sims,
                    "accuracy": accuracy,
                    "avg_value_difference": avg_value_diff,
                    "position_results": position_results,
                }
            )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy vs simulation count
    ax1 = axes[0, 0]
    if results:
        sim_counts = [r["num_simulations"] for r in results]
        accuracies = [r["accuracy"] for r in results]

        ax1.plot(sim_counts, accuracies, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Simulations")
        ax1.set_ylabel("Accuracy (% Optimal Actions)")
        ax1.set_title("MCTS Accuracy vs Simulation Count")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)

    # Value difference vs simulation count
    ax2 = axes[0, 1]
    if results:
        value_diffs = [r["avg_value_difference"] for r in results]

        ax2.plot(sim_counts, value_diffs, "ro-", linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Simulations")
        ax2.set_ylabel("Average Value Difference")
        ax2.set_title("MCTS Value Error vs Simulation Count")
        ax2.grid(True, alpha=0.3)

    # Classification breakdown for highest simulation count
    ax3 = axes[1, 0]
    if results:
        best_result = results[-1]  # Highest simulation count
        classifications = [r["classification"] for r in best_result["position_results"]]

        from collections import Counter

        class_counts = Counter(classifications)

        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = {"optimal": "green", "suboptimal": "orange", "blunder": "red"}
        pie_colors = [colors.get(label, "gray") for label in labels]

        ax3.pie(
            sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90
        )
        ax3.set_title(
            f"Action Quality Distribution\n({best_result['num_simulations']} simulations)"
        )

    # Performance summary table
    ax4 = axes[1, 1]
    ax4.set_title("Performance Summary")

    if results:
        table_data = []
        for r in results:
            row = [
                f"{r['num_simulations']}",
                f"{r['accuracy']:.1%}",
                f"{r['avg_value_difference']:.3f}",
            ]
            table_data.append(row)

        headers = ["Simulations", "Accuracy", "Avg Error"]

        ax4.axis("tight")
        ax4.axis("off")
        table = ax4.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved benchmark comparison to {save_path}")

    plt.show()

    return results


def create_all_visualizations(
    key: jnp.ndarray, output_dir: str = "examples/programmable_mcts/figs"
) -> None:
    """Create all visualization examples for Programmable MCTS.

    Args:
        key: Random key for reproducibility
        output_dir: Directory to save figures
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Creating Programmable MCTS visualizations...")

    # 1. Empty board analysis
    print("1. Empty board MCTS analysis...")
    key1, key = jrand.split(key)
    empty_board = create_empty_board()

    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)
    action, root = mcts.search(key1, empty_board, 100)

    action_stats = {
        act: (child.visits, child.average_reward)
        for act, child in root.children.items()
    }

    visualize_board_with_heatmap(
        empty_board,
        action_stats,
        title="MCTS on Empty Board (100 simulations)",
        save_path=f"{output_dir}/empty_board_analysis.png",
    )

    # 2. Mid-game position
    print("2. Mid-game position analysis...")
    key2, key = jrand.split(key)
    mid_game = create_empty_board()
    mid_game = mid_game.at[1, 1].set(1)  # X in center
    mid_game = mid_game.at[0, 0].set(-1)  # O in corner

    action, root = mcts.search(key2, mid_game, 100)
    action_stats = {
        act: (child.visits, child.average_reward)
        for act, child in root.children.items()
    }

    visualize_board_with_heatmap(
        mid_game,
        action_stats,
        title="MCTS on Mid-Game Position (O to move)",
        save_path=f"{output_dir}/mid_game_analysis.png",
    )

    # 3. Search tree visualization
    print("3. Search tree structure...")
    key3, key = jrand.split(key)
    action, root = mcts.search(key3, empty_board, 50)

    visualize_search_tree(
        root,
        max_depth=2,
        title="MCTS Search Tree (50 simulations, depth 2)",
        save_path=f"{output_dir}/search_tree.png",
    )

    # 4. Consistency analysis
    print("4. Multi-run consistency analysis...")
    key4, key = jrand.split(key)
    compare_mcts_runs(
        key4,
        mid_game,
        num_runs=5,
        num_simulations=50,
        title="MCTS Consistency (5 runs, 50 simulations each)",
        save_path=f"{output_dir}/consistency_analysis.png",
    )

    # 5. Convergence analysis
    print("5. Simulation count convergence...")
    key5, key = jrand.split(key)
    demonstrate_mcts_progression(
        key5,
        mid_game,
        simulation_counts=[10, 25, 50, 100, 200],
        title="MCTS Convergence with Simulation Count",
        save_path=f"{output_dir}/convergence_analysis.png",
    )

    # 6. MCTS vs Exact solution comparison (empty board)
    print("6. MCTS vs Exact solution comparison (empty board)...")
    key6, key = jrand.split(key)

    # Validate theoretical solver first
    print("   Validating theoretical solver...")
    if validate_theoretical_solver():
        visualize_mcts_vs_theoretical(
            empty_board,
            key6,
            num_simulations=100,
            title="MCTS vs Theoretical Optimal: Empty Board",
            save_path=f"{output_dir}/mcts_vs_theoretical_empty.png",
        )
    else:
        print("   WARNING: Theoretical solver validation failed - skipping comparison")

    # 7. MCTS vs Exact solution comparison (mid-game)
    print("7. MCTS vs Exact solution comparison (mid-game)...")
    key7, key = jrand.split(key)
    visualize_mcts_vs_exact(
        mid_game,
        key7,
        num_simulations=100,
        title="MCTS vs Exact Solution: Mid-Game Position",
        save_path=f"{output_dir}/mcts_vs_exact_midgame.png",
    )

    # 8. MCTS accuracy analysis across simulation counts
    print("8. MCTS accuracy analysis across simulation counts...")
    key8, key = jrand.split(key)
    benchmark_comparison_analysis(
        key8,
        simulation_counts=[10, 25, 50, 100, 200],
        num_positions=8,
        title="MCTS Accuracy vs Simulation Count",
        save_path=f"{output_dir}/mcts_accuracy_analysis.png",
    )

    print(f"\nAll visualizations saved to {output_dir}/")
    print("Available figures:")
    print("- empty_board_analysis.png: MCTS evaluation on empty board")
    print("- mid_game_analysis.png: MCTS evaluation on mid-game position")
    print("- search_tree.png: MCTS search tree structure")
    print("- consistency_analysis.png: Multiple run comparison")
    print("- convergence_analysis.png: Effect of simulation count")
    print("- mcts_vs_exact_empty.png: MCTS vs exact solution on empty board")
    print("- mcts_vs_exact_midgame.png: MCTS vs exact solution on mid-game")
    print("- mcts_accuracy_analysis.png: MCTS accuracy across simulation counts")
