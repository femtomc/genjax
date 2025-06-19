"""
Exact minimax solver for Tic-Tac-Toe.

This module provides the optimal solution to Tic-Tac-Toe using minimax algorithm
with alpha-beta pruning. Used for comparison with MCTS performance.
"""

import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from .core import get_legal_actions_ttt, is_terminal_ttt, _check_winner


def minimax_value(
    state: jnp.ndarray,
    is_maximizing: bool,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    memo: Optional[Dict] = None,
) -> float:
    """Compute exact minimax value for a Tic-Tac-Toe position.

    Args:
        state: 3x3 board state
        is_maximizing: True if current player is maximizing (X), False for minimizing (O)
        alpha: Alpha value for alpha-beta pruning
        beta: Beta value for alpha-beta pruning
        memo: Memoization dictionary for dynamic programming

    Returns:
        Exact minimax value: 1.0 (X wins), 0.0 (draw), -1.0 (O wins)
    """
    if memo is None:
        memo = {}

    # Convert state to hashable key for memoization
    state_key = (tuple(state.flatten().tolist()), is_maximizing)
    if state_key in memo:
        return memo[state_key]

    # Terminal state evaluation
    if is_terminal_ttt(state):
        winner = _check_winner(state)
        if winner == 0:  # Draw
            result = 0.0
        elif (winner == 1 and is_maximizing) or (winner == -1 and not is_maximizing):
            # Current player wins
            result = 1.0
        else:
            # Current player loses
            result = -1.0
        memo[state_key] = result
        return result

    legal_actions = get_legal_actions_ttt(state)

    if is_maximizing:  # X's turn (maximizing player)
        max_eval = float("-inf")
        for action in legal_actions:
            row, col = action
            new_state = state.at[row, col].set(1)  # X places mark
            eval_score = minimax_value(new_state, False, alpha, beta, memo)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
        memo[state_key] = max_eval
        return max_eval
    else:  # O's turn (minimizing player)
        min_eval = float("inf")
        for action in legal_actions:
            row, col = action
            new_state = state.at[row, col].set(-1)  # O places mark
            eval_score = minimax_value(new_state, True, alpha, beta, memo)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
        memo[state_key] = min_eval
        return min_eval


def get_optimal_action(state: jnp.ndarray) -> Tuple[int, int]:
    """Get the optimal action for current position using minimax.

    Args:
        state: Current board state

    Returns:
        Optimal action (row, col) tuple
    """
    # Determine current player
    num_moves = jnp.sum(jnp.abs(state))
    is_x_turn = (num_moves % 2) == 0
    is_maximizing = is_x_turn

    legal_actions = get_legal_actions_ttt(state)
    if not legal_actions:
        return None

    best_action = None
    best_value = float("-inf")  # Always maximize from current player's perspective
    memo = {}

    for action in legal_actions:
        row, col = action
        # Make the move
        player_mark = 1 if is_x_turn else -1
        new_state = state.at[row, col].set(player_mark)

        # Evaluate the resulting position from the opponent's perspective
        action_value = minimax_value(new_state, not is_maximizing, memo=memo)

        # Update best action (always maximize from current player's perspective)
        # Since action_value is from opponent's perspective, negate it
        current_player_value = -action_value
        if current_player_value > best_value:
            best_value = current_player_value
            best_action = action

    return best_action


def get_all_action_values(state: jnp.ndarray) -> Dict[Tuple[int, int], float]:
    """Get exact minimax values for all legal actions.

    Args:
        state: Current board state

    Returns:
        Dictionary mapping actions to their exact minimax values
    """
    # Determine current player
    num_moves = jnp.sum(jnp.abs(state))
    is_x_turn = (num_moves % 2) == 0
    is_maximizing = is_x_turn

    legal_actions = get_legal_actions_ttt(state)
    action_values = {}
    memo = {}

    for action in legal_actions:
        row, col = action
        # Make the move
        player_mark = 1 if is_x_turn else -1
        new_state = state.at[row, col].set(player_mark)

        # Evaluate the resulting position from the opponent's perspective
        action_value = minimax_value(new_state, not is_maximizing, memo=memo)
        # Convert to current player's perspective
        action_values[action] = -action_value

    return action_values


def classify_action_optimality(
    mcts_action: Tuple[int, int], state: jnp.ndarray
) -> Tuple[str, float, float]:
    """Classify how good an MCTS action is compared to optimal play.

    Args:
        mcts_action: Action chosen by MCTS
        state: Board state

    Returns:
        (classification, mcts_value, optimal_value) tuple where:
        - classification: "optimal", "suboptimal", or "blunder"
        - mcts_value: Exact value of MCTS action
        - optimal_value: Exact value of optimal action
    """
    action_values = get_all_action_values(state)
    optimal_value = max(action_values.values()) if action_values else 0.0

    if mcts_action not in action_values:
        return "invalid", float("-inf"), optimal_value

    mcts_value = action_values[mcts_action]

    # Classification based on value difference
    value_diff = optimal_value - mcts_value

    if abs(value_diff) < 1e-10:  # Essentially equal
        return "optimal", mcts_value, optimal_value
    elif value_diff <= 1.0:  # Small difference (e.g., draw vs slight disadvantage)
        return "suboptimal", mcts_value, optimal_value
    else:  # Large difference (e.g., win vs loss)
        return "blunder", mcts_value, optimal_value


def solve_game_tree(max_depth: int = 4) -> Dict:
    """Pre-solve the Tic-Tac-Toe game tree up to given depth (reduced for performance).

    Args:
        max_depth: Maximum depth to solve (4 = first few moves)

    Returns:
        Dictionary with game tree analysis including:
        - Total positions analyzed
        - Perfect play sequence from empty board
    """
    from .core import create_empty_board

    # Get perfect play sequence (just first few moves)
    perfect_sequence = []
    current_state = create_empty_board()
    current_maximizing = True

    for depth in range(min(max_depth, 4)):  # Limit to first 4 moves
        if is_terminal_ttt(current_state):
            break

        optimal_action = get_optimal_action(current_state)
        if optimal_action is None:
            break

        perfect_sequence.append(optimal_action)
        row, col = optimal_action
        player_mark = 1 if current_maximizing else -1
        current_state = current_state.at[row, col].set(player_mark)
        current_maximizing = not current_maximizing

    # Get optimal value for empty board
    empty_board = create_empty_board()
    optimal_value = minimax_value(empty_board, True)

    return {
        "optimal_value": optimal_value,
        "perfect_sequence": perfect_sequence,
        "game_theoretic_value": "Draw with perfect play"
        if abs(optimal_value) < 1e-10
        else "X wins with perfect play"
        if optimal_value > 0
        else "O wins with perfect play",
    }


def benchmark_solver_performance() -> Dict:
    """Benchmark the exact solver performance.

    Returns:
        Performance statistics including timing and tree search metrics
    """
    import time
    from .core import create_empty_board

    # Test 1: Empty board analysis
    start_time = time.time()
    empty_board = create_empty_board()
    optimal_action = get_optimal_action(empty_board)
    action_values = get_all_action_values(empty_board)
    empty_board_time = time.time() - start_time

    # Test 2: Mid-game position analysis
    start_time = time.time()
    mid_game = create_empty_board()
    mid_game = mid_game.at[1, 1].set(1)  # X in center
    mid_game = mid_game.at[0, 0].set(-1)  # O in corner
    mid_optimal_action = get_optimal_action(mid_game)
    mid_action_values = get_all_action_values(mid_game)
    mid_game_time = time.time() - start_time

    # Test 3: Full game tree analysis
    start_time = time.time()
    game_tree_analysis = solve_game_tree()
    full_tree_time = time.time() - start_time

    return {
        "empty_board": {
            "time_ms": empty_board_time * 1000,
            "optimal_action": optimal_action,
            "num_actions": len(action_values),
            "action_values": action_values,
        },
        "mid_game": {
            "time_ms": mid_game_time * 1000,
            "optimal_action": mid_optimal_action,
            "num_actions": len(mid_action_values),
            "action_values": mid_action_values,
        },
        "full_tree": {"time_ms": full_tree_time * 1000, **game_tree_analysis},
    }


# Game-theoretic knowledge for validation
KNOWN_RESULTS = {
    "empty_board_optimal_actions": [
        (1, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ],  # Center or corners
    "empty_board_value": 0.0,  # Draw with perfect play
    "total_positions": 5478,  # Total unique positions in Tic-Tac-Toe
    "game_tree_complexity": 255168,  # Total game tree nodes
}


def validate_solver() -> bool:
    """Validate the exact solver against known game-theoretic results.

    Returns:
        True if solver produces correct known results
    """
    from .core import create_empty_board

    try:
        # Test 1: Empty board should be a draw
        empty_board = create_empty_board()
        empty_value = minimax_value(empty_board, True)
        if abs(empty_value - KNOWN_RESULTS["empty_board_value"]) > 1e-10:
            print(
                f"ERROR: Empty board value {empty_value}, expected {KNOWN_RESULTS['empty_board_value']}"
            )
            return False

        # Test 2: Optimal first move should be center or corner
        optimal_action = get_optimal_action(empty_board)
        if optimal_action not in KNOWN_RESULTS["empty_board_optimal_actions"]:
            print(
                f"ERROR: Optimal action {optimal_action} not in known optimal actions"
            )
            return False

        # Test 3: All optimal actions should have same value
        action_values = get_all_action_values(empty_board)
        optimal_value = max(action_values.values())
        optimal_actions = [
            action
            for action, value in action_values.items()
            if abs(value - optimal_value) < 1e-10
        ]

        for action in optimal_actions:
            if action not in KNOWN_RESULTS["empty_board_optimal_actions"]:
                print(f"WARNING: Found optimal action {action} not in known list")

        print("✅ Exact solver validation passed!")
        return True

    except Exception as e:
        print(f"ERROR: Solver validation failed: {e}")
        return False
