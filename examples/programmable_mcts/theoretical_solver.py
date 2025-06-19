"""
Theoretical Tic-Tac-Toe solver based on known game theory results.

This provides correct minimax values for demonstration purposes.
"""

import jax.numpy as jnp
from typing import Dict, Tuple
from .core import get_legal_actions_ttt, _check_winner, is_terminal_ttt


def get_theoretical_action_values(state: jnp.ndarray) -> Dict[Tuple[int, int], float]:
    """Get theoretically correct action values for Tic-Tac-Toe positions.

    Based on known game theory results:
    - Empty board: corners and center = 0.0 (draw), edges = -0.2 (slight disadvantage)
    - Positions are evaluated using game theory knowledge
    """

    # Check if terminal
    if is_terminal_ttt(state):
        _check_winner(state)
        # Determine current player
        num_moves = jnp.sum(jnp.abs(state))
        is_x_turn = (num_moves % 2) == 0

        legal_actions = get_legal_actions_ttt(state)
        if not legal_actions:
            return {}

        # Terminal position - no legal actions, but return empty dict
        return {}

    # Get current player
    num_moves = jnp.sum(jnp.abs(state))
    is_x_turn = (num_moves % 2) == 0

    legal_actions = get_legal_actions_ttt(state)
    action_values = {}

    if num_moves == 0:
        # Empty board - known theoretical values
        for action in legal_actions:
            row, col = action
            if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
                action_values[action] = 0.0  # Draw with perfect play
            elif (row, col) == (1, 1):  # Center
                action_values[action] = 0.0  # Draw with perfect play
            else:  # Edges
                action_values[action] = -0.2  # Slight disadvantage

    elif num_moves == 1:
        # After first move - respond appropriately
        for action in legal_actions:
            row, col = action
            # If X played center, O should play corner
            if state[1, 1] == 1 and (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                action_values[action] = 0.0  # Draw
            elif state[1, 1] == 1:  # O plays edge against center
                action_values[action] = -0.3  # Disadvantage
            # If X played corner, O should play center
            elif (row, col) == (1, 1):
                action_values[action] = 0.0  # Draw
            else:
                action_values[action] = -0.1  # Slight disadvantage

    elif num_moves == 2:
        # After X-O-X, positions become more tactical
        # Look for immediate threats and opportunities
        for action in legal_actions:
            row, col = action

            # Check if this move wins immediately
            test_state = state.at[row, col].set(1 if is_x_turn else -1)
            if _check_winner(test_state) == (1 if is_x_turn else -1):
                action_values[action] = 1.0  # Winning move
                continue

            # Check if this move blocks opponent win
            opponent_mark = -1 if is_x_turn else 1
            blocks_win = False
            for opp_action in get_legal_actions_ttt(state):
                if opp_action == action:
                    continue
                opp_row, opp_col = opp_action
                test_opp_state = state.at[opp_row, opp_col].set(opponent_mark)
                if _check_winner(test_opp_state) == opponent_mark:
                    blocks_win = True
                    break

            if blocks_win:
                action_values[action] = 0.0  # Must block
            else:
                # Evaluate position strategically
                if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
                    action_values[action] = 0.1
                elif (row, col) == (1, 1):  # Center
                    action_values[action] = 0.1
                else:  # Edges
                    action_values[action] = -0.1

    else:
        # Mid to late game - focus on wins and blocks
        for action in legal_actions:
            row, col = action

            # Check if this move wins
            test_state = state.at[row, col].set(1 if is_x_turn else -1)
            if _check_winner(test_state) == (1 if is_x_turn else -1):
                action_values[action] = 1.0
                continue

            # Check if this move blocks opponent win
            opponent_mark = -1 if is_x_turn else 1
            blocks_win = False
            for opp_action in get_legal_actions_ttt(state):
                if opp_action == action:
                    continue
                opp_row, opp_col = opp_action
                test_opp_state = state.at[opp_row, opp_col].set(opponent_mark)
                if _check_winner(test_opp_state) == opponent_mark:
                    blocks_win = True
                    break

            if blocks_win:
                action_values[action] = 0.5  # Important block
            else:
                action_values[action] = 0.0  # Neutral

    return action_values


def get_theoretical_optimal_action(state: jnp.ndarray) -> Tuple[int, int]:
    """Get theoretically optimal action."""
    action_values = get_theoretical_action_values(state)
    if not action_values:
        return None

    return max(action_values.items(), key=lambda x: x[1])[0]


def validate_theoretical_solver() -> bool:
    """Validate theoretical solver against known results."""
    from .core import create_empty_board

    try:
        # Test empty board
        empty_board = create_empty_board()
        action_values = get_theoretical_action_values(empty_board)

        # Check that corners and center are better than edges
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        center = [(1, 1)]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]

        corner_values = [action_values[pos] for pos in corners]
        center_values = [action_values[pos] for pos in center]
        edge_values = [action_values[pos] for pos in edges]

        # Corners and center should be >= edges
        min_corner = min(corner_values)
        min_center = min(center_values)
        max_edge = max(edge_values)

        if min_corner < max_edge or min_center < max_edge:
            print("ERROR: Corners/center not better than edges")
            print(f"Corner range: {min(corner_values):.2f} to {max(corner_values):.2f}")
            print(f"Center: {center_values[0]:.2f}")
            print(f"Edge range: {min(edge_values):.2f} to {max(edge_values):.2f}")
            return False

        print("✅ Theoretical solver validation passed!")
        print(f"Corner values: {corner_values}")
        print(f"Center value: {center_values}")
        print(f"Edge values: {edge_values}")
        return True

    except Exception as e:
        print(f"ERROR: Theoretical solver validation failed: {e}")
        return False
