"""
Test specific Tic-Tac-Toe positions to debug minimax.
"""

from .core import create_empty_board, print_board
from .exact_solver import get_all_action_values


def test_winning_position():
    """Test a position where X can win in one move."""

    print("=== Testing Near-Win Position ===")

    # Create position where X can win
    state = create_empty_board()
    state = state.at[0, 0].set(1)  # X
    state = state.at[0, 1].set(1)  # X
    state = state.at[1, 0].set(-1)  # O
    state = state.at[1, 1].set(-1)  # O
    # X can win by playing (0, 2)

    print("Board position (X to move):")
    print_board(state)

    action_values = get_all_action_values(state)
    print("\nAction values:")
    for action, value in action_values.items():
        print(f"  {action}: {value:.3f}")

    # (0, 2) should have value 1.0 (X wins)
    # Other moves should have lower values


def test_losing_position():
    """Test a position where the current player will lose."""

    print("\n=== Testing Near-Loss Position ===")

    # Create position where O is about to win, X to move
    state = create_empty_board()
    state = state.at[0, 0].set(-1)  # O
    state = state.at[0, 1].set(-1)  # O
    state = state.at[1, 0].set(1)  # X
    state = state.at[1, 1].set(1)  # X
    state = state.at[2, 0].set(1)  # X
    # O threatens (0, 2), X must block but then O can win with (2, 2)

    print("Board position (X to move, but O threatens win):")
    print_board(state)

    action_values = get_all_action_values(state)
    print("\nAction values:")
    for action, value in action_values.items():
        print(f"  {action}: {value:.3f}")

    # All moves should be negative (X will lose)


def test_simple_positions():
    """Test very simple positions."""

    print("\n=== Testing Simple Positions ===")

    # Position 1: X played center, O to move
    state1 = create_empty_board()
    state1 = state1.at[1, 1].set(1)  # X in center

    print("X played center, O to move:")
    print_board(state1)

    action_values1 = get_all_action_values(state1)
    print("Action values for O:")
    for action, value in sorted(action_values1.items()):
        row, col = action
        position_type = (
            "corner" if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)] else "edge"
        )
        print(f"  {action} ({position_type}): {value:.3f}")

    # O should prefer corners over edges
    corner_values = [
        v
        for (r, c), v in action_values1.items()
        if (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]
    ]
    edge_values = [
        v
        for (r, c), v in action_values1.items()
        if (r, c) in [(0, 1), (1, 0), (1, 2), (2, 1)]
    ]

    if corner_values and edge_values:
        print(f"Corner values: {corner_values}")
        print(f"Edge values: {edge_values}")
        print(f"Corners better than edges: {min(corner_values) >= max(edge_values)}")


if __name__ == "__main__":
    test_winning_position()
    test_losing_position()
    test_simple_positions()
