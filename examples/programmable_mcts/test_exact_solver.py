"""
Test script to debug the exact minimax solver.
"""

from .core import create_empty_board
from .exact_solver import get_all_action_values, get_optimal_action, minimax_value


def test_empty_board_values():
    """Test exact values for empty board positions."""

    print("=== Testing Exact Solver on Empty Board ===")

    empty_board = create_empty_board()

    # Get all action values
    action_values = get_all_action_values(empty_board)
    optimal_action = get_optimal_action(empty_board)

    print(f"Empty board optimal action: {optimal_action}")
    print(f"Empty board root value: {minimax_value(empty_board, True)}")
    print("\nAll action values:")

    # Sort by position for easy reading
    for row in range(3):
        for col in range(3):
            action = (row, col)
            if action in action_values:
                value = action_values[action]
                position_type = get_position_type(row, col)
                print(f"  {action} ({position_type}): {value:.3f}")

    # Check known game theory
    print("\n=== Game Theory Validation ===")

    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    center = [(1, 1)]
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]

    corner_values = [action_values[pos] for pos in corners if pos in action_values]
    center_values = [action_values[pos] for pos in center if pos in action_values]
    edge_values = [action_values[pos] for pos in edges if pos in action_values]

    print(f"Corner values: {corner_values}")
    print(f"Center values: {center_values}")
    print(f"Edge values: {edge_values}")

    # Validate expectations
    expected_optimal_value = 0.0  # Draw with perfect play

    if corner_values:
        avg_corner = sum(corner_values) / len(corner_values)
        print(f"Average corner value: {avg_corner:.3f} (should be ~0.0)")

    if center_values:
        avg_center = sum(center_values) / len(center_values)
        print(f"Average center value: {avg_center:.3f} (should be ~0.0)")

    if edge_values:
        avg_edge = sum(edge_values) / len(edge_values)
        print(f"Average edge value: {avg_edge:.3f} (should be < 0.0)")

    # Check if edges are worse than corners
    if corner_values and edge_values:
        corners_better = all(c >= e for c in corner_values for e in edge_values)
        print(f"Corners ≥ edges: {corners_better} (should be True)")

    return action_values


def get_position_type(row, col):
    """Get position type: corner, center, or edge."""
    if (row, col) == (1, 1):
        return "center"
    elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        return "corner"
    else:
        return "edge"


def debug_minimax_step_by_step():
    """Debug minimax calculation step by step."""

    print("\n=== Debugging Minimax Step by Step ===")

    empty_board = create_empty_board()

    # Test a specific position: corner vs edge
    print("Testing corner move (0,0):")
    corner_state = empty_board.at[0, 0].set(1)  # X plays corner
    corner_value = minimax_value(corner_state, False)  # O's turn
    print(f"  After X plays (0,0), minimax value: {corner_value:.3f}")

    print("Testing edge move (0,1):")
    edge_state = empty_board.at[0, 1].set(1)  # X plays edge
    edge_value = minimax_value(edge_state, False)  # O's turn
    print(f"  After X plays (0,1), minimax value: {edge_value:.3f}")

    print("Testing center move (1,1):")
    center_state = empty_board.at[1, 1].set(1)  # X plays center
    center_value = minimax_value(center_state, False)  # O's turn
    print(f"  After X plays (1,1), minimax value: {center_value:.3f}")

    # From X's perspective (maximizing), we want the move that gives highest value
    print("\nFrom X's perspective:")
    print(f"  Corner (0,0): {-corner_value:.3f}")  # Negate because it's O's value
    print(f"  Edge (0,1): {-edge_value:.3f}")
    print(f"  Center (1,1): {-center_value:.3f}")


if __name__ == "__main__":
    action_values = test_empty_board_values()
    debug_minimax_step_by_step()

    print("\n=== EXPECTED BEHAVIOR ===")
    print("For perfect Tic-Tac-Toe play from empty board:")
    print("- Game result: Draw (0.0)")
    print("- Optimal moves: Corners (0,0), (0,2), (2,0), (2,2) and center (1,1)")
    print("- Suboptimal moves: Edges (0,1), (1,0), (1,2), (2,1)")
    print("- Corner/center values should be 0.0 (draw)")
    print("- Edge values should be negative (losing)")
