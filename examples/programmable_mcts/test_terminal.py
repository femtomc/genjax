"""
Test terminal positions to verify minimax signs.
"""

from .core import create_empty_board, print_board, _check_winner, is_terminal_ttt
from .exact_solver import minimax_value


def test_terminal_positions():
    """Test minimax values on terminal positions."""

    print("=== Testing Terminal Positions ===")

    # X wins
    x_wins = create_empty_board()
    x_wins = x_wins.at[0, 0].set(1)
    x_wins = x_wins.at[0, 1].set(1)
    x_wins = x_wins.at[0, 2].set(1)  # X wins top row
    x_wins = x_wins.at[1, 0].set(-1)
    x_wins = x_wins.at[1, 1].set(-1)

    print("X wins position:")
    print_board(x_wins)
    print(f"Winner: {_check_winner(x_wins)}")
    print(f"Terminal: {is_terminal_ttt(x_wins)}")
    print(f"Minimax value (X perspective): {minimax_value(x_wins, True)}")
    print(f"Minimax value (O perspective): {minimax_value(x_wins, False)}")

    # O wins
    o_wins = create_empty_board()
    o_wins = o_wins.at[0, 0].set(-1)
    o_wins = o_wins.at[0, 1].set(-1)
    o_wins = o_wins.at[0, 2].set(-1)  # O wins top row
    o_wins = o_wins.at[1, 0].set(1)
    o_wins = o_wins.at[1, 1].set(1)

    print("\nO wins position:")
    print_board(o_wins)
    print(f"Winner: {_check_winner(o_wins)}")
    print(f"Terminal: {is_terminal_ttt(o_wins)}")
    print(f"Minimax value (X perspective): {minimax_value(o_wins, True)}")
    print(f"Minimax value (O perspective): {minimax_value(o_wins, False)}")

    # Draw
    draw = create_empty_board()
    draw = draw.at[0, 0].set(1)  # X
    draw = draw.at[0, 1].set(-1)  # O
    draw = draw.at[0, 2].set(1)  # X
    draw = draw.at[1, 0].set(-1)  # O
    draw = draw.at[1, 1].set(1)  # X
    draw = draw.at[1, 2].set(-1)  # O
    draw = draw.at[2, 0].set(-1)  # O
    draw = draw.at[2, 1].set(1)  # X
    draw = draw.at[2, 2].set(1)  # X

    print("\nDraw position:")
    print_board(draw)
    print(f"Winner: {_check_winner(draw)}")
    print(f"Terminal: {is_terminal_ttt(draw)}")
    print(f"Minimax value (X perspective): {minimax_value(draw, True)}")
    print(f"Minimax value (O perspective): {minimax_value(draw, False)}")


if __name__ == "__main__":
    test_terminal_positions()
