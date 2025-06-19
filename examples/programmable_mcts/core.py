"""
Programmable Monte Carlo Tree Search implementation.

This module provides MCTS that is parametric over probabilistic programs.
The key insight: a "game model" is a GenJAX generative function that takes
(action, state) and returns (reward, observation).

This allows agents to learn and improve their models through experience!
"""

import jax.numpy as jnp
import jax.random as jrand
from typing import Any, List, Optional, Dict
import math
from genjax.distributions import normal

from genjax.core import gen


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================


class GameModelProgram:
    """Wrapper for a probabilistic game model (GenJAX generative function).

    The core idea: a game model is a generative function with signature:
        @gen
        def game_model(action, state) -> (reward, observation)

    This allows the agent to:
    1. Use the model for MCTS planning (simulate outcomes)
    2. Update the model using inference when new data arrives
    3. Express uncertainty about game dynamics
    """

    def __init__(self, generative_function, get_legal_actions_fn, is_terminal_fn):
        """Initialize with a GenJAX generative function and utility functions.

        Args:
            generative_function: @gen function with signature (action, state) -> (reward, obs)
            get_legal_actions_fn: function to get legal actions from a state
            is_terminal_fn: function to check if a state is terminal
        """
        self.generative_function = generative_function
        self.get_legal_actions = get_legal_actions_fn
        self.is_terminal = is_terminal_fn

    def simulate(self, key: jnp.ndarray, action: Any, state: Any):
        """Simulate taking an action from a state using the probabilistic model.

        Returns:
            trace: GenJAX trace containing (reward, observation) and choices
        """
        return self.generative_function.simulate(action, state)

    def assess(self, choices: Dict, action: Any, state: Any):
        """Assess the probability of given choices under the model."""
        return self.generative_function.assess(choices, action, state)


# =============================================================================
# MCTS NODE REPRESENTATION
# =============================================================================


class MCTSNode:
    """Simple MCTS node - no need for JAX compatibility here."""

    def __init__(
        self, state: Any, parent: Optional["MCTSNode"] = None, action_taken: Any = None
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0

        # Tree structure
        self.children = {}  # action -> MCTSNode
        self.untried_actions = []

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all legal actions have been tried."""
        return len(self.untried_actions) == 0

    @property
    def average_reward(self) -> float:
        """Get average reward for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """Compute UCB1 value for action selection."""
        if self.visits == 0:
            return float("inf")  # Unvisited nodes have infinite value

        if self.parent is None or self.parent.visits == 0:
            return self.average_reward

        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration


# =============================================================================
# MCTS ALGORITHM
# =============================================================================


class MCTS:
    """Monte Carlo Tree Search parametric over probabilistic game models."""

    def __init__(
        self,
        model_program: GameModelProgram,
        exploration_constant: float = 1.414,
        simulation_depth: int = 50,
    ):
        """Initialize MCTS with a probabilistic game model.

        Args:
            model_program: GameModelProgram wrapping a GenJAX generative function
            exploration_constant: UCB1 exploration parameter
            simulation_depth: Maximum depth for rollout simulations
        """
        self.model = model_program
        self.exploration_constant = exploration_constant
        self.simulation_depth = simulation_depth

    def search(self, key: jnp.ndarray, root_state: Any, num_simulations: int = 1000):
        """Run MCTS to find the best action from root_state.

        Args:
            key: Random key for stochastic simulations
            root_state: Starting game state
            num_simulations: Number of MCTS simulations to run

        Returns:
            (best_action, root_node) tuple
        """
        # Initialize root node
        root = MCTSNode(state=root_state)
        root.untried_actions = self.model.get_legal_actions(root_state)

        # Run simulations
        keys = jrand.split(key, num_simulations)
        for i in range(num_simulations):
            self._simulate_once(keys[i], root)

        # Select best action (most visited child)
        if not root.children:
            # No expansions happened - return random legal action
            legal_actions = self.model.get_legal_actions(root_state)
            if legal_actions:
                return legal_actions[0], root
            else:
                return None, root

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)

        return best_action, root

    def _simulate_once(self, key: jnp.ndarray, root: MCTSNode) -> None:
        """Run one MCTS simulation: selection, expansion, rollout, backprop."""

        # 1. Selection: traverse tree using UCB1
        node = self._select(root)

        # 2. Expansion: add new child if possible
        key1, key2 = jrand.split(key)
        expanded_node = self._expand(key1, node)

        # 3. Simulation: rollout from expanded node using probabilistic model
        reward = self._rollout(key2, expanded_node)

        # 4. Backpropagation: update statistics
        self._backpropagate(expanded_node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node for expansion using UCB1."""
        current = node

        while (
            current.is_fully_expanded
            and current.children
            and not self.model.is_terminal(current.state)
        ):
            # Select child with highest UCB1 value
            best_action = max(
                current.children.keys(),
                key=lambda a: current.children[a].ucb1_value(self.exploration_constant),
            )
            current = current.children[best_action]

        return current

    def _expand(self, key: jnp.ndarray, node: MCTSNode) -> MCTSNode:
        """Expand the tree by adding a new child node."""

        # If terminal or no untried actions, return the node itself
        if self.model.is_terminal(node.state) or not node.untried_actions:
            return node

        # Pick an untried action
        action = node.untried_actions.pop()

        # Use probabilistic model to simulate the action
        trace = self.model.simulate(key, action, node.state)
        reward, observation = trace.get_retval()

        # Create new state from observation (this is game-specific)
        # For now, assume observation IS the new state
        new_state = observation

        child = MCTSNode(state=new_state, parent=node, action_taken=action)
        child.untried_actions = self.model.get_legal_actions(new_state)

        node.children[action] = child
        return child

    def _rollout(self, key: jnp.ndarray, node: MCTSNode) -> float:
        """Simulate random play using the probabilistic model."""

        current_state = node.state
        current_key = key
        total_reward = 0.0

        for _ in range(self.simulation_depth):
            if self.model.is_terminal(current_state):
                break

            # Choose random legal action
            legal_actions = self.model.get_legal_actions(current_state)
            if not legal_actions:
                break

            current_key, action_key, sim_key = jrand.split(current_key, 3)
            action_idx = jrand.randint(action_key, (), 0, len(legal_actions))
            action = legal_actions[action_idx]

            # Use probabilistic model to simulate action
            trace = self.model.simulate(sim_key, action, current_state)
            reward, observation = trace.get_retval()

            total_reward += reward
            current_state = observation  # Next state is the observation

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        current = node

        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent


# =============================================================================
# EXAMPLE: SIMPLE PROBABILISTIC TIC-TAC-TOE MODEL
# =============================================================================


@gen
def tic_tac_toe_model(action, state):
    """Probabilistic Tic-Tac-Toe model.

    Args:
        action: (row, col) tuple for where to place the mark
        state: 3x3 board state

    Returns:
        (reward, new_state) tuple
    """
    row, col = action

    # Determine current player
    num_moves = jnp.sum(jnp.abs(state))
    current_player = int(num_moves % 2)
    player_mark = 1 if current_player == 0 else -1

    # Place the mark (deterministic for now)
    new_state = state.at[row, col].set(player_mark)

    # Check if game is terminal and compute reward
    winner = _check_winner(new_state)
    is_terminal = winner != 0 or jnp.sum(jnp.abs(new_state)) == 9

    if is_terminal:
        if winner == 0:  # Draw
            reward = 0.0
        elif (winner == 1 and current_player == 0) or (
            winner == -1 and current_player == 1
        ):
            reward = 1.0  # Current player wins
        else:
            reward = -1.0  # Current player loses
    else:
        reward = 0.0  # Game continues

    # Add some noise to the observation (model imperfection)
    noise_scale = 0.01
    noisy_state = new_state + normal(0.0, noise_scale) @ "noise"

    return reward, jnp.round(noisy_state).astype(jnp.int32)


def _check_winner(state: jnp.ndarray) -> int:
    """Check for winner: 1 for X, -1 for O, 0 for no winner."""
    # Check rows
    for i in range(3):
        if abs(jnp.sum(state[i, :])) == 3:
            return int(jnp.sign(jnp.sum(state[i, :])))

    # Check columns
    for j in range(3):
        if abs(jnp.sum(state[:, j])) == 3:
            return int(jnp.sign(jnp.sum(state[:, j])))

    # Check diagonals
    if abs(jnp.sum(jnp.diag(state))) == 3:
        return int(jnp.sign(jnp.sum(jnp.diag(state))))

    if abs(jnp.sum(jnp.diag(jnp.fliplr(state)))) == 3:
        return int(jnp.sign(jnp.sum(jnp.diag(jnp.fliplr(state)))))

    return 0


def get_legal_actions_ttt(state: jnp.ndarray) -> List:
    """Get legal actions (empty positions) for Tic-Tac-Toe."""
    actions = []
    for i in range(3):
        for j in range(3):
            if state[i, j] == 0:  # Empty cell
                actions.append((i, j))
    return actions


def is_terminal_ttt(state: jnp.ndarray) -> bool:
    """Check if Tic-Tac-Toe game is terminal."""
    return _check_winner(state) != 0 or jnp.sum(jnp.abs(state)) == 9


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_empty_board() -> jnp.ndarray:
    """Create empty 3x3 Tic-Tac-Toe board."""
    return jnp.zeros((3, 3), dtype=jnp.int32)


def print_board(state: jnp.ndarray) -> None:
    """Print a human-readable Tic-Tac-Toe board."""
    symbols = {0: ".", 1: "X", -1: "O"}

    print("  0 1 2")
    for i in range(3):
        row = f"{i} "
        for j in range(3):
            row += symbols[int(state[i, j])] + " "
        print(row)
    print()


def create_tic_tac_toe_model() -> GameModelProgram:
    """Create a probabilistic Tic-Tac-Toe model."""
    return GameModelProgram(
        generative_function=tic_tac_toe_model,
        get_legal_actions_fn=get_legal_actions_ttt,
        is_terminal_fn=is_terminal_ttt,
    )


def demonstrate_probabilistic_mcts(key: jnp.ndarray, num_simulations: int = 100):
    """Demonstrate MCTS with a probabilistic model."""
    print("=== Demonstrating Probabilistic MCTS ===")

    # Create probabilistic model
    model_program = create_tic_tac_toe_model()
    mcts = MCTS(model_program, exploration_constant=1.414)

    # Start with empty board
    state = create_empty_board()
    print("Initial board:")
    print_board(state)

    # Run MCTS to select action
    best_action, root = mcts.search(key, state, num_simulations)

    print(f"After {num_simulations} simulations, MCTS recommends: {best_action}")
    print(f"Root node visits: {root.visits}")

    # Show action statistics
    if root.children:
        print("\nAction evaluation:")
        for action, child in root.children.items():
            print(
                f"  {action}: {child.visits} visits, {child.average_reward:.3f} avg reward"
            )

    return best_action, root
