"""Core physics simulation and rational agent model for intuitive physics case study.

Implements a 2D physics environment where an agent must reach a goal, possibly with a wall obstacle.
Models agent behavior as rational action selection under utility maximization.
Enables inference about hidden environmental constraints from observed trajectories.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import seed
from genjax import categorical, flip, gen, Const, const
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import timing


# Physics simulation parameters
GRAVITY = -0.5
FRICTION = 0.95
TIME_STEPS = 100
DT = 0.02

# Environment parameters
AGENT_START = jnp.array([0.0, 0.1])
AGENT_RADIUS = 0.05
GOAL_CENTER = jnp.array([1.5, 0.25])
GOAL_SIZE = jnp.array([1.0, 0.5])
WALL_CENTER = jnp.array([0.7, 0.3])
WALL_SIZE = jnp.array([0.05, 0.6])

# Action space discretization
ANGLE_GRID = jnp.linspace(0, jnp.pi / 3, 25)  # 0 to 60 degrees
IMPULSE_GRID = jnp.linspace(0.1, 3.0, 25)  # 0.1 to 3.0 impulse magnitude


def physics_step(state, wall_present):
    """Single physics step with gravity, friction, and collision detection.

    Args:
        state: [x, y, vx, vy] agent state
        wall_present: boolean indicating if wall exists

    Returns:
        Updated state after physics step
    """
    x, y, vx, vy = state

    # Apply gravity
    vy = vy + GRAVITY * DT

    # Update position first
    new_x = x + vx * DT
    new_y = y + vy * DT

    # Collision detection with wall (if present)
    def check_wall_collision():
        # Check if agent would intersect wall
        wall_left = WALL_CENTER[0] - WALL_SIZE[0] / 2
        wall_right = WALL_CENTER[0] + WALL_SIZE[0] / 2
        wall_bottom = WALL_CENTER[1] - WALL_SIZE[1] / 2
        wall_top = WALL_CENTER[1] + WALL_SIZE[1] / 2

        agent_left = new_x - AGENT_RADIUS
        agent_right = new_x + AGENT_RADIUS
        agent_bottom = new_y - AGENT_RADIUS
        agent_top = new_y + AGENT_RADIUS

        # Check overlap
        x_overlap = (agent_right >= wall_left) & (agent_left <= wall_right)
        y_overlap = (agent_top >= wall_bottom) & (agent_bottom <= wall_top)
        collision = x_overlap & y_overlap

        # If collision, reflect velocity and keep previous position
        reflected_vx = jax.lax.cond(collision, lambda: -vx * 0.3, lambda: vx)
        reflected_vy = jax.lax.cond(collision, lambda: -vy * 0.3, lambda: vy)
        adjusted_x = jax.lax.cond(
            collision, lambda: x, lambda: new_x
        )  # Stay at old position if collision
        adjusted_y = jax.lax.cond(collision, lambda: y, lambda: new_y)

        return adjusted_x, adjusted_y, reflected_vx, reflected_vy

    def no_wall_collision():
        return new_x, new_y, vx, vy

    # Apply wall collision only if wall exists
    final_x, final_y, final_vx, final_vy = jax.lax.cond(
        wall_present, check_wall_collision, no_wall_collision
    )

    # Apply friction after collision
    final_vx = final_vx * FRICTION
    final_vy = final_vy * FRICTION

    # Ground collision (y = 0)
    final_y = jnp.maximum(final_y, AGENT_RADIUS)
    final_vy = jax.lax.cond(
        final_y <= AGENT_RADIUS, lambda: jnp.maximum(final_vy, 0.0), lambda: final_vy
    )

    return jnp.array([final_x, final_y, final_vx, final_vy])


def simulate_trajectory(theta, impulse, wall_present):
    """Simulate complete agent trajectory given action and environment.

    Args:
        theta: Launch angle (radians)
        impulse: Launch impulse magnitude
        wall_present: Boolean indicating if wall exists

    Returns:
        Final x position after simulation
    """
    # Initial velocity from action
    initial_vx = impulse * jnp.cos(theta)
    initial_vy = impulse * jnp.sin(theta)
    initial_state = jnp.array([AGENT_START[0], AGENT_START[1], initial_vx, initial_vy])

    # Simulate physics for TIME_STEPS
    def step_fn(state, _):
        new_state = physics_step(state, wall_present)
        return new_state, new_state

    final_state, _ = jax.lax.scan(step_fn, initial_state, None, length=TIME_STEPS)
    return final_state[0]  # Return final x position


def compute_utility(theta, impulse, wall_present, goal_weight):
    """Compute utility for action given environment and agent preferences.

    Args:
        theta: Launch angle
        impulse: Launch impulse magnitude
        wall_present: Boolean indicating if wall exists
        goal_weight: Agent's utility weight for reaching goal vs. minimizing effort

    Returns:
        Utility score (higher is better)
    """
    final_x = simulate_trajectory(theta, impulse, wall_present)

    # Goal achievement reward (reached goal region)
    goal_left = GOAL_CENTER[0] - GOAL_SIZE[0] / 2
    goal_right = GOAL_CENTER[0] + GOAL_SIZE[0] / 2
    goal_achieved = (final_x >= goal_left) & (final_x <= goal_right)
    goal_reward = jax.lax.cond(goal_achieved, lambda: 10.0, lambda: 0.0)

    # Distance penalty (closer to goal is better)
    distance_to_goal = jnp.abs(final_x - GOAL_CENTER[0])
    distance_penalty = -distance_to_goal

    # Effort cost (higher impulse costs more)
    effort_cost = -impulse

    # Weighted utility
    utility = (
        goal_weight * (goal_reward + distance_penalty) + (1 - goal_weight) * effort_cost
    )
    return utility


@gen
def rational_agent(wall_present: Const[bool], goal_weight: Const[float]):
    """Rational agent model that chooses actions to maximize expected utility.

    Args:
        wall_present: Whether wall is present in environment
        goal_weight: Agent's preference weight for goal achievement vs. effort minimization

    Returns:
        Tuple of (theta_index, impulse_index) action indices
    """
    # Use a smaller action space for efficiency in demonstrations
    # Subsample the action grid for faster computation
    n_actions = 10  # Reduced from 25x25=625 to 10x10=100
    theta_indices = jnp.linspace(0, len(ANGLE_GRID) - 1, n_actions, dtype=int)
    impulse_indices = jnp.linspace(0, len(IMPULSE_GRID) - 1, n_actions, dtype=int)

    # Create meshgrids for vectorized computation
    theta_mesh, impulse_mesh = jnp.meshgrid(
        theta_indices, impulse_indices, indexing="ij"
    )
    theta_values = ANGLE_GRID[theta_mesh]
    impulse_values = IMPULSE_GRID[impulse_mesh]

    # Vectorized utility computation using jax.vmap
    # Flatten the meshgrids and vectorize over the flattened arrays
    theta_flat = theta_values.flatten()
    impulse_flat = impulse_values.flatten()

    def compute_single_utility(theta, impulse):
        return compute_utility(theta, impulse, wall_present.value, goal_weight.value)

    # Vectorize over flattened arrays
    vectorized_utility = jax.vmap(compute_single_utility, in_axes=(0, 0))
    utilities_flat = vectorized_utility(theta_flat, impulse_flat)
    utilities = utilities_flat.reshape((n_actions, n_actions))

    # Softmax action selection with temperature=3.0 (matching original)
    temperature = 3.0
    action_probs = jax.nn.softmax(utilities.flatten() / temperature)

    # Sample action index
    action_idx = categorical(action_probs) @ "action"

    # Convert flat index to subsampled indices, then to full grid indices
    sub_theta_idx = action_idx // n_actions
    sub_impulse_idx = action_idx % n_actions

    theta_idx = theta_indices[sub_theta_idx]
    impulse_idx = impulse_indices[sub_impulse_idx]

    return theta_idx, impulse_idx


@gen
def intuitive_physics_model():
    """Full generative model for intuitive physics inference.

    Models the generative process:
    1. Wall may or may not be present (50% prior)
    2. Agent has utility preferences (uniform over goal weights)
    3. Agent rationally chooses actions given beliefs and preferences
    4. Observer sees action and infers wall presence
    """
    # Prior over wall presence (50/50)
    wall_present = flip(0.5) @ "wall_present"

    # Prior over agent's goal weight (uniform)
    goal_weight = flip(0.8) @ "goal_weight_high"  # 80% chance of high goal weight
    goal_weight_value = jax.lax.cond(goal_weight, lambda: 0.8, lambda: 0.2)

    # Agent chooses action rationally
    theta_idx, impulse_idx = (
        rational_agent(const(wall_present), const(goal_weight_value)) @ "agent_action"
    )

    return {
        "wall_present": wall_present,
        "goal_weight": goal_weight_value,
        "theta_idx": theta_idx,
        "impulse_idx": impulse_idx,
        "theta": ANGLE_GRID[theta_idx],
        "impulse": IMPULSE_GRID[impulse_idx],
    }


# Inference utilities


def wall_inference_from_action(theta_idx, impulse_idx, num_samples=1000):
    """Infer wall presence probability given observed action.

    Args:
        theta_idx: Observed theta action index
        impulse_idx: Observed impulse action index
        num_samples: Number of importance samples

    Returns:
        Probability that wall was present
    """
    # Constrain observed action - need to trace through the model structure
    # The rational_agent returns (theta_idx, impulse_idx), so we constrain the categorical choice
    action_flat_idx = theta_idx * len(IMPULSE_GRID) + impulse_idx
    constraints = {"agent_action": {"action": action_flat_idx}}

    # Generate importance samples
    key = jrand.key(42)

    def importance_sample(key, constraints):
        trace, weight = intuitive_physics_model.generate(constraints)
        wall_present = trace.get_choices()["wall_present"]
        return wall_present, weight

    # Vectorized sampling
    keys = jrand.split(key, num_samples)
    wall_samples, log_weights = jax.vmap(importance_sample, in_axes=(0, None))(
        keys, constraints
    )

    # Normalize weights
    max_weight = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_weight)
    weights = weights / jnp.sum(weights)

    # Compute weighted probability of wall presence
    wall_prob = jnp.sum(weights * wall_samples)
    return wall_prob


def generate_action_wall_grid(num_samples=1000):
    """Generate probability grid of wall presence for all actions.

    Args:
        num_samples: Number of samples per action

    Returns:
        Grid of shape (len(ANGLE_GRID), len(IMPULSE_GRID)) with wall probabilities
    """
    wall_probs = jnp.zeros((len(ANGLE_GRID), len(IMPULSE_GRID)))

    for i in range(len(ANGLE_GRID)):
        for j in range(len(IMPULSE_GRID)):
            wall_prob = wall_inference_from_action(i, j, num_samples)
            wall_probs = wall_probs.at[i, j].set(wall_prob)

    return wall_probs


# Timing utilities


def genjax_timing(num_samples=1000, repeats=50):
    """Time GenJAX importance sampling for wall inference."""

    # Apply seed transformation to the model before timing
    seeded_model = seed(intuitive_physics_model.simulate)

    def get_wall_presence(key):
        trace = seeded_model(key)
        return trace.get_retval()["wall_present"]

    jitted_fn = jax.jit(get_wall_presence)

    # Warm-up
    key = jrand.key(42)
    _ = jitted_fn(key)
    _ = jitted_fn(key)

    times, (time_mu, time_std) = timing(
        lambda: jitted_fn(key).block_until_ready(), repeats=repeats, auto_sync=False
    )
    return times, (time_mu, time_std)


def physics_timing(num_trajectories=1000, repeats=50):
    """Time physics simulation component."""
    # Random actions
    key = jrand.key(123)
    thetas = jrand.choice(key, ANGLE_GRID, (num_trajectories,))
    key = jrand.split(key)[1]
    impulses = jrand.choice(key, IMPULSE_GRID, (num_trajectories,))
    wall_presents = jrand.bernoulli(key, 0.5, (num_trajectories,))

    def simulate_batch():
        return jax.vmap(simulate_trajectory)(thetas, impulses, wall_presents)

    # JIT compile
    jitted_fn = jax.jit(simulate_batch)
    _ = jitted_fn()  # Warm-up
    _ = jitted_fn()  # Warm-up

    times, (time_mu, time_std) = timing(
        lambda: jitted_fn().block_until_ready(), repeats=repeats, auto_sync=False
    )
    return times, (time_mu, time_std)
