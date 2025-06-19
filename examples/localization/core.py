"""
Localization case study using GenJAX.

Simplified probabilistic robot localization with particle filtering.
Focuses on the probabilistic aspects without complex physics.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import gen, normal, Pytree, modular_vmap as vmap, seed, Vmap, Const


# Data structures
@Pytree.dataclass
class Pose(Pytree):
    """Robot pose: position (x, y) and heading (theta)."""

    x: float
    y: float
    theta: float


@Pytree.dataclass
class Control(Pytree):
    """Robot control command: velocity and angular velocity."""

    velocity: float
    angular_velocity: float


@Pytree.dataclass
class Wall(Pytree):
    """A wall segment defined by two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float


@Pytree.dataclass
class World(Pytree):
    """Multi-room world with internal walls and geometry.

    Invariant: Must have at least 1 internal wall (num_walls >= 1).
    """

    width: float
    height: float
    # Internal walls as JAX arrays for vectorization
    # walls_x1, walls_y1, walls_x2, walls_y2 are arrays of wall endpoints
    walls_x1: jnp.ndarray = None
    walls_y1: jnp.ndarray = None
    walls_x2: jnp.ndarray = None
    walls_y2: jnp.ndarray = None
    num_walls: int = 0

    def __post_init__(self):
        """Validate that world has at least 1 internal wall."""
        if self.num_walls < 1:
            raise ValueError(
                f"World must have at least 1 internal wall, got {self.num_walls}"
            )

        # Validate array shapes match num_walls
        expected_shape = (self.num_walls,)
        for wall_array, name in [
            (self.walls_x1, "walls_x1"),
            (self.walls_y1, "walls_y1"),
            (self.walls_x2, "walls_x2"),
            (self.walls_y2, "walls_y2"),
        ]:
            if wall_array.shape != expected_shape:
                raise ValueError(
                    f"Array {name} has shape {wall_array.shape}, expected {expected_shape}"
                )


# Geometry helper functions
def compute_ray_wall_intersection(
    px: float,
    py: float,
    dx: float,
    dy: float,
    walls_x1: jnp.ndarray,
    walls_y1: jnp.ndarray,
    walls_x2: jnp.ndarray,
    walls_y2: jnp.ndarray,
) -> float:
    """Compute intersection of ray with walls using vectorized operations.

    Args:
        px, py: Ray origin
        dx, dy: Ray direction (unit vector)
        walls_*: Wall endpoints as arrays

    Returns:
        Distance to nearest wall intersection, or inf if no intersection
    """
    # Wall vectors
    wall_dx = walls_x2 - walls_x1
    wall_dy = walls_y2 - walls_y1

    # Vector from ray origin to wall start
    to_wall_x = walls_x1 - px
    to_wall_y = walls_y1 - py

    # Solve for intersection using parametric line equations
    # Ray: (px, py) + t * (dx, dy)
    # Wall: (x1, y1) + s * (wall_dx, wall_dy)

    # Cross product for denominator
    denom = dx * wall_dy - dy * wall_dx

    # Avoid division by zero (parallel lines)
    denom = jnp.where(jnp.abs(denom) < 1e-10, 1e-10, denom)

    # Solve for parameters
    t = (to_wall_x * wall_dy - to_wall_y * wall_dx) / denom
    s = (to_wall_x * dy - to_wall_y * dx) / denom

    # Check if intersection is valid (ray forward, within wall segment)
    valid = (t >= 0) & (s >= 0) & (s <= 1)

    # Return minimum valid distance
    distances = jnp.where(valid, t, jnp.inf)
    return jnp.min(distances)


def point_to_walls_distance_vectorized(
    px: float,
    py: float,
    walls_x1: jnp.ndarray,
    walls_y1: jnp.ndarray,
    walls_x2: jnp.ndarray,
    walls_y2: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized computation of distance from point to multiple line segments."""
    # Vector from wall start to wall end (vectorized)
    wall_dx = walls_x2 - walls_x1
    wall_dy = walls_y2 - walls_y1

    # Wall length squared (vectorized)
    wall_length_sq = wall_dx * wall_dx + wall_dy * wall_dy

    # Handle zero-length wall case
    wall_length_sq = jnp.maximum(wall_length_sq, 1e-10)

    # Vector from wall start to point (vectorized)
    point_dx = px - walls_x1
    point_dy = py - walls_y1

    # Project point onto line (parametric t) - vectorized
    t = (point_dx * wall_dx + point_dy * wall_dy) / wall_length_sq

    # Clamp t to [0, 1] to stay on line segment
    t = jnp.clip(t, 0.0, 1.0)

    # Find closest point on line segment (vectorized)
    closest_x = walls_x1 + t * wall_dx
    closest_y = walls_y1 + t * wall_dy

    # Return distance from point to closest point on each segment
    dist_x = px - closest_x
    dist_y = py - closest_y
    return jnp.sqrt(dist_x * dist_x + dist_y * dist_y)


def point_to_line_distance(px: float, py: float, wall: Wall) -> float:
    """Compute distance from point to line segment - kept for compatibility."""
    # This function is kept for any remaining non-vectorized uses
    wall_dx = wall.x2 - wall.x1
    wall_dy = wall.y2 - wall.y1
    wall_length_sq = jnp.maximum(wall_dx * wall_dx + wall_dy * wall_dy, 1e-10)
    point_dx = px - wall.x1
    point_dy = py - wall.y1
    t = jnp.clip((point_dx * wall_dx + point_dy * wall_dy) / wall_length_sq, 0.0, 1.0)
    closest_x = wall.x1 + t * wall_dx
    closest_y = wall.y1 + t * wall_dy
    dist_x = px - closest_x
    dist_y = py - closest_y
    return jnp.sqrt(dist_x * dist_x + dist_y * dist_y)


def check_wall_collision_vectorized(pose: Pose, new_pose: Pose, world: World) -> tuple:
    """Check if movement from pose to new_pose collides with any wall using vectorization.

    Note: World always has at least 1 wall due to validation.
    """
    # Vectorized collision check
    wall_distances = point_to_walls_distance_vectorized(
        new_pose.x,
        new_pose.y,
        world.walls_x1,
        world.walls_y1,
        world.walls_x2,
        world.walls_y2,
    )

    min_dist = jnp.min(wall_distances)
    collision = min_dist < 0.15  # Collision threshold

    # If collision, keep old position; otherwise use new position
    final_x = jnp.where(collision, pose.x, new_pose.x)
    final_y = jnp.where(collision, pose.y, new_pose.y)
    final_theta = jnp.where(collision, pose.theta, new_pose.theta)

    return collision, Pose(final_x, final_y, final_theta)


# Physics functions (simplified)
def apply_control(pose: Pose, control: Control, world: World, dt: float = 0.1) -> Pose:
    """Apply control command with wall collision detection for multi-room world."""
    # Update heading
    new_theta = pose.theta + control.angular_velocity * dt

    # Calculate intended new position
    intended_x = pose.x + control.velocity * jnp.cos(new_theta) * dt
    intended_y = pose.y + control.velocity * jnp.sin(new_theta) * dt

    # Check collision with world boundaries first
    bounce_margin = 0.2

    # Handle boundary bouncing
    hit_left = intended_x < bounce_margin
    hit_right = intended_x > world.width - bounce_margin
    hit_bottom = intended_y < bounce_margin
    hit_top = intended_y > world.height - bounce_margin

    # Reflect positions for boundary bouncing
    reflected_x = jnp.where(hit_left, bounce_margin, intended_x)
    reflected_x = jnp.where(hit_right, world.width - bounce_margin, reflected_x)
    reflected_y = jnp.where(hit_bottom, bounce_margin, intended_y)
    reflected_y = jnp.where(hit_top, world.height - bounce_margin, reflected_y)

    # Update theta for boundary bouncing
    theta_after_x_bounce = jnp.where(
        hit_left | hit_right, jnp.pi - new_theta, new_theta
    )
    final_theta = jnp.where(
        hit_bottom | hit_top, -theta_after_x_bounce, theta_after_x_bounce
    )

    # Check for internal wall collisions using vectorization
    # Since world always has at least 1 wall, we can simplify this
    wall_distances = point_to_walls_distance_vectorized(
        reflected_x,
        reflected_y,
        world.walls_x1,
        world.walls_y1,
        world.walls_x2,
        world.walls_y2,
    )

    # Check if any wall distance is below threshold
    min_wall_dist = jnp.min(wall_distances)
    collision_detected = min_wall_dist < 0.15  # Collision threshold for internal walls

    # If collision with internal wall, just stop (don't move)
    final_x = jnp.where(collision_detected, pose.x, reflected_x)
    final_y = jnp.where(collision_detected, pose.y, reflected_y)
    final_theta_val = jnp.where(collision_detected, pose.theta, final_theta)

    return Pose(final_x, final_y, final_theta_val)


def distance_to_wall_lidar(
    pose: Pose, world: World, n_angles: int = 8, max_range: float = 10.0
) -> jnp.ndarray:
    """Compute LIDAR-style distance measurements at multiple angles around the robot.

    Args:
        pose: Robot pose (x, y, theta)
        world: World object with boundaries and internal walls
        n_angles: Number of angular measurements (default 8)
        max_range: Maximum sensor range

    Returns:
        Array of distances at each angle, shape (n_angles,)
    """
    # Create angular grid around robot (relative to robot's heading)
    angles = jnp.linspace(0, 2 * jnp.pi, n_angles, endpoint=False)

    # Convert to world coordinates (absolute angles)
    world_angles = pose.theta + angles

    # Vectorized ray directions
    dx = jnp.cos(world_angles)
    dy = jnp.sin(world_angles)

    # Vectorized boundary intersection calculations
    t_right = jnp.where(dx > 0, (world.width - pose.x) / dx, jnp.inf)
    t_left = jnp.where(dx < 0, -pose.x / dx, jnp.inf)
    t_top = jnp.where(dy > 0, (world.height - pose.y) / dy, jnp.inf)
    t_bottom = jnp.where(dy < 0, -pose.y / dy, jnp.inf)

    # Time to hit boundary (vectorized minimum)
    t_boundary = jnp.minimum(jnp.minimum(t_right, t_left), jnp.minimum(t_top, t_bottom))

    # Vectorized wall intersection using vmap over angles
    def compute_single_ray_wall_intersection(dx_single, dy_single):
        return compute_ray_wall_intersection(
            pose.x,
            pose.y,
            dx_single,
            dy_single,
            world.walls_x1,
            world.walls_y1,
            world.walls_x2,
            world.walls_y2,
        )

    # Use vmap to vectorize over all angles at once
    vectorized_wall_intersection = jax.vmap(
        compute_single_ray_wall_intersection, in_axes=(0, 0)
    )
    t_wall = vectorized_wall_intersection(dx, dy)

    # Take minimum distance (vectorized)
    t_min = jnp.minimum(t_boundary, t_wall)
    distances = jnp.minimum(t_min, max_range)

    return distances


def distance_to_wall(pose: Pose, world: World) -> float:
    """Compute minimum distance to any wall (backward compatibility).

    This function is kept for compatibility with existing code that expects
    a single distance value.
    """
    lidar_distances = distance_to_wall_lidar(pose, world, n_angles=8)
    return jnp.min(lidar_distances)


# Generative functions for rejuvenation_smc API
@gen
def localization_model(prev_pose, world):
    """Simplified localization model for rejuvenation_smc API.

    This follows the autoregressive pattern where the model takes the previous
    pose and generates the next pose with observations.

    Args:
        prev_pose: Previous robot pose (Pose object)
        world: World geometry for sensor computations

    Returns:
        Current pose (sampled from motion model) with sensor observation
    """
    # Simple motion model: robot moves with some noise
    # Assume random walk with small steps
    velocity = normal(0.5, 0.2) @ "velocity"
    angular_velocity = normal(0.0, 0.1) @ "angular_velocity"

    # Apply simple motion (no world collision for now)
    dt = 0.1
    new_theta = prev_pose.theta + angular_velocity * dt
    new_x = prev_pose.x + velocity * jnp.cos(new_theta) * dt
    new_y = prev_pose.y + velocity * jnp.sin(new_theta) * dt

    # Add motion noise
    x = normal(new_x, 0.3) @ "x"
    y = normal(new_y, 0.3) @ "y"
    theta = normal(new_theta, 0.1) @ "theta"

    # Keep within world boundaries
    x = jnp.clip(x, 0.1, world.width - 0.1)
    y = jnp.clip(y, 0.1, world.height - 0.1)

    current_pose = Pose(x, y, theta)

    # LIDAR sensor observations (8 rays) using Vmap - now fixed!
    # Get true distances for all 8 rays
    true_distances = distance_to_wall_lidar(current_pose, world, n_angles=8)

    # Use GenJAX Vmap to vectorize sensor observations
    ray_indices = jnp.arange(8)
    vectorized_sensor = Vmap(
        sensor_model_single_ray,
        in_axes=Const((0, 0)),  # true_distance=0, ray_idx=0
        axis_size=Const(8),
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    # Apply vectorized sensor model - this now works with rejuvenation_smc!
    observed_distances = vectorized_sensor(true_distances, ray_indices) @ "measurements"

    return current_pose, world


@gen
def sensor_model_single_ray(true_distance: float, ray_idx: int):
    """Generative model for a single LIDAR ray observation."""
    # Each ray has independent Gaussian noise - increased for particle diversity
    obs_dist = (
        normal(true_distance, 0.8) @ "distance"
    )  # Increased from 0.5 to 0.8 for better particle diversity
    # Constrain to non-negative values
    obs_dist = jnp.maximum(0.0, obs_dist)
    return obs_dist


@gen
def localization_proposal(prev_pose, world):
    """Proposal function for MCMC moves in localization.

    This takes the previous pose and world geometry, proposes a new pose
    based on simple motion assumptions.

    Args:
        prev_pose: Previous robot pose (Pose object)
        world: World geometry (for boundary constraints)
    """
    # Simple motion proposal around the previous pose
    x = normal(prev_pose.x, 0.5) @ "x"
    y = normal(prev_pose.y, 0.5) @ "y"
    theta = normal(prev_pose.theta, 0.2) @ "theta"

    # Keep within world boundaries
    x = jnp.clip(x, 0.1, world.width - 0.1)
    y = jnp.clip(y, 0.1, world.height - 0.1)

    return Pose(x, y, theta), world


@gen
def sensor_model(pose: Pose, world: World):
    """Generative model for LIDAR-style sensor observations.

    Returns vector of noisy distance measurements at multiple angles.
    """
    n_angles = 8  # Fixed number of angles
    # True LIDAR distances at multiple angles
    true_distances = distance_to_wall_lidar(pose, world, n_angles)

    # Create vectorized sensor model using Vmap with required parameters
    # Vmap over the true_distances array and ray indices
    ray_indices = jnp.arange(n_angles)
    vectorized_sensor = Vmap(
        sensor_model_single_ray,
        in_axes=Const((0, 0)),  # true_distance=0, ray_idx=0
        axis_size=Const(n_angles),
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    # Apply vectorized sensor model - pass arguments directly, not as nested tuple
    observed_distances = vectorized_sensor(true_distances, ray_indices) @ "measurements"

    return observed_distances


@gen
def sensor_model_single(pose: Pose, world: World):
    """Generative model for single distance sensor (backward compatibility)."""
    # True distance to wall (minimum)
    true_distance = distance_to_wall(pose, world)

    # Sample observed distance as a random variable
    observed_distance = normal(true_distance, 0.2) @ "observed_distance"

    # Constrain to non-negative values
    observed_distance = jnp.maximum(0.0, observed_distance)

    return observed_distance


@gen
def initial_model(world: World):
    """Initial pose model with random variables for x, y, theta.

    Uses broader priors to allow particles to start anywhere in the world,
    which is important for cross-room navigation scenarios.
    """
    # Random initial position within the world (broader for multi-room scenarios)
    x = normal(world.width / 2, 2.0) @ "x"  # Increased variance for multi-room
    y = normal(world.height / 2, 2.0) @ "y"
    theta = normal(0.0, 0.5) @ "theta"

    # Constrain to world boundaries
    x = jnp.clip(x, 0.5, world.width - 0.5)
    y = jnp.clip(y, 0.5, world.height - 0.5)

    return Pose(x=x, y=y, theta=theta)


# Particle filter implementation
def resample_particles(particles, weights, key):
    """Resample particles based on weights using systematic resampling."""
    n_particles = len(particles)

    # Normalize weights
    weights = weights / jnp.sum(weights)

    # Systematic resampling
    indices = jrand.categorical(key, jnp.log(weights), shape=(n_particles,))

    # Return resampled particles (manual indexing for list of Pose objects)
    resampled_particles = [particles[int(idx)] for idx in indices]
    return resampled_particles


def particle_filter_step(particles, weights, observation, world, key):
    """Single step of particle filter using step model with control inference.

    Now the step model infers controls internally, so we don't need to provide them.
    """

    # Predict: apply step model to each particle using vectorization
    seeded_step = seed(step_model.simulate)
    particle_keys = jrand.split(key, len(particles))

    # Vectorized particle prediction using vmap
    def predict_single_particle(particle_key, particle):
        trace = seeded_step(particle_key, particle, world)  # Pass as separate arguments
        return trace.get_retval()

    # Use vmap to vectorize over particles
    vectorized_predict = vmap(predict_single_particle, in_axes=(0, 0))

    # Convert particles list to arrays for vmap
    particle_xs = jnp.array([p.x for p in particles])
    particle_ys = jnp.array([p.y for p in particles])
    particle_thetas = jnp.array([p.theta for p in particles])
    particle_array = Pose(x=particle_xs, y=particle_ys, theta=particle_thetas)

    # Apply vectorized prediction
    predicted_pose_array = vectorized_predict(particle_keys, particle_array)

    # Convert back to list of individual Pose objects
    predicted_poses = [
        Pose(
            predicted_pose_array.x[i],
            predicted_pose_array.y[i],
            predicted_pose_array.theta[i],
        )
        for i in range(len(particles))
    ]

    # Update: weight particles by observation likelihood (LIDAR vector)
    def compute_weight(pose):
        # For vectorized LIDAR observations using Vmap
        # The Vmap combinator expects choices in the structure: {"measurements": {"distance": [array of distances]}}
        choices = {"measurements": {"distance": observation}}
        try:
            log_weight, _ = sensor_model.assess(choices, pose, world)
            # log_weight should now be a scalar (joint log density from Vmap.assess)
            weight = jnp.exp(log_weight)
            # Avoid numerical issues using JAX conditionals
            is_valid = jnp.logical_and(jnp.isfinite(weight), weight > 0)
            return jnp.where(is_valid, jnp.maximum(weight, 1e-10), 1e-10)
        except Exception as e:
            print(f"Error computing weight: {e}")
            return 1e-10

    # Compute weights for all particles
    new_weights = jnp.array([compute_weight(pose) for pose in predicted_poses])
    print(f"    Weight range: [{jnp.min(new_weights):.2e}, {jnp.max(new_weights):.2e}]")
    print(f"    Weight variance: {jnp.var(new_weights):.2e}")
    print(f"    Non-uniform weights: {jnp.sum(new_weights > 1e-9)}/{len(new_weights)}")

    return predicted_poses, new_weights


def initialize_particles(n_particles, world, key, initial_pose=None):
    """Initialize particles using the initial_model with generate or around a known pose."""
    print(f"Initializing {n_particles} particles...")

    if initial_pose is not None:
        # Create choices that constrain particles around the initial pose
        particle_keys = jrand.split(key, n_particles)
        particles = []

        for i in range(n_particles):
            # Generate particles around the initial pose with some noise
            choices = {
                "x": initial_pose.x + jrand.normal(particle_keys[i]) * 0.5,
                "y": initial_pose.y + jrand.normal(particle_keys[i]) * 0.5,
                "theta": initial_pose.theta + jrand.normal(particle_keys[i]) * 0.2,
            }

            # Use generate to create particles constrained around initial pose
            trace, weight = initial_model.generate(choices, world)
            particles.append(trace.get_retval())

        print(f"Initialized {len(particles)} particles around initial pose")
    else:
        # Initialize particles broadly across the entire world for multi-room exploration
        key_x, key_y, key_theta = jrand.split(key, 3)

        # Uniform distribution across the world (with margins)
        x_values = jrand.uniform(
            key_x, shape=(n_particles,), minval=0.5, maxval=world.width - 0.5
        )
        y_values = jrand.uniform(
            key_y, shape=(n_particles,), minval=0.5, maxval=world.height - 0.5
        )
        theta_values = jrand.uniform(
            key_theta, shape=(n_particles,), minval=-jnp.pi, maxval=jnp.pi
        )

        # Convert to list of individual Pose objects
        particles = [
            Pose(x_values[i], y_values[i], theta_values[i]) for i in range(n_particles)
        ]

        print(f"Initialized {len(particles)} particles across entire world")

    return particles


def run_particle_filter(n_particles, observations, world, key, initial_pose=None):
    """Run particle filter using rejuvenation_smc API.

    Args:
        n_particles: Number of particles to use
        observations: List of sensor observations
        world: World object
        key: Random key
        initial_pose: Optional initial pose (not used in new API, handled by model)

    Returns:
        particle_history: List of particle states over time
        weight_history: List of particle weights over time
    """
    # Import rejuvenation_smc
    from genjax.inference import rejuvenation_smc, mh
    from genjax.core import sel, const

    # Prepare observations for rejuvenation_smc API
    # observations is a list of 8-element arrays (LIDAR measurements)
    # With Vmap, we need to structure this as {"measurements": {"distance": [T, 8]}}
    obs_array = jnp.array(observations)  # Shape: (T, 8)

    # rejuvenation_smc expects nested dict format for Vmap
    obs_sequence = {"measurements": {"distance": obs_array}}  # Shape: (T, 8)

    # Initial arguments for localization_model: (prev_pose, world)
    initial_pose_arg = Pose(x=world.width / 2, y=world.height / 2, theta=0.0)
    initial_args = (initial_pose_arg, world)

    # Define MCMC kernel for rejuvenation moves
    def mcmc_kernel(trace):
        return mh(trace, sel("x") | sel("y") | sel("theta"))

    print(f"Running rejuvenation_smc with {n_particles} particles...")

    # Run rejuvenation_smc
    result = seed(rejuvenation_smc)(
        key,
        localization_model,
        localization_proposal,
        const(mcmc_kernel),
        obs_sequence,
        initial_args,
        const(n_particles),
    )

    print("Particle filter completed successfully!")

    # Extract particle history from result
    # For now, return final particles in expected format
    final_traces = result.traces
    final_weights = result.log_weights  # Use log_weights instead of weights

    # Convert traces back to Pose objects
    choices = final_traces.get_choices()
    final_particles = []

    if len(choices["x"].shape) == 1:
        # Multiple particles
        for i in range(len(choices["x"])):
            pose = Pose(x=choices["x"][i], y=choices["y"][i], theta=choices["theta"][i])
            final_particles.append(pose)
    else:
        # Single particle
        pose = Pose(x=choices["x"], y=choices["y"], theta=choices["theta"])
        final_particles = [pose]

    # For compatibility, create mock history (just initial and final)
    initial_particles = [
        Pose(x=world.width / 2, y=world.height / 2, theta=0.0)
        for _ in range(n_particles)
    ]
    particle_history = [initial_particles, final_particles]
    weight_history = [jnp.ones(n_particles) / n_particles, final_weights]

    return particle_history, weight_history


# Utility functions
def create_multi_room_world():
    """Create a multi-room world with internal walls and doorways using JAX arrays."""
    width, height = 12.0, 10.0

    # Define internal walls to create a 3-room layout using JAX arrays
    # Each wall is defined by (x1, y1) -> (x2, y2)
    wall_coords = [
        # Vertical wall between Room 1 and Room 2 (with doorway)
        [4.0, 0.0, 4.0, 3.0],  # Bottom part of wall
        [4.0, 5.0, 4.0, 10.0],  # Top part of wall (doorway from y=3 to y=5)
        # Vertical wall between Room 2 and Room 3 (with doorway)
        [8.0, 0.0, 8.0, 4.0],  # Bottom part of wall
        [8.0, 6.0, 8.0, 10.0],  # Top part of wall (doorway from y=4 to y=6)
        # Horizontal internal wall in Room 2 (creates a small alcove)
        [5.0, 7.0, 7.0, 7.0],  # Horizontal wall
        # Small obstacle in Room 3
        [9.0, 2.0, 10.0, 2.0],  # Bottom of obstacle
        [10.0, 2.0, 10.0, 3.0],  # Right side of obstacle
        [10.0, 3.0, 9.0, 3.0],  # Top of obstacle
        [9.0, 3.0, 9.0, 2.0],  # Left side of obstacle (completes rectangle)
    ]

    # Convert to JAX arrays
    wall_array = jnp.array(wall_coords)
    walls_x1 = wall_array[:, 0]
    walls_y1 = wall_array[:, 1]
    walls_x2 = wall_array[:, 2]
    walls_y2 = wall_array[:, 3]

    return World(
        width=width,
        height=height,
        walls_x1=walls_x1,
        walls_y1=walls_y1,
        walls_x2=walls_x2,
        walls_y2=walls_y2,
        num_walls=len(wall_coords),
    )


def create_simple_world():
    """Create a simple rectangular world with a single internal wall."""
    # Single internal wall in the middle of the world
    wall_coords = [
        [5.0, 2.0, 5.0, 8.0]  # Vertical wall in center with gaps at top/bottom
    ]

    # Convert to JAX arrays
    wall_array = jnp.array(wall_coords)
    walls_x1 = wall_array[:, 0]
    walls_y1 = wall_array[:, 1]
    walls_x2 = wall_array[:, 2]
    walls_y2 = wall_array[:, 3]

    return World(
        width=10.0,
        height=10.0,
        walls_x1=walls_x1,
        walls_y1=walls_y1,
        walls_x2=walls_x2,
        walls_y2=walls_y2,
        num_walls=1,
    )


# Data generation functions moved to data.py
