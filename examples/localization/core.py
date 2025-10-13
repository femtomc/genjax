"""
Localization case study using GenJAX.

Simplified probabilistic robot localization with particle filtering.
Focuses on the probabilistic aspects without complex physics.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import gen, normal, Pytree, seed, Vmap, Const, const


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
    pose: Pose, world: World, n_angles: int = 128, max_range: float = 10.0
) -> jnp.ndarray:
    """Compute LIDAR-style distance measurements at multiple angles around the robot.

    Args:
        pose: Robot pose (x, y, theta)
        world: World object with boundaries and internal walls
        n_angles: Number of angular measurements (default 32)
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


# Generative functions for rejuvenation_smc API
@gen
def localization_model(prev_pose, time_index, world, n_rays=Const(8)):
    """Localization model using drift-only dynamics for improved convergence.

    This model uses simple positional drift without velocity variables,
    which has been shown to provide better SMC convergence properties.

    Args:
        prev_pose: Previous robot pose (Pose object, dummy for t=0)
        time_index: Current timestep (0 for initialization, >0 for transitions)
        world: World geometry for sensor computations
        n_rays: Number of LIDAR rays for distance measurements (Const)

    Returns:
        Current pose (sampled from appropriate distribution) with sensor observation
    """
    is_initial = time_index == 0

    # Initial distribution parameters (near actual start position)
    initial_x = 1.5
    initial_y = 1.5
    initial_theta = 0.0

    # Drift parameters (no velocity, just positional drift)
    drift_noise_x = 0.25  # Increased from 0.15
    drift_noise_y = 0.25  # Increased from 0.15
    drift_noise_theta = 0.1  # Increased from 0.05

    # Initial uncertainty
    initial_noise_x = 0.5
    initial_noise_y = 0.5
    initial_noise_theta = 0.3

    # Use JAX conditionals to select mean and noise based on timestep
    mean_x = jax.lax.select(is_initial, initial_x, prev_pose.x)
    mean_y = jax.lax.select(is_initial, initial_y, prev_pose.y)
    mean_theta = jax.lax.select(is_initial, initial_theta, prev_pose.theta)

    # Noise parameters - higher for initial, lower for transitions
    noise_x = jax.lax.select(is_initial, initial_noise_x, drift_noise_x)
    noise_y = jax.lax.select(is_initial, initial_noise_y, drift_noise_y)
    noise_theta = jax.lax.select(is_initial, initial_noise_theta, drift_noise_theta)

    # Sample current pose with drift
    x = normal(mean_x, noise_x) @ "x"
    y = normal(mean_y, noise_y) @ "y"
    theta = normal(mean_theta, noise_theta) @ "theta"

    # Keep within world boundaries
    x = jnp.clip(x, 0.1, world.width - 0.1)
    y = jnp.clip(y, 0.1, world.height - 0.1)

    current_pose = Pose(x, y, theta)

    # LIDAR sensor observations using Vmap
    # Get true distances for all rays
    true_distances = distance_to_wall_lidar(current_pose, world, n_angles=n_rays.value)

    # Use GenJAX Vmap to vectorize sensor observations
    ray_indices = jnp.arange(n_rays.value)
    vectorized_sensor = Vmap(
        sensor_model_single_ray,
        in_axes=Const((0, 0)),  # true_distance=0, ray_idx=0
        axis_size=n_rays,
        axis_name=Const(None),
        spmd_axis_name=Const(None),
    )

    # Apply vectorized sensor model
    vectorized_sensor(true_distances, ray_indices) @ "measurements"

    return current_pose, time_index + 1, world, n_rays


@gen
def sensor_model_single_ray(true_distance: float, ray_idx: int):
    """Generative model for a single LIDAR ray observation."""
    # Each ray has independent Gaussian noise - reduced for better tracking
    obs_dist = (
        normal(true_distance, 0.3) @ "distance"
    )  # Reduced sensor noise for drift-only model
    # Constrain to non-negative values
    obs_dist = jnp.maximum(0.0, obs_dist)
    return obs_dist


@gen
def sensor_model(pose: Pose, world: World, n_angles: int = 128):
    """Generative model for LIDAR-style sensor observations.

    Returns vector of noisy distance measurements at multiple angles.
    """
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
def initial_model(world: World):
    """Initial pose model with random variables for x, y, theta.

    Uses tighter priors centered near the actual starting position
    for better convergence with the drift-only model.
    """
    # Initial position near actual start (1.2, 1.2)
    x = normal(1.5, 0.5) @ "x"  # Centered near true start
    y = normal(1.5, 0.5) @ "y"  # Centered near true start
    theta = normal(0.0, 0.3) @ "theta"  # Moderate heading uncertainty

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


def run_particle_filter(
    n_particles, observations, world, key, n_rays=8, collect_diagnostics=False
):
    """Run particle filter using rejuvenation_smc API.

    Args:
        n_particles: Number of particles to use
        observations: List of sensor observations
        world: World object
        key: Random key
        n_rays: Number of LIDAR rays
        collect_diagnostics: Whether to collect diagnostic weights from SMC

    Returns:
        particle_history: List of particle states over time
        weight_history: List of particle weights over time
        diagnostic_weights: Log normalized diagnostic weights [T, n_particles] if collect_diagnostics=True, else None
    """
    # Import rejuvenation_smc
    from genjax.inference import rejuvenation_smc
    from genjax.core import const

    # Prepare observations for rejuvenation_smc API
    # The model generates poses and observes from them, so observations[t] should correspond to
    # the observation taken from the pose generated at timestep t
    # observations is a list of n_rays-element arrays (LIDAR measurements)

    # With Vmap, we need to structure this as {"measurements": {"distance": [T, n_rays]}}
    obs_array = jnp.array(observations)  # Shape: (T, n_rays)

    # rejuvenation_smc expects nested dict format for Vmap
    obs_sequence = {"measurements": {"distance": obs_array}}  # Shape: (T, n_rays)

    # Initial arguments for localization_model: (prev_pose, time_index, world, n_rays)
    # For t=0, prev_pose is dummy (will be ignored), time_index starts at 0
    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)  # Dummy - ignored for t=0
    initial_args = (
        dummy_pose,
        jnp.array(0),
        world,
        const(n_rays),
    )  # (prev_pose, time_index, world, n_rays)

    print(f"Running rejuvenation_smc with {n_particles} particles...")

    # Run rejuvenation_smc
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,  # model
        observations=obs_sequence,  # observations
        initial_model_args=initial_args,  # initial_model_args
        n_particles=const(n_particles),  # n_particles
        return_all_particles=const(
            True
        ),  # return_all_particles=True to get all timesteps
    )

    print("Particle filter completed successfully!")

    # Extract particle history from result - now includes all timesteps!
    all_traces = particles_smc.traces  # Shape: [T, n_particles, ...]
    all_weights = particles_smc.log_weights  # Shape: [T, n_particles]

    # Convert traces back to Pose objects for all timesteps
    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]  # Number of timesteps

    print(f"Extracted {n_timesteps} timesteps of particle history")

    # Simple extraction - keep it compatible with existing code
    particle_history = []
    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=float(choices["x"][t, i]),
                y=float(choices["y"][t, i]),
                theta=float(choices["theta"][t, i]),
            )
            timestep_particles.append(pose)
        particle_history.append(timestep_particles)

    # Vectorized weight processing
    # Convert from log weights and normalize
    weights = jnp.exp(all_weights)  # Shape: (n_timesteps, n_particles)
    weight_sums = jnp.sum(weights, axis=1, keepdims=True)  # Shape: (n_timesteps, 1)
    normalized_weights = weights / weight_sums  # Shape: (n_timesteps, n_particles)

    # Convert to list for compatibility
    weight_history = [normalized_weights[t] for t in range(n_timesteps)]

    # Extract diagnostic weights if requested
    diagnostic_weights_history = None
    if collect_diagnostics:
        diagnostic_weights_history = (
            particles_smc.diagnostic_weights
        )  # Shape: [T, n_particles]

    return particle_history, weight_history, diagnostic_weights_history


def run_smc_basic(n_particles, observations, world, key, n_rays=8):
    """Run basic SMC without rejuvenation (just resampling)."""
    from genjax.inference import rejuvenation_smc
    from genjax.core import const

    # Prepare observations
    obs_array = jnp.array(observations)
    obs_sequence = {"measurements": {"distance": obs_array}}

    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)
    initial_args = (dummy_pose, jnp.array(0), world, const(n_rays))

    # Run basic SMC (no rejuvenation)
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,
        observations=obs_sequence,
        initial_model_args=initial_args,
        n_particles=const(n_particles),
        return_all_particles=const(True),
    )

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights


def run_smc_with_mh(
    n_particles, observations, world, key, n_rays=8, K: Const[int] = const(10)
):
    """Run SMC with Metropolis-Hastings rejuvenation."""
    from genjax.inference import rejuvenation_smc, mh
    from genjax.core import const, sel

    # Prepare observations
    obs_array = jnp.array(observations)
    obs_sequence = {"measurements": {"distance": obs_array}}

    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)
    initial_args = (dummy_pose, jnp.array(0), world, const(n_rays))

    # Define MH kernel function that takes a trace and returns a trace
    def mh_kernel(trace):
        # Select all latent variables for MH updates
        selection = sel("x") | sel("y") | sel("theta")
        return mh(trace, selection)

    # Run SMC with MH rejuvenation
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,
        mcmc_kernel=const(mh_kernel),
        observations=obs_sequence,
        initial_model_args=initial_args,
        n_particles=const(n_particles),
        return_all_particles=const(True),
        n_rejuvenation_moves=K,
    )

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights


def run_smc_with_hmc(
    n_particles, observations, world, key, n_rays=8, K: Const[int] = const(10)
):
    """Run SMC with Hamiltonian Monte Carlo rejuvenation."""
    from genjax.inference import rejuvenation_smc, hmc
    from genjax.core import const, sel

    # Prepare observations
    obs_array = jnp.array(observations)
    obs_sequence = {"measurements": {"distance": obs_array}}

    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)
    initial_args = (dummy_pose, jnp.array(0), world, const(n_rays))

    # Define HMC kernel function that takes a trace and returns a trace
    def hmc_kernel(trace):
        # Select all continuous latent variables for HMC updates
        selection = sel("x") | sel("y") | sel("theta")
        return hmc(trace, selection, step_size=0.01, n_steps=10)

    # Run SMC with HMC rejuvenation
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,
        mcmc_kernel=const(hmc_kernel),
        observations=obs_sequence,
        initial_model_args=initial_args,
        n_particles=const(n_particles),
        return_all_particles=const(True),
        n_rejuvenation_moves=K,
    )

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights


def create_locally_optimal_proposal(
    world, grid_size=15, noise_std=0.1, grid_radius=1.5
):
    """Create a locally optimal proposal using grid evaluation and max selection.

    Args:
        world: World object for bounds and constraints
        grid_size: Number of grid points per dimension (3D grid)
        noise_std: Standard deviation for Gaussian noise around selected grid points
        grid_radius: Radius around current position to search (for x,y)

    Returns:
        Generative function for locally optimal proposal
    """

    @gen
    def locally_optimal_proposal(obs, prev_choices, *args):
        """Locally optimal proposal for drift-only model (no velocity variables).

        Args:
            obs: Current observation constraints (what we're conditioning on)
            prev_choices: Previous particle's choices
            *args: Model arguments for this timestep
        """
        # Get current particle position from prev_choices
        current_x = prev_choices.get("x", 1.5)  # Default near start
        current_y = prev_choices.get("y", 1.5)
        current_theta = prev_choices.get("theta", 0.0)

        # Create grid centered around current position
        # For x and y: search within grid_radius
        x_min = jnp.maximum(0.5, current_x - grid_radius)
        x_max = jnp.minimum(world.width - 0.5, current_x + grid_radius)
        y_min = jnp.maximum(0.5, current_y - grid_radius)
        y_max = jnp.minimum(world.height - 0.5, current_y + grid_radius)

        # Create coordinate grids centered on current position
        x_grid = jnp.linspace(x_min, x_max, grid_size)
        y_grid = jnp.linspace(y_min, y_max, grid_size)

        # For theta: search within pi/6 radians of current angle (smaller for drift model)
        theta_radius = jnp.pi / 6
        theta_grid = jnp.linspace(
            current_theta - theta_radius, current_theta + theta_radius, grid_size
        )

        # Create meshgrid for all combinations (3D grid only - no velocity)
        X, Y, THETA = jnp.meshgrid(x_grid, y_grid, theta_grid, indexing="ij")
        grid_points = jnp.stack(
            [X.flatten(), Y.flatten(), THETA.flatten()], axis=1
        )  # Shape: (grid_size^3, 3)

        # Use simpler vectorized assessment over all grid points
        def assess_single_point(grid_point):
            """Assess the model at a single grid point."""
            x_prop, y_prop, theta_prop = grid_point

            # Create constraints for drift-only model
            constraints = {"x": x_prop, "y": y_prop, "theta": theta_prop, **obs}

            # Assess the model
            score, _ = localization_model.assess(constraints, *args)
            return score

        # Vectorize over all grid points
        vectorized_assess = jax.vmap(assess_single_point, in_axes=0)
        log_probs = vectorized_assess(grid_points)

        # Find the maximum probability grid point
        max_idx = jnp.argmax(log_probs)
        selected_point = grid_points[max_idx]

        # Sample with Gaussian noise around the selected point
        x_prop = normal(selected_point[0], noise_std) @ "x"
        y_prop = normal(selected_point[1], noise_std) @ "y"
        theta_prop = normal(selected_point[2], noise_std * 0.5) @ "theta"

        return x_prop, y_prop, theta_prop

    return locally_optimal_proposal


def run_smc_with_locally_optimal(
    n_particles, observations, world, key, n_rays=8, K: Const[int] = const(10)
):
    """Run SMC with locally optimal proposal rejuvenation."""
    from genjax.inference import rejuvenation_smc
    from genjax.core import const

    # Prepare observations
    obs_array = jnp.array(observations)
    obs_sequence = {"measurements": {"distance": obs_array}}

    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)
    initial_args = (dummy_pose, jnp.array(0), world, const(n_rays))

    # Create locally optimal proposal with 25x25x25 grid centered on particle
    locally_optimal_proposal = create_locally_optimal_proposal(
        world, grid_size=25, noise_std=0.15, grid_radius=2.0
    )

    # Run SMC with locally optimal proposal (no mcmc_kernel needed)
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,
        transition_proposal=locally_optimal_proposal,
        observations=obs_sequence,
        initial_model_args=initial_args,
        n_particles=const(n_particles),
        return_all_particles=const(True),
        n_rejuvenation_moves=K,
    )

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights


def run_smc_with_locally_optimal_big_grid(
    n_particles, observations, world, key, n_rays=8, K: Const[int] = const(10)
):
    """Run SMC with locally optimal proposal using a bigger grid (25x25x25)."""
    from genjax.inference import rejuvenation_smc
    from genjax.core import const

    # Prepare observations
    obs_array = jnp.array(observations)
    obs_sequence = {"measurements": {"distance": obs_array}}

    dummy_pose = Pose(x=0.0, y=0.0, theta=0.0)
    initial_args = (dummy_pose, jnp.array(0), world, const(n_rays))

    # Create locally optimal proposal with bigger grid and wider search radius
    locally_optimal_proposal = create_locally_optimal_proposal(
        world, grid_size=25, noise_std=0.1, grid_radius=3.0
    )

    # Run SMC with locally optimal proposal (no mcmc_kernel needed)
    particles_smc = seed(rejuvenation_smc)(
        key,
        localization_model,
        transition_proposal=locally_optimal_proposal,
        observations=obs_sequence,
        initial_model_args=initial_args,
        n_particles=const(n_particles),
        return_all_particles=const(True),
        n_rejuvenation_moves=K,
    )

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights

    # Extract particle history and weights from SMC result
    all_traces = particles_smc.traces
    all_weights = particles_smc.log_weights

    choices = all_traces.get_choices()
    n_timesteps = choices["x"].shape[0]

    particle_history = []
    weight_history = []

    for t in range(n_timesteps):
        timestep_particles = []
        for i in range(n_particles):
            pose = Pose(
                x=choices["x"][t, i], y=choices["y"][t, i], theta=choices["theta"][t, i]
            )
            timestep_particles.append(pose)

        particle_history.append(timestep_particles)

        timestep_weights = jnp.exp(all_weights[t])
        timestep_weights = timestep_weights / jnp.sum(timestep_weights)
        weight_history.append(timestep_weights)

    diagnostic_weights = particles_smc.diagnostic_weights
    return particle_history, weight_history, diagnostic_weights


def benchmark_smc_methods(
    n_particles,
    observations,
    world,
    key,
    n_rays=8,
    repeats=5,
    K=20,
    K_hmc=25,
    n_particles_big_grid=5,
):
    """Benchmark different SMC methods and return timing results."""
    import jax
    from genjax.timing import benchmark_with_warmup
    from genjax.core import const

    # Define methods with their specific particle counts
    methods = {
        "smc_basic": (run_smc_basic, n_particles),
        "smc_hmc": (run_smc_with_hmc, n_particles),
        "smc_locally_optimal": (run_smc_with_locally_optimal, n_particles),
        "smc_locally_optimal_big_grid": (
            run_smc_with_locally_optimal_big_grid,
            n_particles_big_grid,
        ),  # Use parameter
    }

    results = {}

    for method_name, (method_func, method_n_particles) in methods.items():
        print(f"Benchmarking {method_name}...")

        # JIT compile the method for fair timing comparison
        # Note: With Const[...] pattern, we don't need K in static_argnames
        jitted_method = jax.jit(method_func, static_argnames=("n_particles", "n_rays"))

        # Prepare function for timing (pass const(K) for methods that need it)
        def run_method():
            if method_name == "smc_hmc":
                return jitted_method(
                    method_n_particles, observations, world, key, n_rays, const(K_hmc)
                )
            elif method_name in ["smc_locally_optimal", "smc_locally_optimal_big_grid"]:
                return jitted_method(
                    method_n_particles, observations, world, key, n_rays, const(K)
                )
            else:
                return jitted_method(
                    method_n_particles, observations, world, key, n_rays
                )

        # Warm up and time the method
        times, (mean_time, std_time) = benchmark_with_warmup(
            run_method,
            repeats=repeats,
            warmup_runs=3,  # More warmup for JIT
            inner_repeats=3,
        )

        # Get results for quality comparison
        particle_history, weight_history, diagnostic_weights = run_method()

        results[method_name] = {
            "timing_stats": (mean_time, std_time),
            "particle_history": particle_history,
            "weight_history": weight_history,
            "diagnostic_weights": diagnostic_weights,
            "n_particles": method_n_particles,  # Store actual particle count used
        }

        print(f"  {method_name}: {mean_time:.3f} ± {std_time:.3f} seconds")

    return results


# Utility functions
def create_basic_multi_room_world():
    """Create a basic multi-room world with simple rectangular walls and doorways."""
    width, height = 12.0, 10.0

    # Define internal walls to create a simple 3-room layout
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
        # Small rectangular obstacle in Room 3
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


def create_complex_multi_room_world():
    """Create a complex multi-room world with slanted walls and complex geometry."""
    width, height = 12.0, 10.0

    # Define internal walls to create a complex 3-room layout with slanted walls
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
        # Complex slanted obstacle in Room 3
        [9.0, 2.0, 10.5, 2.5],  # Slanted bottom edge
        [10.5, 2.5, 10.0, 3.5],  # Slanted right edge
        [10.0, 3.5, 8.5, 3.0],  # Slanted top edge
        [8.5, 3.0, 9.0, 2.0],  # Slanted left edge (completes diamond)
        # Additional slanted walls in Room 1 for complexity
        [1.0, 1.0, 2.5, 0.5],  # Slanted wall in lower-left
        [3.0, 8.5, 3.5, 9.5],  # Small slanted wall in upper area
        # Slanted barriers in Room 2
        [5.5, 2.0, 6.5, 1.0],  # Diagonal barrier
        [6.0, 4.5, 7.0, 5.5],  # Another diagonal element
        # Complex geometry near Room 3 entrance
        [8.5, 0.5, 9.5, 1.5],  # Slanted guide wall
        [10.5, 8.0, 11.0, 9.0],  # Slanted corner feature
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


def create_multi_room_world(world_type="basic"):
    """Create a multi-room world with configurable complexity.

    Args:
        world_type: "basic" for simple rectangular walls, "complex" for slanted walls

    Returns:
        World object with specified geometry
    """
    if world_type == "basic":
        return create_basic_multi_room_world()
    elif world_type == "complex":
        return create_complex_multi_room_world()
    else:
        raise ValueError(f"Unknown world_type: {world_type}. Use 'basic' or 'complex'")


# Data generation functions moved to data.py
