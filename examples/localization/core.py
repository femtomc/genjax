"""
Localization case study using GenJAX.

Simplified probabilistic robot localization with particle filtering.
Focuses on the probabilistic aspects without complex physics.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import gen, normal, Pytree, modular_vmap as vmap
from genjax.core import Scan


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
class World(Pytree):
    """Simplified world with boundaries."""

    width: float
    height: float


# Physics functions (simplified)
def apply_control(pose: Pose, control: Control, dt: float = 0.1) -> Pose:
    """Apply control command to pose with deterministic motion model."""
    # Update heading
    new_theta = pose.theta + control.angular_velocity * dt

    # Update position
    new_x = pose.x + control.velocity * jnp.cos(new_theta) * dt
    new_y = pose.y + control.velocity * jnp.sin(new_theta) * dt

    return Pose(new_x, new_y, new_theta)


def distance_to_wall(pose: Pose, world: World) -> float:
    """Compute distance to nearest wall (simplified sensor model)."""
    # Distance to each wall
    dist_left = pose.x
    dist_right = world.width - pose.x
    dist_bottom = pose.y
    dist_top = world.height - pose.y

    # Return minimum distance
    return jnp.minimum(
        jnp.minimum(dist_left, dist_right), jnp.minimum(dist_bottom, dist_top)
    )


# Generative functions
@gen
def step_model(pose: Pose, control: Control, world: World):
    """Generative model for a single robot step with motion noise."""
    # Apply deterministic motion
    ideal_next_pose = apply_control(pose, control)

    # Add motion noise
    noise_x = normal(0.0, 0.1) @ "noise_x"
    noise_y = normal(0.0, 0.1) @ "noise_y"
    noise_theta = normal(0.0, 0.05) @ "noise_theta"

    # Constrain to world boundaries
    next_x = jnp.clip(ideal_next_pose.x + noise_x, 0.0, world.width)
    next_y = jnp.clip(ideal_next_pose.y + noise_y, 0.0, world.height)
    next_theta = ideal_next_pose.theta + noise_theta

    next_pose = Pose(next_x, next_y, next_theta)
    return next_pose


@gen
def sensor_model(pose: Pose, world: World):
    """Generative model for sensor observations."""
    # True distance to wall
    true_distance = distance_to_wall(pose, world)

    # Add sensor noise
    sensor_noise = normal(0.0, 0.2) @ "sensor_noise"
    observed_distance = jnp.maximum(0.0, true_distance + sensor_noise)

    return observed_distance


@gen
def trajectory_model(initial_pose: Pose, controls, world: World):
    """Full trajectory model using Scan combinator."""

    # Create step function for scan that matches the expected interface
    @gen
    def scan_step(carry, control_data):
        # carry is the previous pose, world is available in closure
        pose = carry
        # Reconstruct control from scan input
        velocity, angular_velocity = control_data
        control = Control(velocity=velocity, angular_velocity=angular_velocity)
        next_pose = step_model(pose, control, world) @ "step"
        observation = sensor_model(next_pose, world) @ "obs"
        return next_pose, (next_pose, observation)

    # Convert controls to JAX arrays for scan
    velocities = jnp.array([c.velocity for c in controls])
    angular_velocities = jnp.array([c.angular_velocity for c in controls])
    control_arrays = (velocities, angular_velocities)

    # Use Scan for trajectory
    scan_fn = Scan(scan_step, length=len(controls))
    final_pose, (poses, observations) = (
        scan_fn(initial_pose, control_arrays) @ "trajectory"
    )

    return poses, observations


# Particle filter implementation
def resample_particles(particles, weights, key):
    """Resample particles based on weights using systematic resampling."""
    n_particles = len(particles)

    # Normalize weights
    weights = weights / jnp.sum(weights)

    # Systematic resampling
    indices = jrand.categorical(key, jnp.log(weights), shape=(n_particles,))

    # Return resampled particles
    return jax.tree.map(lambda x: x[indices], particles)


def particle_filter_step(particles, weights, control, observation, world):
    """Single step of particle filter."""

    # Predict: apply motion model to each particle
    @gen
    def predict_particle():
        # We'll pass particle via vmap
        return None

    # Use vmap to apply step_model to all particles
    def predict_single_particle(particle):
        return step_model.simulate((particle, control, world))

    predicted_traces = vmap(predict_single_particle)(particles)
    predicted_poses = jax.tree.map(lambda tr: tr.get_retval(), predicted_traces)

    # Update: weight particles by observation likelihood
    def compute_weight(pose):
        log_weight, _ = sensor_model.assess((pose, world), observation)
        return jnp.exp(log_weight)

    new_weights = vmap(compute_weight)(predicted_poses)

    return predicted_poses, new_weights


def run_particle_filter(initial_particles, controls, observations, world, key):
    """Run full particle filter on trajectory."""
    n_steps = len(controls)
    n_particles = len(initial_particles)

    particles = initial_particles
    weights = jnp.ones(n_particles) / n_particles

    particle_history = []
    weight_history = []

    keys = jrand.split(key, n_steps)

    for t in range(n_steps):
        # Particle filter step
        particles, weights = particle_filter_step(
            particles, weights, controls[t], observations[t], world
        )

        # Store results
        particle_history.append(particles)
        weight_history.append(weights)

        # Resample if needed (simple criterion: effective sample size)
        eff_sample_size = 1.0 / jnp.sum(weights**2)
        if eff_sample_size < n_particles / 2:
            particles = resample_particles(particles, weights, keys[t])
            weights = jnp.ones(n_particles) / n_particles

    return particle_history, weight_history


# Utility functions
def create_simple_world():
    """Create a simple rectangular world."""
    return World(width=10.0, height=10.0)


def create_test_trajectory():
    """Create a simple test trajectory."""
    # Square trajectory - return as structured arrays that JAX can handle
    velocities = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    angular_velocities = jnp.array(
        [0.0, 0.0, jnp.pi / 2, 0.0, 0.0, jnp.pi / 2, 0.0, 0.0]
    )

    # Create array of Control objects (will be indexed during scan)
    controls = []
    for v, av in zip(velocities, angular_velocities):
        controls.append(Control(velocity=v, angular_velocity=av))

    return controls


def generate_ground_truth_data(world, key):
    """Generate ground truth trajectory and observations."""
    initial_pose = Pose(x=2.0, y=2.0, theta=0.0)
    controls = create_test_trajectory()

    # Generate true trajectory
    trace = trajectory_model.simulate((initial_pose, controls, world))
    poses, observations = trace.get_retval()

    return initial_pose, controls, poses, observations
