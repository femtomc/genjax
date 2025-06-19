"""
Synthetic data generation for the localization case study.

Contains trajectory creation and ground truth data generation functions.
"""

import jax.numpy as jnp
import jax.random as jrand
from genjax import seed

from .core import (
    Pose,
    Control,
    sensor_model,
    distance_to_wall_lidar,
    apply_control,
)
from genjax import gen, normal


def create_wall_to_wall_trajectory():
    """Create a trajectory that moves aggressively toward walls to trigger bouncing."""
    # More aggressive movement that should definitely hit walls and bounce

    controls = [
        # Move fast right toward wall
        Control(velocity=2.0, angular_velocity=0.0),  # Step 1: fast right
        Control(velocity=2.0, angular_velocity=0.0),  # Step 2: fast right
        Control(velocity=2.0, angular_velocity=0.0),  # Step 3: fast right
        Control(
            velocity=2.0, angular_velocity=0.0
        ),  # Step 4: fast right (should hit wall)
        Control(velocity=2.0, angular_velocity=0.0),  # Step 5: continue (should bounce)
        # Move down toward bottom wall
        Control(velocity=0.5, angular_velocity=-jnp.pi / 2),  # Step 6: turn down
        Control(velocity=2.0, angular_velocity=0.0),  # Step 7: fast down
        Control(
            velocity=2.0, angular_velocity=0.0
        ),  # Step 8: fast down (should hit bottom)
        Control(velocity=2.0, angular_velocity=0.0),  # Step 9: continue (should bounce)
        # Move up toward top wall
        Control(velocity=0.5, angular_velocity=jnp.pi),  # Step 10: turn around
        Control(velocity=2.0, angular_velocity=0.0),  # Step 11: fast up
        Control(
            velocity=2.0, angular_velocity=0.0
        ),  # Step 12: fast up (should hit top)
    ]

    return controls


def create_bouncing_test_trajectory():
    """Create a simple bouncing test with just a few aggressive moves."""
    return [
        # Start at (2,2) and move fast right - should hit right wall and bounce
        Control(velocity=3.0, angular_velocity=0.0),  # Step 1: very fast right
        Control(
            velocity=3.0, angular_velocity=0.0
        ),  # Step 2: very fast right (hit wall)
        Control(velocity=3.0, angular_velocity=0.0),  # Step 3: should bounce back left
        Control(velocity=3.0, angular_velocity=0.0),  # Step 4: continue bouncing
    ]


def create_room_navigation_trajectory():
    """Create a trajectory that navigates from Room 1 to Room 3.

    Goes from lower-left corner of Room 1 (0.5, 0.5) to
    upper-right corner of Room 3 (11.5, 9.5).

    Room layout:
    - Room 1: x=[0, 4], y=[0, 10]
    - Room 2: x=[4, 8], y=[0, 10]
    - Room 3: x=[8, 12], y=[0, 10]
    - Doorway 1→2: x=4, y=[3, 5]
    - Doorway 2→3: x=8, y=[4, 6]
    """
    return [
        # Phase 1: Move from Room 1 lower-left to doorway (0.5,0.5) → (4,4)
        Control(
            velocity=1.2, angular_velocity=jnp.pi / 6
        ),  # Step 1: Turn slightly up-right
        Control(velocity=1.8, angular_velocity=0.0),  # Step 2: Move toward doorway
        Control(velocity=1.8, angular_velocity=0.0),  # Step 3: Continue toward doorway
        Control(
            velocity=1.5, angular_velocity=0.0
        ),  # Step 4: Approach doorway entrance
        # Phase 2: Navigate through Room 2 to reach second doorway (4,4) → (8,5)
        Control(velocity=1.5, angular_velocity=0.0),  # Step 5: Enter Room 2
        Control(
            velocity=1.2, angular_velocity=jnp.pi / 8
        ),  # Step 6: Turn slightly up toward Room 3 doorway
        Control(
            velocity=1.8, angular_velocity=0.0
        ),  # Step 7: Move toward Room 3 doorway
        Control(velocity=1.5, angular_velocity=0.0),  # Step 8: Approach Room 3 doorway
        # Phase 3: Enter Room 3 and navigate to upper-right corner (8,5) → (11.5,9.5)
        Control(velocity=1.5, angular_velocity=0.0),  # Step 9: Enter Room 3
        Control(
            velocity=1.2, angular_velocity=jnp.pi / 4
        ),  # Step 10: Turn up-right toward corner
        Control(velocity=1.8, angular_velocity=0.0),  # Step 11: Move toward upper-right
        Control(velocity=1.8, angular_velocity=0.0),  # Step 12: Continue toward corner
        Control(
            velocity=1.5, angular_velocity=0.0
        ),  # Step 13: Reach upper-right corner
    ]


def create_exploration_trajectory():
    """Create a trajectory that explores Room 1 systematically."""
    return [
        # Systematic exploration pattern within Room 1
        Control(velocity=1.2, angular_velocity=0.0),  # Move forward
        Control(velocity=1.0, angular_velocity=jnp.pi / 2),  # Turn right
        Control(velocity=1.2, angular_velocity=0.0),  # Move forward
        Control(velocity=1.0, angular_velocity=jnp.pi / 2),  # Turn right
        Control(velocity=1.2, angular_velocity=0.0),  # Move forward
        Control(velocity=1.0, angular_velocity=jnp.pi / 2),  # Turn right
        Control(velocity=1.2, angular_velocity=0.0),  # Move forward
        Control(velocity=1.0, angular_velocity=jnp.pi / 2),  # Complete square
    ]


def create_test_trajectory():
    """Create a test trajectory - defaults to room navigation."""
    return create_room_navigation_trajectory()


def create_waypoint_trajectory_room1_to_room3():
    """Create a trajectory using explicit waypoints from Room 1 to Room 3.

    Returns a list of (x, y) coordinates that define the path from
    lower-left corner of Room 1 to upper-right corner of Room 3.

    Room layout:
    - Room 1: x=[0, 4], y=[0, 10]
    - Room 2: x=[4, 8], y=[0, 10]
    - Room 3: x=[8, 12], y=[0, 10]
    - Doorway 1→2: x=4, y=[3, 5]
    - Doorway 2→3: x=8, y=[4, 6]
    """
    waypoints = [
        # Start in lower-left corner of Room 1
        (0.5, 0.5),
        # Move toward first doorway center
        (1.5, 1.5),  # Diagonal movement toward doorway
        (2.5, 2.5),  # Continue diagonal
        (3.5, 3.5),  # Approach doorway center
        (3.9, 4.0),  # Just before doorway
        # Pass through first doorway into Room 2
        (4.1, 4.0),  # Just inside Room 2
        (4.5, 4.0),  # Firmly in Room 2
        # Navigate through Room 2 toward second doorway
        (5.5, 4.2),  # Move through Room 2
        (6.5, 4.5),  # Continue toward Room 3 doorway
        (7.5, 5.0),  # Approach second doorway center
        (7.9, 5.0),  # Just before second doorway
        # Pass through second doorway into Room 3
        (8.1, 5.0),  # Just inside Room 3
        (8.5, 5.0),  # Firmly in Room 3
        # Navigate to upper-right corner of Room 3
        (9.5, 6.0),  # Move toward corner
        (10.5, 7.5),  # Continue toward corner
        (11.0, 8.5),  # Approach corner
        (11.5, 9.5),  # Reach upper-right corner
    ]

    return waypoints


def create_waypoint_trajectory_exploration_room1():
    """Create exploration trajectory within Room 1 using waypoints."""
    waypoints = [
        # Start in center of Room 1
        (2.0, 2.0),
        # Explore corners and edges of Room 1
        (1.0, 1.0),  # Lower-left area
        (3.0, 1.0),  # Lower-right area
        (3.0, 3.0),  # Upper-right area
        (1.0, 3.0),  # Upper-left area
        (2.0, 5.0),  # Center-upper
        (1.0, 7.0),  # Left side upper
        (3.0, 8.0),  # Right side upper
        (2.0, 9.0),  # Top center
        (2.0, 6.0),  # Return toward center
    ]

    return waypoints


def generate_synthetic_data_from_waypoints(waypoints, world, key, noise_std=0.15):
    """Generate synthetic trajectory data from waypoints using LIDAR sensor model.

    Args:
        waypoints: List of (x, y) coordinate tuples
        world: World object for distance calculations
        key: JAX random key for sensor noise
        noise_std: Standard deviation for sensor noise

    Returns:
        tuple: (initial_pose, poses, observations)
    """
    # Convert waypoints to poses (theta=0.0 since we'll use LIDAR in all directions)
    poses = [Pose(x=x, y=y, theta=0.0) for x, y in waypoints]

    # Generate true LIDAR distances at each waypoint (8 angles)
    true_lidar_distances = [
        distance_to_wall_lidar(pose, world, n_angles=8) for pose in poses
    ]

    # Add noise to create realistic sensor observations
    noise_keys = jrand.split(key, len(waypoints))
    observations = []

    for i, true_lidar in enumerate(true_lidar_distances):
        # Add independent noise to each of the 8 distance measurements
        angle_keys = jrand.split(noise_keys[i], 8)
        noisy_lidar = []
        for j in range(8):
            noise = jrand.normal(angle_keys[j]) * noise_std
            observed_dist = jnp.maximum(
                0.0, true_lidar[j] + noise
            )  # Ensure non-negative
            noisy_lidar.append(observed_dist)
        observations.append(jnp.array(noisy_lidar))

    # Return initial pose separately and remaining poses
    initial_pose = poses[0]
    trajectory_poses = poses[1:]
    trajectory_observations = observations[
        1:
    ]  # Skip first observation since it's at initial pose

    return initial_pose, trajectory_poses, trajectory_observations


def generate_ground_truth_data(world, key, trajectory_type="room_navigation"):
    """Generate ground truth trajectory and observations.

    Args:
        world: World object defining the environment
        key: JAX random key for reproducible generation
        trajectory_type: Type of trajectory to generate
                        ("room_navigation", "wall_bouncing", "exploration", "simple_bounce")

    Returns:
        tuple: (initial_pose, controls, poses, observations)
    """
    # For room_navigation, use waypoint-based approach for reliable cross-room travel
    if trajectory_type == "room_navigation":
        waypoints = create_waypoint_trajectory_room1_to_room3()
        key1, key2 = jrand.split(key)
        initial_pose, poses, observations = generate_synthetic_data_from_waypoints(
            waypoints, world, key1, noise_std=0.15
        )

        # Create dummy controls (not used in waypoint approach but needed for compatibility)
        controls = [
            Control(velocity=1.0, angular_velocity=0.0) for _ in range(len(poses))
        ]
        return initial_pose, controls, poses, observations

    elif trajectory_type == "exploration":
        waypoints = create_waypoint_trajectory_exploration_room1()
        key1, key2 = jrand.split(key)
        initial_pose, poses, observations = generate_synthetic_data_from_waypoints(
            waypoints, world, key1, noise_std=0.15
        )

        # Create dummy controls
        controls = [
            Control(velocity=1.0, angular_velocity=0.0) for _ in range(len(poses))
        ]
        return initial_pose, controls, poses, observations

    # For other trajectory types, use the original control-based approach
    elif trajectory_type == "wall_bouncing":
        controls = create_wall_to_wall_trajectory()
    elif trajectory_type == "simple_bounce":
        controls = create_bouncing_test_trajectory()
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    # Default initial pose for control-based trajectories
    initial_pose = Pose(x=2.0, y=2.0, theta=0.0)

    # Sequential generation using individual models (original approach)
    # For control-based trajectories, we need to create a compatible step model
    # that takes controls as arguments (for backward compatibility)

    @gen
    def step_model_with_controls(pose, control, world):
        """Temporary step model that accepts control arguments for compatibility."""
        # Apply deterministic motion with bouncing
        ideal_next_pose = apply_control(pose, control, world)

        # Add motion noise
        next_x = normal(ideal_next_pose.x, 0.1) @ "x"
        next_y = normal(ideal_next_pose.y, 0.1) @ "y"
        next_theta = normal(ideal_next_pose.theta, 0.05) @ "theta"

        # Keep within world boundaries
        bounce_threshold = 0.3
        next_x = jnp.clip(next_x, bounce_threshold, world.width - bounce_threshold)
        next_y = jnp.clip(next_y, bounce_threshold, world.height - bounce_threshold)

        return Pose(next_x, next_y, next_theta)

    seeded_step = seed(step_model_with_controls.simulate)
    seeded_sensor = seed(sensor_model.simulate)

    poses = [initial_pose]
    observations = []
    current_pose = initial_pose

    for i, control in enumerate(controls):
        # Generate next pose
        step_key, key = jrand.split(key)
        step_trace = seeded_step(step_key, current_pose, control, world)
        current_pose = step_trace.get_retval()

        # Generate observation at new pose
        obs_key, key = jrand.split(key)
        obs_trace = seeded_sensor(obs_key, current_pose, world)
        observation = obs_trace.get_retval()

        poses.append(current_pose)
        observations.append(observation)

    return (
        initial_pose,
        controls,
        poses[1:],
        observations,
    )  # Return initial pose separately


def generate_multiple_trajectories(world, key, n_trajectories=5, trajectory_types=None):
    """Generate multiple ground truth trajectories for comparison.

    Args:
        world: World object
        key: JAX random key
        n_trajectories: Number of trajectories to generate
        trajectory_types: List of trajectory types to cycle through

    Returns:
        list: List of (initial_pose, controls, poses, observations) tuples
    """
    if trajectory_types is None:
        trajectory_types = ["room_navigation", "exploration", "wall_bouncing"]

    trajectories = []
    keys = jrand.split(key, n_trajectories)

    for i in range(n_trajectories):
        traj_type = trajectory_types[i % len(trajectory_types)]
        traj_data = generate_ground_truth_data(
            world, keys[i], trajectory_type=traj_type
        )
        trajectories.append((traj_type, traj_data))

    return trajectories
