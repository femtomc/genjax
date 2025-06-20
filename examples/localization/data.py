"""
Synthetic data generation for the localization case study.

Contains trajectory creation and ground truth data generation functions.
"""

import jax.numpy as jnp
import jax.random as jrand

from .core import (
    Pose,
    Control,
    distance_to_wall_lidar,
)


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
        # Start offset from walls in Room 1
        (1.2, 1.2),
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


def generate_synthetic_data_from_waypoints(
    waypoints, world, key, noise_std=0.15, n_rays=8
):
    """Generate synthetic trajectory data from waypoints using LIDAR sensor model.

    Args:
        waypoints: List of (x, y) coordinate tuples
        world: World object for distance calculations
        key: JAX random key for sensor noise
        noise_std: Standard deviation for sensor noise
        n_rays: Number of LIDAR rays for distance measurements

    Returns:
        tuple: (initial_pose, poses, observations)
    """
    # Convert waypoints to poses (theta=0.0 since we'll use LIDAR in all directions)
    poses = [Pose(x=x, y=y, theta=0.0) for x, y in waypoints]

    # Generate true LIDAR distances at each waypoint
    true_lidar_distances = [
        distance_to_wall_lidar(pose, world, n_angles=n_rays) for pose in poses
    ]

    # Add noise to create realistic sensor observations
    noise_keys = jrand.split(key, len(waypoints))
    observations = []

    for i, true_lidar in enumerate(true_lidar_distances):
        # Add independent noise to each distance measurement
        angle_keys = jrand.split(noise_keys[i], n_rays)
        noisy_lidar = []
        for j in range(n_rays):
            noise = jrand.normal(angle_keys[j]) * noise_std
            observed_dist = jnp.maximum(
                0.0, true_lidar[j] + noise
            )  # Ensure non-negative
            noisy_lidar.append(observed_dist)
        observations.append(jnp.array(noisy_lidar))

    # Return all poses and observations - no need to separate initial pose
    # For rejuvenation_smc, we need observations for each pose including the initial one
    return poses, observations


def generate_ground_truth_data(
    world, key, trajectory_type="room_navigation", n_steps=16, n_rays=8
):
    """Generate ground truth trajectory and observations.

    Args:
        world: World object defining the environment
        key: JAX random key for reproducible generation
        trajectory_type: Type of trajectory to generate
                        ("room_navigation", "exploration")
        n_steps: Number of trajectory steps to generate
        n_rays: Number of LIDAR rays for distance measurements

    Returns:
        tuple: (all_poses, controls, observations)
    """
    # For room_navigation, use waypoint-based approach for reliable cross-room travel
    if trajectory_type == "room_navigation":
        waypoints = create_waypoint_trajectory_room1_to_room3()
        key1, key2 = jrand.split(key)
        all_poses, observations = generate_synthetic_data_from_waypoints(
            waypoints, world, key1, noise_std=0.15, n_rays=n_rays
        )

        # Create dummy controls (not used in waypoint approach but needed for compatibility)
        controls = [
            Control(velocity=1.0, angular_velocity=0.0)
            for _ in range(len(all_poses) - 1)
        ]
        return all_poses, controls, observations

    elif trajectory_type == "exploration":
        waypoints = create_waypoint_trajectory_exploration_room1()
        key1, key2 = jrand.split(key)
        all_poses, observations = generate_synthetic_data_from_waypoints(
            waypoints, world, key1, noise_std=0.15, n_rays=n_rays
        )

        # Create dummy controls
        controls = [
            Control(velocity=1.0, angular_velocity=0.0)
            for _ in range(len(all_poses) - 1)
        ]
        return all_poses, controls, observations

    else:
        raise ValueError(
            f"Unknown trajectory_type: {trajectory_type}. Supported types: 'room_navigation', 'exploration'"
        )


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
        trajectory_types = ["room_navigation", "exploration"]

    trajectories = []
    keys = jrand.split(key, n_trajectories)

    for i in range(n_trajectories):
        traj_type = trajectory_types[i % len(trajectory_types)]
        traj_data = generate_ground_truth_data(
            world, keys[i], trajectory_type=traj_type
        )
        trajectories.append((traj_type, traj_data))

    return trajectories
