"""
GenJAX Localization Case Study

Probabilistic robot localization using particle filtering.
Demonstrates GenJAX usage for sequential inference and state estimation.
"""

from .core import (
    Pose,
    Control,
    World,
    run_particle_filter,
)

from .data import (
    generate_ground_truth_data,
    generate_multiple_trajectories,
)

from .figs import (
    plot_world,
    plot_trajectory,
    plot_particle_filter_step,
    plot_particle_filter_evolution,
    plot_estimation_error,
    plot_sensor_observations,
    plot_ground_truth_trajectory,
    plot_multiple_trajectories,
)

__all__ = [
    "Pose",
    "Control",
    "World",
    "generate_ground_truth_data",
    "generate_multiple_trajectories",
    "run_particle_filter",
    "plot_world",
    "plot_trajectory",
    "plot_particle_filter_step",
    "plot_particle_filter_evolution",
    "plot_estimation_error",
    "plot_sensor_observations",
    "plot_ground_truth_trajectory",
    "plot_multiple_trajectories",
]
