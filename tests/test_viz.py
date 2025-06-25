"""
Tests for GenJAX visualization module.

This module tests the visualization utilities in the viz module,
particularly the raincloud plotting functionality.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from genjax.viz.raincloud import (
    horizontal_raincloud,
    raincloud,
    diagnostic_raincloud,
    _estimate_density,
    _scale_density,
    _draw_half_violin,
    _draw_boxplot,
    _draw_stripplot,
)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


def test_estimate_density():
    """Test kernel density estimation helper function."""
    # Normal data
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    support, density = _estimate_density(data)

    assert len(support) == 100  # Default gridsize
    assert len(density) == 100
    assert jnp.all(jnp.isfinite(density))
    assert jnp.all(density >= 0)

    # Single point data - special case returns just the unique value
    single_data = jnp.array([1.0])
    support_single, density_single = _estimate_density(single_data)
    assert len(support_single) == 1  # Returns unique values only
    assert len(density_single) == 1
    assert jnp.all(jnp.isfinite(density_single))

    # Empty data - special case returns empty arrays
    empty_data = jnp.array([])
    support_empty, density_empty = _estimate_density(empty_data)
    assert len(support_empty) == 0  # Returns empty arrays
    assert len(density_empty) == 1  # Single 1.0 value
    assert density_empty[0] == 1.0


def test_scale_density():
    """Test density scaling helper function."""
    density = jnp.array([0.1, 0.2, 0.5, 0.3, 0.1])

    # Area scaling (default)
    scaled = _scale_density(density, scale="area")
    assert jnp.max(scaled) == 1.0

    # Width scaling
    scaled_width = _scale_density(density, scale="width")
    assert jnp.max(scaled_width) == 1.0

    # Count scaling
    scaled_count = _scale_density(density, scale="count")
    assert jnp.max(scaled_count) == 1.0


# =============================================================================
# RAINCLOUD PLOT TESTS
# =============================================================================


def test_horizontal_raincloud_single_distribution():
    """Test raincloud plot with single distribution."""
    # Generate test data
    data = jnp.array(np.random.normal(0, 1, 100))

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    result_ax = horizontal_raincloud(data, ax=ax)

    # Check that we got back the same axis
    assert result_ax is ax

    # Check that the plot has some content
    assert len(ax.collections) > 0  # Should have violin plots
    assert len(ax.lines) > 0  # Should have box plot lines

    plt.close(fig)


def test_horizontal_raincloud_multiple_distributions():
    """Test raincloud plot with multiple distributions."""
    # Generate test data with different means
    data = [
        jnp.array(np.random.normal(0, 1, 100)),
        jnp.array(np.random.normal(2, 1.5, 150)),
        jnp.array(np.random.normal(-1, 0.5, 80)),
    ]
    labels = ["Group A", "Group B", "Group C"]
    colors = ["red", "blue", "green"]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    result_ax = horizontal_raincloud(data, labels=labels, colors=colors, ax=ax)

    # Check plot elements
    assert result_ax is ax
    assert len(ax.collections) > 0
    assert len(ax.lines) > 0

    # Check that we have correct number of y-tick labels
    assert len(ax.get_yticklabels()) == 3

    plt.close(fig)


def test_horizontal_raincloud_with_jax_arrays():
    """Test that raincloud plots work with JAX arrays."""
    # JAX array input
    jax_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0])

    fig, ax = plt.subplots()
    result_ax = horizontal_raincloud(jax_data, ax=ax)

    assert result_ax is ax
    assert len(ax.collections) > 0

    plt.close(fig)


def test_horizontal_raincloud_empty_data():
    """Test raincloud plot behavior with empty data."""
    # Empty data should not crash
    empty_data = jnp.array([])

    fig, ax = plt.subplots()
    result_ax = horizontal_raincloud(empty_data, ax=ax)

    assert result_ax is ax

    plt.close(fig)


def test_horizontal_raincloud_orientation():
    """Test both horizontal and vertical orientations."""
    data = jnp.array(np.random.normal(0, 1, 50))

    # Horizontal orientation
    fig, ax = plt.subplots()
    horizontal_raincloud(data, ax=ax, orient="h")
    plt.close(fig)

    # Vertical orientation
    fig, ax = plt.subplots()
    horizontal_raincloud(data, ax=ax, orient="v")
    plt.close(fig)


def test_raincloud_convenience_function():
    """Test the convenience raincloud function."""
    # Simple usage
    data = jnp.array(np.random.normal(0, 1, 100))

    fig, ax = plt.subplots()
    result_ax = raincloud(data, ax=ax)

    assert result_ax is ax
    assert len(ax.collections) > 0

    plt.close(fig)


# =============================================================================
# DIAGNOSTIC RAINCLOUD TESTS
# =============================================================================


def test_diagnostic_raincloud():
    """Test diagnostic raincloud for particle filter weights."""
    # Simulate particle weights (should sum to 1)
    n_particles = 100
    weights = jnp.array(np.random.dirichlet(np.ones(n_particles)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Test diagnostic raincloud
    ess, ess_color = diagnostic_raincloud(
        ax, weights, position=0, n_particles=n_particles
    )

    # Check ESS computation
    assert 0 <= ess <= 1  # ESS ratio should be between 0 and 1
    assert ess_color.startswith("#")  # Should be a hex color code
    assert len(ess_color) == 7  # Hex color format #RRGGBB

    # Check plot elements
    assert len(ax.collections) > 0  # Should have scatter plots

    plt.close(fig)


def test_diagnostic_raincloud_ess_coloring():
    """Test ESS-based coloring in diagnostic raincloud."""
    n_particles = 100

    # Test different ESS scenarios
    # High ESS (uniform weights)
    uniform_weights = jnp.ones(n_particles) / n_particles
    fig, ax = plt.subplots()
    ess_high, color_high = diagnostic_raincloud(
        ax, uniform_weights, 0, n_particles=n_particles
    )
    assert color_high == "#2E8B57"  # Should be sea green for good ESS
    plt.close(fig)

    # Low ESS (one dominant weight)
    concentrated_weights = jnp.zeros(n_particles).at[0].set(0.99)
    concentrated_weights = concentrated_weights.at[1:].set(0.01 / (n_particles - 1))
    fig, ax = plt.subplots()
    ess_low, color_low = diagnostic_raincloud(
        ax, concentrated_weights, 0, n_particles=n_particles
    )
    assert color_low == "#DC143C"  # Should be crimson for poor ESS
    plt.close(fig)


def test_diagnostic_raincloud_multiple_timesteps():
    """Test diagnostic raincloud with multiple timesteps."""
    n_particles = 50
    n_timesteps = 5

    fig, ax = plt.subplots(figsize=(10, 6))

    ess_values = []
    colors = []

    for t in range(n_timesteps):
        # Different weight distributions over time
        if t == 0:
            weights = jnp.ones(n_particles) / n_particles  # Uniform
        else:
            # Increasingly concentrated
            alpha = jnp.ones(n_particles) * (6 - t)  # Decreasing concentration
            weights = jnp.array(np.random.dirichlet(alpha))

        ess, color = diagnostic_raincloud(
            ax, weights, position=t, n_particles=n_particles, width=0.3
        )

        ess_values.append(ess)
        colors.append(color)

    # Should have decreasing ESS over time
    assert ess_values[0] > ess_values[-1]

    # Check plot has multiple elements
    assert len(ax.collections) >= n_timesteps

    plt.close(fig)


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


def test_horizontal_raincloud_parameters():
    """Test parameter validation and defaults."""
    data = jnp.array(np.random.normal(0, 1, 50))

    # Test width parameters
    fig, ax = plt.subplots()
    horizontal_raincloud(
        data,
        ax=ax,
        width_violin=0.6,
        width_box=0.2,
        jitter=0.1,
        point_size=30,
        alpha=0.8,
    )
    plt.close(fig)

    # Test with None labels (should work)
    fig, ax = plt.subplots()
    horizontal_raincloud(data, labels=None, ax=ax)
    plt.close(fig)


def test_draw_component_functions():
    """Test individual drawing component functions."""
    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    support, density = _estimate_density(data)

    fig, ax = plt.subplots()

    # Test violin component
    _draw_half_violin(
        ax,
        position=0,
        support=support,
        density=density,
        width=0.4,
        color="blue",
        alpha=0.7,
        orient="h",
        offset=0,
    )

    # Test boxplot component
    _draw_boxplot(
        ax, position=0, data=data, width=0.2, color="blue", orient="h", offset=0
    )

    # Test stripplot component
    _draw_stripplot(
        ax,
        position=0,
        data=data,
        jitter=0.05,
        size=20,
        color="blue",
        alpha=0.7,
        orient="h",
        offset=0,
    )

    # Should have created plot elements
    assert len(ax.collections) > 0 or len(ax.patches) > 0 or len(ax.lines) > 0

    plt.close(fig)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_single_value_data():
    """Test behavior with single repeated value."""
    # All same value
    data = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])

    fig, ax = plt.subplots()
    result_ax = horizontal_raincloud(data, ax=ax)

    # Should handle gracefully without crashing
    assert result_ax is ax

    plt.close(fig)


def test_very_small_data():
    """Test with very small datasets."""
    # Two points
    small_data = jnp.array([1.0, 2.0])

    fig, ax = plt.subplots()
    horizontal_raincloud(small_data, ax=ax)
    plt.close(fig)

    # Single point
    single_data = jnp.array([1.0])

    fig, ax = plt.subplots()
    horizontal_raincloud(single_data, ax=ax)
    plt.close(fig)


if __name__ == "__main__":
    # Run basic functionality tests
    test_estimate_density()
    test_scale_density()

    # Run raincloud plot tests
    test_horizontal_raincloud_single_distribution()
    test_horizontal_raincloud_multiple_distributions()
    test_horizontal_raincloud_with_jax_arrays()
    test_horizontal_raincloud_empty_data()
    test_horizontal_raincloud_orientation()
    test_raincloud_convenience_function()

    # Run diagnostic tests
    test_diagnostic_raincloud()
    test_diagnostic_raincloud_ess_coloring()
    test_diagnostic_raincloud_multiple_timesteps()

    # Run parameter validation tests
    test_horizontal_raincloud_parameters()
    test_draw_component_functions()

    # Run edge case tests
    test_single_value_data()
    test_very_small_data()

    print("All visualization tests passed!")
