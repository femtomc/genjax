"""
Custom raincloud plot implementation for GenJAX.

Adapted from PtitPrince library to avoid seaborn compatibility issues.
Provides horizontal raincloud plots combining violin plots, box plots, and scatter plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax.numpy as jnp
from scipy import stats
from beartype.typing import Optional, List, Any


def _estimate_density(
    data: np.ndarray, bw: str = "scott", cut: float = 2.0, gridsize: int = 100
):
    """Estimate kernel density for a data vector."""
    if len(data) == 0:
        return np.array([]), np.array([1.0])

    if len(np.unique(data)) == 1:
        return np.unique(data), np.array([1.0])

    # Fit KDE
    try:
        kde = stats.gaussian_kde(data, bw_method=bw)
    except Exception:
        kde = stats.gaussian_kde(data)

    # Get bandwidth
    bw_used = kde.factor * data.std(ddof=1)

    # Define support grid
    support_min = data.min() - bw_used * cut
    support_max = data.max() + bw_used * cut
    support = np.linspace(support_min, support_max, gridsize)

    # Evaluate density
    density = kde.evaluate(support)

    return support, density


def _scale_density(density: np.ndarray, scale: str = "area"):
    """Scale density curve."""
    if len(density) <= 1 or density.max() == 0:
        return density

    if scale == "area":
        return density / density.max()
    elif scale == "width":
        return density / density.max()
    elif scale == "count":
        return density / density.max()
    else:
        return density / density.max()


def _draw_half_violin(
    ax,
    position: float,
    support: np.ndarray,
    density: np.ndarray,
    width: float = 0.4,
    color: str = "lightblue",
    alpha: float = 0.7,
    orient: str = "h",
    offset: float = 0.0,
):
    """Draw half violin (cloud part of raincloud)."""
    if len(support) == 0 or len(density) == 0:
        return

    # Scale density to width
    scaled_density = density * width

    if orient == "h":
        # Horizontal orientation: violin extends in y-direction from position
        y_violin = position + offset + scaled_density
        ax.fill_between(
            support, position + offset, y_violin, color=color, alpha=alpha, zorder=1
        )
        ax.plot(support, y_violin, color="gray", linewidth=0.5, alpha=0.8, zorder=2)
    else:
        # Vertical orientation: violin extends in x-direction from position
        x_violin = position + offset + scaled_density
        ax.fill_betweenx(
            support, position + offset, x_violin, color=color, alpha=alpha, zorder=1
        )
        ax.plot(x_violin, support, color="gray", linewidth=0.5, alpha=0.8, zorder=2)


def _draw_boxplot(
    ax,
    position: float,
    data: np.ndarray,
    width: float = 0.15,
    color: str = "black",
    orient: str = "h",
    offset: float = 0.0,
):
    """Draw box plot (umbrella part of raincloud)."""
    if len(data) == 0:
        return

    # Calculate quartiles
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    whisker_lim = 1.5 * iqr

    # Whisker limits
    lower_whisker = np.min(data[data >= (q25 - whisker_lim)])
    upper_whisker = np.max(data[data <= (q75 + whisker_lim)])

    if orient == "h":
        # Horizontal box plot
        y_center = position + offset

        # Box
        box = patches.Rectangle(
            (q25, y_center - width / 2),
            q75 - q25,
            width,
            linewidth=1,
            edgecolor=color,
            facecolor="white",
            zorder=3,
        )
        ax.add_patch(box)

        # Median line
        ax.plot(
            [q50, q50],
            [y_center - width / 2, y_center + width / 2],
            color=color,
            linewidth=2,
            zorder=4,
        )

        # Whiskers
        ax.plot(
            [lower_whisker, q25],
            [y_center, y_center],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [q75, upper_whisker],
            [y_center, y_center],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [lower_whisker, lower_whisker],
            [y_center - width / 4, y_center + width / 4],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [upper_whisker, upper_whisker],
            [y_center - width / 4, y_center + width / 4],
            color=color,
            linewidth=1,
            zorder=3,
        )
    else:
        # Vertical box plot
        x_center = position + offset

        # Box
        box = patches.Rectangle(
            (x_center - width / 2, q25),
            width,
            q75 - q25,
            linewidth=1,
            edgecolor=color,
            facecolor="white",
            zorder=3,
        )
        ax.add_patch(box)

        # Median line
        ax.plot(
            [x_center - width / 2, x_center + width / 2],
            [q50, q50],
            color=color,
            linewidth=2,
            zorder=4,
        )

        # Whiskers
        ax.plot(
            [x_center, x_center],
            [lower_whisker, q25],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [x_center, x_center],
            [q75, upper_whisker],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [x_center - width / 4, x_center + width / 4],
            [lower_whisker, lower_whisker],
            color=color,
            linewidth=1,
            zorder=3,
        )
        ax.plot(
            [x_center - width / 4, x_center + width / 4],
            [upper_whisker, upper_whisker],
            color=color,
            linewidth=1,
            zorder=3,
        )


def _draw_stripplot(
    ax,
    position: float,
    data: np.ndarray,
    jitter: float = 0.05,
    size: float = 20,
    color: str = "darkblue",
    alpha: float = 0.6,
    orient: str = "h",
    offset: float = 0.0,
):
    """Draw strip plot (rain part of raincloud)."""
    if len(data) == 0:
        return

    # Add jitter
    n_points = len(data)
    jitter_vals = np.random.uniform(-jitter, jitter, n_points)

    if orient == "h":
        # Horizontal strip plot
        y_positions = position + offset - 0.2 + jitter_vals  # Below the violin/box
        ax.scatter(
            data,
            y_positions,
            s=size,
            color=color,
            alpha=alpha,
            zorder=2,
            edgecolors="white",
            linewidth=0.5,
        )
    else:
        # Vertical strip plot
        x_positions = position + offset - 0.2 + jitter_vals  # Left of the violin/box
        ax.scatter(
            x_positions,
            data,
            s=size,
            color=color,
            alpha=alpha,
            zorder=2,
            edgecolors="white",
            linewidth=0.5,
        )


def horizontal_raincloud(
    data: Any,  # Accept JAX/NumPy arrays flexibly
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    colors: Optional[List[str]] = None,
    width_violin: float = 0.4,
    width_box: float = 0.15,
    jitter: float = 0.05,
    point_size: float = 20,
    alpha: float = 0.7,
    orient: str = "h",
) -> plt.Axes:
    """
    Create a horizontal raincloud plot.

    Parameters:
    -----------
    data : list of arrays or single array
        Data to plot. If list, each array is a separate category.
    labels : list of str, optional
        Labels for categories
    ax : matplotlib Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size if creating new figure
    colors : list of str, optional
        Colors for each category
    width_violin : float
        Width of violin plots (clouds)
    width_box : float
        Width of box plots (umbrellas)
    jitter : float
        Amount of jitter for strip plots (rain)
    point_size : float
        Size of points in strip plots
    alpha : float
        Transparency
    orient : str
        Orientation: "h" for horizontal, "v" for vertical

    Returns:
    --------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Convert JAX arrays to NumPy arrays for compatibility
    def _to_numpy(arr):
        if hasattr(arr, "__array__"):  # JAX arrays
            return np.array(arr)
        return arr

    # Handle single array input and convert to NumPy
    if not isinstance(data, list):
        data = [_to_numpy(data)]
    else:
        data = [_to_numpy(arr) for arr in data]

    n_categories = len(data)

    # Default labels
    if labels is None:
        labels = [f"Category {i + 1}" for i in range(n_categories)]

    # Default colors
    if colors is None:
        # Use a colormap for variety
        cmap = plt.cm.Set3
        colors = [cmap(i / max(1, n_categories - 1)) for i in range(n_categories)]

    # Plot each category
    for i, (category_data, label, color) in enumerate(zip(data, labels, colors)):
        position = i

        if len(category_data) == 0:
            continue

        # 1. Draw cloud (half violin)
        support, density = _estimate_density(category_data)
        density = _scale_density(density)
        _draw_half_violin(
            ax, position, support, density, width_violin, color, alpha, orient
        )

        # 2. Draw umbrella (box plot)
        _draw_boxplot(ax, position, category_data, width_box, "black", orient)

        # 3. Draw rain (strip plot)
        _draw_stripplot(
            ax,
            position,
            category_data,
            jitter,
            point_size,
            "darkblue",
            alpha * 0.8,
            orient,
        )

    # Set labels and formatting
    if orient == "h":
        ax.set_yticks(range(n_categories))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Value")
        ax.set_ylabel("Category")
        # Invert y-axis so first category is at top
        ax.invert_yaxis()
    else:
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


# Convenience function matching PtitPrince API more closely
def raincloud(x=None, y=None, data=None, orient="h", **kwargs):
    """
    Create a raincloud plot with pandas DataFrame support.

    This is a simplified version of the PtitPrince RainCloud function.
    """
    if data is not None and x is not None and y is not None:
        # Extract data from DataFrame
        if hasattr(data, "groupby"):  # pandas DataFrame
            groups = data.groupby(x)[y].apply(lambda x: x.values).tolist()
            labels = list(data[x].unique())
        else:
            raise ValueError("Data must be a pandas DataFrame for x/y interface")
    elif isinstance(y, list):
        groups = y
        labels = x if x is not None else None
    else:
        groups = [y] if y is not None else [x]
        labels = None

    return horizontal_raincloud(groups, labels=labels, orient=orient, **kwargs)


def diagnostic_raincloud(
    ax,
    timestep_data,
    position,
    color="#606060",
    width=0.3,
    n_particles=None,
    ess_thresholds=None,
):
    """
    Create a layered raincloud visualization for diagnostic data at a specific timestep.

    This creates a three-layer visualization:
    1. Bottom: Scatter plot (cloud) of individual data points
    2. Middle: Vertical lines showing histogram-based density (rain)
    3. Top: Smooth KDE density curve with fill (cloud on top)

    Parameters:
    -----------
    ax : matplotlib Axes
        Axes to plot on
    timestep_data : array-like
        Data values for this timestep
    position : float
        Y-coordinate for the timestep (horizontal raincloud)
    color : str
        Color for the visualization
    width : float
        Vertical width scaling factor for density components
    n_particles : int, optional
        Total number of particles (for ESS quality assessment)
    ess_thresholds : dict, optional
        Custom ESS quality thresholds. Default: {"good": 0.5, "medium": 0.25}
        Values are fractions of n_particles.

    Returns:
    --------
    tuple : (ess_value, ess_color) - ESS value and color based on quality
    """
    import numpy as np
    from scipy import stats

    # Convert to numpy for scipy compatibility
    if hasattr(timestep_data, "__array__"):
        data = np.array(timestep_data)
    else:
        data = timestep_data

    if len(data) == 0:
        return 0.0

    # Compute ESS (Effective Sample Size)
    ess = 1.0 / jnp.sum(data**2)
    
    # Compute ESS ratio (ESS / n_particles)
    ess_ratio = ess / n_particles if n_particles is not None else ess

    # Determine ESS quality and color based on ratio
    ess_color = "#000000"  # Default black
    if n_particles is not None:
        # Set default thresholds as fractions (already ratios)
        if ess_thresholds is None:
            ess_thresholds = {"good": 0.5, "medium": 0.25}

        if ess_ratio >= ess_thresholds["good"]:
            ess_color = "#2E8B57"  # Sea green - good
        elif ess_ratio >= ess_thresholds["medium"]:
            ess_color = "#FF8C00"  # Dark orange - medium
        else:
            ess_color = "#DC143C"  # Crimson - bad

    # 1. Bottom layer: Scatter plot (cloud)
    y_positions = np.full_like(data, position - 0.15)
    ax.scatter(data, y_positions, alpha=0.4, s=8, color=color, zorder=2)

    # 2. Middle layer: Histogram-based density (rain)
    if len(data) > 1:
        bins = np.linspace(0, np.max(data), 20)
        hist, bin_edges = np.histogram(data, bins=bins)
        if np.max(hist) > 0:
            hist_norm = hist / np.max(hist) * (width * 0.5)  # Scale to half width

            # Draw vertical lines for histogram density
            for i, (bin_center, height) in enumerate(zip(bin_edges[:-1], hist_norm)):
                if height > 0:
                    ax.plot(
                        [bin_center, bin_center],
                        [position - height, position + height],
                        color=color,
                        alpha=0.6,
                        linewidth=1,
                        zorder=3,
                    )

    # 3. Top layer: Smooth KDE density curve (cloud on top)
    if len(data) > 5:  # Need enough points for KDE
        try:
            kde = stats.gaussian_kde(data)
            x_dense = np.linspace(0, np.max(data), 100)
            density = kde(x_dense)
            density_norm = density / np.max(density) * width  # Scale to full width

            # Plot density curve above the timestep
            ax.plot(
                x_dense,
                position + density_norm,
                color=color,
                alpha=0.8,
                linewidth=2,
                zorder=4,
            )
            ax.fill_between(
                x_dense,
                position,
                position + density_norm,
                color=color,
                alpha=0.3,
                zorder=1,
            )
        except Exception:
            # Fallback if KDE fails
            pass

    return float(ess_ratio), ess_color
