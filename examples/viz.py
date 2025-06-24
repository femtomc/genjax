"""
GenJAX Research Visualization Standards (GRVS)

Shared styling utilities for consistent figure generation across all GenJAX case studies.
Based on the curvefit aesthetic system with publication-quality standards.

Usage:
    from examples.viz import setup_publication_fonts, FIGURE_SIZES, get_method_color
    
    setup_publication_fonts()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    ax.scatter(x, y, color=get_method_color("data_points"))
    apply_grid_style(ax)
    save_publication_figure(fig, "output.pdf")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
from typing import Dict, Tuple, Optional, Any

# =============================================================================
# FIGURE SIZES - Standardized dimensions for LaTeX integration
# =============================================================================

FIGURE_SIZES = {
    # Single-panel figures (4:3 aspect ratio for readability)
    "single_small": (4.33, 3.25),     # 1/3 textwidth, inline figures
    "single_medium": (6.5, 4.875),    # 1/2 textwidth, standard single
    "single_large": (8.66, 6.5),      # 2/3 textwidth, important results
    
    # Multi-panel layouts
    "two_panel_horizontal": (12, 5),   # Side-by-side panels
    "two_panel_vertical": (6.5, 8),    # Stacked panels
    "three_panel_horizontal": (18, 5), # Three panels in a row
    "four_panel_grid": (10, 8),        # 2x2 grid layout
    
    # Specialized layouts
    "framework_comparison": (12, 8),    # Main comparison figure
    "parameter_posterior": (15, 10),    # 3D parameter visualizations
    "inference_scaling": (18, 3.5),     # Three-panel scaling (wide, short)
    
    # Case study specific sizes
    "lidar_demo": (12, 10),            # LIDAR demonstration (localization)
    "trajectory_plot": (10, 8),        # Trajectory visualization (localization)
    "smc_comparison": (16, 12),        # 4-row SMC method comparison (localization)
}

# =============================================================================
# COLOR PALETTE - Colorblind-friendly, consistent across methods
# =============================================================================

PRIMARY_COLORS = {
    # Core method colors
    "genjax_is": "#0173B2",        # Medium blue (primary IS color)
    "genjax_hmc": "#DE8F05",       # Orange (HMC)
    "numpyro_hmc": "#029E73",      # Green (NumPyro)
    "pyro_hmc": "#029E73",         # Green (Pyro, same as NumPyro)
    "pyro_vi": "#CC3311",          # Red (Pyro VI)
    
    # Data and truth markers
    "data_points": "#CC3311",      # Red (data points, observations)
    "true_values": "#CC3311",      # Red (ground truth indicators)
    "curves": "#0173B2",           # Blue (polynomial curves)
    "background": "#333333",       # Dark gray (neutral elements)
    
    # State and particle colors
    "particles": "#0173B2",        # Blue (particle clouds)
    "robot_pose": "#CC3311",       # Red (robot position)
    "trajectory": "#DE8F05",       # Orange (trajectory paths)
    "lidar_rays": "#029E73",       # Green (LIDAR measurements)
}

# IS particle count variations (light to dark progression)
IS_VARIANT_COLORS = {
    "is_50": "#B19CD9",           # Light purple (N=50)
    "is_500": "#0173B2",          # Medium blue (N=500) 
    "is_5000": "#029E73",         # Dark green (N=5000)
}

# SMC method colors (for localization case study)
SMC_METHOD_COLORS = {
    "smc_basic": "#0173B2",       # Blue (bootstrap filter)
    "smc_mh": "#DE8F05",          # Orange (SMC + MH)
    "smc_hmc": "#029E73",         # Green (SMC + HMC)
    "smc_locally_optimal": "#CC3311",  # Red (locally optimal)
}

# =============================================================================
# TYPOGRAPHY STANDARDS
# =============================================================================

FONT_HIERARCHY = {
    "main_text": 18,              # Base text size
    "axis_labels": 18,            # X/Y axis labels (bold)
    "tick_labels": 16,            # Tick mark labels
    "legends": 16,                # Legend text
    "titles": 20,                 # Figure titles (when used)
    "density_values": 20,         # Log density annotations (bold)
    "timing_labels": 16,          # Timing value labels (bold)
}

FONT_WEIGHTS = {
    "axis_labels": "bold",        # Always bold for clarity
    "density_values": "bold",     # Emphasis for key values
    "timing_labels": "bold",      # Performance metrics emphasis
    "default": "normal",          # Everything else normal weight
}

# =============================================================================
# VISUAL ELEMENT SPECIFICATIONS
# =============================================================================

LINE_SPECS = {
    "curve_main": {"linewidth": 3, "alpha": 0.9},        # Main polynomial curves
    "curve_secondary": {"linewidth": 2.5, "alpha": 0.9}, # Secondary curves
    "curve_samples": {"linewidth": 1, "alpha": 0.1},     # Posterior samples (many)
    "curve_light": {"linewidth": 0.8, "alpha": 0.05},    # Very light samples
    "truth_lines": {"linewidth": 3, "alpha": 0.8},       # Ground truth lines
    "truth_vertical": {"linewidth": 4, "alpha": 0.9},    # Vertical truth markers
    "grid_lines": {"linewidth": 1.5, "alpha": 0.3},      # Grid lines
    "trajectory": {"linewidth": 3, "alpha": 0.9},        # Robot trajectories
    "lidar_rays": {"linewidth": 2.0, "alpha": 0.3},      # LIDAR ray lines
}

MARKER_SPECS = {
    "data_points": {
        "s": 120,                 # Size for main data points
        "zorder": 10,             # Layer order
        "edgecolor": "white", 
        "linewidth": 2,
        "alpha": 0.9
    },
    "truth_stars": {
        "s": 400,                 # Large size for ground truth
        "marker": "*",
        "edgecolor": "black",
        "linewidth": 3,
        "zorder": 100,
        "alpha": 0.9
    },
    "secondary_points": {
        "s": 100,                 # Smaller secondary points
        "zorder": 10,
        "edgecolor": "white",
        "linewidth": 2,
        "alpha": 0.8
    },
    "particles": {
        "s": 80,                  # Particle cloud points
        "alpha": 0.6,
        "edgecolor": "black",
        "linewidth": 1.0,
        "zorder": 5
    },
    "lidar_endpoints": {
        "s": 80,                  # LIDAR measurement endpoints
        "alpha": 0.8,
        "edgecolor": "black",
        "linewidth": 1.0,
        "zorder": 3
    }
}

# Transparency hierarchy
ALPHA_VALUES = {
    "main_elements": 0.9,         # Primary curves, markers
    "secondary_elements": 0.8,    # Secondary importance
    "many_samples": 0.1,          # When showing many posterior samples
    "very_light": 0.05,           # Background sample clouds
    "truth_markers": 0.6,         # Ground truth reference lines
    "grid": 0.3,                  # Subtle grid lines
    "bar_charts": 0.8,            # Bar chart transparency
    "particles": 0.6,             # Particle clouds
    "lidar_rays": 0.3,            # LIDAR ray lines
}

# =============================================================================
# LEGEND STYLING
# =============================================================================

LEGEND_SPECS = {
    "framealpha": 0.9,            # Semi-transparent background
    "fancybox": True,             # Rounded corners
    "shadow": False,              # No shadow for clean look
    "handlelength": 3,            # Length of legend lines
    "handletextpad": 1,           # Space between line and text
    "columnspacing": 2,           # Space between columns
    "facecolor": "white",         # White background
    "edgecolor": "black",         # Black border
    "linewidth": 1.5,             # Border thickness
}

LEGEND_POSITIONS = {
    "default": "best",            # Matplotlib auto-positioning
    "upper_left": "upper left",   # For outlier plots
    "upper_right": "upper right", # For metrics plots
    "center": "center",           # For standalone legend figures
}

# =============================================================================
# GRID AND TICK CONFIGURATION
# =============================================================================
#
# GRVS STANDARD: 3-TICK CONFIGURATION
# All figures should use exactly 3 tick marks per axis for optimal readability
# and clean appearance. This standard provides the best balance between:
# - Giving readers orientation without overwhelming the plot
# - Maintaining clean, uncluttered visual design
# - Ensuring consistency across all GenJAX research figures
# 
# Only deviate from 3 ticks when absolutely necessary for data clarity.

GRID_SETTINGS = {
    "enable": True,
    "alpha": 0.3,                 # Subtle grid lines
    "which": "major",             # Only major grid lines
    "axis": "both",               # Both x and y (or "x" only for bar charts)
}

TICK_SPECS = {
    "standard": {"x": 3, "y": 3},        # Standard 3-tick configuration (ENFORCED)
    "detailed": {"x": 4, "y": 4},        # More detailed for complex plots (use sparingly)
    "scaling": {"x": 5, "y": 5},         # Maximum detail for scaling plots (avoid if possible)
}

# =============================================================================
# SAVE CONFIGURATION
# =============================================================================

SAVE_SPECS = {
    "dpi": 300,                   # High resolution for publication
    "bbox_inches": "tight",       # Tight bounding box
    "format": "pdf",              # Vector format preferred
    "pad_inches": 0.05,           # Small padding for legends
}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def setup_publication_fonts():
    """Apply GenJAX Research Visualization Standards typography."""
    plt.rcParams.update({
        'font.size': FONT_HIERARCHY['main_text'],
        'axes.titlesize': FONT_HIERARCHY['titles'],
        'axes.labelsize': FONT_HIERARCHY['axis_labels'],
        'xtick.labelsize': FONT_HIERARCHY['tick_labels'],
        'ytick.labelsize': FONT_HIERARCHY['tick_labels'],
        'legend.fontsize': FONT_HIERARCHY['legends'],
        'axes.linewidth': 1.5,       # Thicker axes
        'xtick.major.width': 1.5,    # Thicker ticks
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,       # Longer ticks
        'ytick.major.size': 6,
        'font.weight': FONT_WEIGHTS['default'],
        'axes.labelweight': FONT_WEIGHTS['axis_labels'],
        'axes.titleweight': FONT_WEIGHTS['default'],
    })


def get_method_color(method_name: str) -> str:
    """Get standardized color for a given method or data type.
    
    Args:
        method_name: Name of method/data type (e.g., "genjax_is", "data_points")
        
    Returns:
        Hex color string
        
    Raises:
        KeyError: If method_name not found in color schemes
    """
    # Check primary colors first
    if method_name in PRIMARY_COLORS:
        return PRIMARY_COLORS[method_name]
    
    # Check IS variants
    if method_name in IS_VARIANT_COLORS:
        return IS_VARIANT_COLORS[method_name]
        
    # Check SMC methods
    if method_name in SMC_METHOD_COLORS:
        return SMC_METHOD_COLORS[method_name]
    
    # Default fallback
    available_colors = list(PRIMARY_COLORS.keys()) + list(IS_VARIANT_COLORS.keys()) + list(SMC_METHOD_COLORS.keys())
    raise KeyError(f"Unknown method '{method_name}'. Available: {available_colors}")


def apply_grid_style(ax, style: str = "default"):
    """Apply consistent grid styling to axes.
    
    Args:
        ax: Matplotlib axes object
        style: Grid style ("default", "x_only", "none")
    """
    if style == "none":
        ax.grid(False)
        return
    
    grid_axis = "both" if style == "default" else "x"
    ax.grid(
        GRID_SETTINGS["enable"],
        alpha=GRID_SETTINGS["alpha"],
        which=GRID_SETTINGS["which"],
        axis=grid_axis
    )


def set_minimal_ticks(ax, x_ticks: int = None, y_ticks: int = None):
    """Set minimal number of ticks for cleaner plots.
    
    GRVS Standard: Use 3 ticks per axis for optimal readability and clean appearance.
    Only deviate from this standard when absolutely necessary for data clarity.
    
    Args:
        ax: Matplotlib axes object
        x_ticks: Number of x-axis ticks (default: 3, following GRVS standard)
        y_ticks: Number of y-axis ticks (default: 3, following GRVS standard)
    """
    # Use GRVS standard if not specified
    if x_ticks is None:
        x_ticks = TICK_SPECS["standard"]["x"]
    if y_ticks is None:
        y_ticks = TICK_SPECS["standard"]["y"]
        
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, prune='both'))


def apply_standard_ticks(ax):
    """Apply GRVS standard 3-tick configuration.
    
    This function enforces the GenJAX Research Visualization Standards
    of using exactly 3 ticks per axis for optimal readability.
    
    Args:
        ax: Matplotlib axes object
    """
    set_minimal_ticks(ax)  # Uses GRVS standard (3, 3) by default


def apply_legend_style(ax, location: str = "default", **kwargs):
    """Apply consistent legend styling.
    
    Args:
        ax: Matplotlib axes object
        location: Legend location ("default", "upper_left", etc.)
        **kwargs: Additional legend parameters
    """
    legend_kwargs = LEGEND_SPECS.copy()
    legend_kwargs.update(kwargs)
    
    loc = LEGEND_POSITIONS.get(location, location)
    ax.legend(loc=loc, **legend_kwargs)


def save_publication_figure(fig, filename: str, **kwargs):
    """Save figure with publication-quality settings and cleanup.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename (should end with .pdf)
        **kwargs: Additional save parameters
    """
    save_kwargs = SAVE_SPECS.copy()
    save_kwargs.update(kwargs)
    
    # Apply tight layout before saving
    fig.tight_layout()
    
    # Save with publication settings
    fig.savefig(filename, **save_kwargs)
    
    # Clean up to prevent memory leaks
    plt.close(fig)


def create_method_legend(methods: list, filename: Optional[str] = None) -> plt.Figure:
    """Create standalone legend figure for method comparison.
    
    Args:
        methods: List of method names to include
        filename: Optional output filename
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_small"])
    
    # Create legend entries
    legend_elements = []
    for method in methods:
        color = get_method_color(method)
        # Clean up method name for display
        display_name = method.replace("_", " ").title()
        legend_elements.append(
            mpatches.Rectangle((0, 0), 3, 2, facecolor=color, label=display_name)
        )
    
    # Create legend
    ax.legend(handles=legend_elements, loc="center", **LEGEND_SPECS)
    ax.axis("off")  # Hide axes for standalone legend
    
    if filename:
        save_publication_figure(fig, filename)
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS
# =============================================================================

def setup_comparison_plot(figsize_key: str = "framework_comparison") -> Tuple[plt.Figure, plt.Axes]:
    """Setup figure for method comparison with GRVS standards.
    
    Args:
        figsize_key: Key from FIGURE_SIZES dictionary
        
    Returns:
        Tuple of (figure, axes) objects
    """
    setup_publication_fonts()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES[figsize_key])
    apply_grid_style(ax)
    return fig, ax


def plot_data_points(ax, x, y, label: str = "Data", **kwargs):
    """Plot data points with GRVS standards.
    
    Args:
        ax: Matplotlib axes object
        x, y: Data coordinates
        label: Legend label
        **kwargs: Additional scatter parameters
    """
    scatter_kwargs = MARKER_SPECS["data_points"].copy()
    scatter_kwargs.update(kwargs)
    scatter_kwargs["color"] = get_method_color("data_points")
    
    ax.scatter(x, y, label=label, **scatter_kwargs)


def plot_method_curve(ax, x, y, method: str, label: Optional[str] = None, **kwargs):
    """Plot method curve with GRVS standards.
    
    Args:
        ax: Matplotlib axes object
        x, y: Curve coordinates
        method: Method name for color lookup
        label: Legend label (defaults to method name)
        **kwargs: Additional plot parameters
    """
    line_kwargs = LINE_SPECS["curve_main"].copy()
    line_kwargs.update(kwargs)
    line_kwargs["color"] = get_method_color(method)
    
    if label is None:
        label = method.replace("_", " ").title()
    
    ax.plot(x, y, label=label, **line_kwargs)


def finalize_comparison_plot(ax, xlabel: str, ylabel: str, 
                           title: Optional[str] = None, 
                           legend: bool = True):
    """Finalize comparison plot with GRVS standards.
    
    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Optional title (generally not recommended)
        legend: Whether to add legend
    """
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    if title is not None:
        ax.set_title(title)  # Note: GRVS recommends no titles
    
    if legend:
        apply_legend_style(ax)
    
    set_minimal_ticks(ax)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_grvs_compliance(fig) -> Dict[str, bool]:
    """Check if figure follows GRVS standards.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Dictionary of compliance checks
    """
    checks = {}
    
    # Check font sizes
    checks["font_size_18"] = plt.rcParams['font.size'] == 18
    checks["label_size_18"] = plt.rcParams['axes.labelsize'] == 18
    checks["legend_size_16"] = plt.rcParams['legend.fontsize'] == 16
    checks["bold_labels"] = plt.rcParams['axes.labelweight'] == 'bold'
    
    # Check figure size against standards
    size = fig.get_size_inches()
    checks["standard_size"] = tuple(size) in FIGURE_SIZES.values()
    
    return checks


def get_grvs_summary() -> str:
    """Return summary of GRVS standards."""
    return """
GenJAX Research Visualization Standards (GRVS) Summary:

Typography:
- Base font: 18pt
- Axis labels: 18pt bold
- Tick labels: 16pt
- Legends: 16pt

Visual Elements:
- Main curves: 3px linewidth
- Data points: 120px markers with white edges
- Grid: 30% alpha, major lines only
- Colors: Colorblind-friendly palette

Output:
- Format: PDF (vector)
- Resolution: 300 DPI
- Layout: Tight bounding box

Usage:
    from examples.viz import setup_publication_fonts, FIGURE_SIZES
    setup_publication_fonts()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
"""


if __name__ == "__main__":
    # Demo of GRVS standards
    print(get_grvs_summary())
    
    # Create example figure
    setup_publication_fonts()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])
    
    # Example data
    x = np.linspace(0, 1, 100)
    y1 = 0.5 + 0.3 * np.sin(10 * x) + 0.1 * np.random.randn(100)
    y2 = 0.5 + 0.2 * np.cos(8 * x)
    
    # Plot with GRVS standards
    plot_data_points(ax, x[::10], y1[::10], "Observations")
    plot_method_curve(ax, x, y2, "genjax_is", "GenJAX IS")
    
    finalize_comparison_plot(ax, "X Variable", "Y Variable")
    
    # Save demo figure
    save_publication_figure(fig, "grvs_demo.pdf")
    print("Demo figure saved as grvs_demo.pdf")