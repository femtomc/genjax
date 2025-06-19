"""Game of Life data generation for gen2d case study."""

import jax
import jax.numpy as jnp
from typing import Tuple


def game_of_life_step(grid: jnp.ndarray) -> jnp.ndarray:
    """
    Single step of Conway's Game of Life.

    Args:
        grid: (H, W) boolean array

    Returns:
        Updated grid after one step
    """
    # Count neighbors using convolution-like operation
    # Pad with periodic boundary conditions (toroidal topology)
    padded = jnp.pad(grid, 1, mode="wrap")

    # Sum neighbors (convert to int to count properly)
    neighbors = (
        padded[:-2, :-2].astype(int)
        + padded[:-2, 1:-1].astype(int)
        + padded[:-2, 2:].astype(int)
        + padded[1:-1, :-2].astype(int)
        + padded[1:-1, 2:].astype(int)
        + padded[2:, :-2].astype(int)
        + padded[2:, 1:-1].astype(int)
        + padded[2:, 2:].astype(int)
    )

    # Apply Game of Life rules
    # Live cells with 2-3 neighbors survive
    # Dead cells with exactly 3 neighbors become alive
    survive = grid & ((neighbors == 2) | (neighbors == 3))
    birth = (~grid) & (neighbors == 3)

    return survive | birth


def create_glider(grid_size: int = 64) -> jnp.ndarray:
    """Create a grid with a glider pattern."""
    grid = jnp.zeros((grid_size, grid_size), dtype=bool)

    # Classic glider pattern
    glider = jnp.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool)

    # Place glider
    y, x = 10, 10
    grid = grid.at[y : y + 3, x : x + 3].set(glider)

    return grid


def create_oscillators(grid_size: int = 64) -> jnp.ndarray:
    """Create a grid with multiple oscillator patterns."""
    grid = jnp.zeros((grid_size, grid_size), dtype=bool)

    # Blinker (period 2)
    blinker = jnp.array([[1, 1, 1]], dtype=bool)
    grid = grid.at[10:11, 10:13].set(blinker)

    # Toad (period 2)
    toad = jnp.array([[0, 1, 1, 1], [1, 1, 1, 0]], dtype=bool)
    grid = grid.at[20:22, 20:24].set(toad)

    # Beacon (period 2) - only if grid is large enough
    if grid_size >= 40:
        beacon = jnp.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool
        )
        grid = grid.at[30:34, 30:34].set(beacon)

    # Pulsar (period 3) - simplified version
    pulsar_quarter = jnp.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    # Place pulsar (symmetric pattern) - only if grid is large enough
    if grid_size >= 60:
        center_y, center_x = 45, 45
        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            y_flip = 1 if dy else -1
            x_flip = 1 if dx else -1
            quarter = pulsar_quarter[::y_flip, ::x_flip]
            y_start = center_y + dy * 6
            x_start = center_x + dx * 6
            grid = grid.at[y_start : y_start + 5, x_start : x_start + 5].set(quarter)

    return grid


def parse_rle_pattern(rle_string: str) -> jnp.ndarray:
    """
    Parse RLE (Run Length Encoded) pattern format used in Conway's Game of Life.

    Args:
        rle_string: RLE pattern string (e.g., "2o$obo$b2o")

    Returns:
        Boolean array representing the pattern
    """
    # Remove end-of-pattern marker
    rle_string = rle_string.replace("!", "")

    # Split into lines
    lines = rle_string.split("$")
    pattern_rows = []

    for line in lines:
        row = []
        i = 0
        while i < len(line):
            # Parse number (run length)
            num_str = ""
            while i < len(line) and line[i].isdigit():
                num_str += line[i]
                i += 1
            count = int(num_str) if num_str else 1

            # Parse character
            if i < len(line):
                char = line[i]
                if char == "o":
                    row.extend([True] * count)
                elif char == "b":
                    row.extend([False] * count)
                i += 1
            else:
                # If we have a number but no character, it's trailing dead cells
                if num_str:
                    row.extend([False] * count)

        pattern_rows.append(row)

    # Remove empty rows at the end
    while pattern_rows and not any(pattern_rows[-1]):
        pattern_rows.pop()

    # Pad all rows to same length
    max_width = max(len(row) for row in pattern_rows) if pattern_rows else 0
    for row in pattern_rows:
        while len(row) < max_width:
            row.append(False)

    return jnp.array(pattern_rows, dtype=bool)


def create_achims_p4(grid_size: int = 64) -> jnp.ndarray:
    """
    Create Achim's p4 (cloverleaf) oscillator pattern.

    This is a period-4 oscillator discovered by Achim Flammenkamp in 1988.
    It's sometimes called "cloverleaf" or "dual 1-2-3-4".

    Pattern from copy.sh/life - exact RLE format.
    """
    # Exact RLE pattern from copy.sh/life
    rle_pattern = (
        "b2o24b2o$b2o23bo2bo$9b2o16bobo$8bo2bo16bo$9b2o14bo$24bobo$27bo$14bo11b"
        "o$12bob3o7b2o$12bo4bo$4b2o7b3obo$3bo11bo$2bo$3bobo$4bo14b2o$bo16bo2bo$"
        "obo16b2o$o2bo23b2o$b2o24b2o"
    )

    return create_pattern_from_rle(rle_pattern, grid_size)


def create_pattern_from_rle(rle_string: str, grid_size: int = 64) -> jnp.ndarray:
    """
    Create a pattern from RLE (Run Length Encoded) format.

    Args:
        rle_string: RLE pattern string
        grid_size: Size of the output grid

    Returns:
        Grid with the pattern placed in the center
    """
    pattern = parse_rle_pattern(rle_string)
    grid = jnp.zeros((grid_size, grid_size), dtype=bool)

    # Center the pattern
    center_y = grid_size // 2 - pattern.shape[0] // 2
    center_x = grid_size // 2 - pattern.shape[1] // 2

    end_y = center_y + pattern.shape[0]
    end_x = center_x + pattern.shape[1]

    if end_y <= grid_size and end_x <= grid_size:
        grid = grid.at[center_y:end_y, center_x:end_x].set(pattern)

    return grid


def create_random_soup(
    grid_size: int = 64,
    density: float = 0.3,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42),
) -> jnp.ndarray:
    """Create a random initial configuration."""
    return jax.random.bernoulli(key, p=density, shape=(grid_size, grid_size))


def simulate_game_of_life(initial_grid: jnp.ndarray, n_steps: int = 20) -> jnp.ndarray:
    """
    Simulate Game of Life for multiple steps.

    Args:
        initial_grid: Initial configuration
        n_steps: Number of steps to simulate

    Returns:
        Array of shape (n_steps + 1, H, W) including initial state
    """

    def scan_fn(grid, _):
        next_grid = game_of_life_step(grid)
        return next_grid, next_grid

    _, grids = jax.lax.scan(scan_fn, initial_grid, jnp.arange(n_steps))

    # Prepend initial grid
    all_grids = jnp.concatenate([initial_grid[None, ...], grids], axis=0)

    return all_grids


def extract_active_pixels(
    grids: jnp.ndarray, max_pixels: int = 1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract coordinates of active (True) pixels from Game of Life grids.

    Args:
        grids: (T, H, W) boolean arrays
        max_pixels: Maximum number of pixels to track per frame

    Returns:
        observations: (T, max_pixels, 2) padded pixel coordinates
        counts: (T,) number of active pixels per frame
    """
    T, H, W = grids.shape
    observations = jnp.zeros((T, max_pixels, 2))
    counts = jnp.zeros(T, dtype=jnp.int32)

    for t in range(T):
        # Get active pixel coordinates
        y_coords, x_coords = jnp.where(grids[t])
        coords = jnp.stack([x_coords, y_coords], axis=-1).astype(jnp.float32)

        # Limit to max_pixels
        n_active = min(len(coords), max_pixels)
        counts = counts.at[t].set(n_active)

        if n_active > 0:
            observations = observations.at[t, :n_active].set(coords[:n_active])

    return observations, counts


def generate_tracking_data(
    pattern: str = "oscillators",
    grid_size: int = 64,
    n_steps: int = 20,
    max_pixels: int = 500,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate Game of Life data for tracking.

    Args:
        pattern: Type of initial pattern ("glider", "oscillators", "random")
        grid_size: Size of the grid
        n_steps: Number of simulation steps
        max_pixels: Maximum pixels to track per frame
        key: Random key for random patterns

    Returns:
        grids: (T, H, W) Game of Life states
        observations: (T, max_pixels, 2) active pixel coordinates
        counts: (T,) number of active pixels per frame
    """
    # Create initial configuration
    if pattern == "glider":
        initial_grid = create_glider(grid_size)
    elif pattern == "oscillators":
        initial_grid = create_oscillators(grid_size)
    elif pattern == "achims_p4":
        initial_grid = create_achims_p4(grid_size)
    elif pattern == "random":
        initial_grid = create_random_soup(grid_size, 0.2, key)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Simulate
    grids = simulate_game_of_life(initial_grid, n_steps)

    # Extract observations
    observations, counts = extract_active_pixels(grids, max_pixels)

    return grids, observations, counts
