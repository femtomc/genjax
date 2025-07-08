"""Local timing utilities to avoid cross-environment dependencies."""

import time
import numpy as np
from typing import Callable, Tuple, Any


def timing(
    fn: Callable[[], Any],
    repeats: int = 20,
    inner_repeats: int = 20,
    auto_sync: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Benchmark function execution time with multiple runs.

    This function provides consistent timing methodology across all benchmarks.
    It uses a double-nested loop structure where the inner loop finds the minimum time
    (to reduce noise) and the outer loop provides statistical samples.

    Args:
        fn: Function to benchmark (should be a no-argument callable)
        repeats: Number of outer timing runs for statistical aggregation
        inner_repeats: Number of inner timing runs per outer run (minimum is taken)
        auto_sync: Whether to automatically synchronize (not used for PyTorch)

    Returns:
        Tuple of:
        - times: Array of minimum times from each outer run
        - (mean_time, std_time): Statistical summary of timing results
    """
    times = []
    for i in range(repeats):
        possible = []
        for j in range(inner_repeats):
            start_time = time.perf_counter()
            result = fn()
            interval = time.perf_counter() - start_time
            possible.append(interval)
        times.append(np.array(possible).min())

    times = np.array(times)
    return times, (float(np.mean(times)), float(np.std(times)))


def benchmark_with_warmup(
    fn: Callable[[], Any],
    warmup_runs: int = 2,
    repeats: int = 10,
    inner_repeats: int = 10,
    auto_sync: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Benchmark function with automatic warm-up runs.

    Convenience function that handles the common pattern of running warm-up
    iterations before timing.

    Args:
        fn: Function to benchmark
        warmup_runs: Number of warm-up runs before timing
        repeats: Number of outer timing runs
        inner_repeats: Number of inner timing runs per outer run
        auto_sync: Whether to automatically synchronize (not used for PyTorch)

    Returns:
        Same as timing(): (times_array, (mean_time, std_time))
    """
    # Warm-up runs
    for _ in range(warmup_runs):
        _ = fn()

    # Actual timing
    return timing(fn, repeats=repeats, inner_repeats=inner_repeats, auto_sync=auto_sync)