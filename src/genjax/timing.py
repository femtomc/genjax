"""Shared utilities for GenJAX examples and case studies."""

import time
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any, Optional


def timing(
    fn: Callable[[], Any],
    repeats: int = 20,
    inner_repeats: int = 20,
    auto_sync: bool = True,
) -> Tuple[jnp.ndarray, Tuple[float, float]]:
    """Benchmark function execution time with multiple runs.

    This function provides consistent timing methodology across all GenJAX case studies.
    It uses a double-nested loop structure where the inner loop finds the minimum time
    (to reduce noise) and the outer loop provides statistical samples.

    Args:
        fn: Function to benchmark (should be a no-argument callable)
        repeats: Number of outer timing runs for statistical aggregation
        inner_repeats: Number of inner timing runs per outer run (minimum is taken)
        auto_sync: Whether to automatically call jax.block_until_ready() on function results

    Returns:
        Tuple of:
        - times: Array of minimum times from each outer run
        - (mean_time, std_time): Statistical summary of timing results

    Usage:
        # Basic usage
        times, (mean, std) = timing(lambda: my_function(args))

        # With custom parameters
        times, (mean, std) = timing(
            lambda: jitted_fn(data).block_until_ready(),
            repeats=100,
            inner_repeats=50,
            auto_sync=False  # disable auto-sync if manually handling
        )

        # Typical pattern for JAX functions with JIT warm-up
        jitted_fn = jax.jit(my_function)
        _ = jitted_fn(data)  # warm-up
        _ = jitted_fn(data)  # warm-up
        times, (mean, std) = timing(lambda: jitted_fn(data), repeats=200)
    """
    times = []
    for i in range(repeats):
        possible = []
        for j in range(inner_repeats):
            start_time = time.perf_counter()
            if auto_sync:
                # Automatically synchronize JAX computations
                result = fn()
                jax.block_until_ready(result)
            else:
                result = fn()
            interval = time.perf_counter() - start_time
            possible.append(interval)
        times.append(jnp.array(possible).min())

    times = jnp.array(times)
    return times, (float(jnp.mean(times)), float(jnp.std(times)))


def benchmark_with_warmup(
    fn: Callable[[], Any],
    warmup_runs: int = 2,
    repeats: int = 10,
    inner_repeats: int = 10,
    auto_sync: bool = True,
) -> Tuple[jnp.ndarray, Tuple[float, float]]:
    """Benchmark function with automatic JIT warm-up runs.

    Convenience function that handles the common pattern of running warm-up
    iterations before timing. This is essential for JAX functions that use JIT
    compilation.

    Args:
        fn: Function to benchmark
        warmup_runs: Number of warm-up runs before timing
        repeats: Number of outer timing runs
        inner_repeats: Number of inner timing runs per outer run
        auto_sync: Whether to automatically call jax.block_until_ready()

    Returns:
        Same as timing(): (times_array, (mean_time, std_time))

    Usage:
        # Automatically handles warm-up for JIT compiled functions
        jitted_fn = jax.jit(my_function)
        times, (mean, std) = benchmark_with_warmup(lambda: jitted_fn(data))
    """
    # Warm-up runs to trigger JIT compilation
    for _ in range(warmup_runs):
        _ = fn()

    # Actual timing
    return timing(fn, repeats=repeats, inner_repeats=inner_repeats, auto_sync=auto_sync)


def compare_timings(*timing_results, labels: Optional[list] = None) -> None:
    """Print a formatted comparison of timing results.

    Args:
        *timing_results: Multiple timing results from timing() or benchmark_with_warmup()
        labels: Optional labels for each timing result

    Usage:
        genjax_times, genjax_stats = timing(lambda: genjax_fn())
        numpyro_times, numpyro_stats = timing(lambda: numpyro_fn())
        compare_timings(
            (genjax_times, genjax_stats),
            (numpyro_times, numpyro_stats),
            labels=["GenJAX", "NumPyro"]
        )
    """
    if labels is None:
        labels = [f"Method {i + 1}" for i in range(len(timing_results))]

    print(f"{'Method':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Relative':<12}")
    print("-" * 55)

    # Find baseline (first method) for relative comparison
    baseline_mean = timing_results[0][1][0]

    for i, ((times, (mean, std)), label) in enumerate(zip(timing_results, labels)):
        mean_ms = mean * 1000
        std_ms = std * 1000
        relative = (mean / baseline_mean) * 100
        print(f"{label:<15} {mean_ms:<12.3f} {std_ms:<12.3f} {relative:<12.1f}%")
