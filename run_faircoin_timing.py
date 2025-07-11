#!/usr/bin/env python
"""Run faircoin timing with 100 repeats and 50 inner repeats."""

from examples.faircoin.figs import combined_comparison_fig

# Run with specific timing parameters
combined_comparison_fig(
    num_obs=50,
    num_samples=2000,        # 2000 samples for posterior plots
    timing_repeats=100,      # 100 outer repeats as requested
    timing_samples=2000,     # 2000 samples for timing
    inner_repeats=50,        # 50 inner repeats as requested
)