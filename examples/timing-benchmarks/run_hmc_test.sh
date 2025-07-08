#!/bin/bash
# Quick test run of HMC benchmarks with fewer repeats

echo "====================================="
echo "Quick HMC Benchmark Test (5 repeats)"
echo "====================================="

cd "$(dirname "$0")"

# Test with fewer repeats and only 2 chain lengths
pixi run python run_hmc_benchmarks.py \
    --frameworks genjax numpyro handcoded_tfp genjl \
    --chain-lengths 100 1000 \
    --n-points 50 \
    --repeats 5 \
    --device cuda

echo ""
echo "Generating test figure..."
pixi run python combine_results.py \
    --frameworks genjax numpyro handcoded_tfp genjl \
    --data-dir data \
    --output-dir figs

echo ""
echo "Test complete! Check figs/benchmark_timings_hmc_all_frameworks.pdf"