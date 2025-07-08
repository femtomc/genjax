#!/bin/bash
# Test HMC benchmarks including Gen.jl

echo "Running HMC benchmarks including Gen.jl..."
echo "======================================="

# Small test run with all frameworks including Gen.jl
python run_hmc_benchmarks.py \
    --frameworks genjax numpyro genjl \
    --chain-lengths 100 500 \
    --n-points 30 \
    --repeats 5 \
    --n-warmup 100 \
    --step-size 0.01 \
    --n-leapfrog 20

echo ""
echo "Test complete! Check the data/ directory for results."