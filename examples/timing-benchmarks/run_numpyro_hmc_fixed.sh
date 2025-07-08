#!/bin/bash
# Re-run NumPyro HMC with fixed leapfrog steps

echo "Re-running NumPyro HMC with fixed leapfrog steps..."
echo "Parameters: step_size=0.01, n_leapfrog=20 (no adaptation)"

# Run NumPyro HMC benchmarks with cuda environment
pixi run -e cuda python run_hmc_benchmarks.py \
    --frameworks numpyro \
    --chain-lengths 100 500 1000 5000 \
    --n-points 50 \
    --repeats 100 \
    --n-warmup 500 \
    --step-size 0.01 \
    --n-leapfrog 20 \
    --device cuda

echo "Regenerating figures..."
pixi run python combine_results.py \
    --frameworks genjax numpyro handcoded_tfp handcoded_torch \
    --data-dir data \
    --output-dir figs

echo "Done! Check figs/benchmark_timings_hmc_all_frameworks.pdf"