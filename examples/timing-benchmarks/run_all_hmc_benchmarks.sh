#!/bin/bash
# Run all HMC benchmarks including Gen.jl

echo "====================================="
echo "Running ALL HMC Benchmarks on GPU"
echo "====================================="

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Set up Julia environment if needed
echo "Setting up Julia environment..."
cd julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
cd ..

# Run the benchmarks
echo ""
echo "Starting HMC benchmarks..."
echo "Frameworks: GenJAX, NumPyro, Handcoded JAX, Handcoded PyTorch, Pyro, Gen.jl"
echo "Chain lengths: 100, 500, 1000, 5000"
echo ""

# Run with pixi cuda environment for GPU support
pixi run -e cuda python run_hmc_benchmarks.py \
    --frameworks genjax numpyro handcoded_tfp handcoded_torch pyro genjl \
    --chain-lengths 100 500 1000 5000 \
    --n-points 50 \
    --repeats 100 \
    --device cuda

echo ""
echo "====================================="
echo "Generating HMC comparison figure..."
echo "====================================="

# Generate the figure (can use default environment for plotting)
pixi run python combine_results.py \
    --frameworks genjax numpyro handcoded_tfp handcoded_torch pyro genjl \
    --data-dir data \
    --output-dir figs

echo ""
echo "====================================="
echo "HMC benchmarking complete!"
echo "====================================="
echo ""
echo "Results saved to:"
echo "  - Individual results: data/{framework}/hmc_n*.json"
echo "  - Summary CSV: figs/benchmark_summary_*.csv"
echo "  - HMC comparison plot: figs/benchmark_timings_hmc_all_frameworks.pdf"
echo "  - LaTeX table: figs/benchmark_table.tex"