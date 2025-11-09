module TimingBenchmarks

using Gen
using CSV
using DataFrames
using JSON
using Random
using Statistics
using LinearAlgebra
using Distributions

# Include benchmark implementations
include("polynomial_regression.jl")
include("polynomial_regression_dynamic.jl")
include("utils.jl")

# Export main benchmark functions
export run_polynomial_is_benchmark
export run_polynomial_hmc_benchmark
export run_polynomial_is_benchmark_dynamic
export PolynomialData
export load_polynomial_data

end # module