# Dynamic DSL implementation - this is the original "unoptimized" version

# Use the existing dynamic implementation but return a different label
function run_polynomial_is_benchmark_dynamic(
    data::PolynomialData,
    n_particles::Int;
    repeats::Int = 100
)
    # Run the regular benchmark
    result = run_polynomial_is_benchmark(data, n_particles; repeats=repeats)
    
    # Change the framework name to distinguish it
    result["framework"] = "genjl_dynamic"
    result["model_type"] = "dynamic"
    
    return result
end

export run_polynomial_is_benchmark_dynamic