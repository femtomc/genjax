# Improved polynomial regression benchmark with better warm-up

using Gen
using Statistics

include("polynomial_regression.jl")

function run_polynomial_is_benchmark_improved(
    data::PolynomialData,
    n_particles::Int;
    repeats::Int = 100,
    warmup_repeats::Int = 10
)
    xs = data.xs
    ys = data.ys
    
    # Create observations
    observations = choicemap()
    for i in 1:length(ys)
        observations[:y => i] = ys[i]
    end
    
    # Multiple warm-up runs to ensure JIT compilation
    println("Running $(warmup_repeats) warm-up iterations...")
    for i in 1:warmup_repeats
        traces, log_weights, _ = importance_sampling(
            polynomial_model, (xs,), observations, n_particles
        )
    end
    
    # Force garbage collection before timing
    GC.gc()
    
    # Timing runs
    println("Running $(repeats) timing iterations...")
    times = Float64[]
    for i in 1:repeats
        # Force GC every 10 runs to avoid GC during timing
        if i % 10 == 0
            GC.gc()
        end
        
        start_time = time_ns()
        traces, log_weights, _ = importance_sampling(
            polynomial_model, (xs,), observations, n_particles
        )
        elapsed = (time_ns() - start_time) / 1e9  # Convert to seconds
        push!(times, elapsed)
    end
    
    # Remove outliers (top and bottom 10%)
    sorted_times = sort(times)
    trim_count = Int(floor(length(times) * 0.1))
    if trim_count > 0
        trimmed_times = sorted_times[(trim_count+1):(end-trim_count)]
    else
        trimmed_times = sorted_times
    end
    
    # Extract final samples for validation
    samples_a = [traces[i][:a] for i in 1:n_particles]
    samples_b = [traces[i][:b] for i in 1:n_particles]
    samples_c = [traces[i][:c] for i in 1:n_particles]
    
    return Dict(
        "framework" => "gen.jl",
        "method" => "is",
        "n_particles" => n_particles,
        "times" => times,
        "mean_time" => mean(trimmed_times),
        "std_time" => std(trimmed_times),
        "trimmed_mean" => mean(trimmed_times),
        "trimmed_std" => std(trimmed_times),
        "raw_mean" => mean(times),
        "raw_std" => std(times),
        "samples" => Dict(
            "a" => samples_a,
            "b" => samples_b,
            "c" => samples_c
        ),
        "log_weights" => log_weights
    )
end