# Polynomial regression benchmark implementations in Gen.jl

struct PolynomialData
    xs::Vector{Float64}
    ys::Vector{Float64}
    true_coeffs::Dict{String, Float64}
    noise_std::Float64
    n_points::Int
end

# Polynomial function (generic to handle AD types)
function polyfn(x, a, b, c)
    return a + b * x + c * x^2
end

# Gen.jl model for polynomial regression
@gen function polynomial_model(xs::Vector{Float64})
    # Prior on polynomial coefficients
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0) 
    c ~ normal(0.0, 1.0)
    
    # Likelihood
    for i in 1:length(xs)
        y_det = polyfn(xs[i], a, b, c)
        {:y => i} ~ normal(y_det, 0.05)
    end
    
    return (a, b, c)
end

# Importance sampling benchmark
function run_polynomial_is_benchmark(
    data::PolynomialData,
    n_particles::Int;
    repeats::Int = 100
)
    xs = data.xs
    ys = data.ys
    
    # Create observations
    observations = choicemap()
    for i in 1:length(ys)
        observations[:y => i] = ys[i]
    end
    
    # Warm-up run
    traces, log_weights, _ = importance_sampling(
        polynomial_model, (xs,), observations, n_particles
    )
    
    # Additional warm-up runs to ensure JIT compilation
    for _ in 1:5
        importance_sampling(polynomial_model, (xs,), observations, n_particles)
    end
    
    # Timing runs
    times = Float64[]
    for _ in 1:repeats
        start_time = time()
        traces, log_weights, _ = importance_sampling(
            polynomial_model, (xs,), observations, n_particles
        )
        push!(times, time() - start_time)
    end
    
    # Extract final samples for validation
    samples_a = [traces[i][:a] for i in 1:n_particles]
    samples_b = [traces[i][:b] for i in 1:n_particles]
    samples_c = [traces[i][:c] for i in 1:n_particles]
    
    return Dict(
        "framework" => "genjl",
        "method" => "is",
        "n_particles" => n_particles,
        "times" => times,
        "mean_time" => mean(times),
        "std_time" => std(times),
        "samples" => Dict(
            "a" => samples_a,
            "b" => samples_b,
            "c" => samples_c
        ),
        "log_weights" => log_weights
    )
end

# HMC benchmark
function run_polynomial_hmc_benchmark(
    data::PolynomialData,
    n_samples::Int;
    n_warmup::Int = 500,
    repeats::Int = 100,
    step_size::Float64 = 0.01,
    n_leapfrog::Int = 20
)
    xs = data.xs
    ys = data.ys
    
    # Create observations
    observations = choicemap()
    for i in 1:length(ys)
        observations[:y => i] = ys[i]
    end
    
    # Selection for continuous parameters
    selection = Gen.select(:a, :b, :c)
    
    # Warm-up run
    trace, _ = Gen.generate(polynomial_model, (xs,), observations)
    
    # Full warm-up chain
    for _ in 1:n_warmup
        trace, _ = Gen.hmc(trace, selection; L=n_leapfrog, eps=step_size)
    end
    
    # Timing runs
    times = Float64[]
    for _ in 1:repeats
        # Reset trace
        trace, _ = Gen.generate(polynomial_model, (xs,), observations)
        
        # Time the full chain
        start_time = time()
        for _ in 1:(n_warmup + n_samples)
            trace, _ = Gen.hmc(trace, selection; L=n_leapfrog, eps=step_size)
        end
        push!(times, time() - start_time)
    end
    
    # Get one final chain for samples
    trace, _ = Gen.generate(polynomial_model, (xs,), observations)
    samples_a = Float64[]
    samples_b = Float64[]
    samples_c = Float64[]
    
    # Burn-in
    for _ in 1:n_warmup
        trace, _ = Gen.hmc(trace, selection; L=n_leapfrog, eps=step_size)
    end
    
    # Collect samples
    for _ in 1:n_samples
        trace, _ = Gen.hmc(trace, selection; L=n_leapfrog, eps=step_size)
        push!(samples_a, trace[:a])
        push!(samples_b, trace[:b])
        push!(samples_c, trace[:c])
    end
    
    return Dict(
        "framework" => "genjl",
        "method" => "hmc",
        "times" => times,
        "mean_time" => mean(times),
        "std_time" => std(times),
        "samples" => Dict(
            "a" => samples_a,
            "b" => samples_b,
            "c" => samples_c
        ),
        "n_samples" => n_samples,
        "n_warmup" => n_warmup
    )
end