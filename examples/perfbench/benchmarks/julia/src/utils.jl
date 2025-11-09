# Utilities for data loading and result export

# Load polynomial data from CSV
function load_polynomial_data(csv_path::String)::PolynomialData
    df = CSV.read(csv_path, DataFrame)
    
    # Extract metadata from first row (stored as comments in Python export)
    # For now, use defaults - can enhance later
    true_coeffs = Dict("a" => 1.0, "b" => -2.0, "c" => 3.0)
    noise_std = 0.05
    
    xs = df.x
    ys = df.y
    n_points = length(xs)
    
    return PolynomialData(xs, ys, true_coeffs, noise_std, n_points)
end

# Export timing results to CSV
function export_timing_results(results::Dict, output_path::String)
    # Create DataFrame from timing results
    df = DataFrame(
        run = 1:length(results["times"]),
        time_seconds = results["times"]
    )
    
    # Add summary statistics as metadata
    metadata = DataFrame(
        metric = ["mean_time", "std_time", "n_runs"],
        value = [results["mean_time"], results["std_time"], length(results["times"])]
    )
    
    # Write main timing data
    CSV.write(output_path, df)
    
    # Write metadata to companion file
    metadata_path = replace(output_path, ".csv" => "_metadata.csv")
    CSV.write(metadata_path, metadata)
end

# Wrapper for running benchmarks from Python
function run_benchmark_from_json(config_json::String)
    config = JSON.parse(config_json)
    
    # Load data
    data = load_polynomial_data(config["data_path"])
    
    # Run appropriate benchmark
    if config["method"] == "is"
        results = run_polynomial_is_benchmark(
            data,
            config["n_particles"];
            repeats = get(config, "repeats", 100)
        )
    elseif config["method"] == "hmc"
        results = run_polynomial_hmc_benchmark(
            data,
            config["n_samples"];
            n_warmup = get(config, "n_warmup", 500),
            repeats = get(config, "repeats", 100)
        )
    else
        error("Unknown method: $(config["method"])")
    end
    
    # Export results
    export_timing_results(results, config["output_path"])
    
    return results
end