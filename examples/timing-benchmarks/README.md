# Timing Benchmarks

Performance benchmarks comparing GenJAX with other probabilistic programming frameworks.

## Installation

```bash
# Install dependencies using pixi
pixi install

# Install package in development mode
pixi run install-dev
```

## Project Structure

```
timing-benchmarks/
├── pyproject.toml          # Package configuration
├── julia/                  # Gen.jl benchmark implementations
├── data/                   # Benchmark results storage
├── figs/                   # Generated figures
└── src/
    └── timing_benchmarks/
        ├── benchmarks/     # Framework-specific implementations
        │   ├── genjax.py   # GenJAX benchmarks
        │   └── pyro.py     # Pyro benchmarks
        ├── data/           # Data generation utilities
        ├── visualization/  # Plotting and figure generation
        ├── export/         # Result export/import utilities
        ├── analysis/       # Benchmark comparison logic
        └── main.py         # CLI interface
```

## Running Benchmarks

### Quick Test
```bash
# Test basic functionality
pixi run test
```

### Importance Sampling Benchmarks
```bash
# Run IS benchmarks with default settings
pixi run benchmark-is --export-data --plot

# Custom particle counts and frameworks
python -m timing_benchmarks.main polynomial-is \
    --n-particles 100 1000 10000 \
    --frameworks genjax gen.jl \
    --export-data --plot
```

### HMC Benchmarks
```bash
# Run HMC benchmarks
pixi run benchmark-hmc --export-data --plot

# Custom settings
python -m timing_benchmarks.main polynomial-hmc \
    --n-samples 2000 --n-warmup 1000 \
    --frameworks genjax gen.jl \
    --export-data --plot
```

### Complete Benchmark Suite
```bash
# Run all benchmarks (IS and HMC)
pixi run benchmark-all --export-data --plot
```

### Pyro Benchmarks (Separate Environment)
```bash
# Switch to Pyro environment
pixi shell -e pyro

# Run Pyro benchmarks
pixi run benchmark-pyro-all --export --device cuda
```

## Generating Figures

```bash
# Generate figures from latest results
python -m timing_benchmarks.main plot

# Generate from specific experiment
python -m timing_benchmarks.main plot --data data/benchmark_20250102_143022
```

## Available Datasets

- `tiny`: 10 data points (for testing)
- `small`: 50 data points
- `medium`: 100 data points (default)
- `large`: 500 data points
- `xlarge`: 1000 data points
- `xxlarge`: 5000 data points

## Framework Support

- **GenJAX**: Full support for IS and HMC
- **Gen.jl**: Requires Julia installation
- **Pyro**: Requires separate environment (`pixi shell -e pyro`)
- **NumPyro**: Coming soon

## Output

- **Benchmark Data**: Saved to `data/benchmark_YYYYMMDD_HHMMSS/`
- **Figures**: Generated in `figs/` directory
- **Formats**: CSV for raw data, JSON for metadata, PDF/PNG for figures