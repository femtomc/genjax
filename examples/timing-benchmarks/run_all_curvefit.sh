#!/bin/bash
# Run all curvefit benchmarks and save to data/curvefit/...

cd /home/femtomc/genjax-popl-2026/genjax/examples/timing-benchmarks

echo "Running curvefit benchmarks..."

# GenJAX
echo "Running GenJAX..."
PYTHONPATH=src pixi run -e cuda python -m "timing_benchmarks.curvefit-benchmarks.genjax" \
    --output-dir data/curvefit/genjax \
    --n-particles 100 1000 10000 100000 \
    --repeats 100

# NumPyro
echo "Running NumPyro..."
PYTHONPATH=src pixi run -e cuda python -m "timing_benchmarks.curvefit-benchmarks.numpyro" \
    --output-dir data/curvefit/numpyro \
    --n-particles 100 1000 10000 100000 \
    --repeats 100

# Handcoded TFP
echo "Running Handcoded TFP..."
PYTHONPATH=src pixi run -e cuda python -m "timing_benchmarks.curvefit-benchmarks.handcoded_tfp" \
    --output-dir data/curvefit/handcoded_tfp \
    --n-particles 100 1000 10000 100000 \
    --repeats 100

# Pyro
echo "Running Pyro..."
PYTHONPATH=src pixi run -e pyro python -m "timing_benchmarks.curvefit-benchmarks.pyro" \
    --output-dir data/curvefit/pyro \
    --n-particles 100 1000 10000 100000 \
    --repeats 50

# Handcoded Torch
echo "Running Handcoded Torch..."
PYTHONPATH=src pixi run -e pyro python -m "timing_benchmarks.curvefit-benchmarks.handcoded_torch" \
    --output-dir data/curvefit/handcoded_torch \
    --n-particles 100 1000 10000 100000 \
    --repeats 100

# Gen.jl (if available)
echo "Running Gen.jl..."
PYTHONPATH=src pixi run -e cuda python -m "timing_benchmarks.curvefit-benchmarks.genjl" \
    --output-dir data/curvefit/genjl \
    --n-particles 100 1000 10000 100000 \
    --repeats 50 || echo "Gen.jl benchmark failed (Julia may not be installed)"

echo "All benchmarks complete!"
echo "Now running combine_results.py..."

# Combine results
python combine_results.py --data-dir data

echo "Done! Check figs/all_frameworks_comparison.pdf"