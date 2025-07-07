"""Main CLI for timing benchmarks.

Note: This is a stub. The actual benchmarking is done through:
- Direct benchmark scripts: python -m timing_benchmarks.benchmarks.{framework}
- Result combination: python combine_results.py

For GPU benchmarks, use the cuda-* tasks:
- pixi run -e cuda cuda-genjax
- pixi run -e cuda cuda-numpyro
- pixi run -e cuda cuda-handcoded-tfp
- pixi run -e cuda cuda-combine
"""

import sys


def main():
    print(__doc__)
    print("\nPlease use the direct benchmark commands or pixi tasks instead.")
    sys.exit(0)


if __name__ == "__main__":
    main()