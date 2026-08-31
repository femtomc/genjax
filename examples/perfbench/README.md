# Performance benchmark case study

- **Purpose:** Reproduce POPL 2026 Figure 16b across GenJAX, NumPyro, Pyro,
  TensorFlow Probability, hand-coded JAX/PyTorch, and Gen.jl.

## Use

- From `research/genjax/`:

```sh
pixi run paper-perfbench
pixi run paper-perfbench --mode cuda
pixi run paper-perfbench --inference is
pixi run paper-perfbench --inference hmc
pixi run paper-perfbench --frameworks genjax numpyro handcoded_jax
pixi run paper-perfbench-clean
```

- Direct orchestration:

```sh
pixi run -e perfbench python examples/perfbench/main.py pipeline --help
```

- CPU output: `data_cpu/` and `figs_cpu/`.
- CUDA output: `data/` and `figs/`.
- Resume with the `--skip-*` flags shown by `--help`.
- Gen.jl lanes require Julia 1.10 or newer.
- Framework-specific environments and repeat caps are encoded in the pipeline.

## Code

- [Pipeline](main.py)
- [Benchmark runners](benchmarks/)
- [Framework adapters](benchmarks/src/timing_benchmarks/curvefit_benchmarks/)
- [Result merge and plotting](benchmarks/combine_results.py)
- [Pixi task registration](../../pyproject.toml)
- [Parent artifact index](../../README.md)

## References

- [POPL 2026 paper source](../../../../press/papers/tex/genjax-popl-2026/README.md)
- Imported timing benchmark baseline: `timing-benchmarks@d4433b0`.

## License

Apache-2.0. See [LICENSE](../../LICENSE.md).
