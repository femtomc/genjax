<p align="center">
<img width="450" src="https://raw.githubusercontent.com/femtomc/genjax/main/logo.png"/>
</p>

[![DOI](https://zenodo.org/badge/971731825.svg)](https://doi.org/10.5281/zenodo.17342547)

- **Purpose:** JAX probabilistic programming with generative functions,
  structured traces, vectorized programmable inference, MCMC, SMC, VI, and ADEV.
- **POPL 2026 artifact:**
  [v1.0.10](https://github.com/femtomc/genjax/releases/tag/v1.0.10)

## Use

```sh
cd research/genjax
pixi install
pixi run test-fast
pixi run paper-figures
pixi run paper-figures-gpu
```

```python
from genjax import gen, normal

@gen
def model():
    return normal(0.0, 1.0) @ "x"

trace = model.simulate()
choices = trace.get_choices()
```

- Generative functions expose `simulate`, `generate`, `assess`, and `update`.
- `vmap` and `modular_vmap` lift model and inference structure over explicit
  array axes.
- Inspect all Pixi tasks in [pyproject.toml](pyproject.toml).

## Paper cases

| Case                      | Figures        | Command                                        |
| ------------------------- | -------------- | ---------------------------------------------- |
| Fair coin                 | 16a            | `pixi run paper-faircoin-gen`                  |
| Curve fitting             | 4–6            | `pixi run paper-curvefit-gen`                  |
| Multi-framework benchmark | 16b            | `pixi run paper-perfbench`                     |
| Game of Life              | 18             | `pixi run assets && pixi run -e gol gol-paper` |
| Localization              | 19             | `pixi run paper-localization-gen`              |
| AIR estimators            | PLDI 2024 port | `pixi run air-compare`                         |

- Add `--mode cuda` to `paper-perfbench` for its CUDA pipeline.
- CPU and GPU execute the same models but do not have the same scaling curves.
- Figure 19 and paper-scale curve fitting require CUDA-like throughput to match
  the published timing/ESS panels.
- Gen.jl benchmark lanes require Julia 1.10 or newer.
- Generated figures land under `figs/`. Perfbench owns separate CPU/CUDA output
  roots.

## Code

- [Public package](src/genjax/__init__.py)
- [Generative-function core](src/genjax/core.py)
- [Distributions](src/genjax/distributions.py)
- [Probabilistic vectorization](src/genjax/pjax.py)
- [Inference](src/genjax/inference/)
- [ADEV](src/genjax/adev/)
- [Tests](tests/)
- [Examples](examples/)
- [Performance benchmark](examples/perfbench/README.md)
- [Citation metadata](CITATION.cff)
- [Package references](src/genjax/REFERENCES.md)

## References

- [Probabilistic Programming with Vectorized Programmable Inference](../../press/papers/tex/genjax-popl-2026/README.md)
- [Artifact DOI](https://doi.org/10.5281/zenodo.17342547)
- [Gen: programmable inference](https://doi.org/10.1145/3314221.3314642)
- [ADEV](https://doi.org/10.1145/3571198)
- [Programmable variational inference](https://doi.org/10.1145/3656463)

## License

Apache-2.0. See [LICENSE](LICENSE.md).
