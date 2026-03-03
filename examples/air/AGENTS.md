# AIR Case Study Guide

This case ports the GenJAX AIR estimator experiment from the PLDI'24 programmable-VI artifact.

## Purpose

Evaluate discrete-gradient estimators in an AIR-style latent-variable model:

- `enum`
- `reinforce`
- `mvd`
- `hybrid` (MVD for early `z_pres` sites, ENUM on the final site)

## Key Files

- `core.py`: model/guide definitions, STN transforms, objectives, training/eval utilities
- `main.py`: CLI (`train`, `compare`)

## CLI Commands

```bash
# Quick smoke comparison on synthetic data
pixi run python -m examples.air.main compare --small-config --num-examples 128 --epochs 2

# Train a single estimator and save history
pixi run python -m examples.air.main train \
  --estimator enum \
  --small-config \
  --num-examples 256 \
  --epochs 4 \
  --history-csv figs/air_enum_history.csv

# Use pre-generated multi-MNIST data (multi_mnist_uint8.npz)
pixi run python -m examples.air.main compare \
  --dataset multi-mnist \
  --data-path /path/to/multi_mnist_uint8.npz \
  --num-examples 2000
```

## Notes

- Default dataset mode is `synthetic` (samples from the AIR prior), so the case runs without Pyro/Torch.
- For `multi-mnist`, supply an existing NPZ file.
- Keep heavy logic in `core.py`; keep `main.py` as orchestration only.

## Tests

- `tests/test_air_example.py`
