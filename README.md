<p align="center">
<img width="450" src="./logo.png"/>
</p>

[![codecov](https://codecov.io/gh/femtomc/genjax/graph/badge.svg?token=V5W1YIYC5P)](https://codecov.io/gh/femtomc/genjax)

> **Note**: This is the research version of GenJAX. A [(more stable) community version can be found here](https://github.com/genjax-community/genjax).

## What is GenJAX?

This research branch powers the POPL'26 artifact submitted alongside the paper *Probabilistic Programming with Vectorized Programmable Inference*. Throughout the paper we use the anonymized name **VPPL**, but VPPL and GenJAX refer to the same system.

### **Probabilistic Programming Language**

GenJAX is a probabilistic programming language (PPL): a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation for expected values, and more.

### **With Programmable Inference**

The design of GenJAX is centered on _programmable inference_: automation which allows users to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions: "_x_ affects _y_, and I observe _y_, what are my new beliefs about _x_?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

### **Core Concepts**

GenJAX's automation is based on two key concepts:
- **_Generative functions_** – GenJAX's version of probabilistic programs
- **_Traces_** – samples from probabilistic programs

GenJAX provides:

- **Modeling language automation** for constructing complex probability distributions from pieces
- **Inference automation** for constructing Monte Carlo samplers using convenient idioms (programs expressed by creating and editing traces), and [variational inference automation](https://dl.acm.org/doi/10.1145/3656463) using [new extensions to automatic differentation for expected values](https://dl.acm.org/doi/10.1145/3571198)

### **Fully Vectorized & Compatible with JAX**

All of GenJAX's automation is fully compatible with JAX, implying that any program written in GenJAX can be `vmap`'d and `jit` compiled.

## 🤖 Working with AI Coding Agents

This repository is optimized for collaboration with AI coding agents. The codebase includes comprehensive `AGENTS.md` files that provide the contextual grounding these tools need to operate safely and effectively within GenJAX.

## POPL'26 Artifact Overview

- **Terminology**  GenJAX ≡ VPPL (the paper name). Whenever you see VPPL in the POPL submission, read it as GenJAX in this repository.
- **Placement**  This directory lives inside the artifact root (`..`). Consult `../README.md` for paper build commands; use the instructions below when you are working directly with GenJAX.

## Environment Setup

1. Install [pixi](https://pixi.sh/) (only prerequisite).
2. From the artifact root run:
   ```bash
   pixi install
   pixi run genjax-setup
   ```
   The second command installs this submodule's dependencies and prepares the case-study environments.

## Quick Validation

Run these commands from the artifact root to confirm the toolchain works end-to-end:

```bash
# Compile the POPL paper (checks LaTeX toolchain)
pixi run paper

# Lightweight fair-coin benchmark smoke test
pixi run --manifest-path genjax/pyproject.toml -e faircoin \
  python -m examples.faircoin.main --combined --num-obs 20 --num-samples 200 --repeats 10
```

Both commands should finish without errors. The smoke test writes PDFs to `genjax/examples/faircoin/figs/`.

## Case Study Commands

Invoke the following from the artifact root. Append `-e <env>-cuda` if you have a CUDA-capable GPU and want the timings reported in the paper.

| Case study (paper section) | Command | Outputs |
| --- | --- | --- |
| Beta–Bernoulli performance survey (§7) | `pixi run --manifest-path genjax/pyproject.toml -e faircoin python -m examples.faircoin.main --combined --num-obs 50 --num-samples 2000 --repeats 200` | PDFs in `genjax/examples/faircoin/figs/` |
| Game of Life inversion showcase (§7) | `pixi run --manifest-path genjax/pyproject.toml -e gol gol-paper` | PDFs in `genjax/examples/gol/figs/` |
| Curve fitting + outliers (§2) | `pixi run --manifest-path genjax/pyproject.toml -e curvefit python -m examples.curvefit.main paper` | PDFs in `genjax/examples/curvefit/figs/` |
| Robot localization SMC comparison (§7) | `pixi run --manifest-path genjax/pyproject.toml -e localization python -m examples.localization.main paper --include-basic-demo --include-smc-comparison --n-particles 200 --n-steps 8 --timing-repeats 3 --n-rays 8` | PDFs in `genjax/examples/localization/figs/` plus CSV/JSON data in `genjax/examples/localization/data/` |

Each CLI provides additional flags (see the per-case `AGENTS.md` files) so you can run exploratory or accelerated configurations before reproducing the full paper setup.

## Regenerating All Paper Figures

The artifact root exposes aggregate helpers:

```bash
# CPU-only regeneration of every paper figure
pixi run paper-figures

# GPU-accelerated regeneration (requires CUDA)
pixi run paper-figures-gpu
```

Rebuild the paper with `pixi run paper` afterwards to incorporate refreshed figures.

## Directory Map (genjax/)

```
examples/        # Case studies used in the evaluation
figs/            # Staging area for generated PDFs copied into ../figs/
src/genjax/      # Library implementation
tests/           # Test suite
pyproject.toml   # pixi environments/tasks for this submodule
README.md        # (this file)
```
