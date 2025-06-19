# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔥 CRITICAL: Initial Context Loading

When starting work in this codebase, ALWAYS read the relevant CLAUDE.md files first:
1. **Core concepts**: Read `src/genjax/CLAUDE.md` for GenJAX fundamentals
2. **Inference algorithms**: Read `src/genjax/inference/CLAUDE.md` for MCMC, SMC, VI
3. **ADEV**: Read `src/genjax/adev/CLAUDE.md` for gradient estimation
4. **Module-specific**: Check for CLAUDE.md in any directory you're working in

## Overview

GenJAX is a probabilistic programming language embedded in Python centered on programmable inference.

## Directory Structure

```
genjax/
├── src/genjax/           # Core GenJAX library
│   ├── core.py          # GFI implementation, traces, generative functions
│   ├── distributions.py # Built-in probability distributions
│   ├── pjax.py          # Probabilistic JAX (PJAX) - probabilistic primitives and interpreters
│   ├── state.py         # State interpreter for tagged value inspection
│   ├── mcmc.py          # MCMC algorithms (Metropolis-Hastings, HMC)
│   ├── smc.py           # Sequential Monte Carlo methods
│   ├── vi.py            # Variational inference algorithms
│   └── adev.py          # Automatic differentiation for variational estimates
├── examples/            # Example applications and case studies
│   ├── faircoin/        # Beta-Bernoulli framework comparison (GenJAX vs NumPyro vs handcoded JAX)
│   ├── curvefit/        # Curve fitting with multiple frameworks
│   ├── localization/    # Particle filter localization
│   └── gol/             # Game of Life inference
├── tests/               # Test suite
│   ├── test_core.py     # Tests for core.py
│   ├── test_distributions.py # Tests for distributions.py
│   ├── test_pjax.py     # Tests for pjax.py
│   ├── test_state.py    # Tests for state.py
│   ├── test_mcmc.py     # Tests for mcmc.py
│   ├── test_smc.py      # Tests for smc.py
│   ├── test_vi.py       # Tests for vi.py
│   ├── test_adev.py     # Tests for adev.py
│   └── discrete_hmm.py  # Discrete HMM test utilities
├── docs/                # Generated documentation
└── quarto/              # Documentation source files
```

## Working in the Codebase

### CRITICAL Guidelines

1. **🔥 ALWAYS write test scripts first**
   - NEVER use command line Python snippets
   - Create test scripts in a temporary directory: `test_feature.py`
   - Run with: `pixi run python test_feature.py`
   - Only add to test suite after validating locally

2. **Prefer localized testing over full suite**
   - Run specific test: `pixi run test -m tests/test_<module>.py -k test_name`
   - Full suite is slow - use only for final validation
   - Example: after changing `src/genjax/inference/mcmc.py`, test with:
     ```bash
     pixi run test -m tests/test_mcmc.py -k test_metropolis
     ```

3. **Check for corresponding test files**
   - Core modules: `src/genjax/*.py` → `tests/test_*.py`
   - Inference: `src/genjax/inference/*.py` → `tests/test_*.py`
   - ADEV: `src/genjax/adev/*.py` → `tests/test_adev.py`

### CRITICAL Claude Code Workflow

Follow this four-step workflow for effective development:

### 1. Explore

- Read relevant files and context first
- Explicitly state "don't write code yet" to focus on understanding
- Use subagents for complex problems requiring extensive exploration

### 2. Plan

- Ask Claude to create a detailed plan before coding
- Use "think" to trigger extended thinking mode for complex solutions
- Ensure the plan addresses all requirements and edge cases

### 3. Code

- Implement the solution in code based on the plan
- Verify the solution's reasonableness during implementation
- Test and validate the implementation works correctly

### 4. Commit

Follow this enhanced commit workflow to avoid failed commits and wasted time:

1. **Check git status** - `git status` to see what files will be committed
2. **Format code** - `pixi run format` to fix linting issues early
3. **Run pre-commit hooks** - `pixi run precommit-run` to catch issues before commit
4. **Stage changes** - `git add .` to stage all changes
5. **Check diff** - `git diff --cached` to review staged changes
6. **Commit with message** - Use proper commit message format
7. **Push if requested** - Only push when user explicitly asks

**Key insight**: Steps 1-2 are crucial - without them, Claude tends to jump straight to coding without proper understanding.

### CRITICAL Development Practices

1. **Testing Protocol**
   - 🔥 NEVER RUN INLINE PYTHON - always write test scripts
   - Create `test_<feature>.py` scripts for all experiments
   - Use localized tests during development
   - Run full suite only before commits

2. **Documentation Requirements**
   - Add paper/website references to `REFERENCES.md` in module directory
   - Keep CLAUDE.md files focused on their specific module
   - Cross-reference related CLAUDE.md files explicitly

3. **Communication Guidelines**
   - Be concise - avoid unnecessary elaboration
   - Ask questions rather than making assumptions
   - Don't commit partial/broken solutions

### CRITICAL Documentation Policy

- **NEVER create documentation files** unless explicitly requested
- Focus on implementation tasks and working code

### Module Organization

```
src/genjax/
├── CLAUDE.md           # Core concepts: gen, traces, distributions, PJAX
├── core.py            # Generative functions, traces, Fixed infrastructure
├── distributions.py   # Probability distributions
├── pjax.py           # Probabilistic JAX primitives
├── state.py          # State inspection interpreter
├── inference/
│   ├── CLAUDE.md     # Inference algorithms guidance
│   ├── mcmc.py       # MCMC algorithms
│   ├── smc.py        # Sequential Monte Carlo
│   └── vi.py         # Variational inference
├── adev/
│   ├── CLAUDE.md     # Gradient estimation guidance
│   └── __init__.py   # ADEV implementation
└── extras/
    ├── CLAUDE.md     # Testing utilities guidance
    └── state_space.py # Exact inference for testing
```

Each CLAUDE.md file contains module-specific guidance. Always read the relevant files before working in a module.

### Workflow Tips

- Before any commit: `pixi run format` → `pixi run precommit-run` → `git add .` → commit
- Use `pixi run test-all` for comprehensive validation (tests + doctests)
- Check examples in the relevant directory for usage patterns
- When unsure about approach, explore first with explicit "don't write code yet" statement

## Development Commands

### Setup

```bash
pixi install              # Install dependencies
```

**JAX Backend**: GenJAX uses CPU-compatible JAX by default for maximum compatibility across environments. JAX will automatically detect and use GPU acceleration when available without requiring special configuration.

### Testing

```bash
pixi run test             # Run tests with coverage
pixi run test-all         # Run tests + doctests
pixi run doctest          # Run doctests only
pixi run coverage         # Generate coverage report
```

### Code Quality

```bash
pixi run format           # Format and lint Python code with ruff
pixi run format-md        # Format Markdown files with prettier
pixi run format-all       # Format both Python and Markdown files
pixi run vulture          # Find unused code
pixi run precommit-run    # Run pre-commit hooks
```

### Examples

```bash
# Faircoin example - Beta-Bernoulli framework comparison
pixi run -e faircoin faircoin-timing      # Timing comparison only
pixi run -e faircoin faircoin-combined    # Combined timing + posterior figure (recommended)
pixi run -e faircoin python -m examples.faircoin.main --posterior  # Posterior comparison only
pixi run -e faircoin python -m examples.faircoin.main --all        # All figures

# Curvefit example
pixi run -e curvefit curvefit
pixi run -e curvefit curvefit-all

# Other examples
pixi run -e localization localization
pixi run -e gol gol
```

### Documentation

```bash
pixi run -e docs preview  # Preview docs locally
pixi run -e docs deploy   # Deploy to GitHub Pages
```
