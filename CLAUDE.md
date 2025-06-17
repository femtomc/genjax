# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Policy

- **NEVER create documentation files** unless explicitly requested
- Focus on implementation tasks and working code

## Overview

GenJAX is a probabilistic programming language embedded in Python centered on programmable inference.

For detailed GenJAX concepts, API patterns, and usage examples, see [src/genjax/CLAUDE.md](src/genjax/CLAUDE.md).

## Directory Structure

```
genjax/
├── src/genjax/           # Core GenJAX library
│   ├── core.py          # GFI implementation, traces, generative functions
│   ├── distributions.py # Built-in probability distributions
│   ├── mcmc.py          # MCMC algorithms (Metropolis-Hastings, HMC)
│   ├── smc.py           # Sequential Monte Carlo methods
│   ├── vi.py            # Variational inference algorithms
│   └── adev.py          # Automatic differentiation for variational estimates
├── examples/            # Example applications and case studies
│   ├── faircoin/        # Bayesian coin flipping example
│   ├── curvefit/        # Curve fitting with multiple frameworks
│   ├── localization/    # Particle filter localization
│   └── gol/             # Game of Life inference
├── tests/               # Test suite
├── docs/                # Generated documentation
└── quarto/              # Documentation source files
```

## Working in the Codebase

### CRITICAL Guidelines

1. **🔥 HIGH PRIORITY: Always read CLAUDE.md files** in directories you're working in
   - Each directory may contain specific guidance and patterns
   - These files contain essential context for that module/example

2. **Always run tests after changes** to `src/genjax/`
   - Run the full test suite: `pixi run test`
   - Or run specific test file: `pixi run python -m pytest tests/test_<module>.py`
   - Example: after changing `src/genjax/mcmc.py`, run `tests/test_mcmc.py`

3. **Check for corresponding test files**
   - `src/genjax/core.py` → `tests/test_core.py`
   - `src/genjax/mcmc.py` → `tests/test_mcmc.py`
   - `src/genjax/smc.py` → `tests/test_smc.py`

### Communication Guidelines

- **Be concise** - avoid unnecessary explanation or elaboration
- **Eliminate sycophancy** - no "I'd be happy to help" or similar pleasantries
- **Ask questions** - clarify requirements rather than making assumptions

### Workflow Tips

- Use `pixi run format` before committing to ensure code style
- Run `pixi run test-all` for comprehensive validation (tests + doctests)
- Check examples in the relevant directory for usage patterns

## Development Commands

### Setup

```bash
pixi install              # Install dependencies
```

### Testing

```bash
pixi run test             # Run tests with coverage
pixi run test-all         # Run tests + doctests
pixi run doctest          # Run doctests only
pixi run coverage         # Generate coverage report
```

### Code Quality

```bash
pixi run format           # Format and lint code with ruff
pixi run vulture          # Find unused code
pixi run precommit-run    # Run pre-commit hooks
```

### Examples

```bash
# Faircoin example
pixi run -e faircoin faircoin-timing
pixi run -e faircoin faircoin-comparison

# Curvefit example
pixi run -e curvefit curvefit
pixi run -e curvefit curvefit-all

# CUDA examples (requires CUDA environment)
pixi run -e cuda localization
pixi run -e cuda gol
```

### Documentation

```bash
pixi run -e docs preview  # Preview docs locally
pixi run -e docs deploy   # Deploy to GitHub Pages
```
