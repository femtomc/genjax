# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- Commit the result and create pull requests when appropriate
- Update documentation (READMEs, changelogs) as needed
- Ensure all tests pass before committing

**Key insight**: Steps 1-2 are crucial - without them, Claude tends to jump straight to coding without proper understanding.

### CRITICAL Communication Guidelines

- **Be concise** - avoid unnecessary explanation or elaboration
- **Eliminate sycophancy** - no "I'd be happy to help" or similar pleasantries
- **Ask questions** - clarify requirements rather than making assumptions
- **Don't commit failures** - if you fail to solve a problem, don't commit partial/broken solutions

### CRITICAL Documentation Policy

- **NEVER create documentation files** unless explicitly requested
- Focus on implementation tasks and working code

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
pixi run format           # Format and lint Python code with ruff
pixi run format-md        # Format Markdown files with prettier
pixi run format-all       # Format both Python and Markdown files
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
