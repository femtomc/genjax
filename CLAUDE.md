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
│   ├── faircoin/        # Beta-Bernoulli framework comparison (GenJAX vs NumPyro vs handcoded JAX)
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

Follow this enhanced commit workflow to avoid failed commits and wasted time:

1. **Check git status** - `git status` to see what files will be committed
2. **Format code** - `pixi run format` to fix linting issues early
3. **Run pre-commit hooks** - `pixi run precommit-run` to catch issues before commit
4. **Stage changes** - `git add .` to stage all changes
5. **Check diff** - `git diff --cached` to review staged changes
6. **Commit with message** - Use proper commit message format
7. **Push if requested** - Only push when user explicitly asks

**Key insight**: Steps 1-2 are crucial - without them, Claude tends to jump straight to coding without proper understanding.

### CRITICAL Communication Guidelines

- **Be concise** - avoid unnecessary explanation or elaboration
- **Eliminate sycophancy** - no "I'd be happy to help" or similar pleasantries
- **Ask questions** - clarify requirements rather than making assumptions
- **Don't commit failures** - if you fail to solve a problem, don't commit partial/broken solutions

### CRITICAL Documentation Policy

- **NEVER create documentation files** unless explicitly requested
- Focus on implementation tasks and working code

### CRITICAL Efficiency Guidelines

1. **Always read CLAUDE.md files first** - Check for directory-specific guidance before working
2. **Use parallel tool calls** - Batch independent operations (git status + git diff, multiple file reads)
3. **Use Task tool for complex searches** - When searches may require multiple rounds of exploration
4. **Check existing patterns** - Read similar code before implementing new features
5. **Test relevant modules** - After changing `src/genjax/X.py`, run `tests/test_X.py`
6. **Avoid unnecessary files** - Only create files when absolutely required for the task
7. **Use proper search tools** - Glob for file patterns, Grep for content, Task for open-ended exploration

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
