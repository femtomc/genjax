# AGENTS.md

This note briefs AI coding agents that assist with the POPL 2026 paper artifact “Probabilistic Programming with Vectorized Programmable Inference.” Use it as the starting point before touching any files so the documentation, figures, and experiments stay aligned with the submission package.

Terminology: within the POPL submission the system is called VPPL; inside this repository we continue to use the GenJAX name. They refer to the same implementation.

## Critical Initial Context Loading

When starting work in this codebase, ALWAYS read the relevant AGENTS.md files first:
1. **Core concepts**: Read `src/genjax/AGENTS.md` for GenJAX fundamentals
2. **Inference algorithms**: Read `src/genjax/inference/AGENTS.md` for MCMC, SMC, VI
3. **ADEV**: Read `src/genjax/adev/AGENTS.md` for unbiased gradient estimation
4. **Module-specific**: Check for AGENTS.md in any directory you're working in

## Overview

GenJAX is a probabilistic programming language embedded in Python centered on programmable inference.

## Inference Highlights

`rejuvenation_smc` supports optional custom proposals while defaulting to the model’s internal transition. Existing call sites continue to work. Diagnostics expose log-normalized weights so downstream code can track ESS (see `src/genjax/inference/AGENTS.md` for usage patterns).

## JAX Best Practices

GenJAX uses JAX extensively. **ALWAYS enforce good JAX idioms**:

### Control Flow Rules
- **NEVER use Python control flow** in JAX-compiled functions:
  - FAIL `if`, `elif`, `else` statements
  - FAIL `for`, `while` loops
  - FAIL `break`, `continue` statements

- **ALWAYS use JAX control flow** instead:
  - OK `jax.lax.cond` for conditionals
  - OK `jax.lax.scan` for loops with carry
  - OK `jax.lax.fori_loop` for simple iteration
  - OK `jax.lax.while_loop` for conditional loops
  - OK `jax.lax.switch` for multiple branches

### Exceptions
- Only use Python control flow if explicitly told "it's okay to use Python control flow"
- Static values (known at compile time) can use Python control flow
- Outside of JIT-compiled functions, normal Python is fine

### Common Patterns
```python
# FAIL WRONG - Python control flow
if condition:
    x = computation_a()
else:
    x = computation_b()

# OK CORRECT - JAX control flow
x = jax.lax.cond(condition,
                  lambda: computation_a(),
                  lambda: computation_b())

# FAIL WRONG - Python loop
for i in range(n):
    x = update(x, i)

# OK CORRECT - JAX loop
def body(i, x):
    return update(x, i)
x = jax.lax.fori_loop(0, n, body, x)
```

### GenJAX-Specific JAX Tips
- Use `Const[T]` for static values that must not become tracers
- Wrap probabilistic callables with `genjax.pjax.seed` so the seeded version accepts a `PRNGKey` before applying `jax.jit`, `jax.vmap`, etc.
- Prefer `modular_vmap` over `jax.vmap` for probabilistic operations
- All GenJAX types inherit from `Pytree` for automatic vectorization

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
│   └── adev             # Automatic Differentiation of Expected Values (ADEV)
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

## Development Workflow

### 1. Explore
- Read relevant AGENTS.md files first (see Initial Context Loading above)
- **HIGH PRIORITY**: Check existing tests and examples for usage patterns
  - `tests/` for API usage and edge cases
  - `examples/` for implementation patterns
- Only refer to source code after understanding usage
- Explicitly state "don't write code yet" to focus on understanding

### 2. Plan
- Create a detailed plan based on patterns found
- Use "think" to trigger extended thinking mode for complex solutions
- Ensure the plan addresses all requirements

### 3. Code
- **Do not use the command line Python interpreter** - always write test scripts
- Create `test_<feature>.py` scripts for experiments
- Run with: `pixi run python test_feature.py`
- Follow patterns from existing tests/examples

### 4. Test
- Use localized testing: `pixi run test -m tests/test_<module>.py -k test_name`
- Full suite only for final validation
- Check corresponding test files exist:
  - `src/genjax/*.py` → `tests/test_*.py`
  - `src/genjax/inference/*.py` → `tests/test_*.py`

### 5. Commit

Follow this enhanced commit workflow to avoid failed commits and wasted time:

1. **Check git status** - `git status` to see what files will be committed
2. **Format code** - `pixi run format` to fix linting issues early
3. **Run pre-commit hooks** - `pixi run precommit-run` to catch issues before commit
4. **Stage changes** - `git add .` to stage all changes
5. **Check diff** - `git diff --cached` to review staged changes
6. **Commit with message** - Use proper commit message format
7. **Push if requested** - Only push when user explicitly asks

## Key Development Practices

### Documentation
- **NEVER create documentation files** unless explicitly requested
- Add paper/website references to `REFERENCES.md` in module directory
- Keep AGENTS.md files focused on their specific module
- Cross-reference related AGENTS.md files explicitly

### Communication
- Be concise - avoid unnecessary elaboration
- Ask questions rather than making assumptions
- Don't commit partial/broken solutions

### Examples and Case Studies
- Follow standardized structure in `examples/AGENTS.md`
- Reuse shared helpers from `genjax.timing` and existing case studies
- See existing examples for patterns before implementing

### Documentation Style for AGENTS.md Files

When working with AGENTS.md files in the codebase:

- **Use method signatures and file references** instead of raw code blocks
- **Format**: `**Function**: name(params) -> return_type`
- **Include location**: `**Location**: filename.py:line_numbers`
- **Describe API contracts** and usage patterns, not implementation details
- **Reference actual source files** for examples and detailed implementation
- **Keep documentation maintainable** by avoiding code duplication

This approach ensures documentation stays in sync with the codebase and reduces maintenance burden.

### Workflow Tips

- Before any commit: `pixi run format` → `pixi run precommit-run` → `git add .` → commit
- Use `pixi run test-all` for comprehensive validation (tests + doctests)
- Each module has a AGENTS.md file - always read it before working in that module

## Development Commands

**Setup**: Run `pixi install` to install dependencies.

**JAX Backend**: GenJAX uses CPU-compatible JAX by default for maximum compatibility across environments. JAX will automatically detect and use GPU acceleration when available without requiring special configuration.

**All available commands**: See `pyproject.toml` for the complete list of available pixi commands. The file is organized into features:

- **Testing**: `test`, `test-all`, `doctest`, `coverage` commands in `[tool.pixi.feature.test.tasks]`
- **Code Quality**: `format`, `format-md`, `format-all`, `vulture`, `precommit-run` commands in `[tool.pixi.feature.format.tasks]`
- **Examples**: Each example has its own feature section with specific commands:
  - `faircoin`: Beta-Bernoulli framework comparison
  - `curvefit`: Curve fitting with multiple frameworks (requires NumPyro)
  - `gol`: Game of Life inference
  - `localization`: Particle filter localization (requires cuda environment for visualization)
  - `state-space`: State space models
  - `gen2d`: 2D generative models
  - `intuitive-physics`: Physics simulation inference
  - `programmable-mcts`: Monte Carlo Tree Search

**Usage Pattern**:
- General commands: `pixi run <command>`
- Example-specific commands: `pixi run -e <example> <command>`
- Many examples have `setup`, `<name>-quick`, `<name>-all` variants for different use cases

**Environment Selection for Case Studies**:
- **Case study-specific environments**: Some examples require specific environments
  - `curvefit`: Requires the curvefit environment for NumPyro dependencies
  - `localization`: Requires the cuda environment for visualization dependencies
- **CPU environments**: Default for most examples, works everywhere
- **CUDA environments**: Use for GPU acceleration when available
  - Access via: `pixi run -e cuda <command>` or `pixi run cuda-<example>`
  - Examples: `cuda-localization`, `cuda-faircoin`, `cuda-curvefit`
- **Usage examples**:
  - `pixi run -e curvefit python -m examples.curvefit.main` (curvefit with NumPyro)
  - `pixi run -e cuda python -m examples.localization.main` (localization with GPU)
