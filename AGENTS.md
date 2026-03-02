# GenJAX Agent Guide

Use this file as the top-level onboarding brief when working in this repository.
Its goal is to help external agents quickly learn GenJAX’s model, APIs, and idioms.

## Start Here (Fast Ramp)

1. Read `src/genjax/AGENTS.md` for the core generative function interface (GFI).
2. Read the module-level guide for the area you are changing:
   - `src/genjax/inference/AGENTS.md`
   - `src/genjax/adev/AGENTS.md`
   - `src/genjax/extras/AGENTS.md`
   - `src/genjax/viz/AGENTS.md`
3. If touching a case study, read both:
   - `examples/AGENTS.md`
   - `examples/<case>/AGENTS.md`
4. If adding/changing behavior, read related tests in `tests/test_*.py`.

## GenJAX Mental Model

GenJAX is a programmable inference PPL built around:

- **Generative functions** (`@gen`) that define probabilistic programs.
- **Traces** that store choices, args, retval, and score (`-log p`).
- **GFI methods** (`simulate`, `assess`, `generate`, `update`, `regenerate`) used to build inference algorithms.

Treat these as the core contract. Most of the repository is either:
- implementing this contract,
- composing it into inference algorithms,
- or validating it in tests/examples.

## Global Coding Idioms (Important)

- Keep traced control flow JAX-safe: use `jax.lax.cond/scan/while_loop/fori_loop` instead of Python branching in traced paths.
- Use static addresses and stable choice structure in `@gen` functions.
- Wrap static values with `Const[...]` / `const(...)` when needed for JAX staging.
- Expose keyless probabilistic functions in library code; seed at call sites:
  - `seeded = genjax.pjax.seed(fn)`
  - `seeded(key, ...)`
- Prefer `genjax.pjax.modular_vmap` (or `genjax.modular_vmap`) when probabilistic sampling appears under vectorization.

## Repository Layout

- `src/genjax/`: library implementation (core, inference, adev, pjax, state, viz, extras)
- `examples/`: case studies used in paper/artifact workflows
- `tests/`: regression + unit/integration coverage

## Development Checklist

1. Read local `AGENTS.md` files for every directory you modify.
2. Make the smallest change that preserves existing idioms.
3. Add/update focused tests in `tests/`.
4. Run targeted tests first, then a broader smoke run.
5. Keep AGENTS docs in sync when APIs or workflows change.

## Useful Commands

- Full tests: `pixi run test`
- Single file: `pixi run test -- tests/test_core.py`
- Single test: `pixi run test -- tests/test_core.py::test_name`

Use `pixi run <task> -- ...` for task-specific flags where needed.
