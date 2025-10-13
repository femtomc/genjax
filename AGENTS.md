# GenJAX Agent Guide

Use this note as the entry point before touching the repository. It points to the canonical module guides and highlights the conventions that apply everywhere.

## Read First
- `src/genjax/AGENTS.md` – core runtime, GFI methods, addressing rules  
- `src/genjax/inference/AGENTS.md` – MCMC/SMC/VI drivers and diagnostics  
- `src/genjax/adev/AGENTS.md` – ADEV expectations and estimator catalog  
- `<module>/AGENTS.md` – every directory you modify has a local brief

## Global Conventions
- Keep control flow JAX-friendly: use `jax.lax.cond/scan/fori_loop/while_loop` instead of Python branching inside traced code.
- Treat static parameters with `Const[...]` and expose probabilistic functions without keys.
- When you need explicit randomness, or before `jax.jit`, `jax.vmap` or using probabilistic sampling within a `jax.scan`, 
  wrap callables with `seeded = genjax.pjax.seed(fn)` and call `seeded(key, ...)`.
- Prefer `genjax.inference.modular_vmap` over raw `jax.vmap` when probabilistic sampling is involved.

## Repository Layout
```
src/genjax/        core library (core.py, pjax.py, inference/, adev/, viz/, ...)
examples/          publication case studies (each with its own AGENTS.md)
tests/             regression and unit tests mirroring the src/ tree
docs/, quarto/     documentation sources
```

## Working Checklist
1. Review the relevant AGENTS guide(s) and existing tests/examples for the feature you touch.
2. Prototype changes in modules or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (`tests/test_*.py`) alongside code changes.
4. Run the scoped pytest command (`pixi run test -m ...`) before submitting.
5. Keep documentation edits minimal and aligned with the per-module format.
