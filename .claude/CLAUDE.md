# GenJAX Agent Context Documentation

This file serves as the index to all AGENTS.md files throughout the GenJAX codebase. These files provide curated context for AI coding assistants like Claude Code and Codex.

## Repository Structure

### Root Level
- [AGENTS.md](../AGENTS.md) - Top-level repository overview and architecture

### Source Code (`src/genjax/`)
- [src/genjax/AGENTS.md](../src/genjax/AGENTS.md) - Core GenJAX implementation
- [src/genjax/adev/AGENTS.md](../src/genjax/adev/AGENTS.md) - Automatic differentiation for expected values
- [src/genjax/extras/AGENTS.md](../src/genjax/extras/AGENTS.md) - Extra functionality (HMM, Kalman filters)
- [src/genjax/inference/AGENTS.md](../src/genjax/inference/AGENTS.md) - MCMC, SMC, and VI algorithms
- [src/genjax/viz/AGENTS.md](../src/genjax/viz/AGENTS.md) - Visualization utilities

### Examples (Case Studies)
- [examples/AGENTS.md](../examples/AGENTS.md) - Overview of all case studies
- [examples/faircoin/AGENTS.md](../examples/faircoin/AGENTS.md) - Beta-Bernoulli conjugate inference
- [examples/curvefit/AGENTS.md](../examples/curvefit/AGENTS.md) - Polynomial regression with outliers
- [examples/gol/AGENTS.md](../examples/gol/AGENTS.md) - Game of Life inverse dynamics
- [examples/localization/AGENTS.md](../examples/localization/AGENTS.md) - Robot localization with SMC

### Tests
- [tests/AGENTS.md](../tests/AGENTS.md) - Testing patterns and infrastructure

## How to Use

When working with GenJAX:
1. Start with the root [AGENTS.md](../AGENTS.md) for overall architecture
2. Navigate to specific module AGENTS.md for detailed context
3. For case studies, check the relevant example's AGENTS.md

Each AGENTS.md file provides:
- Module/directory purpose and design
- Key concepts and patterns
- Usage examples
- Common pitfalls and best practices
