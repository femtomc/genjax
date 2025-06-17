# tests/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX test suite.

## Test Structure

The test suite mirrors the main library structure:

```
tests/
├── test_core.py         # Tests for src/genjax/core.py (GFI, traces, generative functions)
├── test_distributions.py # Tests for src/genjax/distributions.py
├── test_mcmc.py         # Tests for src/genjax/mcmc.py (Metropolis-Hastings, HMC)
├── test_smc.py          # Tests for src/genjax/smc.py (Sequential Monte Carlo)
├── test_vi.py           # Tests for src/genjax/vi.py (Variational inference)
└── test_adev.py         # Tests for src/genjax/adev.py (Automatic differentiation)
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests with coverage
pixi run test

# Run specific test file
pixi run python -m pytest tests/test_<module>.py

# Run tests with verbose output
pixi run python -m pytest tests/test_<module>.py -v

# Run specific test function
pixi run python -m pytest tests/test_<module>.py::test_function_name
```

### Test Writing Patterns

1. **JAX Integration Tests**
   - Use `jax.random.PRNGKey` for reproducible randomness
   - Test both CPU and compilation (JIT) behavior where relevant
   - Use `jax.numpy` for array operations

2. **Probabilistic Programming Tests**
   - Test generative functions with known distributions
   - Verify trace structure and addressing
   - Check inference algorithm convergence properties

3. **MCMC Algorithm Tests**
   - Test acceptance rates are reasonable
   - Verify chain mixing and convergence
   - Check posterior estimates against known values

4. **SMC Tests**
   - Test particle filtering accuracy
   - Verify resampling behavior
   - Check effective sample size calculations

### Test Naming Conventions

- `test_<functionality>_<specific_case>`
- Use descriptive names that explain what is being tested
- Group related tests using classes when appropriate

### Fixtures and Utilities

Common test utilities and fixtures should be placed in `conftest.py` if they're used across multiple test files.

### Performance Testing

For computationally intensive algorithms:

- Use smaller problem sizes in tests
- Focus on correctness over performance
- Add performance benchmarks separately if needed

## Critical Testing Requirements

1. **Always test after changes** to corresponding source files
2. **Ensure tests pass** before committing changes
3. **Add tests for new functionality** - don't just modify existing code
4. **Test edge cases** and error conditions
5. **Use appropriate tolerances** for floating-point comparisons with probabilistic algorithms
