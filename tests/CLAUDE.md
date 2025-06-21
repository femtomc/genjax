# tests/CLAUDE.md

This file provides guidance for Claude Code when working with the GenJAX test suite.

## Test Structure

The test suite mirrors the main library structure:

```
tests/
├── test_core.py         # Tests for src/genjax/core.py (GFI, traces, generative functions)
├── test_distributions.py # Tests for src/genjax/distributions.py (built-in distributions)
├── test_pjax.py         # Tests for src/genjax/pjax.py (PJAX primitives and interpreters)
├── test_state.py        # Tests for src/genjax/state.py (state interpreter)
├── test_mcmc.py         # Tests for src/genjax/mcmc.py (Metropolis-Hastings, MALA)
├── test_smc.py          # Tests for src/genjax/smc.py (Sequential Monte Carlo)
├── test_vi.py           # Tests for src/genjax/vi.py (Variational inference)
├── test_adev.py         # Tests for src/genjax/adev.py (Automatic differentiation)
├── test_gp.py           # Tests for src/genjax/gp/ (Gaussian Processes)
├── discrete_hmm.py      # Discrete HMM test utilities
└── conftest.py          # Test configuration and shared fixtures
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests with coverage
pixi run test

# Run specific test file
pixi run test -m tests/test_<module>.py

# Run tests with verbose output
pixi run test -m tests/test_<module>.py -v

# Run specific test function
pixi run test -m tests/test_<module>.py::test_function_name
```

### Test File Organization

Use **sectional headers** to organize tests within each file for better readability and navigation:

```python
# =============================================================================
# GENERATIVE FUNCTION (@gen) TESTS
# =============================================================================

@pytest.mark.core
def test_fn_simulate_vs_manual_density():
    """Test @gen function simulation."""
    pass

# =============================================================================
# SCAN COMBINATOR TESTS
# =============================================================================

@pytest.mark.core
def test_scan_simulate_vs_manual_density():
    """Test Scan combinator functionality."""
    pass
```

**Guidelines for sectional headers:**

- Use descriptive section names that clearly identify the functionality being tested
- Group related tests under the same header
- Use consistent formatting with `# =============================================================================`
- Place headers before test functions/classes, not within them
- Keep sections focused - split large sections if they become unwieldy

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
   - **Always test monotonic convergence** with increasing chain lengths
   - **Example**: See `test_mala_chain_monotonic_convergence` in `test_mcmc.py`

4. **SMC Tests**
   - **CRITICAL: Always test log marginal likelihood convergence** with increasing particle counts
   - Test particle filtering accuracy
   - Verify resampling behavior
   - Check effective sample size calculations
   - Validate against analytical marginal likelihoods when available
   - **Example**: See `test_smc_*_marginal_likelihood` tests in `test_smc.py`

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

### Gaussian Process Testing

The GP module tests (`test_gp.py`) cover:

1. **Kernel Functionality** (`test_gp_kernels`):
   - RBF, Matern52, and other kernel implementations
   - Kernel evaluation on test inputs
   - Proper variance values on diagonal

2. **Mean Functions** (`test_gp_mean_functions`):
   - Zero, Constant, and Linear mean functions
   - Correct output shapes and values

3. **GP as Generative Function**:
   - `test_gp_simulate`: Forward sampling from GP prior/posterior
   - `test_gp_conditioning`: GP conditioning on observations
   - `test_gp_assess`: Density evaluation
   - `test_gp_generate_with_constraints`: Constrained generation
   - `test_gp_in_gen_function`: Integration with `@gen` functions

4. **Special Considerations**:
   - GP uses exact inference as internal proposal (weight = 0)
   - GPTrace inherits from Trace for proper unwrapping
   - Integration with PJAX via `seed()` transformation
   - Proper handling of Fixed wrappers not needed for GP

## Inference Algorithm Testing Strategies

### Convergence Testing Philosophy

Inference algorithms should demonstrate **monotonic improvement** as computational resources increase. This principle applies across all GenJAX inference methods:

- **MCMC**: Longer chains → better posterior approximation
- **SMC**: More particles → better marginal likelihood estimates (**CRITICAL: This must be tested in every SMC implementation**)
- **Variational Inference**: More optimization steps → better ELBO convergence

**Special Note on SMC**: Log marginal likelihood convergence is the fundamental validation test for SMC algorithms. Unlike MCMC (which approximates posteriors) or VI (which approximates with known divergence), SMC provides unbiased estimates of marginal likelihoods. Testing this convergence property is essential to verify the SMC implementation is mathematically correct.

### Testing Patterns by Algorithm

#### MCMC Convergence Testing

**Examples in `test_mcmc.py`:**
- `test_mala_chain_monotonic_convergence`: Tests error decrease with chain lengths [100, 500, 1000]
- `test_mh_chain_multiple_chains`: Tests R-hat and ESS diagnostics with multiple chains
- `test_mala_step_size_effects`: Tests acceptance rate ordering with different step sizes

**Key Pattern**: Test with increasing computational resources and verify monotonic improvement trends.

#### SMC Convergence Testing

**CRITICAL Requirement**: Every SMC test must include log marginal likelihood convergence testing.

**Examples in `test_smc.py`:**
- `test_*_marginal_likelihood`: Tests log marginal accuracy with increasing particle counts
- `test_smc_*_convergence`: Tests monotonic improvement in marginal likelihood estimates

**Key Patterns**:
1. **With analytical solution**: Compare estimated vs. true log marginal likelihood
2. **Without analytical solution**: Test stability and finite bounds of estimates
3. **Always test**: Monotonic improvement with particle counts [100, 500, 1000, 2000]

#### Variational Inference Testing

**Examples in `test_vi.py`:**
- `test_vi_elbo_*`: Tests ELBO monotonic improvement during optimization
- `test_variational_inference_*`: Tests convergence to known posteriors

**Key Pattern**: ELBO should improve monotonically and converge (not still rapidly improving).

### Testing Guidelines for Inference Algorithms

**Computational Resource Scaling**:
1. **Always test with increasing compute**: More chains, longer chains, more particles, more optimization steps
2. **Expect monotonic improvement**: Errors should decrease (with reasonable tolerance for stochasticity)
3. **CRITICAL for SMC**: Every SMC test must include log marginal likelihood convergence with increasing particles
4. **Test algorithm-specific properties**: Step size effects (MALA), particle degeneracy (SMC), ELBO convergence (VI)
5. **Validate against known solutions**: Use conjugate models with analytical posteriors when possible

**Tolerance Management**:
```python
# Tolerance hierarchy for different computational budgets
strict_tolerance = 1e-3      # High-compute scenarios
standard_tolerance = 1e-2    # Medium-compute scenarios
practical_tolerance = 1e-1   # Low-compute or difficult problems
convergence_tolerance = 0.1  # Convergence trend detection
```

**Stochasticity Handling**:
- Use **multiple random seeds** for robustness testing
- Allow **reasonable variance** in convergence patterns (not strictly monotonic)
- Focus on **overall trends** rather than step-by-step improvement
- Test **worst-case scenarios** with challenging initializations

## Testing Patterns

### Density Validation

**Basic Density Consistency** (Example: `test_*_simulate_assess_consistency`):
- Generate trace with `model.simulate()`
- Validate trace structure with `helpers.assert_valid_trace()`
- Check `simulate`/`assess` consistency: `score = -log_density`

**Manual Density Computation** (Example: `test_*_vs_manual_density`):
- Compare `model.assess()` with manual distribution calculations
- Validate against analytical density formulas

### GFI Method Testing

**Update Method Testing** (Example: `test_*_update_weight_invariant`):
- Test weight invariant: `weight = -new_score + old_score`
- Verify traces remain valid after updates

**Regenerate Method Testing** (Example: `test_*_regenerate_*`):
- Test selective resampling of addresses
- Verify non-selected addresses remain unchanged
- Check `discard` contains old values of regenerated addresses

**Generate Method Testing** (Example: `test_*_generate_*`):
- Test constrained generation with partial constraints
- Verify importance weight computation

### Selection Testing

**Selection Combinators** (Example: `test_selection_*`):
- Test `sel("x") | sel("y")` (OR), `sel("x") & sel("y")` (AND)
- Test empty and all selections
- Verify selection behavior in regenerate context

### Tolerance Guidelines

**Floating Point Comparisons**:
```python
strict_tolerance = 1e-10      # For exact mathematical relationships
standard_tolerance = 1e-6     # For numerical computations
mcmc_tolerance = 1e-2         # For Monte Carlo methods
convergence_tolerance = 0.1   # For convergence tests
```

### Error Testing

**Exception Handling** (Example: `test_*_error_*`):
- Test invalid choice maps raise `KeyError`
- Test invalid arguments raise `ValueError`/`TypeError`
- Test beartype violations raise `TypeError`

## Test Suite Optimization

### Performance Analysis

**Key Metrics**:
- Total suite time can be reduced by 40-50% with JIT compilation
- Slowest files: `test_discrete_hmm.py` (7.93s), `test_distributions.py` (6.15s)
- See `analyze_test_timing.py` for identifying bottlenecks

### JIT Optimization Strategy

**Available Fixtures** (in `conftest.py`):
- `jit_compiler`: Cache and compile functions with `@jax.jit`
- `batch_tester`: Convert single tests to batch tests with `vmap`
- `jitted_distributions`: Pre-compiled distribution operations
- `jitted_hmm_ops`: Pre-compiled HMM operations

**Usage Pattern**:
```python
def test_something(jitted_distributions):
    # Use pre-compiled batch operations
    logprobs = jitted_distributions['binomial_batch'](samples, n, p)
```

**Best Practices**:
1. **Warm up JIT**: First call is slow, use fixtures for warmup
2. **Batch operations**: Use `vmap` for testing multiple inputs
3. **Static arguments**: Use `static_argnames` for unchanging parameters
4. **Cache compilation**: Module-scoped fixtures prevent recompilation

**References**:
- Implementation examples: `test_distributions_optimized.py`, `test_discrete_hmm_optimized.py`
- Benchmarking tool: `benchmark_test_optimization.py`
- Timing analysis: `analyze_test_timing.py`

## Critical Testing Requirements

1. **Always test after changes** to corresponding source files
2. **Ensure tests pass** before committing changes
3. **Add tests for new functionality** - don't just modify existing code
4. **Test edge cases** and error conditions
5. **Use appropriate tolerances** for floating-point comparisons with probabilistic algorithms
6. **Test convergence properties** - algorithms should improve with more compute
7. **MANDATORY for SMC**: Include log marginal likelihood convergence test in every SMC test suite
8. **Validate against analytical solutions** when available (conjugate models)
9. **Optimize slow tests**: Use JIT compilation fixtures for tests >0.5s
