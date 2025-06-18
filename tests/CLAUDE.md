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
├── test_mcmc.py         # Tests for src/genjax/mcmc.py (Metropolis-Hastings, HMC)
├── test_smc.py          # Tests for src/genjax/smc.py (Sequential Monte Carlo)
├── test_vi.py           # Tests for src/genjax/vi.py (Variational inference)
├── test_adev.py         # Tests for src/genjax/adev.py (Automatic differentiation)
├── discrete_hmm.py      # Discrete HMM test utilities
└── conftest.py          # Test configuration and shared fixtures
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

# =============================================================================
# GENERATIVE FUNCTION INTERFACE (GFI) METHOD TESTS
# =============================================================================

class TestGenerateConsistency:
    """Test generate method consistency."""
    pass

class TestUpdateAndRegenerate:
    """Test update and regenerate methods."""
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

## Testing Patterns

### Density Validation

**Basic Density Consistency**:

```python
def test_model_density_validation():
    """Test that simulate and assess are consistent."""
    # Generate trace
    trace = model.simulate(*args)
    choices = trace.get_choices()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Check simulate/assess consistency
    assess_density, assess_retval = model.assess(choices, *args)
    simulate_score = trace.get_score()

    # Score should be negative log density
    assert jnp.allclose(simulate_score, -assess_density, rtol=tolerance)
    assert jnp.allclose(trace.get_retval(), assess_retval, rtol=tolerance)
```

**Manual Density Computation**:

```python
def test_manual_density_computation():
    """Validate against manual probability calculations."""
    trace = model.simulate(*args)
    choices = trace.get_choices()

    # Compute expected density using Distribution.assess
    expected_log_density = 0.0
    for addr, choice_value in choices.items():
        dist_log_density, _ = distribution.assess(choice_value, *dist_params)
        expected_log_density += dist_log_density

    # Compare with model.assess
    actual_log_density, _ = model.assess(choices, *args)
    assert jnp.allclose(actual_log_density, expected_log_density, rtol=tolerance)
    assert jnp.allclose(trace.get_score(), -expected_log_density, rtol=tolerance)
```

### GFI Method Testing

**Update Method Testing**:

```python
def test_update_weight_invariant():
    """Test that update satisfies the weight invariant."""
    # Create initial trace
    initial_trace = model.simulate(*args)
    old_score = initial_trace.get_score()

    # Update with new arguments/constraints
    new_trace, weight, discarded = model.update(initial_trace, constraints, *new_args)
    new_score = new_trace.get_score()

    # Test weight invariant: weight = -new_score + old_score (for simple cases)
    expected_weight = -new_score + old_score
    assert jnp.allclose(weight, expected_weight, rtol=tolerance)

    # Validate traces
    helpers.assert_valid_trace(initial_trace)
    helpers.assert_valid_trace(new_trace)
```

**Regenerate Method Testing**:

```python
def test_regenerate_selective_resampling():
    """Test that regenerate only changes selected addresses."""
    # Create initial trace
    initial_trace = model.simulate(*args)
    old_choices = initial_trace.get_choices()
    old_score = initial_trace.get_score()

    # Regenerate subset of choices
    selection = sel("x") | sel("y")  # Select specific addresses
    new_trace, weight, discarded = model.regenerate(initial_trace, selection, *args)
    new_choices = new_trace.get_choices()
    new_score = new_trace.get_score()

    # Test that non-selected addresses remain unchanged
    for addr in old_choices:
        if not is_selected(addr, selection):
            assert jnp.allclose(new_choices[addr], old_choices[addr], rtol=1e-10)

    # Test that discarded contains old values of regenerated addresses
    for addr in ["x", "y"]:  # Selected addresses
        assert addr in discarded
        assert jnp.allclose(discarded[addr], old_choices[addr], rtol=tolerance)

    # Test weight invariant for regenerate
    expected_weight = -new_score + old_score  # Simplified for prior regeneration
    assert jnp.allclose(weight, expected_weight, rtol=tolerance)
```

**Generate Method Testing**:

```python
def test_generate_with_constraints():
    """Test generate method with partial constraints."""
    constraints = {"x": 1.5, "y": 2.0}

    # Generate with constraints
    trace, weight = model.generate(constraints, *args)
    choices = trace.get_choices()

    # Verify constraints are satisfied
    for addr, value in constraints.items():
        assert jnp.allclose(choices[addr], value, rtol=1e-10)

    # Verify weight computation (importance weight)
    # For fully constrained case: weight = log P(choices; args)
    expected_density, _ = model.assess(choices, *args)
    # Weight should account for constraint satisfaction
    assert jnp.isfinite(weight)
    assert jnp.isfinite(expected_density)
```

### Scan and Combinator Testing

**Scan with Const Pattern**:

```python
def test_scan_with_const_pattern():
    """Test scan using Const[...] pattern for static values."""
    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(step_fn, length=length.value)
        return scan_gf(init_carry, xs) @ "scan"

    # Test with static length parameter
    args = (const(4), init_carry, xs)

    # Test simulate with seed transformation
    trace = seed(scan_model.simulate)(key, *args)
    choices = trace.get_choices()

    # Validate addressing structure
    assert "scan" in choices
    assert "sample" in choices["scan"]  # From step_fn

    # Test simulate/assess consistency
    assess_density, assess_retval = scan_model.assess(choices, *args)
    assert jnp.allclose(trace.get_score(), -assess_density, rtol=tolerance)
```

### Selection Testing

**Selection Combinators**:

```python
def test_selection_combinations():
    """Test various selection combinations."""
    # Basic selections
    sel_x = sel("x")
    sel_y = sel("y")

    # Test OR combination
    or_selection = sel_x | sel_y
    assert or_selection.match("x")[0] is True
    assert or_selection.match("y")[0] is True
    assert or_selection.match("z")[0] is False

    # Test empty selection
    empty_selection = sel()
    assert empty_selection.match("x")[0] is False

    # Test in regenerate context
    new_trace, weight, discarded = model.regenerate(trace, or_selection, *args)
    assert "x" in discarded or "y" in discarded  # At least one regenerated
```

### Tolerance Guidelines

**Floating Point Comparisons**:

```python
# Standard tolerances for different test types
strict_tolerance = 1e-10      # For exact mathematical relationships
standard_tolerance = 1e-6     # For numerical computations
mcmc_tolerance = 1e-2         # For Monte Carlo methods
convergence_tolerance = 0.1   # For convergence tests

# Usage examples
assert jnp.allclose(computed_value, expected_value, rtol=standard_tolerance)
helpers.assert_finite_and_close(value1, value2, rtol=tolerance, msg="Custom message")
```

### Error Testing

**Exception Handling**:

```python
def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Test invalid choice maps
    with pytest.raises(KeyError):
        model.assess(invalid_choices, *args)

    # Test invalid arguments
    with pytest.raises((ValueError, TypeError)):
        model.simulate(*invalid_args)

    # Test beartype violations
    with pytest.raises(TypeError):
        model.assess(wrong_type_choices, *args)
```

## Inference Algorithm Testing Strategies

### Convergence Testing Philosophy

Inference algorithms should demonstrate **monotonic improvement** as computational resources increase. This principle applies across all GenJAX inference methods:

- **MCMC**: Longer chains → better posterior approximation
- **SMC**: More particles → better marginal likelihood estimates
- **Variational Inference**: More optimization steps → better ELBO convergence

### MCMC Convergence Testing

**Monotonic Chain Length Testing**:
```python
def test_mcmc_monotonic_convergence():
    """Test that longer MCMC chains produce better posterior estimates."""
    chain_lengths = [100, 500, 1000]  # Increasing compute
    mean_errors = []

    true_posterior_mean = 0.0  # Known analytical value

    for length in chain_lengths:
        result = run_mcmc_chain(length=length)
        samples = apply_burn_in(result.traces)

        sample_mean = jnp.mean(samples["parameter"])
        error = jnp.abs(sample_mean - true_posterior_mean)
        mean_errors.append(error)

    # Test for overall improvement trend
    short_error, medium_error, long_error = mean_errors

    # Allow for stochastic variation but require overall progress
    monotonic_improvement = (medium_error <= short_error) or (long_error <= medium_error)
    overall_improvement = long_error <= short_error * 1.5  # 50% tolerance

    assert monotonic_improvement or overall_improvement, (
        f"No convergence: errors {mean_errors} should show decreasing trend"
    )
```

**Multi-Chain Diagnostic Testing**:
```python
def test_mcmc_multi_chain_convergence():
    """Test convergence diagnostics improve with more chains."""
    chain_counts = [2, 4, 8]
    rhat_values = []

    for n_chains in chain_counts:
        result = run_mcmc_chain(n_chains=n_chains, n_steps=1000)
        rhat_values.append(result.rhat["parameter"])

    # R-hat should approach 1.0 with more chains (better convergence assessment)
    final_rhat = rhat_values[-1]
    assert final_rhat < 1.1, f"Poor convergence: R-hat = {final_rhat:.3f}"

    # ESS should be reasonable
    final_ess = result.ess_bulk["parameter"]
    assert final_ess > 100, f"Low effective sample size: {final_ess:.0f}"
```

**Algorithm-Specific Testing**:
```python
def test_mala_step_size_effects():
    """Test MALA acceptance rates vary appropriately with step size."""
    step_sizes = [0.01, 0.1, 1.0, 10.0]  # Small to large
    acceptance_rates = []

    for step_size in step_sizes:
        result = run_mala_chain(step_size=step_size)
        acceptance_rates.append(result.acceptance_rate)

    # Smaller step sizes should generally have higher acceptance
    small_acceptance = acceptance_rates[0]  # step_size=0.01
    large_acceptance = acceptance_rates[-1]  # step_size=10.0

    assert small_acceptance >= large_acceptance, (
        f"Step size ordering violated: {small_acceptance:.3f} vs {large_acceptance:.3f}"
    )

    # Very large step sizes should definitely reject some proposals
    assert large_acceptance < 1.0, "Large step size should not accept everything"
```

### SMC Convergence Testing

**Particle Count Scaling**:
```python
def test_smc_particle_scaling():
    """Test SMC accuracy improves with more particles."""
    particle_counts = [100, 500, 1000]
    log_marginal_errors = []

    true_log_marginal = compute_exact_log_marginal()  # Analytical value

    for n_particles in particle_counts:
        particles = run_smc(n_particles=n_particles)
        estimated_log_marginal = particles.log_marginal_likelihood()
        error = jnp.abs(estimated_log_marginal - true_log_marginal)
        log_marginal_errors.append(error)

    # More particles should reduce Monte Carlo error
    final_error = log_marginal_errors[-1]
    assert final_error < 1.0, f"SMC convergence too slow: error {final_error:.3f}"

    # Test overall improvement trend
    assert log_marginal_errors[-1] <= log_marginal_errors[0] * 1.5, (
        "No improvement with more particles"
    )
```

**Effective Sample Size Testing**:
```python
def test_smc_degeneracy_management():
    """Test SMC maintains particle diversity."""
    particles = run_smc_with_resampling(n_particles=1000)

    # ESS should not be too low (severe degeneracy)
    ess = particles.effective_sample_size()
    assert ess > 100, f"Severe particle degeneracy: ESS = {ess:.0f}"

    # Weight entropy should be reasonable
    weights = particles.get_weights()
    normalized_weights = weights / jnp.sum(weights)
    entropy = -jnp.sum(normalized_weights * jnp.log(normalized_weights + 1e-10))

    # High entropy indicates good particle diversity
    max_entropy = jnp.log(len(weights))
    relative_entropy = entropy / max_entropy
    assert relative_entropy > 0.1, f"Low particle diversity: {relative_entropy:.3f}"
```

### Variational Inference Convergence Testing

**ELBO Convergence**:
```python
def test_vi_elbo_monotonic_improvement():
    """Test ELBO improves monotonically during optimization."""
    n_steps_list = [100, 500, 1000]
    final_elbos = []

    for n_steps in n_steps_list:
        vi_result = run_variational_inference(n_steps=n_steps)
        final_elbos.append(vi_result.elbo_history[-1])

    # Longer optimization should achieve better ELBO
    short_elbo, medium_elbo, long_elbo = final_elbos

    assert long_elbo >= medium_elbo >= short_elbo, (
        f"ELBO not improving: {short_elbo:.3f} → {medium_elbo:.3f} → {long_elbo:.3f}"
    )

    # ELBO should converge (not still improving rapidly)
    elbo_history = vi_result.elbo_history
    final_improvement = elbo_history[-1] - elbo_history[-100]  # Last 100 steps
    assert final_improvement < 0.1, f"ELBO not converged: still improving by {final_improvement:.3f}"
```

### Cross-Algorithm Validation

**Posterior Moment Comparison**:
```python
def test_inference_algorithm_consistency():
    """Test different algorithms converge to same posterior."""
    # Run multiple inference methods on same model
    mcmc_result = run_mcmc_chain(n_steps=5000, burn_in=1000)
    smc_result = run_smc(n_particles=1000)
    vi_result = run_variational_inference(n_steps=1000)

    # Extract posterior samples/approximations
    mcmc_samples = mcmc_result.traces.get_choices()["parameter"]
    smc_samples = smc_result.traces.get_choices()["parameter"]
    vi_samples = vi_result.sample_posterior(n_samples=1000)

    # Compare posterior moments
    mcmc_mean = jnp.mean(mcmc_samples)
    smc_mean = jnp.mean(smc_samples)
    vi_mean = jnp.mean(vi_samples)

    # All methods should agree on posterior mean (within tolerance)
    tolerance = 0.1
    assert jnp.abs(mcmc_mean - smc_mean) < tolerance
    assert jnp.abs(mcmc_mean - vi_mean) < tolerance
    assert jnp.abs(smc_mean - vi_mean) < tolerance
```

### Testing Guidelines for Inference Algorithms

**Computational Resource Scaling**:
1. **Always test with increasing compute**: More chains, longer chains, more particles, more optimization steps
2. **Expect monotonic improvement**: Errors should decrease (with reasonable tolerance for stochasticity)
3. **Test algorithm-specific properties**: Step size effects (MALA), particle degeneracy (SMC), ELBO convergence (VI)
4. **Validate against known solutions**: Use conjugate models with analytical posteriors when possible

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

## Critical Testing Requirements

1. **Always test after changes** to corresponding source files
2. **Ensure tests pass** before committing changes
3. **Add tests for new functionality** - don't just modify existing code
4. **Test edge cases** and error conditions
5. **Use appropriate tolerances** for floating-point comparisons with probabilistic algorithms
6. **Test convergence properties** - algorithms should improve with more compute
7. **Validate against analytical solutions** when available (conjugate models)
