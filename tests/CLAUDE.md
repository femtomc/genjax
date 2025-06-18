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

## Critical Testing Requirements

1. **Always test after changes** to corresponding source files
2. **Ensure tests pass** before committing changes
3. **Add tests for new functionality** - don't just modify existing code
4. **Test edge cases** and error conditions
5. **Use appropriate tolerances** for floating-point comparisons with probabilistic algorithms
