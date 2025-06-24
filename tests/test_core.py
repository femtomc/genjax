"""
Test cases for GenJAX core functionality.

These tests validate the core GenJAX components including:
- Generative functions (@gen)
- Scan and other combinators
- Distribution classes
- Pytree functionality
- Trace operations
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
from jax.lax import scan
import tensorflow_probability.substrates.jax as tfp

from genjax.core import (
    gen,
    Scan,
    Cond,
    sel,
    Const,
    const,
    distribution,
    Pytree,
    tfp_distribution,
)
from genjax.pjax import seed, modular_vmap
from genjax.distributions import normal, exponential, flip


# =============================================================================
# GENERATIVE FUNCTION (@gen) TESTS
# =============================================================================


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_simulate_vs_manual_density(
    bivariate_normal_model, standard_tolerance, helpers
):
    """Test that @gen Fn.simulate produces correct densities compared to manual computation."""
    # Generate trace
    trace = bivariate_normal_model.simulate()
    choices = trace.get_choices()
    fn_score = trace.get_score()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Extract individual choices
    x_choice = choices["x"]
    y_choice = choices["y"]

    # Compute densities using Distribution.assess
    x_log_density, _ = normal.assess(x_choice, 0.0, 1.0)
    y_log_density, _ = normal.assess(y_choice, x_choice, 0.5)

    manual_total_log_density = x_log_density + y_log_density
    expected_fn_score = -manual_total_log_density

    helpers.assert_finite_and_close(
        fn_score,
        expected_fn_score,
        rtol=standard_tolerance,
        msg="Fn score does not match manual computation",
    )

    # Verify return value
    expected_retval = (x_choice, y_choice)
    assert jnp.allclose(
        trace.get_retval()[0], expected_retval[0], rtol=standard_tolerance
    )
    assert jnp.allclose(
        trace.get_retval()[1], expected_retval[1], rtol=standard_tolerance
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_assess_vs_manual_density(standard_tolerance, helpers):
    """Test that @gen Fn.assess produces correct densities compared to manual computation."""

    @gen
    def exponential_fn(rate1, rate2):
        x = exponential(rate1) @ "x"
        y = exponential(rate2 * x) @ "y"  # rate depends on x
        return x * y

    # Test parameters
    rate1 = 2.0
    rate2 = 0.5
    args = (rate1, rate2)

    # Fixed choices to assess
    choices = {"x": 0.8, "y": 1.2}

    # Assess using Fn
    fn_density, fn_retval = exponential_fn.assess(choices, *args)

    # Manual computation
    x_val = choices["x"]
    y_val = choices["y"]

    # Compute densities using Distribution.assess
    x_log_density, _ = exponential.assess(x_val, rate1)
    y_rate = rate2 * x_val
    y_log_density, _ = exponential.assess(y_val, y_rate)

    manual_total_log_density = x_log_density + y_log_density
    manual_retval = x_val * y_val

    helpers.assert_finite_and_close(
        fn_density,
        manual_total_log_density,
        rtol=standard_tolerance,
        msg="Fn density does not match manual computation",
    )
    helpers.assert_finite_and_close(
        fn_retval,
        manual_retval,
        rtol=standard_tolerance,
        msg="Fn return value does not match manual computation",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_simulate_assess_consistency(standard_tolerance, helpers):
    """Test that @gen Fn simulate and assess are consistent."""

    @gen
    def complex_fn(mu, sigma):
        # Chain of dependencies
        x = normal(mu, sigma) @ "x"
        y = normal(x * 0.5, 1.0) @ "y"
        z = exponential(jnp.exp(y * 0.1)) @ "z"
        return (x, y, z)

    args = (2.0, 0.8)

    # Generate trace
    trace = complex_fn.simulate(*args)
    choices = trace.get_choices()
    simulate_score = trace.get_score()
    simulate_retval = trace.get_retval()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Assess same choices
    assess_density, assess_retval = complex_fn.assess(choices, *args)
    helpers.assert_valid_density(assess_density)

    # Should be consistent
    expected_score = -assess_density
    helpers.assert_finite_and_close(
        simulate_score,
        expected_score,
        rtol=standard_tolerance,
        msg="Simulate score inconsistent with assess density",
    )

    # Return values should match
    helpers.assert_finite_and_close(
        simulate_retval[0],
        assess_retval[0],
        rtol=standard_tolerance,
        msg="Return value x component mismatch",
    )
    helpers.assert_finite_and_close(
        simulate_retval[1],
        assess_retval[1],
        rtol=standard_tolerance,
        msg="Return value y component mismatch",
    )
    helpers.assert_finite_and_close(
        simulate_retval[2],
        assess_retval[2],
        rtol=standard_tolerance,
        msg="Return value z component mismatch",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_nested_addressing(standard_tolerance, helpers):
    """Test @gen functions with nested addressing and data flow."""

    @gen
    def inner_fn(scale):
        return normal(0.0, scale) @ "inner_sample"

    @gen
    def outer_fn():
        x = normal(1.0, 0.5) @ "x"
        y = inner_fn(jnp.abs(x)) @ "inner"  # scale depends on x
        z = normal(y, 0.2) @ "z"  # location depends on y
        return x + y + z

    # Generate trace
    trace = outer_fn.simulate()
    choices = trace.get_choices()
    fn_score = trace.get_score()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Extract nested choices
    x_choice = choices["x"]
    y_choice = choices["inner"]["inner_sample"]  # This is the inner function's result
    z_choice = choices["z"]

    # Compute densities using Distribution.assess
    # x ~ normal(1.0, 0.5)
    x_log_density, _ = normal.assess(x_choice, 1.0, 0.5)

    # y ~ normal(0.0, abs(x)) (from inner function)
    y_scale = jnp.abs(x_choice)
    y_log_density, _ = normal.assess(y_choice, 0.0, y_scale)

    # z ~ normal(y, 0.2)
    z_log_density, _ = normal.assess(z_choice, y_choice, 0.2)

    manual_total_log_density = x_log_density + y_log_density + z_log_density
    expected_fn_score = -manual_total_log_density

    helpers.assert_finite_and_close(
        fn_score,
        expected_fn_score,
        rtol=standard_tolerance,
        msg="Nested Fn score does not match manual computation",
    )

    # Verify return value
    expected_retval = x_choice + y_choice + z_choice
    helpers.assert_finite_and_close(
        trace.get_retval(),
        expected_retval,
        rtol=standard_tolerance,
        msg="Nested function return value mismatch",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_with_deterministic_computation(standard_tolerance, helpers):
    """Test @gen functions with deterministic computations mixed with sampling."""

    @gen
    def mixed_fn(base):
        # Deterministic computation
        scaled_base = base * 2.0
        offset = jnp.sin(scaled_base)

        # Probabilistic computation
        x = normal(offset, 1.0) @ "x"

        # More deterministic computation using random variable
        processed = x**2 + jnp.cos(x)

        # More probabilistic computation
        y = exponential(1.0 / (jnp.abs(processed) + 0.1)) @ "y"

        return processed + y

    args = (0.5,)

    # Test consistency
    trace = mixed_fn.simulate(*args)
    choices = trace.get_choices()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    assess_density, assess_retval = mixed_fn.assess(choices, *args)
    helpers.assert_valid_density(assess_density)
    simulate_score = trace.get_score()

    helpers.assert_finite_and_close(
        simulate_score,
        -assess_density,
        rtol=standard_tolerance,
        msg="Mixed function simulate/assess inconsistency",
    )
    helpers.assert_finite_and_close(
        trace.get_retval(),
        assess_retval,
        rtol=standard_tolerance,
        msg="Mixed function return value mismatch",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_empty_program(standard_tolerance, helpers):
    """Test @gen function with no probabilistic choices."""

    @gen
    def deterministic_fn(x, y):
        # Only deterministic computation
        result = x * y + jnp.sin(x) - jnp.cos(y)
        return result

    args = (2.0, 3.0)

    # Should work and have zero score (no probabilistic choices)
    trace = deterministic_fn.simulate(*args)
    choices = trace.get_choices()

    assert trace.get_score() == 0.0, "Deterministic function should have zero score"
    assert len(choices) == 0, "Deterministic function should have no choices"

    # Should compute same result deterministically
    expected_result = 2.0 * 3.0 + jnp.sin(2.0) - jnp.cos(3.0)
    helpers.assert_finite_and_close(
        trace.get_retval(),
        expected_result,
        rtol=standard_tolerance,
        msg="Deterministic function return value mismatch",
    )

    # Assess should also work
    density, retval = deterministic_fn.assess({}, *args)
    assert density == 0.0, "Deterministic assess should have zero density"
    helpers.assert_finite_and_close(
        retval,
        expected_result,
        rtol=standard_tolerance,
        msg="Deterministic assess return value mismatch",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_fn_conditional_sampling(standard_tolerance, helpers):
    """Test @gen function with conditional sampling patterns using Cond combinator."""

    # Define the two branches as separate @gen functions
    @gen
    def high_branch(x):
        y = exponential(2.0) @ "y_high"
        return x + y

    @gen
    def low_branch(x):
        y = exponential(2.0) @ "y_low"
        return x + y

    @gen
    def conditional_fn(threshold):
        x = normal(0.0, 1.0) @ "x"

        # Use Cond combinator for conditional logic
        condition = x > threshold
        cond_gf = Cond(high_branch, low_branch)
        result = cond_gf(condition, x) @ "cond"
        return result

    # Test both branches
    args_high = (-1.0,)  # threshold low, likely to take first branch
    args_low = (1.0,)  # threshold high, likely to take second branch

    for args in [args_high, args_low]:
        trace = conditional_fn.simulate(*args)
        choices = trace.get_choices()

        # Validate trace structure
        helpers.assert_valid_trace(trace)

        # Should be consistent between simulate and assess
        assess_density, assess_retval = conditional_fn.assess(choices, *args)
        helpers.assert_valid_density(assess_density)
        simulate_score = trace.get_score()

        helpers.assert_finite_and_close(
            -simulate_score,
            assess_density,
            rtol=standard_tolerance,
            msg="Conditional function simulate/assess inconsistency",
        )
        helpers.assert_finite_and_close(
            trace.get_retval(),
            assess_retval,
            rtol=standard_tolerance,
            msg="Conditional function return value mismatch",
        )


# =============================================================================
# COND COMBINATOR TESTS
# =============================================================================


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.regression
def test_cond_update_with_vmap_regression(base_key, standard_tolerance, helpers):
    """Regression test for Cond.update bug with vmap (Issue #XXX).
    
    This test verifies that Cond.update works correctly in vectorized contexts.
    The bug was that jnp.select was used incorrectly with scalar conditions.
    """
    
    # Define a simple conditional model
    @gen
    def branch_a(value):
        return normal(value, 0.1) @ "obs"
    
    @gen
    def branch_b(value):
        return normal(value, 1.0) @ "obs"
    
    @gen
    def conditional_point(value, use_branch_a):
        cond_model = Cond(branch_b, branch_a)
        return cond_model(use_branch_a, value) @ "result"
    
    @gen
    def vectorized_model(values, conditions):
        # Vectorize the conditional model
        results = conditional_point.vmap(in_axes=(0, 0))(
            values, conditions
        ) @ "points"
        return results
    
    # Test data
    n_points = 3
    values = jnp.array([1.0, 2.0, 3.0])
    conditions = jnp.array([True, False, True])
    
    # Generate initial trace
    trace = seed(vectorized_model.simulate)(base_key, values, conditions)
    helpers.assert_valid_trace(trace)
    
    # Create new observations for update
    new_obs = jnp.array([1.1, 2.2, 2.9])
    constraints = {"points": {"result": {"obs": new_obs}}}
    
    # This should NOT raise TypeError anymore
    new_trace, weight, discard = vectorized_model.update(
        trace, constraints, values, conditions
    )
    
    # Verify the update worked correctly
    assert new_trace is not None
    assert weight.shape == ()  # Should be a scalar
    helpers.assert_valid_density(weight)
    
    # Check that the new observations were incorporated
    new_choices = new_trace.get_choices()
    updated_obs = new_choices["points"]["result"]["obs"]
    helpers.assert_finite_and_close(
        updated_obs, new_obs, rtol=standard_tolerance,
        msg="Updated observations should match constraints"
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_cond_with_same_addresses_in_branches(base_key, standard_tolerance, helpers):
    """Test that Cond works correctly with same addresses in both branches.
    
    This is the mixture model pattern that should be supported.
    """
    
    @gen
    def normal_component(x, mean):
        return normal(mean, 0.1) @ "y"  # Same address in both branches!
    
    @gen 
    def outlier_component(x, mean):
        return normal(mean, 1.0) @ "y"  # Same address in both branches!
    
    @gen
    def mixture_model(x, outlier_prob=0.1):
        mean = normal(0.0, 1.0) @ "mean"
        is_outlier = flip(outlier_prob) @ "is_outlier"
        
        # Conditional with same addresses
        cond = Cond(outlier_component, normal_component)
        observation = cond(is_outlier, x, mean) @ "obs"
        
        return observation
    
    # Test simulate
    trace = seed(mixture_model.simulate)(base_key, 1.0)
    choices = trace.get_choices()
    helpers.assert_valid_trace(trace)
    
    # Verify we have all expected addresses
    assert "mean" in choices
    assert "is_outlier" in choices
    assert "obs" in choices
    assert "y" in choices["obs"]  # The observation from the selected branch
    
    # Test assess - should handle same addresses correctly
    density, retval = mixture_model.assess(choices, 1.0)
    helpers.assert_valid_density(density)
    
    # Test update
    new_choices = {"obs": {"y": 1.5}}
    new_trace, weight, discard = mixture_model.update(trace, new_choices, 1.0)
    helpers.assert_valid_density(weight)
    
    # Verify the observation was updated
    updated_choices = new_trace.get_choices()
    helpers.assert_finite_and_close(
        updated_choices["obs"]["y"], 1.5, rtol=standard_tolerance,
        msg="Updated observation should match constraint"
    )


# =============================================================================
# SCAN COMBINATOR TESTS
# =============================================================================


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_scan_simulate_vs_manual_density(base_key, standard_tolerance, helpers):
    """Test that Scan.simulate produces correct densities compared to manual computation."""

    # Create a simple callee that adds a normal sample to the carry
    @gen
    def add_normal_step(carry, input_val):
        noise = normal(0.0, 1.0) @ "noise"
        new_carry = carry + noise + input_val
        output = new_carry * 2.0
        return new_carry, output

    # Create scan generative function with Const[int] for static length
    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(add_normal_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    # Test parameters
    init_carry = 1.0
    xs = jnp.array([0.5, -0.2, 1.1, 0.8])
    args = (const(4), init_carry, xs)

    # Generate a trace using simulate with seed transformation
    trace = seed(scan_model.simulate)(base_key, *args)
    choices = trace.get_choices()
    scan_score = trace.get_score()  # This is log(1/density), so negative log density

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Manually compute the density using the same choices
    def manual_scan_fn(carry, input_and_choice):
        input_val, choice = input_and_choice
        noise = choice["noise"]  # Use the same sampled value
        new_carry = carry + noise + input_val
        output = new_carry * 2.0

        # Compute log density using Distribution.assess
        log_density, _ = normal.assess(noise, 0.0, 1.0)
        return new_carry, (output, log_density)

    # Run manual scan with the same choices
    final_carry, (outputs, log_densities) = scan(
        manual_scan_fn,
        init_carry,
        (xs, choices["scan"]),
        length=len(xs),
    )

    # Sum log densities to get total log density
    manual_total_log_density = jnp.sum(log_densities)

    # scan_score should be -manual_total_log_density (since score is log(1/density))
    expected_scan_score = -manual_total_log_density

    # Compare with tolerance for numerical precision
    helpers.assert_finite_and_close(
        scan_score,
        expected_scan_score,
        rtol=standard_tolerance,
        msg="Scan score does not match manual computation",
    )

    # Also verify the outputs match
    trace_outputs = trace.get_retval()[1]
    helpers.assert_finite_and_close(
        trace_outputs,
        outputs,
        rtol=standard_tolerance,
        msg="Scan outputs do not match manual computation",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_scan_assess_vs_manual_density(standard_tolerance, helpers):
    """Test that Scan.assess produces correct densities compared to manual computation."""

    # Create a callee that uses exponential distribution
    @gen
    def exponential_step(carry, input_val):
        rate = jnp.exp(carry + input_val)  # Ensure positive rate
        sample = exponential(rate) @ "exp_sample"
        new_carry = carry + sample * 0.1  # Small update to carry
        output = sample + input_val
        return new_carry, output

    # Create scan generative function with Const[int] for static length
    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(exponential_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    # Test parameters
    init_carry = 0.5
    xs = jnp.array([0.1, 0.3, -0.2])
    args = (const(3), init_carry, xs)

    # Create some fixed choices to assess
    choices = {"scan": {"exp_sample": jnp.array([0.8, 1.2, 0.4])}}

    # Assess using Scan
    scan_density, scan_retval = scan_model.assess(choices, *args)
    helpers.assert_valid_density(scan_density)

    # Manually compute density
    def manual_assess_fn(carry, input_and_choice):
        input_val, choice = input_and_choice
        rate = jnp.exp(carry + input_val)
        sample = choice["exp_sample"]
        new_carry = carry + sample * 0.1
        output = sample + input_val

        # Compute log density using Distribution.assess
        log_density, _ = exponential.assess(sample, rate)
        return new_carry, (output, log_density)

    # Run manual assessment
    final_carry, (outputs, log_densities) = scan(
        manual_assess_fn,
        init_carry,
        (xs, choices["scan"]),
        length=len(xs),
    )

    manual_total_log_density = jnp.sum(log_densities)
    manual_retval = (final_carry, outputs)

    # Compare densities (scan_density should equal manual_total_log_density)
    helpers.assert_finite_and_close(
        scan_density,
        manual_total_log_density,
        rtol=standard_tolerance,
        msg="Scan density does not match manual computation",
    )

    # Compare return values
    helpers.assert_finite_and_close(
        scan_retval[0],
        manual_retval[0],
        rtol=standard_tolerance,
        msg="Scan final carry does not match manual computation",
    )
    helpers.assert_finite_and_close(
        scan_retval[1],
        manual_retval[1],
        rtol=standard_tolerance,
        msg="Scan outputs do not match manual computation",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_scan_simulate_assess_consistency(base_key, standard_tolerance, helpers):
    """Test that simulate and assess are consistent for the same choices."""

    # Create a more complex callee
    @gen
    def complex_step(carry, input_val):
        # Use carry to parameterize distributions
        loc = carry * 0.5
        scale = jnp.exp(input_val * 0.2) + 0.1  # Ensure positive scale

        sample1 = normal(loc, scale) @ "normal_sample"
        sample2 = exponential(1.0 / (jnp.abs(sample1) + 0.1)) @ "exp_sample"

        new_carry = carry + sample1 * 0.3 + sample2 * 0.1
        output = (sample1, sample2)
        return new_carry, output

    # Create scan generative function with Const[int] for static length
    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(complex_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    # Test parameters
    init_carry = 0.2
    xs = jnp.array([0.5, -0.3, 0.8, 1.0])
    args = (const(4), init_carry, xs)

    # Generate trace with simulate using seed transformation
    trace = seed(scan_model.simulate)(base_key, *args)
    choices = trace.get_choices()
    simulate_score = trace.get_score()
    simulate_retval = trace.get_retval()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Assess the same choices
    assess_density, assess_retval = scan_model.assess(choices, *args)
    helpers.assert_valid_density(assess_density)

    # simulate_score should be -assess_density (score is log(1/density))
    expected_score = -assess_density

    helpers.assert_finite_and_close(
        simulate_score,
        expected_score,
        rtol=standard_tolerance,
        msg="Scan simulate/assess score inconsistency",
    )

    # Return values should be identical
    helpers.assert_finite_and_close(
        simulate_retval[0],
        assess_retval[0],
        rtol=standard_tolerance,
        msg="Scan final carries do not match",
    )
    helpers.assert_finite_and_close(
        simulate_retval[1][0],
        assess_retval[1][0],
        rtol=standard_tolerance,
        msg="Scan normal outputs do not match",
    )
    helpers.assert_finite_and_close(
        simulate_retval[1][1],
        assess_retval[1][1],
        rtol=standard_tolerance,
        msg="Scan exponential outputs do not match",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_empty_scan(helpers):
    """Test scan with empty input sequence."""

    @gen
    def simple_step(carry, input_val):
        sample = normal(0.0, 1.0) @ "sample"
        return carry + sample, sample

    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(simple_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    # Empty inputs
    init_carry = 1.0
    xs = jnp.array([])  # Empty array
    args = (const(0), init_carry, xs)

    # Should work and return initial carry with empty outputs
    trace = scan_model.simulate(*args)
    final_carry, outputs = trace.get_retval()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    assert jnp.allclose(final_carry, init_carry), (
        "Final carry should equal initial carry"
    )
    assert outputs.shape[0] == 0, "Outputs should be empty"
    assert trace.get_score() == 0.0, "Score should be 0 for empty scan"


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
def test_single_step_scan(base_key, standard_tolerance, helpers):
    """Test scan with single step."""

    @gen
    def single_step(carry, input_val):
        sample = normal(input_val, 0.5) @ "sample"
        new_carry = carry + sample
        return new_carry, sample**2

    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(single_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    # Single input
    init_carry = 2.0
    xs = jnp.array([1.5])
    args = (const(1), init_carry, xs)

    # Test simulate with seed transformation
    trace = seed(scan_model.simulate)(base_key, *args)
    choices = trace.get_choices()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Test assess with same choice
    density, retval = scan_model.assess(choices, *args)
    helpers.assert_valid_density(density)

    # Compute expected density using Distribution.assess
    sample = choices["scan"]["sample"][0]
    expected_log_density, _ = normal.assess(sample, 1.5, 0.5)

    helpers.assert_finite_and_close(
        density,
        expected_log_density,
        rtol=standard_tolerance,
        msg="Single step scan density mismatch",
    )


@pytest.mark.core
@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("length", [1, 3, 5, 10])
def test_scan_with_different_lengths(length, base_key, standard_tolerance, helpers):
    """Test scan with various sequence lengths."""

    @gen
    def accumulating_step(carry, input_val):
        sample = normal(0.0, 0.1) @ "sample"  # Small noise
        new_carry = carry + input_val + sample
        return new_carry, new_carry

    @gen
    def scan_model(length: Const[int], init_carry, xs):
        scan_gf = Scan(accumulating_step, length=length)
        return scan_gf(init_carry, xs) @ "scan"

    init_carry = 0.0
    xs = jnp.ones(length) * 0.1  # Small constant inputs
    args = (const(length), init_carry, xs)

    # Test that simulate and assess are consistent
    # Use different key for each length to ensure varied test conditions
    key_list = jrand.split(base_key, 2)
    test_key = key_list[0] if length % 2 == 0 else key_list[1]
    trace = seed(scan_model.simulate)(test_key, *args)
    choices = trace.get_choices()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    assess_density, assess_retval = scan_model.assess(choices, *args)
    helpers.assert_valid_density(assess_density)
    simulate_score = trace.get_score()

    helpers.assert_finite_and_close(
        simulate_score,
        -assess_density,
        rtol=standard_tolerance,
        msg=f"Scan length {length} simulate/assess inconsistency",
    )

    # Check that we have the right number of choices and outputs
    assert choices["scan"]["sample"].shape[0] == length, (
        f"Wrong number of choices: {choices['scan']['sample'].shape[0]} != {length}"
    )
    assert trace.get_retval()[1].shape[0] == length, "Wrong number of outputs"


# =============================================================================
# GENERATIVE FUNCTION INTERFACE (GFI) METHOD TESTS
# =============================================================================


class TestGenerateConsistency:
    """Test that generate method is consistent with assess when given full samples."""

    def test_distribution_generate_assess_consistency(self, strict_tolerance, helpers):
        """Test that Distribution.generate weight equals assess density for full samples."""
        # Test with normal distribution
        args = (0.0, 1.0)  # mu=0.0, sigma=1.0
        sample_value = 1.5

        # Test generate with full sample
        trace, weight = normal.generate(sample_value, *args)
        helpers.assert_valid_trace(trace)

        # Test assess with same sample
        density, retval = normal.assess(sample_value, *args)
        helpers.assert_valid_density(density)

        # For distributions: generate weight should equal assess density
        helpers.assert_finite_and_close(
            weight,
            density,
            rtol=strict_tolerance,
            msg="Generate weight does not match assess density",
        )

        # Check that trace score equals negative density
        helpers.assert_finite_and_close(
            trace.get_score(),
            -density,
            rtol=strict_tolerance,
            msg="Trace score inconsistent with density",
        )

        # Check return values match
        helpers.assert_finite_and_close(
            retval,
            sample_value,
            rtol=strict_tolerance,
            msg="Assess return value mismatch",
        )
        helpers.assert_finite_and_close(
            trace.get_retval(),
            sample_value,
            rtol=strict_tolerance,
            msg="Trace return value mismatch",
        )

    def test_fn_generate_assess_consistency_simple(self, strict_tolerance, helpers):
        """Test that Fn.generate weight equals assess density for full samples."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        # Create full sample (covering all random choices)
        full_sample = {"x": 1.0, "y": 2.0}
        args = ()

        # Test generate with full sample
        trace, weight = simple_model.generate(full_sample, *args)
        helpers.assert_valid_trace(trace)

        # Test assess with same sample
        density, retval = simple_model.assess(full_sample, *args)
        helpers.assert_valid_density(density)

        # Generate weight should equal assess density for full samples
        helpers.assert_finite_and_close(
            weight,
            density,
            rtol=strict_tolerance,
            msg="Generate weight does not match assess density for simple model",
        )

        # Check that trace score equals negative density
        helpers.assert_finite_and_close(
            trace.get_score(),
            -density,
            rtol=strict_tolerance,
            msg="Trace score inconsistent with density for simple model",
        )

        # Check return values match
        expected_retval = 1.0 + 2.0
        helpers.assert_finite_and_close(
            retval,
            expected_retval,
            rtol=strict_tolerance,
            msg="Assess return value mismatch for simple model",
        )
        helpers.assert_finite_and_close(
            trace.get_retval(),
            expected_retval,
            rtol=strict_tolerance,
            msg="Trace return value mismatch for simple model",
        )

    def test_fn_generate_assess_consistency_hierarchical(
        self, strict_tolerance, helpers
    ):
        """Test generate/assess consistency for hierarchical model."""

        @gen
        def hierarchical_model(prior_mean, prior_std, obs_std):
            mu = normal(prior_mean, prior_std) @ "mu"
            y = normal(mu, obs_std) @ "y"
            return y

        # Model parameters
        args = (0.0, 1.0, 0.5)  # prior_mean=0.0, prior_std=1.0, obs_std=0.5

        # Create full sample
        full_sample = {"mu": 1.5, "y": 2.0}

        # Test generate with full sample
        trace, weight = hierarchical_model.generate(full_sample, *args)
        helpers.assert_valid_trace(trace)

        # Test assess with same sample
        density, retval = hierarchical_model.assess(full_sample, *args)
        helpers.assert_valid_density(density)

        # Generate weight should equal assess density
        helpers.assert_finite_and_close(
            weight,
            density,
            rtol=strict_tolerance,
            msg="Generate weight does not match assess density for hierarchical model",
        )

        # Check consistency of scores
        helpers.assert_finite_and_close(
            trace.get_score(),
            -density,
            rtol=strict_tolerance,
            msg="Trace score inconsistent with density for hierarchical model",
        )

    def test_fn_generate_assess_consistency_with_scan(self):
        """Test generate/assess consistency for model with Scan combinator."""

        @gen
        def step_function(carry, x):
            sample = normal(carry, 0.1) @ "sample"
            return sample, sample

        @gen
        def scan_model():
            init_carry = 0.0
            inputs = jnp.array([0.1, 0.2, 0.3])
            scan_gf = Scan(step_function, length=const(3))
            final_carry, outputs = scan_gf(init_carry, inputs) @ "scan_result"
            return outputs

        # Create full sample matching the scan structure
        full_sample = {
            "scan_result": {
                "sample": jnp.array([0.05, 0.15, 0.25])  # 3 samples for length=3
            }
        }
        args = ()

        # Test generate with full sample
        trace, weight = scan_model.generate(full_sample, *args)

        # Test assess with same sample
        density, retval = scan_model.assess(full_sample, *args)

        # Generate weight should equal assess density
        assert jnp.allclose(weight, density, rtol=1e-6), (
            f"Generate weight {weight} != assess density {density}"
        )

    def test_distribution_generate_with_partial_sample(self):
        """Test that Distribution.generate handles None (no constraints) correctly."""
        args = (0.0, 1.0)  # mu=0.0, sigma=1.0

        # Test generate with None (simulate)
        trace, weight = normal.generate(None, *args)

        # Weight should be 0.0 when no constraints are provided (pure simulation)
        assert jnp.allclose(weight, 0.0, rtol=1e-10), (
            f"Generate weight with None should be 0.0, got {weight}"
        )

        # Score should be negative log density
        sample = trace.get_choices()
        density, _ = normal.assess(sample, *args)
        assert jnp.allclose(trace.get_score(), -density, rtol=1e-10)

    def test_fn_generate_with_partial_sample(self):
        """Test that Fn.generate handles partial samples correctly."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        # Test with partial sample (only y constrained)
        partial_sample = {"y": 2.0}
        args = ()

        # Generate should fill in missing choices and return appropriate weight
        trace, weight = simple_model.generate(partial_sample, *args)
        choices = trace.get_choices()

        # Should have both x and y in choices
        assert "x" in choices, "Missing x choice"
        assert "y" in choices, "Missing y choice"
        assert jnp.allclose(choices["y"], 2.0, rtol=1e-10), "y should match constraint"

        # Weight should be the density of the constrained variable (y)
        # given the generated value of x
        x_val = choices["x"]
        y_density, _ = normal.assess(2.0, x_val, 0.5)
        assert jnp.allclose(weight, y_density, rtol=1e-6), (
            f"Weight {weight} should equal y density {y_density}"
        )


class TestPytreeAndDataClasses:
    """Test Pytree functionality and dataclass integration."""

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_pytree_dataclass_basic(self, helpers):
        """Test basic Pytree dataclass functionality."""

        @Pytree.dataclass
        class SimpleModel(Pytree):
            param1: float
            param2: int = Pytree.static()
            param3: str = Pytree.static(default="default")

        # Create instance
        model = SimpleModel(param1=3.14, param2=42, param3="test")

        # Test that it's a valid pytree
        flat, treedef = jax.tree_util.tree_flatten(model)
        assert len(flat) == 1  # Only param1 should be flattened (dynamic)
        assert flat[0] == 3.14

        # Test reconstruction
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed.param1 == model.param1
        assert reconstructed.param2 == model.param2
        assert reconstructed.param3 == model.param3

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_pytree_static_vs_dynamic_fields(self):
        """Test distinction between static and dynamic fields."""

        @Pytree.dataclass
        class MixedModel(Pytree):
            weights: jnp.ndarray  # Dynamic - can be traced
            learning_rate: float = Pytree.static()  # Static - constant
            name: str = Pytree.static(default="model")

        weights = jnp.array([1.0, 2.0, 3.0])
        model = MixedModel(weights=weights, learning_rate=0.01, name="test")

        # Test JAX transformations only affect dynamic fields
        def scale_weights(model, factor):
            return MixedModel(
                weights=model.weights * factor,
                learning_rate=model.learning_rate,
                name=model.name,
            )

        # Should work with vmap
        factors = jnp.array([1.0, 2.0, 3.0])
        scaled_models = jax.vmap(scale_weights, in_axes=(None, 0))(model, factors)

        assert scaled_models.weights.shape == (3, 3)  # Vectorized over factors
        assert jnp.allclose(scaled_models.weights[0], weights * 1.0)
        assert jnp.allclose(scaled_models.weights[1], weights * 2.0)
        assert jnp.allclose(scaled_models.weights[2], weights * 3.0)

        # Static fields should remain unchanged across all instances
        assert scaled_models.learning_rate == 0.01
        assert scaled_models.name == "test"

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_pytree_field_vs_static_annotations(self):
        """Test different field annotation types."""

        @Pytree.dataclass
        class AnnotationTest(Pytree):
            dynamic_field: float  # No annotation = dynamic
            explicit_field: int = Pytree.field()  # Explicit dynamic
            static_field: str = Pytree.static()  # Static
            static_with_default: bool = Pytree.static(default=True)

        instance = AnnotationTest(
            dynamic_field=1.5,
            explicit_field=42,
            static_field="static",
            static_with_default=False,
        )

        # Check pytree flattening
        flat, treedef = jax.tree_util.tree_flatten(instance)
        assert len(flat) == 2  # dynamic_field and explicit_field
        assert 1.5 in flat
        assert 42 in flat

        # Verify reconstruction preserves static fields
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert reconstructed.static_field == "static"
        assert not reconstructed.static_with_default


class TestDistributionClass:
    """Test Distribution class functionality and TFP integration."""

    def test_distribution_basic_functionality(self):
        """Test basic Distribution operations."""

        # Create simple normal distribution
        def sample_normal(mu, sigma):
            return jrand.normal(jrand.key(42)) * sigma + mu

        def logpdf_normal(x, mu, sigma):
            return (
                -0.5 * ((x - mu) / sigma) ** 2
                - jnp.log(sigma)
                - 0.5 * jnp.log(2 * jnp.pi)
            )

        normal_dist = distribution(sample_normal, logpdf_normal, name="normal")

        # Test simulate
        args = (0.0, 1.0)
        trace = normal_dist.simulate(*args)

        assert hasattr(trace, "get_choices")
        assert hasattr(trace, "get_score")
        assert hasattr(trace, "get_retval")

        # Test assess
        test_value = 1.5
        density, retval = normal_dist.assess(test_value, *args)
        assert jnp.isfinite(density)
        assert retval == test_value

        # Test consistency
        expected_density = logpdf_normal(test_value, *args)
        assert jnp.allclose(density, expected_density)

    def test_distribution_generate_method(self):
        """Test Distribution.generate with various inputs."""

        def sample_exponential(rate):
            return jrand.exponential(jrand.key(42)) / rate

        def logpdf_exponential(x, rate):
            return jnp.log(rate) - rate * x

        exp_dist = distribution(
            sample_exponential, logpdf_exponential, name="exponential"
        )
        args = (2.0,)

        # Test generate with None (should simulate)
        trace, weight = exp_dist.generate(None, *args)
        assert jnp.allclose(weight, 0.0)  # Weight should be 0 for unconstrained
        assert trace.get_score() < 0  # Score should be negative log prob

        # Test generate with fixed value
        test_value = 0.5
        trace, weight = exp_dist.generate(test_value, *args)
        expected_density = logpdf_exponential(test_value, *args)
        assert jnp.allclose(weight, expected_density)
        assert jnp.allclose(trace.get_score(), -expected_density)

    def test_distribution_update_method(self):
        """Test Distribution.update with different scenarios."""

        def dummy_sample(mu):
            return jnp.array(mu)  # Return JAX array

        def logpdf_delta(x, mu):
            return jnp.array(0.0) if jnp.allclose(x, mu) else jnp.array(-jnp.inf)

        delta_dist = distribution(dummy_sample, logpdf_delta, name="delta")

        # Create initial trace
        args = (1.0,)
        initial_trace = delta_dist.simulate(*args)

        # Test update with new args, same choice
        new_args = (2.0,)
        new_trace, weight, discard = delta_dist.update(initial_trace, None, *new_args)

        assert jnp.allclose(new_trace.get_choices(), initial_trace.get_choices())
        assert jnp.allclose(discard, initial_trace.get_retval())

        # Test update with new choice
        new_choice = jnp.array(3.0)
        new_trace, weight, discard = delta_dist.update(
            initial_trace, new_choice, *new_args
        )
        assert jnp.allclose(new_trace.get_choices(), new_choice)

    def test_tfp_distribution_integration(self):
        """Test tfp_distribution wrapper functionality."""

        # Create TFP-based normal distribution
        normal_tfp = tfp_distribution(
            lambda mu, sigma: tfp.distributions.Normal(mu, sigma), name="normal_tfp"
        )

        # Test basic operations
        args = (0.0, 1.0)
        trace = normal_tfp.simulate(*args)

        assert hasattr(trace, "get_choices")
        assert jnp.isfinite(trace.get_score())

        # Test assess
        test_value = 0.5
        density, retval = normal_tfp.assess(test_value, *args)
        assert jnp.isfinite(density)
        assert retval == test_value

        # Compare with TFP directly
        tfp_dist = tfp.distributions.Normal(*args)
        expected_density = tfp_dist.log_prob(test_value)
        assert jnp.allclose(density, expected_density, rtol=1e-6)

    def test_distribution_error_handling(self):
        """Test Distribution error cases."""

        def dummy_sample():
            return 1.0

        def dummy_logpdf(x):
            return 0.0

        dist = distribution(dummy_sample, dummy_logpdf)

        # Test merge raises exception
        try:
            dist.merge(1.0, 2.0)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Can't merge" in str(e)


class TestVmapAndVectorization:
    """Test Vmap class and modular_vmap functionality."""

    def test_modular_vmap_basic(self):
        """Test modular_vmap function."""

        def sample_normal(mu, sigma):
            return normal.sample(mu, sigma)

        # Test vectorization over first argument
        vmap_sample = modular_vmap(sample_normal, in_axes=(0, None))

        mus = jnp.array([0.0, 1.0, 2.0])
        sigma = 1.0
        samples = vmap_sample(mus, sigma)

        assert samples.shape == (3,)

    def test_modular_vmap_with_axis_size(self):
        """Test modular_vmap with explicit axis_size."""

        def simple_func(x):
            return x * 2

        # Test with axis_size parameter
        vmap_func = modular_vmap(simple_func, axis_size=5)
        inputs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = vmap_func(inputs)

        expected = inputs * 2
        assert jnp.allclose(results, expected)


class TestTraceAndSelectors:
    """Test Trace class and selector functionality."""

    def test_trace_basic_operations(self):
        """Test basic Trace operations."""
        from genjax.core import get_choices, get_score, get_retval

        # Create a trace using normal distribution
        trace = normal.simulate(0.0, 1.0)

        # Test accessor functions work
        choices = get_choices(trace)
        score = get_score(trace)
        retval = get_retval(trace)

        assert jnp.isfinite(choices)
        assert jnp.isfinite(score)
        assert jnp.isfinite(retval)

        # Test trace methods
        assert jnp.allclose(trace.get_choices(), choices)
        assert jnp.allclose(trace.get_score(), score)
        assert jnp.allclose(trace.get_retval(), retval)
        assert trace.get_args() == ((0.0, 1.0), {})
        assert trace.get_gen_fn() == normal


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_address_error(self):
        """Test behavior with missing addresses."""
        from genjax.core import get_choices

        # Test get_choices with non-trace input
        non_trace_value = 42.0
        result = get_choices(non_trace_value)
        assert result == non_trace_value

    def test_basic_core_functionality(self):
        """Test basic core functionality works."""
        from genjax.core import get_choices

        # Test basic functionality without complex APIs
        trace = normal.simulate(0.0, 1.0)
        choices = get_choices(trace)
        assert jnp.isfinite(choices)

    def test_distribution_with_empty_args(self):
        """Test Distribution behavior with empty arguments."""

        def sample_func():
            return jnp.array(1.0)

        def logpdf_func(x):
            return jnp.array(0.0)

        dist = distribution(sample_func, logpdf_func)

        # Test with empty args
        trace = dist.simulate()
        assert jnp.allclose(trace.get_retval(), jnp.array(1.0))


class TestAddressCollisionDetection:
    """Test address collision detection in @gen functions."""

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_address_collision_simulate(self):
        """Test that simulate detects address collisions."""

        @gen
        def bad_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(2.0, 3.0) @ "x"  # Same address - should error
            return x + y

        with pytest.raises(
            ValueError, match="Address collision detected: 'x' is used multiple times"
        ):
            bad_model.simulate()

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_address_collision_assess(self):
        """Test that assess detects address collisions."""

        @gen
        def bad_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(2.0, 3.0) @ "x"  # Same address - should error
            return x + y

        choices = {"x": 1.0}

        with pytest.raises(
            ValueError, match="Address collision detected: 'x' is used multiple times"
        ):
            bad_model.assess(choices)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_address_collision_generate(self):
        """Test that generate detects address collisions."""

        @gen
        def bad_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(2.0, 3.0) @ "x"  # Same address - should error
            return x + y

        constraints = {"x": 1.5}

        with pytest.raises(
            ValueError, match="Address collision detected: 'x' is used multiple times"
        ):
            bad_model.generate(constraints)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_no_collision_with_unique_addresses(self):
        """Test that unique addresses work correctly."""

        @gen
        def good_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(2.0, 3.0) @ "y"  # Different address - should work
            return x + y

        # All GFI methods should work without error
        trace = good_model.simulate()
        assert "x" in trace.get_choices()
        assert "y" in trace.get_choices()

        choices = trace.get_choices()
        log_density, retval = good_model.assess(choices)
        assert jnp.isfinite(log_density)

        trace2, weight = good_model.generate(choices)
        assert jnp.isfinite(weight)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_nested_addressing_no_collision(self):
        """Test that nested addressing doesn't trigger false positives."""

        @gen
        def inner_model():
            return normal(0.0, 1.0) @ "x"

        @gen
        def outer_model():
            a = inner_model() @ "inner1"
            b = inner_model() @ "inner2"  # Different top-level addresses
            return a + b

        # Should work fine - the "x" addresses are at different levels
        trace = outer_model.simulate()
        choices = trace.get_choices()

        # Should have nested structure
        assert "inner1" in choices
        assert "inner2" in choices
        assert "x" in choices["inner1"]
        assert "x" in choices["inner2"]

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_collision_error_message_contains_address(self):
        """Test that error message contains the colliding address name."""

        @gen
        def bad_model():
            value1 = normal(0.0, 1.0) @ "my_special_address"
            value2 = normal(2.0, 3.0) @ "my_special_address"  # Same address
            return value1 + value2

        with pytest.raises(ValueError) as exc_info:
            bad_model.simulate()

        error_message = str(exc_info.value)
        assert "my_special_address" in error_message
        assert "Address collision detected" in error_message
        assert "used multiple times" in error_message


class TestUpdateAndRegenerate:
    """Test update and regenerate methods for GFI implementations."""

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_distribution_update_weight_invariant(self, standard_tolerance, helpers):
        """Test that Distribution.update weight equals -(new_score) + old_score."""
        args = (1.0, 0.5)  # mu=1.0, sigma=0.5

        # Create initial trace
        initial_trace = normal.simulate(*args)
        old_score = initial_trace.get_score()
        helpers.assert_valid_trace(initial_trace)

        # Update with new choice
        new_choice = 2.5
        new_trace, weight, discarded = normal.update(initial_trace, new_choice, *args)
        new_score = new_trace.get_score()

        # Test weight invariant: weight = -(new_score) + old_score
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight,
            expected_weight,
            rtol=standard_tolerance,
            msg="Update weight does not satisfy invariant",
        )

        # Test that new trace has the new choice
        assert jnp.allclose(
            new_trace.get_choices(), new_choice, rtol=standard_tolerance
        )
        assert jnp.allclose(new_trace.get_retval(), new_choice, rtol=standard_tolerance)

        # Test that discarded value is the old choice
        assert jnp.allclose(
            discarded, initial_trace.get_choices(), rtol=standard_tolerance
        )

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_distribution_update_with_same_choice(self, standard_tolerance, helpers):
        """Test Distribution.update when new choice equals old choice."""
        args = (0.0, 1.0)

        # Create initial trace
        initial_trace = normal.simulate(*args)
        old_choice = initial_trace.get_choices()
        old_score = initial_trace.get_score()

        # Update with same choice
        new_trace, weight, discarded = normal.update(initial_trace, old_choice, *args)
        new_score = new_trace.get_score()

        # Weight should still satisfy invariant
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight, expected_weight, rtol=standard_tolerance
        )

        # Scores should be equal (or very close due to numerical precision)
        helpers.assert_finite_and_close(old_score, new_score, rtol=standard_tolerance)

        # Weight should be approximately zero
        helpers.assert_finite_and_close(weight, 0.0, rtol=standard_tolerance)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_fn_update_weight_invariant_simple(self, standard_tolerance, helpers):
        """Test that Fn.update weight satisfies invariant for simple model."""

        @gen
        def simple_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x + y

        # Create initial trace
        initial_trace = simple_model.simulate()
        old_score = initial_trace.get_score()
        old_choices = initial_trace.get_choices()
        helpers.assert_valid_trace(initial_trace)

        # Update y choice
        new_choices = {"y": 3.0}
        new_trace, weight, discarded = simple_model.update(initial_trace, new_choices)
        new_score = new_trace.get_score()

        # Test weight invariant
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight,
            expected_weight,
            rtol=standard_tolerance,
            msg="Update weight does not satisfy invariant for Fn",
        )

        # Test that y was updated but x remained the same
        assert jnp.allclose(new_trace.get_choices()["y"], 3.0, rtol=standard_tolerance)
        assert jnp.allclose(
            new_trace.get_choices()["x"], old_choices["x"], rtol=standard_tolerance
        )

        # Test that discarded contains old y
        assert jnp.allclose(discarded["y"], old_choices["y"], rtol=standard_tolerance)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_fn_update_multiple_choices(self, standard_tolerance, helpers):
        """Test Fn.update with multiple choice updates."""

        @gen
        def multi_choice_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(1.0, 1.0) @ "y"
            z = normal(x + y, 0.5) @ "z"
            return x + y + z

        # Create initial trace
        initial_trace = multi_choice_model.simulate()
        old_score = initial_trace.get_score()
        old_choices = initial_trace.get_choices()

        # Update multiple choices
        new_choices = {"x": 2.0, "z": -1.0}
        new_trace, weight, discarded = multi_choice_model.update(
            initial_trace, new_choices
        )
        new_score = new_trace.get_score()

        # Test weight invariant
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight, expected_weight, rtol=standard_tolerance
        )

        # Test that specified choices were updated
        assert jnp.allclose(new_trace.get_choices()["x"], 2.0, rtol=standard_tolerance)
        assert jnp.allclose(new_trace.get_choices()["z"], -1.0, rtol=standard_tolerance)

        # Test that unspecified choice remained the same
        assert jnp.allclose(
            new_trace.get_choices()["y"], old_choices["y"], rtol=standard_tolerance
        )

        # Test discarded choices
        assert jnp.allclose(discarded["x"], old_choices["x"], rtol=standard_tolerance)
        assert jnp.allclose(discarded["z"], old_choices["z"], rtol=standard_tolerance)
        # Note: discarded may contain all choices, not just updated ones

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_scan_update_weight_invariant(self, base_key, standard_tolerance, helpers):
        """Test that Scan.update weight satisfies invariant."""

        @gen
        def scan_step(carry, x):
            noise = normal(0.0, 0.1) @ "noise"
            new_carry = carry + noise + x
            return new_carry, noise

        scan_gf = Scan(scan_step, length=const(3))
        init_carry = 1.0
        xs = jnp.array([0.1, 0.2, 0.3])
        args = (init_carry, xs)

        # Create initial trace with seed
        initial_trace = seed(scan_gf.simulate)(base_key, *args)
        old_score = initial_trace.get_score()
        old_choices = initial_trace.get_choices()
        helpers.assert_valid_trace(initial_trace)

        # Update some scan choices
        new_choices = {"noise": jnp.array([0.5, old_choices["noise"][1], -0.2])}
        new_trace, weight, discarded = scan_gf.update(initial_trace, new_choices, *args)
        new_score = new_trace.get_score()

        # Test weight invariant
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight, expected_weight, rtol=standard_tolerance
        )

        # Test that specified indices were updated
        assert jnp.allclose(
            new_trace.get_choices()["noise"][0], 0.5, rtol=standard_tolerance
        )
        assert jnp.allclose(
            new_trace.get_choices()["noise"][2], -0.2, rtol=standard_tolerance
        )

        # Test that unspecified index remained the same
        assert jnp.allclose(
            new_trace.get_choices()["noise"][1],
            old_choices["noise"][1],
            rtol=standard_tolerance,
        )

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_distribution_update_with_different_args(self, standard_tolerance, helpers):
        """Test Distribution.update with different arguments."""
        old_args = (0.0, 1.0)  # mu=0.0, sigma=1.0
        new_args = (2.0, 0.5)  # mu=2.0, sigma=0.5

        # Create initial trace with old args
        initial_trace = normal.simulate(*old_args)
        old_score = initial_trace.get_score()
        old_choice = initial_trace.get_choices()

        # Update with new args and same choice value
        new_trace, weight, discarded = normal.update(
            initial_trace, old_choice, *new_args
        )
        new_score = new_trace.get_score()

        # Test weight invariant: weight = -(new_score) + old_score
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight, expected_weight, rtol=standard_tolerance
        )

        # The choice should be the same, but the score should be different
        # because we're evaluating the same choice under different parameters
        assert jnp.allclose(
            new_trace.get_choices(), old_choice, rtol=standard_tolerance
        )

        # Verify the new score is correct for the new parameters
        expected_new_density, _ = normal.assess(old_choice, *new_args)
        expected_new_score = -expected_new_density
        helpers.assert_finite_and_close(
            new_score, expected_new_score, rtol=standard_tolerance
        )

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_weight_invariant_numerical_stability(self, standard_tolerance, helpers):
        """Test weight invariant holds under numerical edge cases."""

        @gen
        def edge_case_model():
            # Use parameters that might cause numerical issues
            x = normal(0.0, 1e-6) @ "x"  # Very small variance
            y = normal(x * 1e6, 1e-6) @ "y"  # Large mean, small variance
            return y

        # Create initial trace
        initial_trace = edge_case_model.simulate()
        old_score = initial_trace.get_score()

        # Update with value that should have very low probability
        new_choices = {"y": 1000.0}  # Very unlikely given the model
        new_trace, weight, discarded = edge_case_model.update(
            initial_trace, new_choices
        )
        new_score = new_trace.get_score()

        # Test weight invariant even for extreme cases
        expected_weight = -new_score + old_score
        helpers.assert_finite_and_close(
            weight, expected_weight, rtol=standard_tolerance
        )

        # All values should be finite
        assert jnp.isfinite(weight)
        assert jnp.isfinite(new_score)
        assert jnp.isfinite(old_score)


# =============================================================================
# SELECTION TESTS
# =============================================================================


class TestSelection:
    """Test Selection class and its various implementations."""

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_str_selection_basic(self):
        """Test basic string selection functionality."""
        selection = sel("x")

        # Should match the exact address
        matches, _ = selection.match("x")
        assert matches is True

        # Should not match different addresses
        matches, _ = selection.match("y")
        assert matches is False

        # Should not match nested addresses
        matches, _ = selection.match(("x", "y"))
        assert matches is False

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_or_selection_combination(self):
        """Test OR combination of selections."""
        selection = sel("x") | sel("y")

        # Should match either address
        matches_x, _ = selection.match("x")
        matches_y, _ = selection.match("y")
        assert matches_x is True
        assert matches_y is True

        # Should not match unrelated addresses
        matches_z, _ = selection.match("z")
        assert matches_z is False

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_complex_selection_combinations(self):
        """Test complex combinations of selections."""
        # Multiple OR combinations
        selection = sel("x") | sel("y") | sel("z")

        for addr in ["x", "y", "z"]:
            matches, _ = selection.match(addr)
            assert matches is True

        matches, _ = selection.match("w")
        assert matches is False

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_selection(self):
        """Test empty selection matches nothing."""
        selection = sel()

        # Should not match any address
        for addr in ["x", "y", ("a", "b"), 42]:
            matches, _ = selection.match(addr)
            assert matches is False

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_all_selection(self):
        """Test selection that matches everything."""
        from genjax.core import Selection, AllSel

        selection = Selection(AllSel())

        # Should match any address
        for addr in ["x", "y", ("a", "b"), 42]:
            matches, _ = selection.match(addr)
            assert matches is True

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.fast
    def test_selection_in_regenerate_context(self, helpers):
        """Test that selections work correctly in regenerate context."""

        @gen
        def test_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            z = normal(y, 0.1) @ "z"
            return z

        # Create initial trace
        initial_trace = test_model.simulate()
        old_choices = initial_trace.get_choices()

        # Test single selection
        selection = sel("y")
        new_trace, weight, discarded = test_model.regenerate(initial_trace, selection)

        # Only y should be in discarded, x and z should remain the same
        assert "y" in discarded
        assert jnp.allclose(new_trace.get_choices()["x"], old_choices["x"], rtol=1e-10)
        assert jnp.allclose(new_trace.get_choices()["z"], old_choices["z"], rtol=1e-10)

        # Test multiple selection
        multi_selection = sel("x") | sel("z")
        new_trace2, weight2, discarded2 = test_model.regenerate(
            initial_trace, multi_selection
        )

        # x and z should be in discarded, y should remain the same
        assert "x" in discarded2
        assert "z" in discarded2
        assert jnp.allclose(new_trace2.get_choices()["y"], old_choices["y"], rtol=1e-10)
