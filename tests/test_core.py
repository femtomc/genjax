"""
Test cases for GenJAX core functionality, particularly the Scan generative function.

These tests compare density computations using the Scan GFI against pure JAX
implementations to validate correctness.
"""

import jax.numpy as jnp
from jax.lax import scan

from genjax.core import gen, Scan, Cond
from genjax.distributions import normal, exponential


def test_fn_simulate_vs_manual_density():
    """Test that @gen Fn.simulate produces correct densities compared to manual computation."""

    @gen
    def simple_fn():
        x = normal(0.0, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return x + y

    # Generate trace
    trace = simple_fn.simulate(())
    choices = trace.get_choices()
    fn_score = trace.get_score()

    # Extract individual choices
    x_choice = choices["x"]
    y_choice = choices["y"]

    # Compute densities using Distribution.assess
    x_log_density, _ = normal.assess((0.0, 1.0), x_choice)
    y_log_density, _ = normal.assess((x_choice, 0.5), y_choice)

    manual_total_log_density = x_log_density + y_log_density
    expected_fn_score = -manual_total_log_density

    assert jnp.allclose(fn_score, expected_fn_score, rtol=1e-6), (
        f"Fn score {fn_score} != expected {expected_fn_score}"
    )

    # Verify return value
    expected_retval = x_choice + y_choice
    assert jnp.allclose(trace.get_retval(), expected_retval, rtol=1e-6)


def test_fn_assess_vs_manual_density():
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
    fn_density, fn_retval = exponential_fn.assess(args, choices)

    # Manual computation
    x_val = choices["x"]
    y_val = choices["y"]

    # Compute densities using Distribution.assess
    x_log_density, _ = exponential.assess((rate1,), x_val)
    y_rate = rate2 * x_val
    y_log_density, _ = exponential.assess((y_rate,), y_val)

    manual_total_log_density = x_log_density + y_log_density
    manual_retval = x_val * y_val

    assert jnp.allclose(fn_density, manual_total_log_density, rtol=1e-6), (
        f"Fn density {fn_density} != manual {manual_total_log_density}"
    )
    assert jnp.allclose(fn_retval, manual_retval, rtol=1e-6)


def test_fn_simulate_assess_consistency():
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
    trace = complex_fn.simulate(args)
    choices = trace.get_choices()
    simulate_score = trace.get_score()
    simulate_retval = trace.get_retval()

    # Assess same choices
    assess_density, assess_retval = complex_fn.assess(args, choices)

    # Should be consistent
    expected_score = -assess_density
    assert jnp.allclose(simulate_score, expected_score, rtol=1e-6), (
        f"Simulate score {simulate_score} != -assess density {expected_score}"
    )

    # Return values should match
    assert jnp.allclose(simulate_retval[0], assess_retval[0], rtol=1e-6)
    assert jnp.allclose(simulate_retval[1], assess_retval[1], rtol=1e-6)
    assert jnp.allclose(simulate_retval[2], assess_retval[2], rtol=1e-6)


def test_fn_nested_addressing():
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
    trace = outer_fn.simulate(())
    choices = trace.get_choices()
    fn_score = trace.get_score()

    # Extract nested choices
    x_choice = choices["x"]
    y_choice = choices["inner"]["inner_sample"]  # This is the inner function's result
    z_choice = choices["z"]

    # Compute densities using Distribution.assess
    # x ~ normal(1.0, 0.5)
    x_log_density, _ = normal.assess((1.0, 0.5), x_choice)

    # y ~ normal(0.0, abs(x)) (from inner function)
    y_scale = jnp.abs(x_choice)
    y_log_density, _ = normal.assess((0.0, y_scale), y_choice)

    # z ~ normal(y, 0.2)
    z_log_density, _ = normal.assess((y_choice, 0.2), z_choice)

    manual_total_log_density = x_log_density + y_log_density + z_log_density
    expected_fn_score = -manual_total_log_density

    assert jnp.allclose(fn_score, expected_fn_score, rtol=1e-6), (
        f"Nested Fn score {fn_score} != expected {expected_fn_score}"
    )

    # Verify return value
    expected_retval = x_choice + y_choice + z_choice
    assert jnp.allclose(trace.get_retval(), expected_retval, rtol=1e-6)


def test_fn_with_deterministic_computation():
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
    trace = mixed_fn.simulate(args)
    choices = trace.get_choices()

    assess_density, assess_retval = mixed_fn.assess(args, choices)
    simulate_score = trace.get_score()

    assert jnp.allclose(simulate_score, -assess_density, rtol=1e-6), (
        f"Mixed Fn inconsistent: score={simulate_score}, density={assess_density}"
    )
    assert jnp.allclose(trace.get_retval(), assess_retval, rtol=1e-6)


def test_fn_empty_program():
    """Test @gen function with no probabilistic choices."""

    @gen
    def deterministic_fn(x, y):
        # Only deterministic computation
        result = x * y + jnp.sin(x) - jnp.cos(y)
        return result

    args = (2.0, 3.0)

    # Should work and have zero score (no probabilistic choices)
    trace = deterministic_fn.simulate(args)
    choices = trace.get_choices()

    assert trace.get_score() == 0.0, "Deterministic function should have zero score"
    assert len(choices) == 0, "Deterministic function should have no choices"

    # Should compute same result deterministically
    expected_result = 2.0 * 3.0 + jnp.sin(2.0) - jnp.cos(3.0)
    assert jnp.allclose(trace.get_retval(), expected_result, rtol=1e-6)

    # Assess should also work
    density, retval = deterministic_fn.assess(args, {})
    assert density == 0.0, "Deterministic assess should have zero density"
    assert jnp.allclose(retval, expected_result, rtol=1e-6)


def test_fn_conditional_sampling():
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
        trace = conditional_fn.simulate(args)
        choices = trace.get_choices()

        # Should be consistent between simulate and assess
        assess_density, assess_retval = conditional_fn.assess(args, choices)
        simulate_score = trace.get_score()

        assert jnp.allclose(-simulate_score, assess_density, rtol=1e-6)
        assert jnp.allclose(trace.get_retval(), assess_retval, rtol=1e-6)


def test_scan_simulate_vs_manual_density():
    """Test that Scan.simulate produces correct densities compared to manual computation."""

    # Create a simple callee that adds a normal sample to the carry
    @gen
    def add_normal_step(carry, input_val):
        noise = normal(0.0, 1.0) @ "noise"
        new_carry = carry + noise + input_val
        output = new_carry * 2.0
        return new_carry, output

    # Create scan generative function
    scan_gf = Scan(add_normal_step, length=4)

    # Test parameters
    init_carry = 1.0
    xs = jnp.array([0.5, -0.2, 1.1, 0.8])
    args = (init_carry, xs)

    # Generate a trace using simulate
    trace = scan_gf.simulate(args)
    choices = trace.get_choices()
    scan_score = trace.get_score()  # This is log(1/density), so negative log density

    # Manually compute the density using the same choices
    def manual_scan_fn(carry, input_and_choice):
        input_val, choice = input_and_choice
        noise = choice["noise"]  # Use the same sampled value
        new_carry = carry + noise + input_val
        output = new_carry * 2.0

        # Compute log density using Distribution.assess
        log_density, _ = normal.assess((0.0, 1.0), noise)
        return new_carry, (output, log_density)

    # Run manual scan with the same choices
    final_carry, (outputs, log_densities) = scan(
        manual_scan_fn,
        init_carry,
        (xs, choices),
        length=len(xs),
    )

    # Sum log densities to get total log density
    manual_total_log_density = jnp.sum(log_densities)

    # scan_score should be -manual_total_log_density (since score is log(1/density))
    expected_scan_score = -manual_total_log_density

    # Compare with tolerance for numerical precision
    assert jnp.allclose(scan_score, expected_scan_score, rtol=1e-6), (
        f"Scan score {scan_score} != expected {expected_scan_score}"
    )

    # Also verify the outputs match
    trace_outputs = trace.get_retval()[1]
    assert jnp.allclose(trace_outputs, outputs, rtol=1e-6), (
        f"Trace outputs {trace_outputs} != manual outputs {outputs}"
    )


def test_scan_assess_vs_manual_density():
    """Test that Scan.assess produces correct densities compared to manual computation."""

    # Create a callee that uses exponential distribution
    @gen
    def exponential_step(carry, input_val):
        rate = jnp.exp(carry + input_val)  # Ensure positive rate
        sample = exponential(rate) @ "exp_sample"
        new_carry = carry + sample * 0.1  # Small update to carry
        output = sample + input_val
        return new_carry, output

    # Create scan generative function
    scan_gf = Scan(exponential_step, length=3)

    # Test parameters
    init_carry = 0.5
    xs = jnp.array([0.1, 0.3, -0.2])
    args = (init_carry, xs)

    # Create some fixed choices to assess
    choices = {"exp_sample": jnp.array([0.8, 1.2, 0.4])}

    # Assess using Scan
    scan_density, scan_retval = scan_gf.assess(args, choices)

    # Manually compute density
    def manual_assess_fn(carry, input_and_choice):
        input_val, choice = input_and_choice
        rate = jnp.exp(carry + input_val)
        sample = choice["exp_sample"]
        new_carry = carry + sample * 0.1
        output = sample + input_val

        # Compute log density using Distribution.assess
        log_density, _ = exponential.assess((rate,), sample)
        return new_carry, (output, log_density)

    # Run manual assessment
    final_carry, (outputs, log_densities) = scan(
        manual_assess_fn,
        init_carry,
        (xs, choices),
        length=len(xs),
    )

    manual_total_log_density = jnp.sum(log_densities)
    manual_retval = (final_carry, outputs)

    # Compare densities (scan_density should equal manual_total_log_density)
    assert jnp.allclose(scan_density, manual_total_log_density, rtol=1e-6), (
        f"Scan density {scan_density} != manual density {manual_total_log_density}"
    )

    # Compare return values
    assert jnp.allclose(scan_retval[0], manual_retval[0], rtol=1e-6), (
        f"Final carry {scan_retval[0]} != manual {manual_retval[0]}"
    )
    assert jnp.allclose(scan_retval[1], manual_retval[1], rtol=1e-6), (
        f"Outputs {scan_retval[1]} != manual {manual_retval[1]}"
    )


def test_scan_simulate_assess_consistency():
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

    # Create scan generative function
    scan_gf = Scan(complex_step, length=4)

    # Test parameters
    init_carry = 0.2
    xs = jnp.array([0.5, -0.3, 0.8, 1.0])
    args = (init_carry, xs)

    # Generate trace with simulate
    trace = scan_gf.simulate(args)
    choices = trace.get_choices()
    simulate_score = trace.get_score()
    simulate_retval = trace.get_retval()

    # Assess the same choices
    assess_density, assess_retval = scan_gf.assess(args, choices)

    # simulate_score should be -assess_density (score is log(1/density))
    expected_score = -assess_density

    assert jnp.allclose(simulate_score, expected_score, rtol=1e-6), (
        f"Simulate score {simulate_score} != -assess density {expected_score}"
    )

    # Return values should be identical
    assert jnp.allclose(simulate_retval[0], assess_retval[0], rtol=1e-6), (
        f"Final carries don't match: {simulate_retval[0]} vs {assess_retval[0]}"
    )
    assert jnp.allclose(simulate_retval[1][0], assess_retval[1][0], rtol=1e-6), (
        "Normal outputs don't match"
    )
    assert jnp.allclose(simulate_retval[1][1], assess_retval[1][1], rtol=1e-6), (
        "Exponential outputs don't match"
    )


def test_empty_scan():
    """Test scan with empty input sequence."""

    @gen
    def simple_step(carry, input_val):
        sample = normal(0.0, 1.0) @ "sample"
        return carry + sample, sample

    scan_gf = Scan(simple_step, length=0)

    # Empty inputs
    init_carry = 1.0
    xs = jnp.array([])  # Empty array
    args = (init_carry, xs)

    # Should work and return initial carry with empty outputs
    trace = scan_gf.simulate(args)
    final_carry, outputs = trace.get_retval()

    assert jnp.allclose(final_carry, init_carry), (
        "Final carry should equal initial carry"
    )
    assert outputs.shape[0] == 0, "Outputs should be empty"
    assert trace.get_score() == 0.0, "Score should be 0 for empty scan"


def test_single_step_scan():
    """Test scan with single step."""

    @gen
    def single_step(carry, input_val):
        sample = normal(input_val, 0.5) @ "sample"
        new_carry = carry + sample
        return new_carry, sample**2

    scan_gf = Scan(single_step, length=1)

    # Single input
    init_carry = 2.0
    xs = jnp.array([1.5])
    args = (init_carry, xs)

    # Test simulate
    trace = scan_gf.simulate(args)
    choices = trace.get_choices()

    # Test assess with same choice
    density, retval = scan_gf.assess(args, choices)

    # Compute expected density using Distribution.assess
    sample = choices["sample"][0]
    expected_log_density, _ = normal.assess((1.5, 0.5), sample)

    assert jnp.allclose(density, expected_log_density, rtol=1e-6), (
        f"Density {density} != expected {expected_log_density}"
    )


def test_scan_with_different_lengths():
    """Test scan with various sequence lengths."""

    @gen
    def accumulating_step(carry, input_val):
        sample = normal(0.0, 0.1) @ "sample"  # Small noise
        new_carry = carry + input_val + sample
        return new_carry, new_carry

    lengths = [1, 3, 5, 10]

    for length in lengths:
        scan_gf = Scan(accumulating_step, length=length)
        init_carry = 0.0
        xs = jnp.ones(length) * 0.1  # Small constant inputs
        args = (init_carry, xs)

        # Test that simulate and assess are consistent
        trace = scan_gf.simulate(args)
        choices = trace.get_choices()

        assess_density, assess_retval = scan_gf.assess(args, choices)
        simulate_score = trace.get_score()

        assert jnp.allclose(simulate_score, -assess_density, rtol=1e-6), (
            f"Inconsistency at length {length}: score={simulate_score}, density={assess_density}"
        )

        # Check that we have the right number of choices and outputs
        assert choices["sample"].shape[0] == length, (
            f"Wrong number of choices: {choices.shape[0]} != {length}"
        )
        assert trace.get_retval()[1].shape[0] == length, "Wrong number of outputs"
