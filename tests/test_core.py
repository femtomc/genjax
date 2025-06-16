"""
Test cases for GenJAX core functionality, particularly the Scan generative function.

These tests compare density computations using the Scan GFI against pure JAX
implementations to validate correctness.
"""

import jax.numpy as jnp
import jax.random as jrand
from jax.lax import scan

from genjax.core import gen, Scan, Cond, seed
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

    # Generate a trace using simulate with seed transformation
    key = jrand.key(42)
    trace = seed(scan_gf.simulate)(key, args)
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

    # Generate trace with simulate using seed transformation
    key = jrand.key(42)
    trace = seed(scan_gf.simulate)(key, args)
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

    # Test simulate with seed transformation
    key = jrand.key(42)
    trace = seed(scan_gf.simulate)(key, args)
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
        key = jrand.key(42 + length)  # Use different key for each length
        trace = seed(scan_gf.simulate)(key, args)
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


class TestGenerateConsistency:
    """Test that generate method is consistent with assess when given full samples."""

    def test_distribution_generate_assess_consistency(self):
        """Test that Distribution.generate weight equals assess density for full samples."""
        # Test with normal distribution
        args = (0.0, 1.0)  # mu=0.0, sigma=1.0
        sample_value = 1.5

        # Test generate with full sample
        trace, weight = normal.generate(args, sample_value)

        # Test assess with same sample
        density, retval = normal.assess(args, sample_value)

        # For distributions: generate weight should equal assess density
        assert jnp.allclose(weight, density, rtol=1e-10), (
            f"Generate weight {weight} != assess density {density}"
        )

        # Check that trace score equals negative density
        assert jnp.allclose(trace.get_score(), -density, rtol=1e-10)

        # Check return values match
        assert jnp.allclose(retval, sample_value, rtol=1e-10)
        assert jnp.allclose(trace.get_retval(), sample_value, rtol=1e-10)

    def test_fn_generate_assess_consistency_simple(self):
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
        trace, weight = simple_model.generate(args, full_sample)

        # Test assess with same sample
        density, retval = simple_model.assess(args, full_sample)

        # Generate weight should equal assess density for full samples
        assert jnp.allclose(weight, density, rtol=1e-10), (
            f"Generate weight {weight} != assess density {density}"
        )

        # Check that trace score equals negative density
        assert jnp.allclose(trace.get_score(), -density, rtol=1e-10)

        # Check return values match
        assert jnp.allclose(retval, 1.0 + 2.0, rtol=1e-10)
        assert jnp.allclose(trace.get_retval(), 1.0 + 2.0, rtol=1e-10)

    def test_fn_generate_assess_consistency_hierarchical(self):
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
        trace, weight = hierarchical_model.generate(args, full_sample)

        # Test assess with same sample
        density, retval = hierarchical_model.assess(args, full_sample)

        # Generate weight should equal assess density
        assert jnp.allclose(weight, density, rtol=1e-10), (
            f"Generate weight {weight} != assess density {density}"
        )

        # Check consistency of scores
        assert jnp.allclose(trace.get_score(), -density, rtol=1e-10)

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
            scan_gf = Scan(step_function, length=3)
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
        trace, weight = scan_model.generate(args, full_sample)

        # Test assess with same sample
        density, retval = scan_model.assess(args, full_sample)

        # Generate weight should equal assess density
        assert jnp.allclose(weight, density, rtol=1e-6), (
            f"Generate weight {weight} != assess density {density}"
        )

    def test_distribution_generate_with_partial_sample(self):
        """Test that Distribution.generate handles None (no constraints) correctly."""
        args = (0.0, 1.0)  # mu=0.0, sigma=1.0

        # Test generate with None (simulate)
        trace, weight = normal.generate(args, None)

        # Weight should be 0.0 when no constraints are provided (pure simulation)
        assert jnp.allclose(weight, 0.0, rtol=1e-10), (
            f"Generate weight with None should be 0.0, got {weight}"
        )

        # Score should be negative log density
        sample = trace.get_choices()
        density, _ = normal.assess(args, sample)
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
        trace, weight = simple_model.generate(args, partial_sample)
        choices = trace.get_choices()

        # Should have both x and y in choices
        assert "x" in choices, "Missing x choice"
        assert "y" in choices, "Missing y choice"
        assert jnp.allclose(choices["y"], 2.0, rtol=1e-10), "y should match constraint"

        # Weight should be the density of the constrained variable (y)
        # given the generated value of x
        x_val = choices["x"]
        y_density, _ = normal.assess((x_val, 0.5), 2.0)
        assert jnp.allclose(weight, y_density, rtol=1e-6), (
            f"Weight {weight} should equal y density {y_density}"
        )


class TestPytreeAndDataClasses:
    """Test Pytree functionality and dataclass integration."""

    def test_pytree_dataclass_basic(self):
        """Test basic Pytree dataclass functionality."""
        from genjax.core import Pytree
        import jax

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

    def test_pytree_static_vs_dynamic_fields(self):
        """Test distinction between static and dynamic fields."""
        from genjax.core import Pytree
        import jax
        import jax.numpy as jnp

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

    def test_pytree_field_vs_static_annotations(self):
        """Test different field annotation types."""
        from genjax.core import Pytree
        import jax

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
        from genjax.core import Distribution
        import jax.numpy as jnp
        import jax.random as jrand

        # Create simple normal distribution
        def sample_normal(mu, sigma):
            return jrand.normal(jrand.key(42)) * sigma + mu

        def logpdf_normal(x, mu, sigma):
            return (
                -0.5 * ((x - mu) / sigma) ** 2
                - jnp.log(sigma)
                - 0.5 * jnp.log(2 * jnp.pi)
            )

        normal_dist = Distribution(sample_normal, logpdf_normal, name="normal")

        # Test simulate
        args = (0.0, 1.0)
        trace = normal_dist.simulate(args)

        assert hasattr(trace, "get_choices")
        assert hasattr(trace, "get_score")
        assert hasattr(trace, "get_retval")

        # Test assess
        test_value = 1.5
        density, retval = normal_dist.assess(args, test_value)
        assert jnp.isfinite(density)
        assert retval == test_value

        # Test consistency
        expected_density = logpdf_normal(test_value, *args)
        assert jnp.allclose(density, expected_density)

    def test_distribution_generate_method(self):
        """Test Distribution.generate with various inputs."""
        from genjax.core import Distribution
        import jax.numpy as jnp
        import jax.random as jrand

        def sample_exponential(rate):
            return jrand.exponential(jrand.key(42)) / rate

        def logpdf_exponential(x, rate):
            return jnp.log(rate) - rate * x

        exp_dist = Distribution(
            sample_exponential, logpdf_exponential, name="exponential"
        )
        args = (2.0,)

        # Test generate with None (should simulate)
        trace, weight = exp_dist.generate(args, None)
        assert jnp.allclose(weight, 0.0)  # Weight should be 0 for unconstrained
        assert trace.get_score() < 0  # Score should be negative log prob

        # Test generate with fixed value
        test_value = 0.5
        trace, weight = exp_dist.generate(args, test_value)
        expected_density = logpdf_exponential(test_value, *args)
        assert jnp.allclose(weight, expected_density)
        assert jnp.allclose(trace.get_score(), -expected_density)

    def test_distribution_update_method(self):
        """Test Distribution.update with different scenarios."""
        from genjax.core import Distribution
        import jax.numpy as jnp

        def dummy_sample(mu):
            return jnp.array(mu)  # Return JAX array

        def logpdf_delta(x, mu):
            return jnp.array(0.0) if jnp.allclose(x, mu) else jnp.array(-jnp.inf)

        delta_dist = Distribution(dummy_sample, logpdf_delta, name="delta")

        # Create initial trace
        args = (1.0,)
        initial_trace = delta_dist.simulate(args)

        # Test update with new args, same choice
        new_args = (2.0,)
        new_trace, weight, discard = delta_dist.update(new_args, initial_trace, None)

        assert jnp.allclose(new_trace.get_choices(), initial_trace.get_choices())
        assert jnp.allclose(discard, initial_trace.get_retval())

        # Test update with new choice
        new_choice = jnp.array(3.0)
        new_trace, weight, discard = delta_dist.update(
            new_args, initial_trace, new_choice
        )
        assert jnp.allclose(new_trace.get_choices(), new_choice)

    def test_tfp_distribution_integration(self):
        """Test tfp_distribution wrapper functionality."""
        from genjax.core import tfp_distribution
        import tensorflow_probability.substrates.jax as tfp

        # Create TFP-based normal distribution
        normal_tfp = tfp_distribution(
            lambda mu, sigma: tfp.distributions.Normal(mu, sigma), name="normal_tfp"
        )

        # Test basic operations
        args = (0.0, 1.0)
        trace = normal_tfp.simulate(args)

        assert hasattr(trace, "get_choices")
        assert jnp.isfinite(trace.get_score())

        # Test assess
        test_value = 0.5
        density, retval = normal_tfp.assess(args, test_value)
        assert jnp.isfinite(density)
        assert retval == test_value

        # Compare with TFP directly
        tfp_dist = tfp.distributions.Normal(*args)
        expected_density = tfp_dist.log_prob(test_value)
        assert jnp.allclose(density, expected_density, rtol=1e-6)

    def test_distribution_error_handling(self):
        """Test Distribution error cases."""
        from genjax.core import Distribution

        def dummy_sample():
            return 1.0

        def dummy_logpdf(x):
            return 0.0

        dist = Distribution(dummy_sample, dummy_logpdf)

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
        from genjax.core import modular_vmap
        from genjax.distributions import normal

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
        from genjax.core import modular_vmap

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
        from genjax.distributions import normal

        # Create a trace using normal distribution
        trace = normal.simulate((0.0, 1.0))

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
        assert trace.get_args() == (0.0, 1.0)
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
        from genjax.distributions import normal

        # Test basic functionality without complex APIs
        trace = normal.simulate((0.0, 1.0))
        choices = get_choices(trace)
        assert jnp.isfinite(choices)

    def test_distribution_with_empty_args(self):
        """Test Distribution behavior with empty arguments."""
        from genjax.core import Distribution
        import jax.numpy as jnp

        def sample_func():
            return jnp.array(1.0)

        def logpdf_func(x):
            return jnp.array(0.0)

        dist = Distribution(sample_func, logpdf_func)

        # Test with empty args
        trace = dist.simulate(())
        assert jnp.allclose(trace.get_retval(), jnp.array(1.0))
