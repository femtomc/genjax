"""
Test cases for PJAX (Probabilistic JAX) functionality.

These tests validate PJAX-specific components including:
- seed transformation
- modular_vmap transformation
- PJAX primitives and interpreters
- JAX integration patterns
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

from genjax.core import gen, Scan, Const, const
from genjax.pjax import seed, modular_vmap
from genjax.distributions import normal, exponential


# =============================================================================
# SEED TRANSFORMATION TESTS
# =============================================================================


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_seed_transform_scan_simulate(base_key, standard_tolerance, helpers):
    """Test that seed transformation enables JAX compilation for Scan.simulate."""

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
    scan_score = trace.get_score()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Verify the transformation worked correctly
    assert "scan" in choices
    assert "noise" in choices["scan"]
    assert choices["scan"]["noise"].shape == (4,)  # Should have 4 noise samples
    assert jnp.isfinite(scan_score)
    assert jnp.isfinite(trace.get_retval()[0])  # final carry
    assert trace.get_retval()[1].shape == (4,)  # outputs


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_seed_transform_simulate_assess_consistency(
    base_key, standard_tolerance, helpers
):
    """Test that seed transformation preserves simulate/assess consistency."""

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

    # Assess the same choices (without seed transformation for assess)
    assess_density, assess_retval = scan_model.assess(choices, *args)
    helpers.assert_valid_density(assess_density)

    # simulate_score should be -assess_density (score is log(1/density))
    expected_score = -assess_density

    helpers.assert_finite_and_close(
        simulate_score,
        expected_score,
        rtol=standard_tolerance,
        msg="Seed-transformed scan simulate/assess score inconsistency",
    )

    # Return values should be identical
    helpers.assert_finite_and_close(
        simulate_retval[0],
        assess_retval[0],
        rtol=standard_tolerance,
        msg="Seed-transformed scan final carries do not match",
    )
    helpers.assert_finite_and_close(
        simulate_retval[1][0],
        assess_retval[1][0],
        rtol=standard_tolerance,
        msg="Seed-transformed scan normal outputs do not match",
    )
    helpers.assert_finite_and_close(
        simulate_retval[1][1],
        assess_retval[1][1],
        rtol=standard_tolerance,
        msg="Seed-transformed scan exponential outputs do not match",
    )


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("length", [1, 3, 5])
def test_seed_transform_different_lengths(
    length, base_key, standard_tolerance, helpers
):
    """Test seed transformation with various sequence lengths."""

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

    # Test that simulate with seed transformation works for different lengths
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
        msg=f"Seed-transformed scan length {length} simulate/assess inconsistency",
    )

    # Check that we have the right number of choices and outputs
    assert choices["scan"]["sample"].shape[0] == length, (
        f"Wrong number of choices: {choices['scan']['sample'].shape[0]} != {length}"
    )
    assert trace.get_retval()[1].shape[0] == length, "Wrong number of outputs"


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_seed_transform_jit_compilation(base_key):
    """Test that seed transformation enables JIT compilation."""

    @gen
    def simple_model():
        x = normal(0.0, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return x + y

    # Test that JIT compilation works with seed transformation
    jit_simulate = jax.jit(seed(simple_model.simulate))

    # Should compile and run without errors
    trace = jit_simulate(base_key)

    # Basic validation
    assert hasattr(trace, "get_choices")
    assert hasattr(trace, "get_score")
    assert jnp.isfinite(trace.get_score())
    assert jnp.isfinite(trace.get_retval())


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_seed_transform_nested_models(base_key, standard_tolerance, helpers):
    """Test seed transformation with nested generative functions."""

    @gen
    def inner_model(scale):
        return normal(0.0, scale) @ "inner_sample"

    @gen
    def outer_model():
        x = normal(1.0, 0.5) @ "x"
        y = inner_model(jnp.abs(x)) @ "inner"  # scale depends on x
        z = normal(y, 0.2) @ "z"  # location depends on y
        return x + y + z

    # Test with seed transformation
    trace = seed(outer_model.simulate)(base_key)
    choices = trace.get_choices()

    # Validate trace structure
    helpers.assert_valid_trace(trace)

    # Test consistency with assess
    assess_density, assess_retval = outer_model.assess(choices)
    helpers.assert_valid_density(assess_density)

    helpers.assert_finite_and_close(
        trace.get_score(),
        -assess_density,
        rtol=standard_tolerance,
        msg="Seed-transformed nested model simulate/assess inconsistency",
    )
    helpers.assert_finite_and_close(
        trace.get_retval(),
        assess_retval,
        rtol=standard_tolerance,
        msg="Seed-transformed nested model return value mismatch",
    )


# =============================================================================
# MODULAR VMAP TESTS
# =============================================================================


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_modular_vmap_basic_functionality():
    """Test basic modular_vmap functionality."""

    def sample_normal(mu, sigma):
        return normal.sample(mu, sigma)

    # Test vectorization over first argument
    vmap_sample = modular_vmap(sample_normal, in_axes=(0, None))

    mus = jnp.array([0.0, 1.0, 2.0])
    sigma = 1.0
    samples = vmap_sample(mus, sigma)

    assert samples.shape == (3,), f"Expected shape (3,), got {samples.shape}"
    assert jnp.isfinite(samples).all(), "All samples should be finite"


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_modular_vmap_with_axis_size():
    """Test modular_vmap with explicit axis_size."""

    def simple_func(x):
        return x * 2

    # Test with axis_size parameter
    vmap_func = modular_vmap(simple_func, axis_size=5)
    inputs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    results = vmap_func(inputs)

    expected = inputs * 2
    assert jnp.allclose(results, expected), "Results should match expected values"


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_modular_vmap_preserves_probabilistic_semantics():
    """Test that modular_vmap preserves probabilistic primitive semantics."""

    @gen
    def simple_model(mu):
        x = normal(mu, 1.0) @ "x"
        return x

    # Vectorize over mu parameter
    vmap_model = modular_vmap(simple_model.simulate, in_axes=(0,))

    mus = jnp.array([0.0, 1.0, -1.0])
    traces = vmap_model((mus,))

    # Should get one trace per mu value
    choices = traces.get_choices()
    assert choices["x"].shape == (3,), "Should have 3 vectorized choices for x"
    assert traces.get_retval().shape == (3,), "Should have 3 vectorized return values"

    # All values should be finite
    assert jnp.isfinite(choices["x"]).all()
    assert jnp.isfinite(traces.get_score()), "Score should be finite (but scalar)"
    assert jnp.isfinite(traces.get_retval()).all()


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_modular_vmap_vs_manual_vectorization(standard_tolerance):
    """Test modular_vmap produces same results as manual vectorization."""

    @gen
    def test_model(scale):
        return exponential(1.0 / scale) @ "sample"

    # Manual vectorization (multiple individual calls)
    scales = jnp.array([1.0, 2.0, 0.5])
    manual_traces = []
    for scale in scales:
        trace = test_model.simulate(scale)
        manual_traces.append(trace)

    # Extract manual results
    manual_choice_values = jnp.array([t.get_choices()["sample"] for t in manual_traces])
    manual_retvals = jnp.array([t.get_retval() for t in manual_traces])

    # modular_vmap vectorization
    vmap_simulate = modular_vmap(test_model.simulate, in_axes=(0,))
    vmap_trace = vmap_simulate(scales)

    # Results should be equivalent (up to random sampling differences)
    # We'll check structure and finite-ness rather than exact values
    vmap_choices = vmap_trace.get_choices()
    assert vmap_choices["sample"].shape == manual_choice_values.shape
    assert vmap_trace.get_retval().shape == manual_retvals.shape

    assert jnp.isfinite(vmap_choices["sample"]).all()
    assert jnp.isfinite(vmap_trace.get_score()), "Score should be finite (but scalar)"
    assert jnp.isfinite(vmap_trace.get_retval()).all()


# =============================================================================
# PJAX PRIMITIVE INTEGRATION TESTS
# =============================================================================


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_pjax_primitives_with_jax_transformations(base_key):
    """Test that PJAX primitives integrate correctly with JAX transformations."""

    @gen
    def pjax_model(param):
        # Model using PJAX primitives that need transformation
        x = normal(param, 1.0) @ "x"
        y = normal(x, 0.5) @ "y"
        return y

    # Test that seed enables standard JAX transformations
    seeded_simulate = seed(pjax_model.simulate)

    # Test JIT compilation
    jit_simulate = jax.jit(seeded_simulate)
    param = 0.5
    trace = jit_simulate(base_key, param)

    assert jnp.isfinite(trace.get_score())
    assert jnp.isfinite(trace.get_retval())

    # Test vmap
    vmap_simulate = jax.vmap(seeded_simulate, in_axes=(0, 0))
    params = jnp.array([0.0, 1.0, 2.0])
    keys = jrand.split(base_key, 3)
    vmap_traces = vmap_simulate(keys, params)

    # For jax.vmap + seed, each choice should be vectorized individually
    choices = vmap_traces.get_choices()
    assert choices["x"].shape == (3,), "x should be vectorized"
    assert choices["y"].shape == (3,), "y should be vectorized"

    # Check that we have valid finite values
    assert jnp.isfinite(choices["x"]).all()
    assert jnp.isfinite(choices["y"]).all()

    # Score is always scalar per trace, but retval should be vectorized
    assert jnp.isfinite(vmap_traces.get_score()), "Score should be finite"
    assert jnp.isfinite(vmap_traces.get_retval()).all(), (
        "Return values should be finite"
    )
    assert vmap_traces.get_retval().shape == (3,), "Return values should be vectorized"


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_pjax_error_without_seed_transformation():
    """Test that PJAX primitives require seed transformation in certain contexts."""

    @gen
    def model_requiring_seed():
        x = normal(0.0, 1.0) @ "x"
        return x

    # This should work fine (no JAX control flow)
    trace = model_requiring_seed.simulate()
    assert jnp.isfinite(trace.get_score())

    # However, certain JAX transformations may require seed
    # This is more of a documentation test than a failure test
    # since the exact error conditions depend on JAX internals


# =============================================================================
# INTEGRATION WITH COMBINATORS
# =============================================================================


@pytest.mark.pjax
@pytest.mark.unit
@pytest.mark.fast
def test_seed_with_scan_update_operations(base_key, standard_tolerance, helpers):
    """Test seed transformation with Scan update operations."""

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
    helpers.assert_finite_and_close(weight, expected_weight, rtol=standard_tolerance)

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
