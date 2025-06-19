"""Test for vmap generate bug in GenJAX core.

This test demonstrates and verifies the fix for the bug in Vmap.generate
where in_axes specification doesn't account for the constraints argument.
"""

import jax.numpy as jnp
from genjax import gen
from genjax.distributions import normal


@gen
def two_param_fn(x: float, y: float) -> float:
    """Simple function with 2 parameters for testing vmap."""
    z = normal(x, y) @ "sample"
    return z


@gen
def three_param_fn(a: float, b: float, c: float) -> float:
    """Function with 3 parameters for testing vmap."""
    result = normal(a + b, c) @ "output"
    return result


class TestVmapGenerate:
    """Test vmap generate method bug and fix."""

    def test_vmap_simulate_works(self):
        """Verify that vmap simulate works (baseline)."""
        vmapped_fn = two_param_fn.vmap(in_axes=(0, None))

        x_values = jnp.array([1.0, 2.0, 3.0])
        y_value = 0.5

        result = vmapped_fn.simulate(x_values, y_value)
        assert result.get_retval().shape == (3,)

    def test_vmap_generate_two_params(self):
        """Test that vmap generate works with 2 parameters."""
        vmapped_fn = two_param_fn.vmap(in_axes=(0, None))

        x_values = jnp.array([1.0, 2.0, 3.0])
        y_value = 0.5
        constraints = {"sample": jnp.array([1.5, 2.5, 3.5])}

        # This should work after the bug fix
        trace, weight = vmapped_fn.generate(constraints, x_values, y_value)
        assert isinstance(weight, (float, jnp.ndarray))  # JAX returns arrays
        assert trace.get_retval().shape == (3,)

    def test_vmap_generate_three_params(self):
        """Test that vmap generate works with 3 parameters."""
        vmapped_fn = three_param_fn.vmap(in_axes=(0, 0, None))

        a_values = jnp.array([1.0, 2.0])
        b_values = jnp.array([0.5, 1.5])
        c_value = 0.1
        constraints = {"output": jnp.array([1.0, 2.0])}

        # This should work after the bug fix
        trace, weight = vmapped_fn.generate(constraints, a_values, b_values, c_value)
        assert isinstance(weight, (float, jnp.ndarray))  # JAX returns arrays
        assert trace.get_retval().shape == (2,)

    def test_vmap_assess_two_params(self):
        """Test that vmap assess works with 2 parameters."""
        vmapped_fn = two_param_fn.vmap(in_axes=(0, None))

        x_values = jnp.array([1.0, 2.0, 3.0])
        y_value = 0.5
        choices = {"sample": jnp.array([1.5, 2.5, 3.5])}

        log_prob, retval = vmapped_fn.assess(choices, x_values, y_value)
        assert isinstance(log_prob, (float, jnp.ndarray))  # JAX returns arrays
        assert retval.shape == (3,)

    def test_vmap_update_two_params(self):
        """Test that vmap update works with 2 parameters."""
        vmapped_fn = two_param_fn.vmap(in_axes=(0, None))

        # First create a trace
        x_values = jnp.array([1.0, 2.0, 3.0])
        y_value = 0.5
        trace = vmapped_fn.simulate(x_values, y_value)

        # Update with new constraints
        new_constraints = {"sample": jnp.array([2.0, 3.0, 4.0])}
        new_x_values = jnp.array([1.5, 2.5, 3.5])
        new_y_value = 0.8

        new_trace, weight, discarded = vmapped_fn.update(
            trace, new_constraints, new_x_values, new_y_value
        )
        assert isinstance(weight, (float, jnp.ndarray))  # JAX returns arrays
        assert new_trace.get_retval().shape == (3,)


if __name__ == "__main__":
    # Run a quick test to verify the bug still exists before fix
    print("Testing vmap generate bug...")

    vmapped_fn = two_param_fn.vmap(in_axes=(0, None))
    x_values = jnp.array([1.0, 2.0, 3.0])
    y_value = 0.5
    constraints = {"sample": jnp.array([1.5, 2.5, 3.5])}

    try:
        trace, weight = vmapped_fn.generate(constraints, x_values, y_value)
        print("✅ vmap generate works (bug is fixed)")
    except ValueError as e:
        if "vmap in_axes must be" in str(e):
            print("🐛 vmap generate bug still exists")
        else:
            print(f"❌ Unexpected error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
