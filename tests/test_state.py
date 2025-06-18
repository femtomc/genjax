"""
Tests for the state interpreter functionality.
"""

import pytest
import jax
import jax.numpy as jnp

from genjax.state import state, tag_state, save, State


class TestTagState:
    """Test the tag_state function."""

    def test_single_value_tagging(self):
        """Test tagging a single value."""

        @state
        def computation(x):
            y = x + 1
            tagged_y = tag_state(y, name="intermediate")
            return tagged_y * 2

        result, state_dict = computation(5)
        assert result == 12
        assert state_dict == {"intermediate": 6}

    def test_multiple_values_tagging(self):
        """Test tagging multiple values at once."""

        @state
        def computation(x):
            y = x + 1
            z = x * 2
            tagged_y, tagged_z = tag_state(y, z, name="pair")
            return tagged_y + tagged_z

        result, state_dict = computation(5)
        assert result == 16  # 6 + 10
        assert state_dict == {"pair": [6, 10]}

    def test_multiple_separate_tags(self):
        """Test tagging multiple separate values."""

        @state
        def computation(x):
            y = x + 1
            z = x * 2
            tagged_y = tag_state(y, name="first")
            tagged_z = tag_state(z, name="second")
            return tagged_y + tagged_z

        result, state_dict = computation(5)
        assert result == 16  # 6 + 10
        assert state_dict == {"first": 6, "second": 10}

    def test_nested_function_calls(self):
        """Test tagging with nested function calls."""

        def inner_computation(x):
            y = x + 1
            tag_state(y, name="inner")
            return y

        @state
        def outer_computation(x):
            inner_result = inner_computation(x)
            tag_state(inner_result, name="outer")
            return inner_result * 2

        result, state_dict = outer_computation(5)
        assert result == 12  # (5 + 1) * 2
        assert state_dict == {"inner": 6, "outer": 6}

    def test_empty_values_error(self):
        """Test that tag_state raises error with no values."""
        with pytest.raises(ValueError, match="tag_state requires at least one value"):
            tag_state(name="test")

    def test_missing_name_error(self):
        """Test that tag_state raises error without name parameter."""
        with pytest.raises(TypeError, match="missing.*required.*argument.*name"):
            tag_state(42)

    def test_jax_arrays(self):
        """Test tagging JAX arrays."""

        @state
        def computation(x):
            y = jnp.array([1, 2, 3]) + x
            tagged_y = tag_state(y, name="array")
            return jnp.sum(tagged_y)

        result, state_dict = computation(1)
        expected_array = jnp.array([2, 3, 4])
        assert result == 9  # sum([2, 3, 4])
        assert jnp.allclose(state_dict["array"], expected_array)

    def test_multiple_jax_arrays(self):
        """Test tagging multiple JAX arrays."""

        @state
        def computation(x):
            y = jnp.array([1, 2, 3]) + x
            z = jnp.array([4, 5, 6]) * x
            tagged_y, tagged_z = tag_state(y, z, name="arrays")
            return jnp.sum(tagged_y) + jnp.sum(tagged_z)

        result, state_dict = computation(2)
        expected_y = jnp.array([3, 4, 5])
        expected_z = jnp.array([8, 10, 12])
        assert result == 42  # sum([3, 4, 5]) + sum([8, 10, 12])
        assert len(state_dict["arrays"]) == 2
        assert jnp.allclose(state_dict["arrays"][0], expected_y)
        assert jnp.allclose(state_dict["arrays"][1], expected_z)


class TestSave:
    """Test the save convenience function."""

    def test_multiple_values_convenience(self):
        """Test the save convenience function."""

        @state
        def computation(x):
            y = x + 1
            z = x * 2
            values = save(first=y, second=z)
            return values["first"] + values["second"]

        result, state_dict = computation(5)
        assert result == 16  # 6 + 10
        assert state_dict == {"first": 6, "second": 10}

    def test_empty_save(self):
        """Test empty save call."""

        @state
        def computation(x):
            save()  # No return value needed for this test
            return x * 2

        result, state_dict = computation(5)
        assert result == 10
        assert state_dict == {}


class TestState:
    """Test the State class directly."""

    def test_direct_interpreter_usage(self):
        """Test using State directly."""

        def computation(x):
            y = x + 1
            tag_state(y, name="direct")
            return y * 2

        interpreter = State(collected_state={})
        result, state_dict = interpreter.eval(computation, 5)
        assert result == 12
        assert state_dict == {"direct": 6}

    def test_interpreter_accumulates_state(self):
        """Test that interpreter accumulates state across calls."""

        def computation1(x):
            tag_state(x, name="first")
            return x

        def computation2(x):
            tag_state(x, name="second")
            return x

        interpreter = State(collected_state={})

        # First call
        result1, state_dict1 = interpreter.eval(computation1, 5)
        assert result1 == 5
        assert state_dict1 == {"first": 5}

        # Second call - should accumulate
        result2, state_dict2 = interpreter.eval(computation2, 10)
        assert result2 == 10
        assert state_dict2 == {"first": 5, "second": 10}


class TestStateWithJAXTransforms:
    """Test state functionality with JAX transformations."""

    def test_state_with_jit(self):
        """Test state with JAX jit compilation."""

        @state
        def computation(x):
            y = x + 1
            tag_state(y, name="jitted")
            return y * 2

        jitted_computation = jax.jit(computation)
        result, state_dict = jitted_computation(5)
        assert result == 12
        assert state_dict == {"jitted": 6}

    def test_state_with_vmap(self):
        """Test state with JAX vmap."""

        @state
        def computation(x):
            y = x + 1
            tag_state(y, name="vmapped")
            return y * 2

        vmapped_computation = jax.vmap(computation)
        x_array = jnp.array([1, 2, 3])
        result, state_dict = vmapped_computation(x_array)

        expected_result = jnp.array([4, 6, 8])  # (1+1)*2, (2+1)*2, (3+1)*2
        expected_state = jnp.array([2, 3, 4])  # 1+1, 2+1, 3+1

        assert jnp.allclose(result, expected_result)
        assert jnp.allclose(state_dict["vmapped"], expected_state)

    def test_state_with_grad(self):
        """Test state with JAX grad."""

        @state
        def computation(x):
            y = x**2
            tag_state(y, name="squared")
            return y

        grad_computation = jax.grad(
            lambda x: computation(x)[0]
        )  # Get result, not state
        gradient = grad_computation(3.0)
        assert jnp.isclose(gradient, 6.0)  # d/dx(x^2) = 2x = 2*3 = 6


class TestComplexStateScenarios:
    """Test complex scenarios with state tagging."""

    def test_multiple_computations_state_collection(self):
        """Test multiple computations with state collection."""

        @state
        def multi_step_computation(x):
            step1 = x + 1
            tag_state(step1, name="step_1")

            step2 = step1 * 2
            tag_state(step2, name="step_2")

            step3 = step2 + 3
            tag_state(step3, name="step_3")

            final = step3**2
            tag_state(final, name="final")

            return final

        result, state_dict = multi_step_computation(3)
        assert (
            result == 121
        )  # ((3 + 1) * 2 + 3) ** 2 = (4 * 2 + 3) ** 2 = (8 + 3) ** 2 = 11 ** 2 = 121
        assert "step_1" in state_dict
        assert "step_2" in state_dict
        assert "step_3" in state_dict
        assert "final" in state_dict
        assert state_dict["step_1"] == 4
        assert state_dict["step_2"] == 8
        assert state_dict["step_3"] == 11
        assert state_dict["final"] == 121

    def test_different_computations_same_input(self):
        """Test different computations with the same input."""

        @state
        def computation_with_tagging(x):
            y = x + 1
            tag_state(y, name="tagged_value")
            return y * 2

        @state
        def computation_without_tagging(x):
            y = x + 1
            return y * 2

        # With tagging
        result1, state_dict1 = computation_with_tagging(5)
        assert result1 == 12
        assert state_dict1 == {"tagged_value": 6}

        # Without tagging
        result2, state_dict2 = computation_without_tagging(5)
        assert result2 == 12
        assert state_dict2 == {}

    def test_mixed_value_types(self):
        """Test tagging mixed value types."""

        @state
        def mixed_computation(x):
            scalar = x + 1
            array = jnp.array([x, x + 1, x + 2])

            tag_state(scalar, name="scalar")
            tag_state(array, name="array")

            return scalar + jnp.sum(array)

        result, state_dict = mixed_computation(3)
        assert result == 16  # 4 + (3+4+5)
        assert state_dict["scalar"] == 4
        assert jnp.allclose(state_dict["array"], jnp.array([3, 4, 5]))

    def test_many_different_tags(self):
        """Test handling several different state tags."""

        @state
        def many_tags_computation(x):
            # Create multiple tagged values using different computations
            a = x + 1
            tag_state(a, name="a")

            b = x * 2
            tag_state(b, name="b")

            c = x**2
            tag_state(c, name="c")

            d = x - 1
            tag_state(d, name="d")

            e = x / 2
            tag_state(e, name="e")

            return a + b + c + d + e

        result, state_dict = many_tags_computation(4)
        assert len(state_dict) == 5
        assert state_dict["a"] == 5  # 4 + 1
        assert state_dict["b"] == 8  # 4 * 2
        assert state_dict["c"] == 16  # 4 ** 2
        assert state_dict["d"] == 3  # 4 - 1
        assert state_dict["e"] == 2  # 4 / 2

    def test_overwrite_same_name(self):
        """Test that same name overwrites previous value."""

        @state
        def overwrite_computation(x):
            y = x + 1
            tag_state(y, name="value")
            z = x + 2
            tag_state(z, name="value")  # Same name, should overwrite
            return y + z

        result, state_dict = overwrite_computation(5)
        assert result == 13  # 6 + 7
        assert state_dict == {"value": 7}  # Should be the second value
