"""
Tests for the state interpreter functionality.
"""

import pytest
import jax
import jax.numpy as jnp
from jax.lax import scan

from genjax.state import state, tag_state, save, State, namespace


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
        assert state_dict == {"pair": (6, 10)}

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

    def test_state_after_vmap(self):
        """Test applying state decorator after vmap."""

        def base_computation(x):
            y = x * 2
            tag_state(y, name="doubled")
            z = x + 3
            tag_state(z, name="plus_three")
            return y + z

        # Apply vmap first, then state
        vmapped_then_state = state(jax.vmap(base_computation))

        x_array = jnp.array([1, 2, 3])
        result, state_dict = vmapped_then_state(x_array)

        expected_result = jnp.array([6, 9, 12])  # [1*2+1+3, 2*2+2+3, 3*2+3+3]
        expected_doubled = jnp.array([2, 4, 6])  # [1*2, 2*2, 3*2]
        expected_plus_three = jnp.array([4, 5, 6])  # [1+3, 2+3, 3+3]

        assert jnp.allclose(result, expected_result)
        assert jnp.allclose(state_dict["doubled"], expected_doubled)
        assert jnp.allclose(state_dict["plus_three"], expected_plus_three)

    def test_state_after_vmap_with_namespaces(self):
        """Test applying state decorator after vmap with namespace functionality."""

        def base_computation(x):
            # Root level state
            save(input=x)

            # Namespaced state
            processing_fn = namespace(
                lambda y: save(step1=y * 2, step2=y + 10), "processing"
            )
            processing_fn(x)

            # Leaf mode namespace
            coords_fn = namespace(
                lambda z: save(z, z * 3),  # Save tuple at leaf
                "coords",
            )
            coords_fn(x)

            return x * 5

        # Apply vmap first, then state
        vmapped_then_state = state(jax.vmap(base_computation))

        x_array = jnp.array([2, 4])
        result, state_dict = vmapped_then_state(x_array)

        expected_result = jnp.array([10, 20])  # [2*5, 4*5]

        assert jnp.allclose(result, expected_result)

        # Check vectorized state collection
        assert jnp.allclose(state_dict["input"], jnp.array([2, 4]))

        # Check vectorized named mode namespace
        assert jnp.allclose(state_dict["processing"]["step1"], jnp.array([4, 8]))
        assert jnp.allclose(state_dict["processing"]["step2"], jnp.array([12, 14]))

        # Check vectorized leaf mode namespace - should be tuple of arrays
        coords_values = state_dict["coords"]
        assert len(coords_values) == 2  # Tuple with 2 elements
        # First element: array of z values, second element: array of z*3 values
        assert jnp.allclose(coords_values[0], jnp.array([2, 4]))  # z values
        assert jnp.allclose(coords_values[1], jnp.array([6, 12]))  # z*3 values


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


class TestStateWithScan:
    """Test state functionality with jax.lax.scan."""

    def test_state_with_simple_scan(self):
        """Test state collection within scan body."""

        @state
        def scan_computation(init_carry, xs):
            def step_fn(carry, x):
                new_carry = carry + x
                tag_state(new_carry, name="step_carry")
                return new_carry, new_carry * 2

            final_carry, ys = scan(step_fn, init_carry, xs)
            return final_carry, ys

        xs = jnp.array([1, 2, 3])
        result, state_dict = scan_computation(0, xs)

        final_carry, ys = result
        assert final_carry == 6  # 0 + 1 + 2 + 3
        assert jnp.allclose(ys, jnp.array([2, 6, 12]))  # (1, 3, 6) * 2

        # Check that state was collected from each scan iteration
        assert "step_carry" in state_dict
        collected_carries = state_dict["step_carry"]
        expected_carries = jnp.array([1, 3, 6])  # Accumulated carries
        assert jnp.allclose(collected_carries, expected_carries)

    def test_state_with_multiple_scan_tags(self):
        """Test multiple state tags within scan body."""

        @state
        def multi_tag_scan(init_carry, xs):
            def step_fn(carry, x):
                intermediate = carry + x
                tag_state(intermediate, name="intermediate")

                doubled = intermediate * 2
                tag_state(doubled, name="doubled")

                return intermediate, doubled

            final_carry, ys = scan(step_fn, init_carry, xs)
            return final_carry, ys

        xs = jnp.array([1, 2, 3])
        result, state_dict = multi_tag_scan(5, xs)

        final_carry, ys = result
        assert final_carry == 11  # 5 + 1 + 2 + 3
        assert jnp.allclose(ys, jnp.array([12, 16, 22]))  # (6, 8, 11) * 2

        # Check both state collections
        assert "intermediate" in state_dict
        assert "doubled" in state_dict

        intermediates = state_dict["intermediate"]
        doubled_vals = state_dict["doubled"]

        expected_intermediates = jnp.array([6, 8, 11])  # 5+1, 6+2, 8+3
        expected_doubled = jnp.array([12, 16, 22])  # intermediates * 2
        assert jnp.allclose(intermediates, expected_intermediates)
        assert jnp.allclose(doubled_vals, expected_doubled)

    def test_state_outside_and_inside_scan(self):
        """Test state collection both outside and inside scan."""

        @state
        def mixed_state_computation(init_carry, xs):
            # Tag state before scan
            tag_state(init_carry, name="initial")

            def step_fn(carry, x):
                new_carry = carry + x
                tag_state(new_carry, name="scan_step")
                return new_carry, new_carry

            final_carry, ys = scan(step_fn, init_carry, xs)

            # Tag state after scan
            result = final_carry * 10
            tag_state(result, name="final")

            return result

        xs = jnp.array([1, 2])
        result, state_dict = mixed_state_computation(3, xs)

        assert result == 60  # (3 + 1 + 2) * 10

        # Check all collected state
        assert "initial" in state_dict
        assert "scan_step" in state_dict
        assert "final" in state_dict

        assert state_dict["initial"] == 3
        expected_scan_steps = jnp.array([4, 6])  # 3+1, 4+2
        assert jnp.allclose(state_dict["scan_step"], expected_scan_steps)
        assert state_dict["final"] == 60


# =============================================================================
# NAMESPACE TESTS
# =============================================================================


class TestNamespace:
    """Test the namespace functionality for organizing state."""

    def test_single_namespace(self):
        """Test basic single namespace functionality."""

        @state
        def computation(x):
            # Root level state
            save(root_val=x)

            # Namespaced state
            inner_fn = namespace(lambda y: save(nested_val=y * 2), "inner")
            inner_fn(x)

            return x * 3

        result, state_dict = computation(5)
        assert result == 15
        assert state_dict == {"root_val": 5, "inner": {"nested_val": 10}}

    def test_nested_namespaces(self):
        """Test nested namespace functionality."""

        @state
        def computation(x):
            # Root level
            save(root=x)

            # Single namespace
            level1_fn = namespace(lambda y: save(l1_val=y + 1), "level1")
            level1_fn(x)

            # Nested namespace: level1.level2
            level2_fn = namespace(
                namespace(lambda z: save(l2_val=z + 2), "level2"), "level1"
            )
            level2_fn(x)

            # Different top-level namespace
            other_fn = namespace(lambda w: save(other_val=w + 3), "other")
            other_fn(x)

            return x

        result, state_dict = computation(10)
        assert result == 10
        assert state_dict == {
            "root": 10,
            "level1": {"l1_val": 11, "level2": {"l2_val": 12}},
            "other": {"other_val": 13},
        }

    def test_multiple_values_in_namespace(self):
        """Test multiple values saved in the same namespace."""

        @state
        def computation(x):
            save(root=x)

            # Multiple values in same namespace
            multi_fn = namespace(
                lambda y: save(first=y * 2, second=y * 3, third=y * 4), "multi"
            )
            multi_fn(x)

            return x

        result, state_dict = computation(3)
        assert result == 3
        assert state_dict == {
            "root": 3,
            "multi": {"first": 6, "second": 9, "third": 12},
        }

    def test_namespace_with_function_calls(self):
        """Test namespace with nested function calls."""

        def inner_computation(x):
            save(inner=x * 2)
            return x + 1

        @state
        def computation(x):
            save(start=x)

            # Namespace a function that calls other functions
            namespaced_fn = namespace(inner_computation, "ns")
            result = namespaced_fn(x)

            save(end=result)
            return result

        result, state_dict = computation(5)
        assert result == 6  # 5 + 1
        assert state_dict == {
            "start": 5,
            "ns": {"inner": 10},  # 5 * 2
            "end": 6,
        }

    def test_namespace_error_handling(self):
        """Test that namespace stack is properly cleaned up on errors."""

        @state
        def computation_with_error(x):
            save(before=x)

            # This should clean up namespace even if error occurs
            error_fn = namespace(
                lambda y: (_ for _ in ()).throw(ValueError("test error")), "error_ns"
            )

            try:
                error_fn(x)
            except ValueError:
                pass  # Expected error

            # This should still work at root level
            save(after=x + 1)
            return x

        result, state_dict = computation_with_error(5)
        assert result == 5
        assert state_dict == {"before": 5, "after": 6}
        # Namespace should not appear since error occurred before save

    def test_namespace_with_jax_arrays(self):
        """Test namespace with JAX arrays."""

        @state
        def computation(x):
            arr = jnp.array([1, 2, 3]) + x
            save(root_array=arr)

            # Namespace with array computation
            array_fn = namespace(
                lambda y: save(doubled=y * 2, summed=jnp.sum(y)), "arrays"
            )
            array_fn(arr)

            return jnp.sum(arr)

        result, state_dict = computation(2)
        expected_arr = jnp.array([3, 4, 5])

        assert result == 12  # sum([3, 4, 5])
        assert jnp.allclose(state_dict["root_array"], expected_arr)
        assert jnp.allclose(state_dict["arrays"]["doubled"], expected_arr * 2)
        assert state_dict["arrays"]["summed"] == 12

    def test_deep_nested_namespaces(self):
        """Test deeply nested namespace structures."""

        @state
        def computation(x):
            save(root=x)

            # Create a deeply nested structure: a.b.c.d
            deep_fn = namespace(
                namespace(
                    namespace(namespace(lambda y: save(deep_val=y * 10), "d"), "c"), "b"
                ),
                "a",
            )
            deep_fn(x)

            return x

        result, state_dict = computation(3)
        assert result == 3
        assert state_dict == {"root": 3, "a": {"b": {"c": {"d": {"deep_val": 30}}}}}

    def test_namespace_with_tag_state_directly(self):
        """Test namespace works with tag_state as well as save."""

        @state
        def computation(x):
            tag_state(x, name="root_tag")

            # Use tag_state inside namespace
            tag_fn = namespace(
                lambda y: tag_state(y * 5, name="namespaced_tag"), "tagged"
            )
            tag_fn(x)

            return x

        result, state_dict = computation(4)
        assert result == 4
        assert state_dict == {"root_tag": 4, "tagged": {"namespaced_tag": 20}}

    def test_namespace_overwrite_same_key(self):
        """Test that namespace handles key overwrites properly."""

        @state
        def computation(x):
            # Same key in different namespaces
            ns1_fn = namespace(lambda y: save(same_key=y * 2), "ns1")
            ns2_fn = namespace(lambda y: save(same_key=y * 3), "ns2")

            ns1_fn(x)
            ns2_fn(x)

            # Same key overwritten in same namespace
            overwrite_fn = namespace(
                lambda y: [save(key=y), save(key=y * 10)], "overwrite"
            )
            overwrite_fn(x)

            return x

        result, state_dict = computation(2)
        assert result == 2
        assert state_dict == {
            "ns1": {"same_key": 4},
            "ns2": {"same_key": 6},
            "overwrite": {"key": 20},  # Second save overwrites first
        }


# =============================================================================
# LEAF MODE SAVE TESTS
# =============================================================================


class TestSaveLeafMode:
    """Test the new save(*args) leaf mode functionality."""

    def test_save_single_value_leaf_mode(self):
        """Test saving a single value at namespace leaf."""

        @state
        def computation(x):
            save(root=x)

            # Leaf mode: save single value at current namespace
            leaf_fn = namespace(lambda y: save(y * 2), "coords")
            leaf_fn(x)

            return x

        result, state_dict = computation(5)
        assert result == 5
        assert state_dict == {
            "root": 5,
            "coords": 10,  # Single value stored directly at namespace
        }

    def test_save_multiple_values_leaf_mode(self):
        """Test saving multiple values as tuple at namespace leaf."""

        @state
        def computation(x):
            save(root=x)

            # Leaf mode: save multiple values as tuple at current namespace
            leaf_fn = namespace(lambda y: save(y, y * 2, y * 3), "coords")
            leaf_fn(x)

            return x

        result, state_dict = computation(3)
        assert result == 3
        assert state_dict == {
            "root": 3,
            "coords": (3, 6, 9),  # Multiple values stored as tuple
        }

    def test_save_leaf_mode_nested_namespaces(self):
        """Test leaf mode with nested namespaces."""

        @state
        def computation(x):
            save(root=x)

            # Nested namespaces with leaf storage
            deep_fn = namespace(namespace(lambda y: save(y * 10), "inner"), "outer")
            deep_fn(x)

            return x

        result, state_dict = computation(4)
        assert result == 4
        assert state_dict == {
            "root": 4,
            "outer": {"inner": 40},  # Value stored at leaf of nested namespace
        }

    def test_save_leaf_mode_mixed_with_named_mode(self):
        """Test mixing leaf mode and named mode in different namespaces."""

        @state
        def computation(x):
            save(root=x)

            # Named mode in one namespace
            named_fn = namespace(lambda y: save(named_val=y * 2), "named")
            named_fn(x)

            # Leaf mode in another namespace
            leaf_fn = namespace(lambda y: save(y * 3), "leaf")
            leaf_fn(x)

            return x

        result, state_dict = computation(5)
        assert result == 5
        assert state_dict == {
            "root": 5,
            "named": {"named_val": 10},  # Named mode
            "leaf": 15,  # Leaf mode
        }

    def test_save_leaf_mode_overwrite(self):
        """Test that leaf mode overwrites values at same namespace."""

        @state
        def computation(x):
            # Multiple leaf saves in same namespace should overwrite
            overwrite_fn = namespace(
                lambda y: [save(y), save(y * 10)],  # Second save overwrites first
                "overwrite",
            )
            overwrite_fn(x)

            return x

        result, state_dict = computation(3)
        assert result == 3
        assert state_dict == {
            "overwrite": 30  # Second value overwrites first
        }

    def test_save_leaf_mode_with_jax_arrays(self):
        """Test leaf mode with JAX arrays."""

        @state
        def computation(x):
            arr = jnp.array([1, 2, 3]) + x
            save(root_array=arr)

            # Leaf mode with array
            array_fn = namespace(lambda y: save(y * 2), "array_ns")
            array_fn(arr)

            return jnp.sum(arr)

        result, state_dict = computation(2)
        expected_arr = jnp.array([3, 4, 5])

        assert result == 12
        assert jnp.allclose(state_dict["root_array"], expected_arr)
        assert jnp.allclose(state_dict["array_ns"], expected_arr * 2)

    def test_save_leaf_mode_error_at_root(self):
        """Test that leaf mode raises error when used at root level."""

        @state
        def computation(x):
            # This should raise an error - can't use leaf mode at root
            save(x)  # No namespace context
            return x

        with pytest.raises(
            ValueError, match="Leaf mode save\\(\\) requires being inside a namespace"
        ):
            computation(5)

    def test_save_mixed_args_kwargs_error(self):
        """Test that mixing *args and **kwargs raises error."""

        @state
        def computation(x):
            # This should raise error - can't mix positional and keyword args
            namespace_fn = namespace(
                lambda y: save(y, named_val=y * 2),  # Mixed args and kwargs
                "mixed",
            )
            return namespace_fn(x)

        with pytest.raises(
            ValueError, match="Cannot use both positional args.*and keyword args"
        ):
            computation(5)

    def test_save_leaf_mode_return_value(self):
        """Test that leaf mode returns the saved values correctly."""

        @state
        def computation(x):
            # Test return value of leaf mode save
            leaf_fn = namespace(
                lambda y: save(y, y * 2),  # Should return tuple (y, y*2)
                "return_test",
            )
            returned = leaf_fn(x)

            # Use the returned value in computation
            save(returned_sum=sum(returned))
            return sum(returned)

        result, state_dict = computation(3)
        assert result == 9  # 3 + 6
        assert state_dict == {
            "return_test": (3, 6),  # Leaf mode storage
            "returned_sum": 9,  # Using returned value
        }

    def test_save_leaf_mode_single_value_return(self):
        """Test that leaf mode with single value returns scalar, not tuple."""

        @state
        def computation(x):
            # Single value should return scalar
            leaf_fn = namespace(
                lambda y: save(y * 5),  # Should return scalar, not tuple
                "single",
            )
            returned = leaf_fn(x)

            # Check that returned is scalar (not tuple) by using it directly
            save(returned_plus_one=returned + 1)
            return returned

        result, state_dict = computation(4)
        assert result == 20
        assert state_dict == {
            "single": 20,  # Scalar storage
            "returned_plus_one": 21,  # Can use scalar directly
        }
