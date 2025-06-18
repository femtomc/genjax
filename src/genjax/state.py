"""
JAX interpreter for inspecting tagged state inside JAX Python functions.

This module provides a State interpreter that can collect tagged values from
within JAX computations using a special `state_p` primitive. The interpreter
follows the same patterns as the Seed interpreter for consistency.

Primary API:
- `save(**tagged_values)`: Recommended way to tag multiple values by name
- `state(f)`: Transform function to collect tagged state values

Lower-level API:
- `tag_state(*values, name="...")`: Tag individual values for collection
"""

from functools import wraps

import jax.extend as jex
import jax.tree_util as jtu
from jax.extend.core import Jaxpr
from jax.util import safe_map, split_list
from jax.lax import scan_p, scan

from genjax.core import (
    Callable,
    Any,
    Pytree,
)
from genjax.pjax import (
    PPPrimitive,
    Environment,
    InitialStylePrimitive,
    stage,
    TerminalStyle,
    initial_style_bind,
)


# State primitive for tagging values to be collected
state_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.GREEN}state.tag{TerminalStyle.RESET}",
)


# The state_p primitive will use initial_style_bind for dynamic rule creation
# No need for static impl/abstract registration


@Pytree.dataclass
class State(Pytree):
    """JAX interpreter that collects tagged state values.

    This interpreter processes JAX computations and collects values that
    are tagged with the `state_p` primitive. Tagged values are accumulated
    and returned alongside the original computation result.
    """

    collected_state: dict[str, Any]

    def eval_jaxpr_state(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        """Evaluate a jaxpr while collecting tagged state values."""
        env = Environment()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, args)

        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            primitive, inner_params = PPPrimitive.unwrap(eqn.primitive)

            if primitive == state_p:
                # Collect the tagged values
                name = params.get("name", inner_params.get("name"))
                if name is None:
                    raise ValueError("tag_state requires a 'name' parameter")
                values = list(invals) if invals else []
                self.collected_state[name] = (
                    values if len(values) > 1 else (values[0] if values else None)
                )
                # The state primitive returns the values as-is due to multiple_results
                outvals = values

            elif primitive == scan_p:
                # Handle scan primitive by transforming body to collect state
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = split_list(
                    invals, [num_consts, num_carry]
                )

                body_fun = jex.core.jaxpr_as_fun(body_jaxpr)

                def new_body(carry, scanned_in):
                    in_carry = carry
                    all_values = const_vals + jtu.tree_leaves((in_carry, scanned_in))
                    # Apply state transformation to the body
                    body_result, body_state = state(body_fun)(*all_values)
                    # Split the body result back into carry and scan parts
                    out_carry, out_scan = split_list(
                        jtu.tree_leaves(body_result), [num_carry]
                    )
                    # Return carry, scan output, and collected state
                    return out_carry, (out_scan, body_state)

                flat_carry_out, (scanned_out, scan_states) = scan(
                    new_body,
                    carry_vals,
                    xs_vals,
                    length=length,
                    reverse=reverse,
                )

                # Merge vectorized scan states into collected state
                # scan_states is already vectorized by scan - just merge it
                for name, vectorized_values in scan_states.items():
                    self.collected_state[name] = vectorized_values

                outvals = jtu.tree_leaves(
                    (flat_carry_out, scanned_out),
                )

            else:
                # For all other primitives, use normal JAX evaluation
                outvals = eqn.primitive.bind(*args, **params)
                if not eqn.outvars:
                    outvals = []
                elif isinstance(outvals, (list, tuple)):
                    outvals = list(outvals)
                else:
                    outvals = [outvals]

            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    def eval(self, fn, *args):
        """Run the interpreter on a function with given arguments."""
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_state(
            jaxpr,
            consts,
            flat_args,
        )
        result = jtu.tree_unflatten(out_tree(), flat_out)
        return result, self.collected_state


def state(f: Callable[..., Any]):
    """Transform a function to collect tagged state values.

    This transformation wraps a function to intercept and collect values
    that are tagged with the `state_p` primitive. The transformed function
    returns both the original result and a dictionary of collected state.

    Args:
        f: Function containing state tags to transform.

    Returns:
        Function that returns a tuple of (original_result, collected_state).

    Example:
        >>> from genjax.state import state, save
        >>>
        >>> @state
        >>> def computation(x):
        ...     y = x + 1
        ...     z = x * 2
        ...     values = save(intermediate=y, doubled=z)
        ...     return values["intermediate"] * 2
        >>>
        >>> result, state_dict = computation(5)
        >>> print(result)  # 12
        >>> print(state_dict)  # {"intermediate": 6, "doubled": 10}
    """

    @wraps(f)
    def wrapped(*args):
        interpreter = State(collected_state={})
        return interpreter.eval(f, *args)

    return wrapped


def tag_state(*values: Any, name: str) -> Any:
    """Tag one or more values to be collected by the StateInterpreter.

    **Note: Consider using `save(**tagged_values)` for most use cases, as it
    provides a more convenient API for tagging multiple values.**

    This function marks values to be collected when the computation
    is run through the `state` transformation. The values are passed
    through unchanged in normal execution.

    Args:
        *values: The values to tag and collect.
        name: Required string identifier for this state value.

    Returns:
        The original values (identity function). If single value, returns
        the value directly. If multiple values, returns a tuple.

    Example:
        >>> x = 42
        >>> y = tag_state(x, name="my_value")  # y == x == 42
        >>> # Multiple values
        >>> a, b = tag_state(1, 2, name="pair")  # a == 1, b == 2
        >>> # When run through state() transformation,
        >>> # values will be collected in state dict
        >>>
        >>> # Prefer save() for multiple named values:
        >>> values = save(x=42, y=24)  # More convenient
    """
    if not values:
        raise ValueError("tag_state requires at least one value")

    # Use initial_style_bind for proper JAX transformation compatibility
    def identity_fn(*args):
        return args if len(args) > 1 else args[0]

    # Create a simple batch rule that preserves the identity operation
    def batch_rule(vector_args, dims, **params):
        # For identity operation, we just return the args with the same dims
        return vector_args, dims

    result = initial_style_bind(
        state_p,
        batch=batch_rule,
    )(identity_fn, name=name)(*values)

    return result


def save(**tagged_values) -> dict[str, Any]:
    """Save multiple values with their corresponding names (primary API).

    **This is the recommended way to tag state values.** It provides a clean,
    convenient interface for tagging multiple values with different names in
    a single call. Each value is tagged separately using the tag_state function.

    Args:
        **tagged_values: Keyword arguments where keys are names and
                        values are the values to save.

    Returns:
        Dictionary of the same saved values (for convenience).

    Example:
        >>> x, y = 1, 2
        >>> values = save(first=x, second=y)
        >>> # values == {"first": 1, "second": 2}
        >>> # When run through state(), both will be collected
        >>>
        >>> @state
        >>> def computation():
        ...     values = save(a=10, b=20, c=30)
        ...     return sum(values.values())
        >>> result, state_dict = computation()
        >>> # state_dict == {"a": 10, "b": 20, "c": 30}
    """
    result = {}
    for name, value in tagged_values.items():
        result[name] = tag_state(value, name=name)
    return result
