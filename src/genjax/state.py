"""
JAX interpreter for inspecting tagged state inside JAX Python functions.

This module provides a State interpreter that can collect tagged values from
within JAX computations using a special `state_p` primitive. The interpreter
follows the same patterns as the Seed interpreter for consistency.
"""

from functools import wraps

import jax.tree_util as jtu
from jax.extend.core import Jaxpr
from jax.util import safe_map

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
        >>> from genjax.state import state, tag_state
        >>>
        >>> def computation(x):
        ...     y = x + 1
        ...     tag_state(y, name="intermediate")
        ...     return y * 2
        >>>
        >>> state_fn = state(computation)
        >>> result, state_dict = state_fn(5)
        >>> print(result)  # 12
        >>> print(state_dict)  # {"intermediate": 6}
    """

    @wraps(f)
    def wrapped(*args):
        interpreter = State(collected_state={})
        return interpreter.eval(f, *args)

    return wrapped


def tag_state(*values: Any, name: str) -> Any:
    """Tag one or more values to be collected by the StateInterpreter.

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


# Convenience function for saving multiple values
def save(**tagged_values) -> dict[str, Any]:
    """Save multiple values with their corresponding names.

    This is a convenience function that allows saving multiple values
    with different names in a single call. Each value is tagged separately
    using the tag_state function.

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
