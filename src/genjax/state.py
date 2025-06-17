"""
JAX interpreter for inspecting tagged state inside JAX Python functions.

This module provides a StateInterpreter that can collect tagged values from
within JAX computations using a special `state_p` primitive. The interpreter
follows the same patterns as the SeedInterpreter for consistency.
"""

from dataclasses import dataclass
from functools import wraps

import jax.tree_util as jtu
from jax.extend.core import Jaxpr
from jax.util import safe_map

from genjax.core import (
    Callable,
    Any,
)
from genjax.pjax import (
    PPPrimitive,
    Environment,
    Pytree,
    InitialStylePrimitive,
    stage,
    style,
)


# State primitive for tagging values to be collected
state_p = InitialStylePrimitive(
    f"{style.BOLD}{style.GREEN}state.tag{style.RESET}",
)


# Implementation for state_p primitive - just passes through the value
def _state_p_impl(value, **params):
    """Implementation for state primitive - identity function."""
    return [value]  # Return as list for multiple_results


def _state_p_abstract(value_aval, **params):
    """Abstract evaluation for state primitive - preserves shape and dtype."""
    return [value_aval]  # Return as list for multiple_results


# Register the implementations
state_p.def_impl(_state_p_impl)
state_p.def_abstract_eval(_state_p_abstract)


@dataclass
class StateInterpreter(Pytree):
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
                # Collect the tagged value
                tag = params.get("tag", inner_params.get("tag", "unnamed"))
                value = invals[0] if invals else None
                self.collected_state[tag] = value
                # The state primitive returns [value] due to multiple_results
                outvals = [value] if value is not None else []

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

    def run_interpreter(self, fn, *args):
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
        ...     tag_state(y, "intermediate")
        ...     return y * 2
        >>>
        >>> state_fn = state(computation)
        >>> result, state_dict = state_fn(5)
        >>> print(result)  # 12
        >>> print(state_dict)  # {"intermediate": 6}
    """

    @wraps(f)
    def wrapped(*args):
        interpreter = StateInterpreter()
        return interpreter.run_interpreter(f, *args)

    return wrapped


def tag_state(value: Any, tag: str = "unnamed") -> Any:
    """Tag a value to be collected by the StateInterpreter.

    This function marks a value to be collected when the computation
    is run through the `state` transformation. The value is passed
    through unchanged in normal execution.

    Args:
        value: The value to tag and collect.
        tag: String identifier for this state value.

    Returns:
        The original value (identity function).

    Example:
        >>> x = 42
        >>> y = tag_state(x, "my_value")  # y == x == 42
        >>> # When run through state() transformation,
        >>> # "my_value" will be collected in state dict
    """
    result = state_p.bind(value, tag=tag)
    # state_p returns a list due to multiple_results, so extract the first element
    return (
        result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
    )


# Convenience function for multiple tags
def tag_states(**tagged_values) -> dict[str, Any]:
    """Tag multiple values with their corresponding tags.

    Args:
        **tagged_values: Keyword arguments where keys are tags and
                        values are the values to tag.

    Returns:
        Dictionary of the same tagged values (for convenience).

    Example:
        >>> x, y = 1, 2
        >>> values = tag_states(first=x, second=y)
        >>> # values == {"first": 1, "second": 2}
        >>> # When run through state(), both will be collected
    """
    result = {}
    for tag, value in tagged_values.items():
        result[tag] = tag_state(value, tag)
    return result
