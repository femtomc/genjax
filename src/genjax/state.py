"""
JAX interpreter for inspecting and organizing tagged state inside JAX Python functions.

This module provides a State interpreter that can collect and hierarchically organize
tagged values from within JAX computations using JAX primitives. The interpreter
works seamlessly with all JAX transformations while providing powerful state
organization capabilities.

Core Features:
============

**State Collection**: Tag intermediate values during computation for inspection
**Hierarchical Organization**: Use namespaces to create nested state structures
**JAX Integration**: Full compatibility with jit, vmap, grad, scan, and other JAX transforms
**Error Safety**: Automatic cleanup of namespace stack on exceptions
**Zero Overhead**: No performance cost when not using the @state decorator

Primary API:
===========

Basic State Collection:
- `state(f)`: Transform function to collect tagged state values
- `save(**tagged_values)`: Tag multiple values by name (named mode)
- `save(*values)`: Save values directly at current namespace leaf (leaf mode)

Hierarchical Organization:
- `namespace(fn, ns)`: Transform function to collect state under namespace
- Supports arbitrary nesting: `namespace(namespace(fn, "inner"), "outer")`

Lower-level API:
===============

- `tag_state(*values, name="...")`: Tag individual values for collection

Usage Examples:
==============

Basic state collection:
```python
@state
def computation(x):
    y = x + 1
    save(intermediate=y, doubled=x*2)
    return y

result, state_dict = computation(5)
# state_dict = {"intermediate": 6, "doubled": 10}
```

Hierarchical organization with named mode:
```python
@state
def complex_computation(x):
    save(input=x)

    # Namespace for processing steps (named mode)
    processing = namespace(
        lambda: save(step1=x*2, step2=x+1),
        "processing"
    )
    processing()

    # Nested namespaces
    analysis = namespace(
        namespace(lambda: save(mean=x), "stats"),
        "analysis"
    )
    analysis()

    return x

result, state_dict = complex_computation(5)
# state_dict = {
#     "input": 5,
#     "processing": {"step1": 10, "step2": 6},
#     "analysis": {"stats": {"mean": 5}}
# }
```

Leaf mode for direct namespace storage:
```python
@state
def leaf_computation(x):
    save(input=x)

    # Leaf mode: save values directly at namespace (no additional keys)
    coords = namespace(lambda: save(x, x*2, x*3), "coordinates")
    coords()

    # Mixed with named mode in different namespace
    stats = namespace(lambda: save(mean=x, variance=x**2), "statistics")
    stats()

    return x

result, state_dict = leaf_computation(5)
# state_dict = {
#     "input": 5,
#     "coordinates": (5, 10, 15),  # Leaf mode: tuple stored directly
#     "statistics": {"mean": 5, "variance": 25}  # Named mode: dict
# }
```

JAX Integration:
```python
# Works with all JAX transformations
jitted_fn = jax.jit(computation)
vmapped_fn = jax.vmap(computation)
grad_fn = jax.grad(lambda x: computation(x)[0])
```

Implementation Details:
======================

The state interpreter uses JAX primitives (`state_p`, `namespace_push_p`,
`namespace_pop_p`) to integrate with JAX's transformation system. This ensures
proper behavior under jit, vmap, grad, and other JAX transforms.

The namespace functionality is implemented using a stack-based approach where
namespace push/pop operations are tracked via JAX primitives, allowing the
interpreter to maintain correct hierarchical structure even under complex
JAX transformations.
"""

from dataclasses import dataclass, field
from functools import wraps

import jax.extend as jex
import jax.tree_util as jtu
from jax.extend.core import Jaxpr
from jax.util import safe_map, split_list
from jax.lax import scan_p, scan

import beartype.typing as btyping

from genjax.pjax import (
    PPPrimitive,
    Environment,
    InitialStylePrimitive,
    stage,
    TerminalStyle,
    initial_style_bind,
)

# Type aliases for convenience
Any = btyping.Any
Callable = btyping.Callable


# State primitive for tagging values to be collected
state_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.GREEN}state.tag{TerminalStyle.RESET}",
)

# Namespace primitives for organizing state
namespace_push_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.PURPLE}namespace.push{TerminalStyle.RESET}",
)

namespace_pop_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.YELLOW}namespace.pop{TerminalStyle.RESET}",
)


# The primitives will use initial_style_bind for dynamic rule creation
# No need for static impl/abstract registration


# Helper functions for nested dictionary operations


def _nested_dict_set(d, path, key, value):
    """Set a value in a nested dictionary using the given path."""
    current = d
    for namespace in path:
        if namespace not in current:
            current[namespace] = {}
        current = current[namespace]
    current[key] = value


def _nested_dict_get(d, path):
    """Get a nested dictionary using the given path."""
    current = d
    for namespace in path:
        if namespace not in current:
            current[namespace] = {}
        current = current[namespace]
    return current


@dataclass
class State:
    """JAX interpreter that collects tagged state values.

    This interpreter processes JAX computations and collects values that
    are tagged with the `state_p` primitive. Tagged values are accumulated
    and returned alongside the original computation result.
    """

    collected_state: dict[str, Any]
    namespace_stack: list[str] = field(default_factory=list)

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
                # Collect the tagged values with namespace support
                name = params.get("name", inner_params.get("name"))
                if name is None:
                    raise ValueError("tag_state requires a 'name' parameter")
                values = list(invals) if invals else []
                value = (
                    tuple(values)
                    if len(values) > 1
                    else (values[0] if values else None)
                )

                # Handle leaf mode storage (special case for save(*args))
                if name == "__NAMESPACE_LEAF__":
                    namespace_path = tuple(self.namespace_stack)
                    if namespace_path:
                        # Store directly at the namespace path (no additional key)
                        current = self.collected_state
                        for namespace in namespace_path[:-1]:
                            if namespace not in current:
                                current[namespace] = {}
                            current = current[namespace]
                        # Store at the final namespace level
                        current[namespace_path[-1]] = value
                    else:
                        # If no namespace, we can't do leaf storage at root
                        raise ValueError(
                            "Leaf mode save() requires being inside a namespace"
                        )
                else:
                    # Handle namespace path using interpreter's stack (named mode)
                    namespace_path = tuple(self.namespace_stack)
                    if namespace_path:
                        _nested_dict_set(
                            self.collected_state, namespace_path, name, value
                        )
                    else:
                        self.collected_state[name] = value

                # The state primitive returns the values as-is due to multiple_results
                outvals = values

            elif primitive == namespace_push_p:
                # Push namespace onto interpreter's stack
                namespace = params.get("namespace", inner_params.get("namespace"))
                if namespace is None:
                    raise ValueError("namespace_push requires a 'namespace' parameter")
                self.namespace_stack.append(namespace)
                # Namespace push doesn't take or return values
                outvals = []

            elif primitive == namespace_pop_p:
                # Pop namespace from interpreter's stack
                if not self.namespace_stack:
                    raise ValueError("namespace_pop called with empty namespace stack")
                self.namespace_stack.pop()
                # Namespace pop doesn't take or return values
                outvals = []

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
        interpreter = State(collected_state={}, namespace_stack=[])
        return interpreter.eval(f, *args)

    return wrapped


def _namespace_push(namespace: str) -> None:
    """Push a namespace onto the interpreter's stack (internal function)."""

    def empty_fn():
        return None

    def batch_rule(vector_args, dims, **params):
        # Re-insert namespace push primitive under vmap
        initial_style_bind(
            namespace_push_p,
            batch=batch_rule,  # Self-reference for nested vmaps
        )(empty_fn, namespace=params.get("namespace"))()
        return (), ()

    initial_style_bind(
        namespace_push_p,
        batch=batch_rule,
    )(empty_fn, namespace=namespace)()


def _namespace_pop() -> None:
    """Pop a namespace from the interpreter's stack (internal function)."""

    def empty_fn():
        return None

    def batch_rule(vector_args, dims, **params):
        # Re-insert namespace pop primitive under vmap
        initial_style_bind(
            namespace_pop_p,
            batch=batch_rule,  # Self-reference for nested vmaps
        )(empty_fn)()
        return (), ()

    initial_style_bind(
        namespace_pop_p,
        batch=batch_rule,
    )(empty_fn)()


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
        return tuple(args) if len(args) > 1 else args[0]

    # Create a batch rule that re-inserts the primitive under vmap
    def batch_rule(vector_args, dims, **params):
        # Re-insert the state primitive with the vectorized args
        def vectorized_identity(*args):
            return tuple(args) if len(args) > 1 else args[0]

        # Apply the primitive to the vectorized args
        result = initial_style_bind(
            state_p,
            batch=batch_rule,  # Self-reference for nested vmaps
        )(vectorized_identity, name=params.get("name"))(*vector_args)

        # Return result with appropriate batching dimensions
        if isinstance(result, tuple):
            # For multiple outputs, each has the same dims as inputs
            return result, tuple(dims[0] if dims else () for _ in result)
        else:
            # For single output, return as tuple (JAX expects a sequence for dims_out)
            return (result,), (dims[0] if dims else (),)

    result = initial_style_bind(
        state_p,
        batch=batch_rule,
    )(identity_fn, name=name)(*values)

    return result


def save(*values, **tagged_values) -> Any:
    """Save values either at current namespace leaf or with explicit names (primary API).

    **This is the recommended way to tag state values.** Supports two modes:

    1. **Leaf mode** (`*args`): Save values directly at current namespace leaf
    2. **Named mode** (`**kwargs`): Save values with explicit names (original behavior)

    Args:
        *values: Values to save at current namespace leaf (mutually exclusive with **tagged_values)
        **tagged_values: Keyword arguments where keys are names and values are the values to save

    Returns:
        - Leaf mode: The values as a tuple (or single value if only one)
        - Named mode: Dictionary of the saved values (for convenience)

    Example:
        Leaf mode (saves at current namespace):
        >>> @state
        >>> def computation():
        ...     namespace_fn = namespace(lambda: save(1, 2, 3), "coords")
        ...     namespace_fn()
        ...     return 42
        >>> result, state_dict = computation()
        >>> # state_dict == {"coords": (1, 2, 3)}

        Named mode (original behavior):
        >>> @state
        >>> def computation():
        ...     values = save(first=1, second=2)
        ...     return sum(values.values())
        >>> result, state_dict = computation()
        >>> # state_dict == {"first": 1, "second": 2}
    """
    if values and tagged_values:
        raise ValueError(
            "Cannot use both positional args (*values) and keyword args (**tagged_values) in save()"
        )

    if values:
        # Leaf mode: save values directly at current namespace leaf
        # Use a special reserved name to indicate leaf storage
        leaf_value = values if len(values) > 1 else values[0]
        tag_state(leaf_value, name="__NAMESPACE_LEAF__")
        return leaf_value
    else:
        # Named mode: original behavior with explicit names
        result = {}
        for name, value in tagged_values.items():
            result[name] = tag_state(value, name=name)
        return result


def namespace(f: Callable[..., Any], ns: str) -> Callable[..., Any]:
    """Transform a function to collect state under a namespace.

    This function wraps another function so that any state collected within
    it will be organized under the specified namespace. Namespaces can be
    nested by applying this function multiple times.

    Args:
        f: Function to wrap with namespace context
        ns: Namespace string to organize state under

    Returns:
        Function that collects state under the specified namespace

    Example:
        >>> @state
        >>> def computation(x):
        ...     # State collected directly at root level
        ...     save(root_val=x)
        ...
        ...     # State collected under "inner" namespace
        ...     inner_fn = namespace(lambda y: save(nested_val=y * 2), "inner")
        ...     inner_fn(x)
        ...
        ...     # Nested namespaces: state under "outer.deep"
        ...     deep_fn = namespace(
        ...         namespace(lambda z: save(deep_val=z * 3), "deep"),
        ...         "outer"
        ...     )
        ...     deep_fn(x)
        ...
        ...     return x
        >>>
        >>> result, state_dict = computation(5)
        >>> # state_dict == {
        >>> #     "root_val": 5,
        >>> #     "inner": {"nested_val": 10},
        >>> #     "outer": {"deep": {"deep_val": 15}}
        >>> # }
    """

    @wraps(f)
    def namespaced_fn(*args, **kwargs):
        # Push namespace using JAX primitive
        _namespace_push(ns)
        try:
            result = f(*args, **kwargs)
            return result
        finally:
            # Always pop namespace, even if function raises
            _namespace_pop()

    return namespaced_fn
