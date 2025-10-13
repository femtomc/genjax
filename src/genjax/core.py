from abc import abstractmethod
from dataclasses import dataclass, field
from typing import overload
import inspect

import beartype.typing as btyping
import jax.numpy as jnp
import jax.tree_util as jtu
import jaxtyping as jtyping
import penzai.pz as pz
from jax.lax import scan
from tensorflow_probability.substrates import jax as tfp
from typing_extensions import dataclass_transform

# Import PJAX functionality that was moved from this file
from .pjax import (
    modular_vmap,
    wrap_sampler,
    wrap_logpdf,
)

tfd = tfp.distributions

##########
# Typing #
##########

Any = btyping.Any
Union = btyping.Union
Addr = btyping.Tuple | str
PRNGKey = jtyping.PRNGKeyArray
Array = jtyping.Array
ArrayLike = jtyping.ArrayLike
FloatArray = jtyping.Float[jtyping.Array, "..."]
IntArray = jtyping.Int[jtyping.Array, "..."]
BoolArray = jtyping.Bool[jtyping.Array, "..."]
Callable = btyping.Callable
TypeAlias = btyping.TypeAlias
Sequence = btyping.Sequence
Iterable = btyping.Iterable
Optional = btyping.Optional
Final = btyping.Final
Generator = btyping.Generator
Generic = btyping.Generic
TypeVar = btyping.TypeVar
Annotated = btyping.Annotated

A = TypeVar("A")
R = TypeVar("R")
X = TypeVar("X")
Rm = TypeVar("Rm")
K = TypeVar("K")

#######################
# Probabilistic types #
#######################

Weight = FloatArray
Score = FloatArray
Density = FloatArray

##########
# Pytree #
##########


class Pytree(pz.Struct):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system. JAX's `Pytree` system tracks how data classes should behave across
    JAX-transformed function boundaries, like `jax.jit` or `jax.vmap`.

    Inheriting this class provides the implementor with the freedom to
    declare how the subfields of a class should behave:

    * `Pytree.static(...)`: the value of the field cannot
    be a JAX traced value, it must be a Python literal, or a constant).
    The values of static fields are embedded in the `PyTreeDef` of any
    instance of the class.
    * `Pytree.field(...)` or no annotation: the value may be a JAX traced
    value, and JAX will attempt to convert it to tracer values inside of
    its transformations.

    If a field _points to another `Pytree`_, it should not be declared as
    `Pytree.static()`, as the `Pytree` interface will automatically handle
    the `Pytree` fields as dynamic fields.
    """

    @staticmethod
    @overload
    def dataclass(
        incoming: None = None,
        /,
        **kwargs,
    ) -> Callable[[type[R]], type[R]]: ...

    @staticmethod
    @overload
    def dataclass(
        incoming: type[R],
        /,
        **kwargs,
    ) -> type[R]: ...

    @dataclass_transform(
        frozen_default=True,
    )
    @staticmethod
    def dataclass(
        incoming: type[R] | None = None,
        /,
        **kwargs,
    ) -> type[R] | Callable[[type[R]], type[R]]:
        """
        Denote that a class (which is inheriting `Pytree`) should be treated
        as a dataclass, meaning it can hold data in fields which are
        declared as part of the class.

        A dataclass is to be distinguished from a "methods only"
        `Pytree` class, which does not have fields, but may define methods.
        The latter cannot be instantiated, but can be inherited from,
        while the former can be instantiated:
        the `Pytree.dataclass` declaration informs the system _how
        to instantiate_ the class as a dataclass,
        and how to automatically define JAX's `Pytree` interfaces
        (`tree_flatten`, `tree_unflatten`, etc.) for the dataclass, based
        on the fields declared in the class, and possibly `Pytree.static(...)`
        or `Pytree.field(...)` annotations (or lack thereof, the default is
        that all fields are `Pytree.field(...)`).

        All `Pytree` dataclasses support pretty printing, as well as rendering
        to HTML.

        Examples
        --------

        >>> from genjax import Pytree
        >>> from jaxtyping import ArrayLike
        >>> import jax.numpy as jnp
        >>>
        >>> @Pytree.dataclass
        ... class MyClass(Pytree):
        ...     my_static_field: int = Pytree.static()
        ...     my_dynamic_field: ArrayLike
        >>>
        >>> instance = MyClass(10, jnp.array(5.0))
        >>> instance.my_static_field
        10
        >>> instance.my_dynamic_field  # doctest: +ELLIPSIS
        Array(5., dtype=float32...)
        """

        return pz.pytree_dataclass(
            incoming,
            overwrite_parent_init=True,
            **kwargs,
        )

    @staticmethod
    def static(**kwargs):
        """Declare a field of a `Pytree` dataclass to be static.
        Users can provide additional keyword argument options,
        like `default` or `default_factory`, to customize how the field is
        instantiated when an instance of
        the dataclass is instantiated.` Fields which are provided with default
        values must come after required fields in the dataclass declaration.

        Examples
        --------

        >>> from genjax import Pytree
        >>> from jaxtyping import ArrayLike
        >>> import jax.numpy as jnp
        >>>
        >>> @Pytree.dataclass
        ... class MyClass(Pytree):
        ...     my_dynamic_field: ArrayLike
        ...     my_static_field: int = Pytree.static(default=0)
        >>>
        >>> instance = MyClass(jnp.array(5.0))
        >>> instance.my_static_field
        0
        >>> instance.my_dynamic_field  # doctest: +ELLIPSIS
        Array(5., dtype=float32...)

        """
        return field(metadata={"pytree_node": False}, **kwargs)

    @staticmethod
    def field(**kwargs):
        """Declare a field of a `Pytree` dataclass to be dynamic.
        Alternatively, one can leave the annotation off in the declaration."""
        return field(**kwargs)


@Pytree.dataclass
class Const(Generic[A], Pytree):
    """A Pytree wrapper for Python literals that should remain static.

    This class wraps Python values that need to stay as literals (not become
    JAX tracers) when used inside JAX transformations. The wrapped value is
    marked as static, ensuring it's embedded in the PyTreeDef rather than
    becoming a traced value.

    Example:
        ```python
        # Instead of: n_steps: int (becomes tracer in JAX transforms)
        # Use: n_steps: Const[int] (stays as Python int)

        def my_function(n_steps: Const[int]):
            for i in range(n_steps.value):  # n_steps.value is Python int
                ...
        ```

    Args:
        value: The Python literal to wrap as static
    """

    value: A = Pytree.static()

    def __add__(self, other):
        """Add two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value + other.value)
        try:
            return const(self.value + other)
        except TypeError:
            raise TypeError(
                f"Cannot add {type(self.value).__name__} and {type(other).__name__}"
            )

    def __radd__(self, other):
        """Right addition for when Const is on the right side."""
        try:
            return const(other + self.value)
        except TypeError:
            raise TypeError(
                f"Cannot add {type(other).__name__} and {type(self.value).__name__}"
            )

    def __sub__(self, other):
        """Subtract two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value - other.value)
        try:
            return const(self.value - other)
        except TypeError:
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from {type(self.value).__name__}"
            )

    def __rsub__(self, other):
        """Right subtraction for when Const is on the right side."""
        try:
            return const(other - self.value)
        except TypeError:
            raise TypeError(
                f"Cannot subtract {type(self.value).__name__} from {type(other).__name__}"
            )

    def __mul__(self, other):
        """Multiply two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value * other.value)
        try:
            return const(self.value * other)
        except TypeError:
            raise TypeError(
                f"Cannot multiply {type(self.value).__name__} and {type(other).__name__}"
            )

    def __rmul__(self, other):
        """Right multiplication for when Const is on the right side."""
        try:
            return const(other * self.value)
        except TypeError:
            raise TypeError(
                f"Cannot multiply {type(other).__name__} and {type(self.value).__name__}"
            )

    def __truediv__(self, other):
        """Divide two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value / other.value)
        try:
            return const(self.value / other)
        except TypeError:
            raise TypeError(
                f"Cannot divide {type(self.value).__name__} by {type(other).__name__}"
            )

    def __rtruediv__(self, other):
        """Right division for when Const is on the right side."""
        try:
            return const(other / self.value)
        except TypeError:
            raise TypeError(
                f"Cannot divide {type(other).__name__} by {type(self.value).__name__}"
            )

    def __floordiv__(self, other):
        """Floor divide two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value // other.value)
        try:
            return const(self.value // other)
        except TypeError:
            raise TypeError(
                f"Cannot floor divide {type(self.value).__name__} by {type(other).__name__}"
            )

    def __rfloordiv__(self, other):
        """Right floor division for when Const is on the right side."""
        try:
            return const(other // self.value)
        except TypeError:
            raise TypeError(
                f"Cannot floor divide {type(other).__name__} by {type(self.value).__name__}"
            )

    def __mod__(self, other):
        """Modulo two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value % other.value)
        try:
            return const(self.value % other)
        except TypeError:
            raise TypeError(
                f"Cannot compute {type(self.value).__name__} modulo {type(other).__name__}"
            )

    def __rmod__(self, other):
        """Right modulo for when Const is on the right side."""
        try:
            return const(other % self.value)
        except TypeError:
            raise TypeError(
                f"Cannot compute {type(other).__name__} modulo {type(self.value).__name__}"
            )

    def __pow__(self, other):
        """Power of two Const values or a Const and a regular value."""
        if isinstance(other, Const):
            return const(self.value**other.value)
        try:
            return const(self.value**other)
        except TypeError:
            raise TypeError(
                f"Cannot raise {type(self.value).__name__} to power {type(other).__name__}"
            )

    def __rpow__(self, other):
        """Right power for when Const is on the right side."""
        try:
            return const(other**self.value)
        except TypeError:
            raise TypeError(
                f"Cannot raise {type(other).__name__} to power {type(self.value).__name__}"
            )

    def __neg__(self):
        """Unary negation."""
        try:
            return const(-self.value)
        except TypeError:
            raise TypeError(f"Cannot negate {type(self.value).__name__}")

    def __pos__(self):
        """Unary positive."""
        try:
            return const(+self.value)
        except TypeError:
            raise TypeError(f"Cannot apply unary + to {type(self.value).__name__}")

    def __abs__(self):
        """Absolute value."""
        try:
            return const(abs(self.value))
        except TypeError:
            raise TypeError(
                f"Cannot compute absolute value of {type(self.value).__name__}"
            )

    # Comparison operations
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, Const):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        """Inequality comparison."""
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than comparison."""
        if isinstance(other, Const):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        """Less than or equal comparison."""
        if isinstance(other, Const):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        """Greater than comparison."""
        if isinstance(other, Const):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        """Greater than or equal comparison."""
        if isinstance(other, Const):
            return self.value >= other.value
        return self.value >= other


def const(a: A) -> Const[A]:
    """Create a Const wrapper for a static value.

    Args:
        a: The Python literal to wrap as static.

    Returns:
        A Const wrapper that keeps the value static in JAX transformations.

    Example:
        >>> length = const(10)  # Static length for scan
        >>> # @gen
        >>> # def model(n: Const[int]):
        >>> #     scan_gf = Scan(step_fn, length=n.value)
        >>> #     return scan_gf(args)
        >>> # trace = model(length, other_args)
        >>> length.value
        10
    """
    return Const(a)


class NotFixedException(Exception):
    """Exception raised when trace verification finds unfixed values.

    This exception provides a clear visualization of which parts of the choice map
    are properly fixed (True) vs. unfixed (False), helping users debug model
    structure issues during constrained inference.
    """

    def __init__(self, choice_map_status: X):
        self.choice_map_status = choice_map_status
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format a helpful error message showing the choice map status."""

        def format_choice_status(status, indent=2):
            """Format choice map status, handling both dict and non-dict types."""
            if isinstance(status, dict):
                lines = []
                for key, value in status.items():
                    if isinstance(value, dict):
                        lines.append(f"{'  ' * (indent // 2)}{key!r}: {{")
                        lines.extend(format_choice_status(value, indent + 2))
                        lines.append(f"{'  ' * (indent // 2)}}}")
                    else:
                        status_str = "Fixed" if value else "NOT_FIXED"
                        lines.append(f"{'  ' * (indent // 2)}{key!r}: {status_str}")
                return lines
            else:
                # For non-dict types (e.g., scalar boolean), format directly
                status_str = "Fixed" if status else "NOT_FIXED"
                return [f"{'  ' * (indent // 2)}{status_str}"]

        lines = [
            "Found unfixed values in trace choices during constrained inference.",
            "All random choices should be wrapped with Fixed during constrained inference.",
            "",
            "Choice map status (Fixed = properly constrained, NOT_FIXED = missing constraint):",
        ]

        if isinstance(self.choice_map_status, dict):
            lines.append("{")
            lines.extend(format_choice_status(self.choice_map_status, 2))
            lines.append("}")
        else:
            # For scalar choice maps, don't use dict notation
            lines.extend(format_choice_status(self.choice_map_status, 0))

        lines.append("")
        lines.append(
            "Unfixed values indicate missing temporal dependencies or incorrect model structure."
        )

        return "\n".join(lines)


@Pytree.dataclass
class Fixed(Generic[A], Pytree):
    """A Pytree wrapper that denotes a random choice was provided (fixed),
    not proposed by a GFI's internal proposal family.

    This wrapper is used internally by Distribution implementations in
    `generate`, `update`, and `regenerate` methods to mark values that
    were constrained or provided externally rather than sampled from the
    distribution's internal proposal.

    The `Fixed` wrapper helps debug model structure issues during inference
    by tracking which random choices were externally constrained vs.
    internally proposed.

    Args:
        value: The fixed value that was provided externally
    """

    value: A

    def __repr__(self):
        return f"Fixed({self.value!r})"


def fixed(a: A) -> Fixed[A]:
    """Create a Fixed wrapper for a constrained value.

    Args:
        a: The value that was provided/constrained externally.

    Returns:
        A Fixed wrapper indicating the value was not proposed internally.
    """
    return Fixed(a)


#######
# GFI #
#######


class Trace(Generic[X, R], Pytree):
    @abstractmethod
    def get_gen_fn(self) -> "GFI[X, R]":
        pass

    @abstractmethod
    def get_choices(self) -> X:
        pass

    @abstractmethod
    def get_fixed_choices(self) -> X:
        """Get choices preserving Fixed wrappers.

        Returns the raw choice structure with Fixed wrappers intact,
        used for verification that values were constrained during inference.
        """
        pass

    @abstractmethod
    def get_args(self) -> Any:
        pass

    @abstractmethod
    def get_retval(self) -> R:
        pass

    @abstractmethod
    def get_score(self) -> Score:
        pass

    def verify(self) -> None:
        """Verify that all leaf values in the trace choices were fixed (constrained).

        Checks that all random choices in the trace are wrapped with Fixed,
        indicating they were provided externally rather than proposed by the
        GFI's internal proposal family. This helps debug model structure issues
        during inference.

        Raises:
            NotFixedException: If any leaf value is not wrapped with Fixed.
                              The exception includes a detailed choice map showing
                              which values are fixed vs. unfixed.
        """
        # Get choices preserving Fixed wrappers
        choice_values = get_fixed_choices(self)

        # Check if value is Fixed
        def check_instance_fixed(x):
            return isinstance(x, Fixed)

        # Flatten the tree to get all leaf choice values
        leaf_values, tree_def = jtu.tree_flatten(
            choice_values, is_leaf=check_instance_fixed
        )

        # Check if all leaves are Fixed
        all_fixed = all(isinstance(leaf, Fixed) for leaf in leaf_values)

        if not all_fixed:
            # Create a boolean choice map showing which values are fixed
            def make_bool_status(x):
                if isinstance(x, Fixed):
                    return True
                else:
                    return False

            choice_map_status = jtu.tree_map(
                make_bool_status, choice_values, is_leaf=check_instance_fixed
            )

            raise NotFixedException(choice_map_status)

    def update(self, x: X, *args, **kwargs):
        gen_fn = self.get_gen_fn()
        if not args and not kwargs:
            # Use original args if none provided
            original_args = self.get_args()
            if (
                isinstance(original_args, tuple)
                and len(original_args) == 2
                and isinstance(original_args[1], dict)
            ):
                # Handle (args, kwargs) tuple format
                return gen_fn.update(self, x, *original_args[0], **original_args[1])
            else:
                # Handle legacy args format
                return gen_fn.update(self, x, original_args)
        else:
            return gen_fn.update(self, x, *args, **kwargs)

    def __getitem__(self, addr):
        choices = self.get_choices()
        return get_choices(choices[addr])  # pyright: ignore


@Pytree.dataclass
class Tr(Trace[X, R], Pytree):
    """Concrete implementation of the Trace interface.

    Stores all components of an execution trace including the generative
    function, arguments, random choices, return value, and score.

    Args:
        _gen_fn: The generative function that produced this trace.
        _args: Arguments passed to the generative function.
        _choices: Random choices made during execution.
        _retval: Return value of the execution.
        _score: Log probability score of the choices.
    """

    _gen_fn: "GFI[X, R]"
    _args: Any
    _choices: X
    _retval: R
    _score: Score

    def get_gen_fn(self) -> "GFI[X, R]":
        assert isinstance(self._gen_fn, GFI)
        return self._gen_fn

    def get_choices(self) -> X:
        return get_choices(self._choices)

    def get_fixed_choices(self) -> X:
        """Get choices preserving Fixed wrappers."""
        return get_fixed_choices(self._choices)

    def get_args(self) -> Any:
        return self._args

    def get_retval(self) -> R:
        return self._retval

    def get_score(self) -> Score:
        if jnp.shape(self._score):
            return jnp.sum(self._score)
        else:
            return self._score


def get_choices(x: Trace[X, R] | X) -> X:
    """Extract choices from a trace or nested structure containing traces.

    Also strips Fixed wrappers from the choices, returning the unwrapped values.
    Fixed wrappers are used internally to track constrained vs. proposed values.

    Args:
        x: A trace object or nested structure that may contain traces.

    Returns:
        The random choices, with any nested traces recursively unwrapped and
        Fixed wrappers stripped.
    """
    x = x.get_choices() if isinstance(x, Trace) else x

    def _get_choices(x):
        if isinstance(x, Trace):
            return get_choices(x)
        else:
            return x

    # First unwrap any nested traces
    x = jtu.tree_map(
        _get_choices,
        x,
        is_leaf=lambda x: isinstance(x, Trace),
    )

    # Then strip Fixed wrappers
    def _strip_fixed(x):
        if isinstance(x, Fixed):
            return x.value  # Unwrap Fixed wrapper
        else:
            return x

    return jtu.tree_map(
        _strip_fixed,
        x,
        is_leaf=lambda x: isinstance(x, Fixed),
    )


def get_fixed_choices(x: Trace[X, R] | X) -> X:
    """Extract choices from a trace or nested structure containing traces, preserving Fixed wrappers.

    Similar to get_choices() but preserves Fixed wrappers around the choices,
    which is needed for verification that values were constrained during inference.

    Args:
        x: A trace object or nested structure that may contain traces.

    Returns:
        The random choices, with any nested traces recursively unwrapped but
        Fixed wrappers preserved.
    """
    x = x.get_fixed_choices() if isinstance(x, Trace) else x

    def _get_fixed_choices(x):
        if isinstance(x, Trace):
            return get_fixed_choices(x)
        else:
            return x

    # Unwrap any nested traces but preserve Fixed wrappers
    # Note: Unlike get_choices(), we do NOT strip Fixed wrappers
    return jtu.tree_map(
        _get_fixed_choices,
        x,
        is_leaf=lambda x: isinstance(x, Trace),
    )


def get_score(x: Trace[X, R]) -> Weight:
    """Extract the log probability score from a trace.

    Args:
        x: Trace object to extract score from.

    Returns:
        The log probability score of the trace.
    """
    return x.get_score()


def get_retval(x: Trace[X, R]) -> R:
    """Extract the return value from a trace.

    Args:
        x: Trace object to extract return value from.

    Returns:
        The return value of the trace.
    """
    return x.get_retval()


##############
# Selections #
##############


@Pytree.dataclass
class AllSel(Pytree):
    """Selection that matches all addresses."""

    def match(self, addr) -> "tuple[bool, AllSel]":
        return True, self


@Pytree.dataclass
class NoneSel(Pytree):
    """Selection that matches no addresses."""

    def match(self, addr) -> "tuple[bool, NoneSel]":
        return False, self


@Pytree.dataclass
class StrSel(Pytree):
    """Selection that matches a specific string address.

    Args:
        s: The string address to match.
    """

    s: Const[str]

    def match(self, addr) -> tuple[bool, AllSel | NoneSel]:
        check = addr == self.s.value
        return check, AllSel() if check else NoneSel()


@Pytree.dataclass
class TupleSel(Pytree):
    """Selection that matches a hierarchical tuple address.

    Tuple addresses represent hierarchical paths like ("outer", "inner", "leaf").
    When matched against a single string address, it checks if that string
    matches the first element of the tuple, and returns a selection for
    the remaining path.

    Args:
        t: Tuple of strings representing the hierarchical path.
    """

    t: Const[tuple[str, ...]]

    def match(self, addr) -> tuple[bool, Union[AllSel, NoneSel, "TupleSel"]]:
        path = self.t.value
        if not path:
            # Empty tuple matches nothing
            return False, NoneSel()

        if len(path) == 1:
            # Single element tuple behaves like StrSel
            check = addr == path[0]
            return check, AllSel() if check else NoneSel()
        else:
            # Multi-element tuple: check first element and return rest
            check = addr == path[0]
            if check:
                return True, TupleSel(const(path[1:]))
            else:
                return False, NoneSel()


@Pytree.dataclass
class DictSel(Pytree):
    """Selection that matches addresses using a dictionary mapping.

    Args:
        d: Dictionary mapping addresses to nested selections.
    """

    d: "dict[str, Any]"

    def match(self, addr) -> "tuple[bool, Any]":
        check = addr in self.d
        return check, self.d[addr] if check else NoneSel()


@Pytree.dataclass
class ComplSel(Pytree):
    """Selection that matches the complement of another selection.

    Args:
        s: The selection to complement.
    """

    s: "Selection"

    def match(self, addr) -> "tuple[bool, ComplSel]":
        check, rest = self.s.match(addr)
        return not check, ComplSel(rest)


@Pytree.dataclass
class InSel(Pytree):
    """Selection representing intersection of two selections.

    Args:
        s1: First selection.
        s2: Second selection.
    """

    s1: "Selection"
    s2: "Selection"

    def match(self, addr) -> "tuple[bool, InSel]":
        check1, r1 = self.s1.match(addr)
        check2, r2 = self.s2.match(addr)
        return check1 and check2, InSel(r1, r2)


@Pytree.dataclass
class OrSel(Pytree):
    s1: "Selection"
    s2: "Selection"

    def match(self, addr) -> "tuple[bool, OrSel]":
        check1, r1 = self.s1.match(addr)
        check2, r2 = self.s2.match(addr)
        return check1 or check2, OrSel(r1, r2)


@Pytree.dataclass
class Selection(Pytree):
    """A Selection acts as a filter to specify which random choices in a trace should
    be regenerated during the `regenerate` method call.

    Selections are used in inference algorithms like MCMC to specify which subset of
    random choices should be updated while keeping others fixed. The Selection determines
    which addresses (random choice names) match the selection criteria.

    A Selection wraps one of several concrete selection types:
    - StrSel: Matches a specific string address
    - DictSel: Matches addresses using a dictionary mapping
    - AllSel: Matches all addresses
    - NoneSel: Matches no addresses
    - ComplSel: Matches the complement of another selection
    - InSel: Matches the intersection of two selections
    - OrSel: Matches the union of two selections

    Example:
        ```python
        from genjax.core import sel

        # Select a specific address
        selection = sel("x")  # Matches address "x"

        # Select all addresses
        selection = sel(())   # Matches all addresses

        # Select nested addresses
        selection = sel({"outer": sel("inner")})  # Matches "outer"/"inner"

        # Use in regenerate
        new_trace, weight, discard = gen_fn.regenerate(args, trace, selection)
        ```

    Args:
        s: The underlying selection implementation (one of the concrete selection types)
    """

    s: NoneSel | AllSel | StrSel | TupleSel | DictSel | ComplSel | InSel | OrSel

    def match(self, addr) -> "tuple[bool, Selection]":
        check, rest = self.s.match(addr)
        return check, Selection(rest) if not isinstance(rest, Selection) else rest

    def __xor__(self, other: "Selection") -> "Selection":
        return Selection(InSel(self, other))

    def __invert__(self) -> "Selection":
        return Selection(ComplSel(self))

    def __contains__(self, addr) -> bool:
        check, _ = self.match(addr)
        return check

    def __or__(self, other) -> "Selection":
        return Selection(OrSel(self, other))

    def __call__(self, addr) -> "tuple[bool, Selection]":
        return self.match(addr)


def sel(*v: tuple[()] | str | tuple[str, ...] | dict[str, Any] | None) -> Selection:
    """Create a Selection from various input types.

    This is a convenience function to create Selection objects from common patterns.
    Selections specify which random choices in a trace should be regenerated during
    inference operations like MCMC.

    Args:
        *v: Variable arguments specifying the selection pattern:
            - str: Select a specific address (e.g., sel("x"))
            - tuple[str, ...]: Select hierarchical address (e.g., sel(("outer", "inner")))
            - (): Select all addresses (e.g., sel(()))
            - dict: Select nested addresses (e.g., sel({"outer": sel("inner")}))
            - None or no args: Select no addresses (e.g., sel() or sel(None))

    Returns:
        Selection object that can be used with regenerate methods

    Examples:
        ```python
        # Select specific address
        sel("x")                    # Matches address "x"

        # Select hierarchical address
        sel(("outer", "inner"))     # Matches hierarchical path outer/inner

        # Select all addresses
        sel(())                     # Matches all addresses

        # Select no addresses
        sel() or sel(None)          # Matches no addresses

        # Select nested addresses
        sel({"outer": sel("inner")}) # Matches "outer"/"inner"
        ```
    """
    assert len(v) <= 1
    if len(v) == 1:
        if v[0] is None:
            return Selection(NoneSel())
        if v[0] == ():
            return Selection(AllSel())
        elif isinstance(v[0], dict):
            return Selection(DictSel(v[0]))
        elif isinstance(v[0], tuple) and all(isinstance(s, str) for s in v[0]):
            # Tuple of strings for hierarchical addresses
            return Selection(TupleSel(const(v[0])))
        else:
            assert isinstance(v[0], str)
            return Selection(StrSel(const(v[0])))
    else:
        return Selection(NoneSel())


def match(addr: Addr, sel: Selection):
    return sel.match(addr)


class GFI(Generic[X, R], Pytree):
    """Generative Function Interface - the core abstraction for probabilistic programs.

    The GFI defines the standard interface that all generative functions must
    implement. It provides methods for simulation, assessment, generation,
    updating, and regeneration of probabilistic computations.

    Mathematical Foundation:
    A generative function bundles three mathematical objects:
    1. Measure kernel P(dx; args) - the probability distribution over choices
    2. Return value function f(x, args) -> R - deterministic computation from choices
    3. Internal proposal family Q(dx; args, context) - for efficient inference

    The GFI methods provide access to these mathematical objects and enable:
    - Forward sampling (simulate)
    - Density evaluation (assess)
    - Constrained generation (generate)
    - Edit moves (update, regenerate)

    All density computations are in log space for numerical stability.
    Weights from generate/update/regenerate enable importance sampling and MCMC.

    Type Parameters:
        X: The type of the random choices (choice map).
        R: The type of the return value.

    Core Methods:
        simulate: Sample (choices, retval) ~ P(·; args)
        assess: Compute log P(choices; args)
        generate: Sample with constraints, return importance weight
        update: Update trace arguments/choices, return incremental importance weight
        regenerate: Resample selected choices, return incremental importance weight

    Additional Methods:
        merge: Combine choice maps (for compositional functions)
        log_density: Convenience method for assess that sums log densities
        vmap/repeat: Vectorization combinators
        cond: Conditional execution combinator
    """

    def __call__(self, *args, **kwargs) -> "Thunk[X, R] | R":
        if handler_stack:
            return Thunk(self, args, kwargs)
        else:
            tr = self.simulate(*args, **kwargs)
            assert isinstance(tr, Tr)
            return tr.get_retval()

    def T(self, *args, **kwargs) -> "Thunk[X, R]":
        return Thunk(self, args, kwargs)

    @abstractmethod
    def simulate(
        self,
        *args,
        **kwargs,
    ) -> Trace[X, R]:
        """Sample an execution trace from the generative function.

        Mathematical specification:
        - Samples (choices, retval) ~ P(·; args) where P is the generative function's measure kernel
        - Returns trace containing choices, return value, score, and arguments

        The score in the returned trace is log(1/P(choices; args)), i.e., the negative
        log probability density of the sampled choices.

        Args:
            *args: Arguments to the generative function.
            **kwargs: Keyword arguments to the generative function.

        Returns:
            A trace containing the sampled choices, return value, score, and arguments.

        Example:
            >>> # model.simulate(mu, sigma)  # Example usage
            >>> # choices = trace.get_choices()
            >>> # score = trace.get_score()  # -log P(choices; mu, sigma)
            >>> pass  # doctest placeholder
        """
        pass

    @abstractmethod
    def generate(
        self,
        x: X | None,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight]:
        """Generate a trace with optional constraints on some choices.

        Mathematical specification:
        - Samples unconstrained choices ~ Q(·; constrained_choices, args)
        - Computes importance weight: log [P(all_choices; args) / Q(unconstrained_choices; constrained_choices, args)]
        - When x=None, equivalent to simulate() but returns weight=0

        The weight enables importance sampling and is crucial for inference algorithms.
        For fully constrained generation, the weight equals the log density.

        Args:
            x: Optional constraints on subset of choices. If None, equivalent to simulate.
            *args: Arguments to the generative function.
            **kwargs: Keyword arguments to the generative function.

        Returns:
            A tuple (trace, weight) where:
            - trace: contains all choices (constrained + sampled) and return value
            - weight: log [P(all_choices; args) / Q(unconstrained_choices; constrained_choices, args)]

        Example:
            >>> # Constrain some choices
            >>> # constraints = {"x": 1.5, "y": 2.0}
            >>> # trace, weight = model.generate(constraints, mu, sigma)
            >>> # weight accounts for probability of constrained choices
            >>> pass  # doctest placeholder
        """
        pass

    @abstractmethod
    def assess(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Density, R]:
        """Compute the log probability density of given choices.

        Mathematical specification:
        - Computes log P(choices; args) where P is the generative function's measure kernel
        - Also computes the return value for the given choices
        - Requires P(choices; args) > 0 (choices must be valid)

        Args:
            x: The choices to evaluate.
            *args: Arguments to the generative function.
            **kwargs: Keyword arguments to the generative function.

        Returns:
            A tuple (log_density, retval) where:
            - log_density: log P(choices; args)
            - retval: return value computed with the given choices

        Example:
            >>> # log_density, retval = model.assess(choices, mu, sigma)
            >>> # log_density = log P(choices; mu, sigma)
            >>> pass  # doctest placeholder
        """
        pass

    @abstractmethod
    def update(
        self,
        tr: Trace[X, R],
        x_: X | None,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        """Update a trace with new arguments and/or choice constraints.

        Mathematical specification:
        - Transforms trace from (old_args, old_choices) to (new_args, new_choices)
        - Computes incremental importance weight (edit move):

        weight = log [P(new_choices; new_args) / Q(new_choices; new_args, old_choices, constraints)]
               - log [P(old_choices; old_args) / Q(old_choices; old_args)]

        where Q is the internal proposal distribution used for updating.

        Args:
            tr: Current trace to update.
            x_: Optional constraints on choices to enforce during update.
            *args: New arguments to the generative function.
            **kwargs: New keyword arguments to the generative function.

        Returns:
            A tuple (new_trace, weight, discarded_choices) where:
            - new_trace: updated trace with new arguments and choices
            - weight: incremental importance weight for the update (enables MCMC, SMC)
            - discarded_choices: old choice values that were changed

        Example:
            >>> # Update trace with new arguments
            >>> # new_trace, weight, discarded = model.update(old_trace, None, new_mu, new_sigma)
            >>> # weight = log P(new_choices; new_args) - log P(old_choices; old_args)
            >>> pass  # doctest placeholder
        """
        pass

    @abstractmethod
    def regenerate(
        self,
        tr: Trace[X, R],
        sel: Selection,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        """Regenerate selected choices in a trace while keeping others fixed.

        Mathematical specification:
        - Resamples choices at addresses selected by 'sel' from their conditional distribution
        - Keeps non-selected choices unchanged
        - Computes incremental importance weight (edit move):

        weight = log P(new_selected_choices | non_selected_choices; args)
               - log P(old_selected_choices | non_selected_choices; args)

        When sel selects all addresses, regenerate becomes equivalent to simulate.
        When sel selects no addresses, weight = 0 and trace unchanged.

        Args:
            tr: Current trace to regenerate from.
            sel: Selection specifying which addresses to regenerate.
            *args: Arguments to the generative function.
            **kwargs: Keyword arguments to the generative function.

        Returns:
            A tuple (new_trace, weight, discarded_choices) where:
            - new_trace: trace with selected choices resampled
            - weight: incremental importance weight for the regeneration
            - discarded_choices: old values of the regenerated choices

        Example:
            >>> # Regenerate choices at addresses "x" and "y"
            >>> # selection = sel("x") | sel("y")
            >>> # new_trace, weight, discarded = model.regenerate(trace, selection, mu, sigma)
            >>> # weight accounts for probability change due to regeneration
            >>> pass  # doctest placeholder
        """
        pass

    @abstractmethod
    def merge(
        self, x: X, x_: X, check: jnp.ndarray | None = None
    ) -> tuple[X, X | None]:
        """Merge two choice maps, with the second taking precedence.

        Used internally for compositional generative functions where choice maps
        from different components need to be combined. The merge operation resolves
        conflicts by preferring choices from x_ over x.

        Args:
            x: First choice map.
            x_: Second choice map (takes precedence in conflicts).
            check: Optional boolean array for conditional selection.
                   If provided, selects x where True, x_ where False.

        Returns:
            Tuple of (merged choice map, discarded values).
            - merged: Combined choices with x_ values overriding x values at conflicts
            - discarded: Values from x that were overridden by x_ (None if no conflicts)
        """
        pass

    @abstractmethod
    def filter(self, x: X, selection: "Selection") -> tuple[X | None, X | None]:
        """Filter choice map into selected and unselected parts.

        Used to partition choices based on a selection, enabling fine-grained manipulation
        of subsets of choices in inference algorithms. Each GFI implementation specializes
        this method for its choice type X.

        Args:
            x: Choice map to filter.
            selection: Selection specifying which addresses to include.

        Returns:
            Tuple of (selected_choices, unselected_choices) where:
            - selected_choices: Choice map containing only selected addresses, or None if no matches
            - unselected_choices: Choice map containing only unselected addresses, or None if no matches
            Both have the same structure as X but contain disjoint subsets of addresses.

        Example:
            >>> # choices = {"mu": 1.0, "sigma": 2.0, "obs": 3.0}
            >>> # selection = sel("mu") | sel("sigma")
            >>> # selected, unselected = model.filter(choices, selection)
            >>> # selected = {"mu": 1.0, "sigma": 2.0}, unselected = {"obs": 3.0}
            >>> pass  # doctest placeholder
        """
        pass

    def vmap(
        self,
        in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
        axis_size=None,
        axis_name: str | None = None,
        spmd_axis_name: str | None = None,
    ) -> "Vmap[X, R]":
        return Vmap(
            self,
            const(in_axes),
            const(axis_size),
            const(axis_name),
            const(spmd_axis_name),
        )

    def repeat(self, n: int):
        return self.vmap(in_axes=None, axis_size=n)

    def cond(
        self,
        callee_: "GFI[X, R]",
    ) -> "Cond[X, R]":
        return Cond(self, callee_)

    def log_density(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> Score:
        logp, _ = self.assess(x, *args, **kwargs)
        return jnp.sum(logp) if jnp.shape(logp) else logp


########################
# Generative functions #
########################


@Pytree.dataclass
class Thunk(Generic[X, R], Pytree):
    """Delayed evaluation wrapper for generative functions.

    A thunk represents a generative function call that has not yet been
    executed. It captures the function and its arguments for later evaluation.

    Args:
        gen_fn: The generative function to call.
        args: Arguments to pass to the generative function.
        kwargs: Keyword arguments to pass to the generative function.
    """

    gen_fn: GFI[X, R]
    args: tuple
    kwargs: dict = Pytree.field(default_factory=dict)

    def __matmul__(self, other: str):
        return trace(other, self.gen_fn, self.args, self.kwargs)


@Pytree.dataclass
class Vmap(Generic[X, R], GFI[X, R]):
    """A `Vmap` is a generative function combinator that vectorizes another generative function.

    `Vmap` applies a generative function across a batch dimension, similar to `jax.vmap`,
    but preserves probabilistic semantics. It uses GenJAX's `modular_vmap` to handle
    the vectorization of probabilistic computations correctly.

    Mathematical ingredients:
    - If callee has measure kernel P_callee(dx; args), then Vmap has kernel
      P_vmap(dX; Args) = ∏_i P_callee(dx_i; args_i) where X = [x_1, ..., x_n]
    - Return value function f_vmap(X, Args) = [f_callee(x_1, args_1), ..., f_callee(x_n, args_n)]
    - Internal proposal family inherits from callee's proposal family

    Attributes:
        gen_fn: The generative function to vectorize
        in_axes: Specifies which axes to vectorize over (same as jax.vmap)
        axis_size: Size of the vectorized axis (if not inferrable from inputs)
        axis_name: Optional name for the vectorized axis
        spmd_axis_name: Optional SPMD axis name for distributed computation

    Example:
        >>> from genjax import normal
        >>>
        >>> # Vectorize a normal distribution
        >>> vectorized_normal = normal.vmap(in_axes=(0, None))  # vectorize over first arg
        >>>
        >>> mus = jnp.array([0.0, 1.0, 2.0])
        >>> sigma = 1.0
        >>> trace = vectorized_normal.simulate(mus, sigma)
        >>> samples = trace.get_choices()  # Array of 3 normal samples
    """

    gen_fn: GFI[X, R]
    in_axes: Const[int | tuple[int | None, ...] | Sequence[Any] | None]
    axis_size: Const[int | None]
    axis_name: Const[str | None]
    spmd_axis_name: Const[str | None]

    def simulate(
        self,
        *args,
        **kwargs,
    ) -> Trace[X, R]:
        return modular_vmap(
            self.gen_fn.simulate,
            in_axes=self.in_axes.value,
            axis_size=self.axis_size.value,
            axis_name=self.axis_name.value,
            spmd_axis_name=self.spmd_axis_name.value,
        )(*args, **kwargs)

    def generate(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight]:
        if self.in_axes.value is None:
            in_axes = (0,) + (None,) * len(args)
        else:
            in_axes = (0,) + self.in_axes.value
        tr, w = modular_vmap(
            self.gen_fn.generate,
            in_axes=in_axes,
            axis_size=self.axis_size.value,
            axis_name=self.axis_name.value,
            spmd_axis_name=self.spmd_axis_name.value,
        )(x, *args, **kwargs)
        return tr, jnp.sum(w)

    def assess(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Density, R]:
        if self.in_axes.value is None:
            in_axes = (0,) + (None,) * len(args)
        else:
            in_axes = (0,) + self.in_axes.value
        density, retval = modular_vmap(
            self.gen_fn.assess,
            in_axes=in_axes,
            axis_size=self.axis_size.value,
            axis_name=self.axis_name.value,
            spmd_axis_name=self.spmd_axis_name.value,
        )(x, *args, **kwargs)
        return jnp.sum(density), retval

    def update(
        self,
        tr: Trace[X, R],
        x_: X,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        if self.in_axes.value is None:
            in_axes = (0, 0) + (None,) * len(args)
        else:
            in_axes = (0, 0) + self.in_axes.value
        new_tr, w, discard = modular_vmap(
            self.gen_fn.update,
            in_axes=in_axes,
            axis_size=self.axis_size.value,
            axis_name=self.axis_name.value,
            spmd_axis_name=self.spmd_axis_name.value,
        )(tr, x_, *args, **kwargs)
        return new_tr, jnp.sum(w), discard

    def regenerate(
        self,
        tr: Trace[X, R],
        s: Selection,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        if self.in_axes.value is None:
            in_axes = (0, None) + (None,) * len(args)
        else:
            in_axes = (0, None) + self.in_axes.value
        new_tr, w, discard = modular_vmap(
            self.gen_fn.regenerate,
            in_axes=in_axes,
            axis_size=self.axis_size.value,
            axis_name=self.axis_name.value,
            spmd_axis_name=self.spmd_axis_name.value,
        )(tr, s, *args, **kwargs)
        return new_tr, jnp.sum(w), discard

    def merge(
        self, x: X, x_: X, check: jnp.ndarray | None = None
    ) -> tuple[X, X | None]:
        # For Vmap, we need to handle the check parameter appropriately
        if check is None:
            merged, discarded = modular_vmap(self.gen_fn.merge, in_axes=(0, 0, None))(
                x, x_, None
            )
        else:
            # Check should be broadcast across the batch dimension
            merged, discarded = modular_vmap(self.gen_fn.merge, in_axes=(0, 0, 0))(
                x, x_, check
            )
        return merged, discarded

    def filter(self, x: X, selection: "Selection") -> tuple[X | None, X | None]:
        """Filter vectorized choices using the underlying generative function's filter.

        For Vmap, choices are vectorized across the batch dimension. We apply
        the underlying GF's filter to each vectorized choice.

        Args:
            x: Vectorized choice to filter.
            selection: Selection specifying which addresses to include.

        Returns:
            Tuple of (selected_choices, unselected_choices) where each is vectorized or None.
        """
        # Use modular_vmap to apply filter across the batch dimension
        selected, unselected = modular_vmap(
            self.gen_fn.filter,
            in_axes=(0, None),
            axis_size=self.axis_size.value,
        )(x, selection)

        return selected, unselected


#################
# Distributions #
#################


@Pytree.dataclass
class Distribution(Generic[X], GFI[X, X]):
    """A `Distribution` is a generative function that implements a probability distribution.

    Distributions are the fundamental building blocks of probabilistic programs.
    They implement the Generative Function Interface (GFI) by wrapping a sampling
    function and a log probability density function (logpdf).

    Mathematical ingredients:
    - A measure kernel P(dx; args) over a measurable space X given arguments args
    - Return value function f(x, args) = x (identity function for distributions)
    - Internal proposal distribution family Q(dx; args, x') = P(dx; args) (prior)

    Attributes:
        sample: A sampling function that takes distribution parameters and returns a sample
        logpdf: A log probability density function that takes (value, *parameters)
        name: Optional name for the distribution (used in pretty printing)

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from genjax import Distribution, const
        >>>
        >>> # Create a custom normal distribution
        >>> def sample_normal(mu, sigma):
        ...     key = jax.random.PRNGKey(0)  # In practice, use proper key management
        ...     return mu + sigma * jax.random.normal(key)
        >>>
        >>> def logpdf_normal(x, mu, sigma):
        ...     return -0.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)
        >>>
        >>> normal = Distribution(const(sample_normal), const(logpdf_normal), const("normal"))
        >>> trace = normal.simulate(0.0, 1.0)  # mu=0.0, sigma=1.0
    """

    _sample: Const[Callable[..., X]]
    _logpdf: Const[Callable[..., Weight]]
    name: Const[str | None]

    def sample(self, *args, **kwargs) -> X:
        """Sample from the distribution."""
        return self._sample.value(*args, **kwargs)

    def logpdf(self, x: X, *args, **kwargs) -> Weight:
        """Compute log probability density."""
        return self._logpdf.value(x, *args, **kwargs)

    def simulate(
        self,
        *args,
        **kwargs,
    ) -> Tr[X, X]:
        x = self.sample(*args, **kwargs)
        log_density = self.logpdf(x, *args, **kwargs)
        return Tr(self, (args, kwargs), x, x, -log_density)

    def generate(
        self,
        x: X | None,
        *args,
        **kwargs,
    ) -> tuple[Tr[X, X], Weight]:
        if x is None:
            tr = self.simulate(*args, **kwargs)
            return tr, jnp.array(0.0)
        else:
            logp, r = self.assess(x, *args, **kwargs)
            return Tr(self, (args, kwargs), x, x, -logp), logp

    def assess(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Density, X]:
        logp = self.logpdf(x, *args, **kwargs)
        return logp, x

    def update(
        self,
        tr: Tr[X, X],
        x_: X | None,
        *args,
        **kwargs,
    ) -> tuple[Tr[X, X], Weight, X | None]:
        if x_ is None:
            x_ = get_choices(tr)
            log_density_ = self.logpdf(x_, *args, **kwargs)
            return (
                Tr(self, (args, kwargs), x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                tr.get_retval(),
            )
        else:
            log_density_ = self.logpdf(x_, *args, **kwargs)
            return (
                Tr(self, (args, kwargs), x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                tr.get_retval(),
            )

    def regenerate(
        self,
        tr: Tr[X, X],
        s: Selection,
        *args,
        **kwargs,
    ) -> tuple[Tr[X, X], Weight, X | None]:
        if () in s:
            # Address is selected for regeneration - generate new value (not fixed)
            tr_ = self.simulate(*args, **kwargs)
            return tr_, jnp.array(0.0), get_choices(tr)
        else:
            # Address not selected - keep existing value
            x_ = get_choices(tr)
            log_density_ = self.logpdf(get_choices(tr), *args, **kwargs)
            return (
                Tr(self, (args, kwargs), x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                None,
            )

    def merge(
        self, x: X, x_: X, check: jnp.ndarray | None = None
    ) -> tuple[X, X | None]:
        """Merge distribution choices with optional conditional selection.

        For distributions, choices are raw values from the sample space.
        When check is provided, we use jnp.where for conditional selection.
        """
        if check is not None:
            # Conditional merge using jnp.where
            merged = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), x, x_)
            # No values are truly "discarded" in conditional selection
            return merged, None
        else:
            # Without check, Distribution doesn't support merge
            raise Exception(
                "Can't merge: the underlying sample space `X` for the type `Distribution` doesn't support merging without a check parameter."
            )

    def filter(self, x: X, selection: "Selection") -> tuple[X | None, X | None]:
        """Filter choice into selected and unselected parts.

        For Distribution, the choice is a single value X. Selection either
        matches the empty address () or it doesn't.

        Args:
            x: Choice value to potentially filter.
            selection: Selection specifying whether to include the choice.

        Returns:
            Tuple of (selected_choice, unselected_choice) where exactly one is x and the other is None.
        """
        is_selected, _ = selection.match(())
        if is_selected:
            return x, None
        else:
            return None, x


def distribution(
    sampler: Callable[..., Any],
    logpdf: Callable[..., Any],
    /,
    name: str | None = None,
) -> Distribution[Any]:
    """Create a Distribution from sampling and log probability functions.

    Args:
        sampler: Function that takes parameters and returns a sample.
        logpdf: Function that takes (value, *parameters) and returns log probability.
        name: Optional name for the distribution.

    Returns:
        A Distribution instance implementing the Generative Function Interface.
    """
    return Distribution(
        _sample=const(sampler),
        _logpdf=const(logpdf),
        name=const(name),
    )


# Mostly, just use TFP.
# This wraps PJAX's `sample_p` correctly.
def tfp_distribution(
    dist: Callable[..., "tfd.Distribution"],
    /,
    name: str | None = None,
) -> Distribution[Any]:
    """Create a Distribution from a TensorFlow Probability distribution.

    Wraps a TFP distribution constructor to create a GenJAX Distribution
    that properly handles PJAX's `sample_p` primitive.

    Args:
        dist: TFP distribution constructor function.
        name: Optional name for the distribution.

    Returns:
        A Distribution that wraps the TFP distribution.

    Example:
        >>> import tensorflow_probability.substrates.jax as tfp
        >>> from genjax import tfp_distribution
        >>>
        >>> # Create a normal distribution from TFP
        >>> normal = tfp_distribution(tfp.distributions.Normal, name="normal")
    """

    def keyful_sampler(key, *args, sample_shape=(), **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key, sample_shape=sample_shape)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.log_prob(v)

    return distribution(
        wrap_sampler(
            keyful_sampler,
            name=name,
        ),
        wrap_logpdf(logpdf),
        name=name,
    )


######
# Fn #
######


def _get_generative_function_info(gen_fn: "GFI") -> str:
    """Extract source information from a generative function.

    Args:
        gen_fn: The generative function to get info from

    Returns:
        String describing the function location and name
    """
    try:
        if hasattr(gen_fn, "source") and hasattr(gen_fn.source, "value"):
            # This is a Fn with a source function
            func = gen_fn.source.value
            if hasattr(func, "__name__"):
                func_name = func.__name__
                try:
                    file_path = inspect.getfile(func)
                    line_no = inspect.findsource(func)[1] + 1
                    return f"function '{func_name}' at {file_path}:{line_no}"
                except (OSError, TypeError):
                    return f"function '{func_name}'"
            else:
                return "anonymous function"
        elif hasattr(gen_fn, "__class__"):
            # This might be a Distribution or other GFI implementation
            class_name = gen_fn.__class__.__name__
            if (
                hasattr(gen_fn, "name")
                and hasattr(gen_fn.name, "value")
                and gen_fn.name.value
            ):
                return f"{class_name} '{gen_fn.name.value}'"
            else:
                return f"{class_name}"
        else:
            return "unknown generative function"
    except Exception:
        # Fallback if anything goes wrong with inspection
        return "generative function"


def _get_current_call_location() -> str:
    """Get the current call location where the @ operator is being used.

    Returns:
        String describing the file and line where addressing is happening
    """
    try:
        # Walk up the stack to find the first frame outside of GenJAX core and beartype
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name

            # Skip frames that are:
            # 1. In the GenJAX core module
            # 2. Beartype wrappers
            # 3. Internal Python machinery
            skip_conditions = [
                filename.endswith("core.py"),
                "beartype" in filename,
                function_name.startswith("__"),
                function_name
                in ["trace", "simulate", "assess", "generate", "update", "regenerate"],
            ]

            if not any(skip_conditions):
                line_no = frame.f_lineno
                # Just show the basename of the file for cleaner output
                basename = filename.split("/")[-1] if "/" in filename else filename
                return f"{basename}:{line_no}"

        # If we can't find a good frame, just return basic info
        return "unknown location"
    except Exception:
        return "unknown location"


def _check_address_collision(
    addr: str, trace_map: dict[str, Any], gen_fn: "GFI" = None
) -> None:
    """Check for address collision and raise ValueError if detected.

    Args:
        addr: The address being used
        trace_map: Dictionary tracking addresses already used
        gen_fn: Optional generative function for context

    Raises:
        ValueError: If the address has already been used
    """
    if addr in trace_map:
        # Get location information
        func_info = (
            _get_generative_function_info(gen_fn) if gen_fn else "generative function"
        )
        call_location = _get_current_call_location()

        raise ValueError(
            f"Address collision detected: '{addr}' is used multiple times at the same level.\n"
            f"Each address in a generative function must be unique.\n"
            f"Function: {func_info}\n"
            f"Location: {call_location}"
        )


def _check_address_collision_visited(
    addr: str, visited_addresses: set[str], gen_fn: "GFI" = None
) -> None:
    """Check for address collision using a visited set and raise ValueError if detected.

    Args:
        addr: The address being used
        visited_addresses: Set tracking addresses already visited
        gen_fn: Optional generative function for context

    Raises:
        ValueError: If the address has already been visited
    """
    if addr in visited_addresses:
        # Get location information
        func_info = (
            _get_generative_function_info(gen_fn) if gen_fn else "generative function"
        )
        call_location = _get_current_call_location()

        raise ValueError(
            f"Address collision detected: '{addr}' is used multiple times at the same level.\n"
            f"Each address in a generative function must be unique.\n"
            f"Function: {func_info}\n"
            f"Location: {call_location}"
        )
    visited_addresses.add(addr)


@dataclass
class Simulate:
    """Handler for simulating generative function executions.

    Tracks the accumulated score and trace map during simulation.

    Args:
        score: Cumulative log probability score.
        trace_map: Mapping from addresses to trace objects.
        parent_fn: Optional reference to the @gen function being executed.
    """

    score: Weight
    trace_map: dict[str, Any]
    parent_fn: "GFI" = None

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
        kwargs=None,
    ) -> R:
        kwargs = kwargs or {}

        # Check for address collision
        _check_address_collision(addr, self.trace_map, self.parent_fn or gen_fn)

        tr = gen_fn.simulate(*args, **kwargs)
        self.score += tr.get_score()
        self.trace_map[addr] = tr
        return tr.get_retval()


@dataclass
class Generate:
    choice_map: dict[str, Any]
    score: Weight
    weight: Weight
    trace_map: dict[str, Any]
    parent_fn: "GFI" = None

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
        kwargs=None,
    ) -> R:
        kwargs = kwargs or {}

        # Check for address collision
        _check_address_collision(addr, self.trace_map, self.parent_fn or gen_fn)

        x = (
            get_choices(
                self.choice_map[addr],
            )
            if addr in self.choice_map
            else None
        )
        tr, weight = gen_fn.generate(x, *args, **kwargs)
        self.score += tr.get_score()
        self.weight += weight
        self.trace_map[addr] = tr
        return tr.get_retval()


@dataclass
class Assess:
    choice_map: dict[str, Any]
    logp: Density
    visited_addresses: set[str] = field(default_factory=set)
    parent_fn: "GFI" = None

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
        kwargs=None,
    ) -> R:
        kwargs = kwargs or {}

        # Check for address collision
        _check_address_collision_visited(
            addr, self.visited_addresses, self.parent_fn or gen_fn
        )

        x = self.choice_map[addr]
        x = get_choices(x)
        logp, r = gen_fn.assess(x, *args, **kwargs)
        self.logp += logp
        return r


@dataclass
class Update(Generic[R]):
    trace: Tr[dict[str, Any], R]
    choice_map: dict[str, Any]
    trace_map: dict[str, Any]
    discard: dict[str, Any]
    score: Score
    weight: Weight
    parent_fn: "GFI" = None

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args_,
        kwargs=None,
    ) -> R:
        kwargs = kwargs or {}

        # Check for address collision
        _check_address_collision(addr, self.trace_map, self.parent_fn or gen_fn)

        # Get the full subtrace (Tr object) from the trace structure
        subtrace = self.trace._choices[
            addr
        ]  # This is the Tr object, not just the value
        x = (
            self.choice_map[addr]
            if addr in self.choice_map
            else self.trace.get_choices()[addr]
        )
        x = get_choices(x)
        tr, w, discard = gen_fn.update(subtrace, x, *args_, **kwargs)
        self.trace_map[addr] = tr
        self.discard[addr] = discard
        self.score += tr.get_score()
        self.weight += w
        return tr.get_retval()


@dataclass
class Regenerate(Generic[R]):
    trace: Tr[dict[str, Any], R]
    s: Selection
    trace_map: dict[str, Any]
    discard: dict[str, Any]
    score: Score
    weight: Weight
    parent_fn: "GFI" = None

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args_,
        kwargs=None,
    ) -> R:
        kwargs = kwargs or {}

        # Check for address collision
        _check_address_collision(addr, self.trace_map, self.parent_fn or gen_fn)

        # Get the full subtrace (Tr object) from the trace structure
        subtrace = self.trace._choices[
            addr
        ]  # This is the Tr object, not just the value
        # Use Selection.match to check if this address is selected
        should_regenerate, subsel = self.s.match(addr)
        tr, w, discard = gen_fn.regenerate(subtrace, subsel, *args_, **kwargs)
        self.trace_map[addr] = tr
        self.discard[addr] = discard
        self.score += tr.get_score()
        self.weight += w
        return tr.get_retval()


handler_stack: list[Simulate | Assess | Generate | Update | Regenerate] = []


# Generative function "FFI" invocation (no staging).
def trace(
    addr: str,
    gen_fn: GFI[X, R],
    args,
    kwargs=None,
) -> R:
    handler = handler_stack[-1]
    retval = handler(addr, gen_fn, args, kwargs or {})
    return retval


@Pytree.dataclass
class Fn(
    Generic[R],
    GFI[dict[str, Any], R],
):
    """A `Fn` is a generative function created from a JAX Python function
    using the `@gen` decorator.

    `Fn` implements the GFI by executing the wrapped function in different execution contexts
    (handlers) that intercept calls to other generative functions via the `@` addressing syntax.

    Mathematical ingredients:
    - Measure kernel P(dx; args) defined by the composition of distributions in the function
    - Return value function f(x, args) defined by the function's logic and return statement
    - Internal proposal distribution family Q(dx; args, x') defined by ancestral sampling

    The choice space X is a dictionary mapping addresses (strings) to the choices made
    at those addresses during execution.

    Attributes:
        source: The original Python function that defines the probabilistic computation

    Example:
        >>> import jax.numpy as jnp
        >>> from genjax import gen, normal
        >>>
        >>> @gen
        >>> def linear_regression(xs):
        ...     slope = normal(0.0, 1.0) @ "slope"
        ...     intercept = normal(0.0, 1.0) @ "intercept"
        ...     noise = normal(0.0, 0.1) @ "noise"
        ...     return normal(slope * xs + intercept, noise) @ "y"
        >>>
        >>> trace = linear_regression.simulate(jnp.array([1.0, 2.0, 3.0]))
        >>> choices = trace.get_choices()  # dict with keys "slope", "intercept", "noise", "y"
    """

    source: Const[Callable[..., R]]

    def simulate(
        self,
        *args,
        **kwargs,
    ) -> Tr[dict[str, Any], R]:
        handler_stack.append(Simulate(jnp.array(0.0), {}, self))
        r = self.source.value(*args, **kwargs)
        handler = handler_stack.pop()
        assert isinstance(handler, Simulate)
        score, trace_map = handler.score, handler.trace_map
        return Tr(self, (args, kwargs), trace_map, r, score)

    def generate(
        self,
        x: dict[str, Any] | None,
        *args,
        **kwargs,
    ) -> tuple[Tr[dict[str, Any], R], Weight]:
        if x is None:
            tr = self.simulate(*args, **kwargs)
            return tr, jnp.array(0.0)
        else:
            handler_stack.append(Generate(x, jnp.array(0.0), jnp.array(0.0), {}, self))
            r = self.source.value(*args, **kwargs)
            handler = handler_stack.pop()
            assert isinstance(handler, Generate)
            score, weight, trace_map = handler.score, handler.weight, handler.trace_map
            return Tr(self, (args, kwargs), trace_map, r, score), weight

    def assess(
        self,
        x: dict[str, Any],
        *args,
        **kwargs,
    ) -> tuple[Density, R]:
        handler_stack.append(Assess(x, jnp.array(0.0), set(), self))
        r = self.source.value(*args, **kwargs)
        handler = handler_stack.pop()
        assert isinstance(handler, Assess)
        logp = handler.logp
        return logp, r

    def update(
        self,
        tr: Tr[dict[str, Any], R],
        x_: dict[str, Any] | None,
        *args,
        **kwargs,
    ) -> tuple[Tr[dict[str, Any], R], Weight, dict[str, Any] | None]:
        x_ = {} if x_ is None else x_
        handler_stack.append(
            Update(tr, x_, {}, {}, jnp.array(0.0), jnp.array(0.0), self)
        )
        r = self.source.value(*args, **kwargs)
        handler = handler_stack.pop()
        assert isinstance(handler, Update)
        trace_map, score, w, discard = (
            handler.trace_map,
            handler.score,
            handler.weight,
            handler.discard,
        )
        return Tr(self, (args, kwargs), trace_map, r, score), w, discard

    def regenerate(
        self,
        tr: Tr[dict[str, Any], R],
        s: Selection,
        *args,
        **kwargs,
    ) -> tuple[Tr[dict[str, Any], R], Weight, dict[str, Any] | None]:
        handler_stack.append(
            Regenerate(tr, s, {}, {}, jnp.array(0.0), jnp.array(0.0), self)
        )
        r = self.source.value(*args, **kwargs)
        handler = handler_stack.pop()
        assert isinstance(handler, Regenerate)
        trace_map, score, w, discard = (
            handler.trace_map,
            handler.score,
            handler.weight,
            handler.discard,
        )
        return Tr(self, (args, kwargs), trace_map, r, score), w, discard

    def merge(
        self,
        x: dict[str, Any],
        x_: dict[str, Any],
        check: jnp.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        # Handle recursive merge for nested dictionaries
        result = {}
        discarded = {}
        all_keys = set(x.keys()) | set(x_.keys())

        for key in all_keys:
            if key in x and key in x_:
                # Both dictionaries have this key
                val_x = x[key]
                val_x_ = x_[key]

                # If both values are dictionaries, recursively merge
                if isinstance(val_x, dict) and isinstance(val_x_, dict):
                    merged_val, disc_val = self.merge(val_x, val_x_, check)
                    result[key] = merged_val
                    if disc_val is not None:
                        discarded[key] = disc_val
                else:
                    # Conflict: same key but values are not both dictionaries
                    if check is not None:
                        # Use conditional selection at the leaf
                        result[key] = jtu.tree_map(
                            lambda v1, v2: jnp.where(check, v1, v2), val_x, val_x_
                        )
                        # In conditional merge, nothing is truly discarded
                    else:
                        # Without check, x_ takes precedence and x is discarded
                        result[key] = val_x_
                        discarded[key] = val_x
            elif key in x:
                # Only in proposal
                result[key] = x[key]
            else:
                # Only in constraints
                result[key] = x_[key]

        return result, discarded if discarded else None

    def filter(
        self, x: dict[str, Any], selection: "Selection"
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Filter choice map into selected and unselected parts.

        For Fn, choices are stored as dict[str, Any] with string addresses.

        Args:
            x: Choice dictionary to filter.
            selection: Selection specifying which addresses to include.

        Returns:
            Tuple of (selected_choices, unselected_choices) where each is a dict or None.
        """
        if not x:
            return None, None

        selected = {}
        unselected = {}
        found_selected = False
        found_unselected = False

        for addr, value in x.items():
            is_selected, subselection = selection.match(addr)
            if is_selected:
                if isinstance(value, dict) and subselection is not None:
                    # Recursively filter nested choices
                    selected_sub, unselected_sub = self.filter(value, subselection)
                    if selected_sub is not None:
                        selected[addr] = selected_sub
                        found_selected = True
                    if unselected_sub is not None:
                        unselected[addr] = unselected_sub
                        found_unselected = True
                else:
                    # Include the entire value in selected
                    selected[addr] = value
                    found_selected = True
            else:
                # Include the entire value in unselected
                unselected[addr] = value
                found_unselected = True

        return (
            selected if found_selected else None,
            unselected if found_unselected else None,
        )


def gen(fn: Callable[..., R]) -> Fn[R]:
    """Convert a function into a generative function.

    The decorated function can use the `@` operator to make addressed
    random choices from distributions and other generative functions.

    Args:
        fn: Function to convert into a generative function.

    Returns:
        A Fn instance that implements the Generative Function Interface.

    Example:
        >>> from genjax import gen, normal
        >>>
        >>> @gen
        ... def model(mu, sigma):
        ...     x = normal(mu, sigma) @ "x"
        ...     y = normal(x, 0.1) @ "y"
        ...     return x + y
        >>>
        >>> trace = model.simulate(0.0, 1.0)
        >>> choices = trace.get_choices()
        >>> # choices will contain {"x": <value>, "y": <value>}
    """
    gf = Fn(source=const(fn))
    # Copy function metadata to preserve name and module information
    try:
        gf.__name__ = fn.__name__
        gf.__qualname__ = fn.__qualname__
        gf.__module__ = fn.__module__
        gf.__doc__ = fn.__doc__
        gf.__annotations__ = getattr(fn, "__annotations__", {})
    except (AttributeError, TypeError):
        # If we can't set these attributes (e.g., on frozen dataclasses), continue anyway
        pass
    return gf


########
# Scan #
########


@Pytree.dataclass
class ScanTr(Generic[X, R], Trace[X, R]):
    gen_fn: "Scan[X, R]"
    args: Any
    traces: Trace[X, R]  # Vectorized trace
    final_carry: Any
    outs: Any

    def get_gen_fn(self) -> "Scan[X, R]":
        return self.gen_fn

    def get_choices(self) -> X:
        return self.traces.get_choices()

    def get_fixed_choices(self) -> X:
        """Get choices preserving Fixed wrappers."""
        return self.traces.get_fixed_choices()

    def get_args(self) -> Any:
        return self.args

    def get_retval(self) -> R:
        return (self.final_carry, self.outs)

    def get_score(self) -> Score:
        return self.traces.get_score()


@Pytree.dataclass
class Scan(Generic[X, R], GFI[X, R]):
    """A `Scan` is a generative function combinator that implements sequential iteration.

    `Scan` repeatedly applies a generative function in a sequential loop, similar to
    `jax.lax.scan`, but preserves probabilistic semantics. The callee function should
    take (carry, x) as input and return (new_carry, output).

    Mathematical ingredients:
    - If callee has measure kernel P_callee(dx; carry, x), then Scan has kernel
      P_scan(dX; init_carry, xs) = ∏_i P_callee(dx_i; carry_i, xs_i)
      where carry_{i+1} = f_callee(x_i, carry_i, xs_i)[0]
    - Return value function returns (final_carry, [output_1, ..., output_n])
    - Internal proposal family inherits from callee's proposal family

    Attributes:
        callee: The generative function to apply sequentially
        length: Fixed length for the scan

    Example:
        >>> from genjax import gen, normal, Scan, seed, const
        >>> import jax.numpy as jnp
        >>> import jax.random as jrand
        >>>
        >>> @gen
        >>> def step(carry, x):
        ...     noise = normal(0.0, 0.1) @ "noise"
        ...     new_carry = carry + x + noise
        ...     return new_carry, new_carry  # output equals new carry
        >>>
        >>> scan_fn = Scan(step, length=const(3))
        >>> init_carry = 0.0
        >>> xs = jnp.array([1.0, 2.0, 3.0])
        >>> # Use seed transformation for PJAX primitives
        >>> key = jrand.key(0)
        >>> trace = seed(scan_fn.simulate)(key, init_carry, xs)
        >>> final_carry, outputs = trace.get_retval()
        >>> assert len(outputs) == 3  # Should have 3 outputs
    """

    callee: GFI[X, R]
    length: Const[int]

    def merge(
        self, x: X, x_: X, check: jnp.ndarray | None = None
    ) -> tuple[X, X | None]:
        # For Scan, delegate to the callee's merge with check parameter
        return self.callee.merge(x, x_, check)

    def filter(self, x: X, selection: "Selection") -> tuple[X | None, X | None]:
        """Filter scan choices using the underlying generative function's filter.

        For Scan, choices are structured according to the scan iterations.
        We delegate to the underlying callee's filter method.

        Args:
            x: Scan choice structure to filter.
            selection: Selection specifying which addresses to include.

        Returns:
            Tuple of (selected_choices, unselected_choices) from the underlying callee.
        """
        return self.callee.filter(x, selection)

    def simulate(
        self,
        *args,
        **kwargs,
    ) -> ScanTr[X, R]:
        init_carry, xs = args[0], args[1]

        def scan_fn(carry, x):
            trace = self.callee.simulate(carry, x, **kwargs)
            new_carry = trace.get_retval()[0]  # (C, Out) -> C
            out = trace.get_retval()[1]  # (C, Out) -> Out
            return new_carry, (trace, out)

        final_carry, (traces, outs) = scan(
            scan_fn,
            init_carry,
            xs,
            length=self.length.value,
        )

        return ScanTr(self, (args, kwargs), traces, final_carry, outs)

    def generate(
        self,
        x: X | None,
        *args,
        **kwargs,
    ) -> tuple[ScanTr[X, R], Weight]:
        carry_args, scanned_args = args[0], args[1]

        def scan_fn(carry, scanned):
            (scanned_args, x) = scanned
            trace, weight = self.callee.generate(x, carry, scanned_args, **kwargs)
            new_carry = trace.get_retval()[0]
            scanned_out = trace.get_retval()[1]
            return new_carry, (trace, scanned_out, weight)

        final_carry, (traces, scanned_outs, weights) = scan(
            scan_fn,
            carry_args,
            (scanned_args, x),
            length=self.length.value,
        )

        total_weight = jnp.sum(weights)

        return ScanTr(
            self, (args, kwargs), traces, final_carry, scanned_outs
        ), total_weight

    def assess(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Density, R]:
        carry_args, scanned_args = args[0], args[1]

        def scan_fn(carry, scanned):
            (scanned_args, x) = scanned
            density, (new_carry, scanned_out) = self.callee.assess(
                x, carry, scanned_args, **kwargs
            )
            return new_carry, (scanned_out, density)

        final_carry, (scanned_outs, density) = scan(
            scan_fn,
            carry_args,
            (scanned_args, x),
            length=self.length.value,
        )
        return jnp.sum(density), (final_carry, scanned_outs)

    def update(
        self,
        tr: ScanTr[X, R],
        x_: X,
        *args,
        **kwargs,
    ) -> tuple[ScanTr[X, R], Weight, X | None]:
        carry_args, scanned_args = args[0], args[1]
        old_traces = tr.traces

        def scan_fn(carry, scanned_in):
            input_x, old_trace, new_choice = scanned_in
            new_trace, weight, discard = self.callee.update(
                old_trace, new_choice, carry, input_x, **kwargs
            )
            new_carry = new_trace.get_retval()[0]
            out = new_trace.get_retval()[1]
            return new_carry, (new_trace, weight, discard, out)

        # Pack xs, old_traces, and x_ together for scan
        scan_inputs = (
            (scanned_args, old_traces, x_)
            if x_ is not None
            else (scanned_args, old_traces, None)
        )

        final_carry, (new_traces, weights, discards, outs) = scan(
            scan_fn,
            carry_args,
            scan_inputs,
            length=self.length.value,
        )

        total_weight = jnp.sum(weights)

        new_tr = ScanTr(self, (args, kwargs), new_traces, final_carry, outs)
        return new_tr, total_weight, discards

    def regenerate(
        self,
        tr: ScanTr[X, R],
        s: Selection,
        *args,
        **kwargs,
    ) -> tuple[ScanTr[X, R], Weight, X | None]:
        init_carry, xs = args[0], args[1]
        old_traces = tr.traces

        def scan_fn(carry, xs_old_trace):
            input_x, old_trace = xs_old_trace
            new_trace, weight, discard = self.callee.regenerate(
                old_trace,
                s,  # selection applies to all steps
                carry,
                input_x,
                **kwargs,
            )
            new_carry = new_trace.get_retval()[0]
            out = new_trace.get_retval()[1]
            return new_carry, (new_trace, weight, discard, out)

        # Pack xs and old_traces together for scan
        scan_inputs = (xs, old_traces)

        final_carry, (new_traces, weights, discards, outs) = scan(
            scan_fn, init_carry, scan_inputs, length=self.length.value
        )

        total_weight = jnp.sum(weights)
        # discards will be vectorized, so we need to handle them appropriately
        any_discards = jnp.any(jtu.tree_map(lambda x: x is not None, discards))

        new_tr = ScanTr(self, (args, kwargs), new_traces, final_carry, outs)
        return new_tr, total_weight, discards if any_discards else None


########
# Cond #
########


@Pytree.dataclass
class CondTr(Generic[X, R], Trace[X, R]):
    gen_fn: "Cond[X, R]"
    check: BoolArray
    trs: list[Trace[X, R]]

    def get_gen_fn(self) -> "GFI[X, R]":
        return self.gen_fn

    def get_choices(self) -> X:
        chm, chm_ = map(get_choices, self.trs)

        # Use merge with check parameter for conditional selection
        merged, _ = self.gen_fn.merge(chm, chm_, self.check)
        return merged

    def get_fixed_choices(self) -> X:
        """Get choices preserving Fixed wrappers."""
        chm, chm_ = map(lambda tr: tr.get_fixed_choices(), self.trs)

        # Use merge with check parameter for conditional selection
        merged, _ = self.gen_fn.merge(chm, chm_, self.check)
        return merged

    def get_args(self) -> Any:
        return (self.check, *self.trs[0].get_args())

    def get_retval(self) -> R:
        return jnp.where(self.check, *map(get_retval, self.trs))

    def get_score(self) -> Score:
        return jnp.where(self.check, *map(get_score, self.trs))


@Pytree.dataclass
class Cond(Generic[X, R], GFI[X, R]):
    """A `Cond` is a generative function combinator that implements conditional branching.

    `Cond` takes a boolean condition and executes one of two generative functions
    based on the condition, similar to `jax.lax.cond`, but preserves probabilistic
    semantics by evaluating both branches and selecting the appropriate one.

    Mathematical ingredients:
    - If branches have measure kernels P_true(dx; args) and P_false(dx; args), then
      Cond has kernel P_cond(dx; check, args) = P_true(dx; args) if check else P_false(dx; args)
    - Return value function f_cond(x, check, args) = f_true(x, args) if check else f_false(x, args)
    - Internal proposal family selects appropriate branch proposal based on condition

    Note: Both branches are always evaluated during simulation/generation to maintain
    JAX compatibility, but only the appropriate branch contributes to the final result.

    Attributes:
        callee: The generative function to execute when condition is True
        callee_: The generative function to execute when condition is False

    Example:
        >>> from genjax import gen, normal, exponential, Cond
        >>>
        >>> @gen
        >>> def positive_branch():
        ...     return exponential(1.0) @ "value"
        >>>
        >>> @gen
        >>> def negative_branch():
        ...     return exponential(2.0) @ "value"
        >>>
        >>> cond_fn = Cond(positive_branch, negative_branch)
        >>>
        >>> # Use in a larger model
        >>> @gen
        >>> def conditional_model():
        ...     x = normal(0.0, 1.0) @ "x"
        ...     condition = x > 0
        ...     result = cond_fn((condition,)) @ "conditional"
        ...     return result
    """

    callee: GFI[X, R]
    callee_: GFI[X, R]

    def merge(
        self, x: X, x_: X, check: jnp.ndarray | None = None
    ) -> tuple[X, X | None]:
        # For Cond, delegate to callee's merge with check parameter
        return self.callee.merge(x, x_, check)

    def filter(self, x: X, selection: "Selection") -> tuple[X | None, X | None]:
        """Filter conditional choices using the underlying generative function's filter.

        For Cond, choices are determined by which branch was executed.
        We delegate to the first callee's filter method.

        Args:
            x: Conditional choice structure to filter.
            selection: Selection specifying which addresses to include.

        Returns:
            Tuple of (selected_choices, unselected_choices) from the underlying callee.
        """
        return self.callee.filter(x, selection)

    def simulate(
        self,
        *args,
        **kwargs,
    ) -> CondTr[X, R]:
        (check, *rest_args) = args
        tr = self.callee.simulate(*rest_args, **kwargs)
        tr_ = self.callee_.simulate(*rest_args, **kwargs)
        return CondTr(self, check, [tr, tr_])

    def assess(
        self,
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Density, R]:
        (check, *rest_args) = args
        logp, r = self.callee.assess(x, *rest_args, **kwargs)
        logp_, r_ = self.callee_.assess(x, *rest_args, **kwargs)
        total_density = jnp.where(check, logp, logp_)
        retval = jnp.where(check, r, r_)
        return total_density, retval

    def generate(
        self,
        x: X | None,
        *args,
        **kwargs,
    ) -> tuple[CondTr[X, R], Weight]:
        (check, *rest_args) = args
        if x is None:
            tr = self.callee.simulate(*rest_args, **kwargs)
            tr_ = self.callee_.simulate(*rest_args, **kwargs)
            return CondTr(self, check, [tr, tr_]), jnp.array(0.0)
        else:
            tr, w = self.callee.generate(x, *rest_args, **kwargs)
            tr_, w_ = self.callee_.generate(x, *rest_args, **kwargs)
            total_weight = jnp.where(check, w, w_)
            return CondTr(self, check, [tr, tr_]), total_weight

    def update(
        self,
        tr: CondTr[X, R],
        x: X,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X]:
        (check, *rest_args) = args
        new_tr, w, discard = self.callee.update(tr.trs[0], x, *rest_args, **kwargs)
        new_tr_, w_, discard_ = self.callee_.update(tr.trs[1], x, *rest_args, **kwargs)
        # Merge discarded values
        merged_discard, _ = self.callee.merge(discard, discard_)
        return (
            CondTr(self, check, [new_tr, new_tr_]),
            jnp.where(check, w, w_),
            merged_discard,
        )

    def regenerate(
        self,
        tr: CondTr[X, R],
        s: Selection,
        *args,
        **kwargs,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        (check, *rest_args) = args
        new_tr, w, discard = self.callee.regenerate(tr.trs[0], s, *rest_args, **kwargs)
        new_tr_, w_, discard_ = self.callee_.regenerate(
            tr.trs[1], s, *rest_args, **kwargs
        )
        if discard is None:
            merged_discard = discard_
        elif discard_ is None:
            merged_discard = discard
        else:
            merged_discard, _ = self.callee.merge(discard, discard_)
        return (
            CondTr(self, check, [new_tr, new_tr_]),
            jnp.where(check, w, w_),
            merged_discard,
        )
