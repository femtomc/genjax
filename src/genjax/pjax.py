"""PJAX: Probabilistic JAX

This module implements PJAX (Probabilistic JAX), which extends JAX with probabilistic
primitives and specialized interpreters for handling probabilistic computations.

PJAX provides the foundational infrastructure for GenJAX's probabilistic programming
capabilities by introducing:

1. **Probabilistic Primitives**: Custom JAX primitives (`assume_p`, `log_density_p`)
   that represent random sampling and density evaluation operations.

2. **JAX-aware Interpreters**: Specialized interpreters that handle probabilistic
   primitives while preserving JAX's transformation semantics:
   - `SeedInterpreter`: Eliminates PJAX's sampling primitive for JAX PRNG implementations
   - `ModularVmapInterpreter`: Vectorizes probabilistic computations

3. **Staging Infrastructure**: Tools for converting Python functions to JAX's
   intermediate representation (Jaxpr) while preserving probabilistic semantics.

Key Concepts:
    - **Assume Primitive**: Represents random sampling operations in Jaxpr
    - **Seed Transformation**: Converts probabilistic functions to accept explicit keys
    - **Modular Vmap**: Vectorizes probabilistic functions while preserving semantics
    - **Elaborated Primitives**: Enhanced primitives with metadata for pretty printing

Usage:
    ```python
    from genjax.pjax import seed, modular_vmap, assume_binder

    # Transform probabilistic function to accept explicit keys
    seeded_fn = seed(probabilistic_function)
    result = seeded_fn(key, args)

    # Vectorize probabilistic computations
    vmap_fn = modular_vmap(probabilistic_function, in_axes=(0,))
    results = vmap_fn(batched_args)
    ```

Technical Details:
    PJAX works by representing sampling and density evaluation as JAX primitives that can be
    interpreted differently depending on the transformation applied. The `seed`
    transformation eliminates the sampling primitive by providing explicit randomness,
    while `modular_vmap` preserves both primitives for probability-aware vectorization.

References:
    - JAX Primitives: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
    - GenJAX Documentation: See src/genjax/CLAUDE.md for PJAX usage patterns
"""

import itertools as it
import warnings
from dataclasses import dataclass, field
from functools import partial, wraps

# Core JAX imports
import jax
import jax.core as jc
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import api_util, tree_util
from jax._src.interpreters.batching import AxisData  # pyright: ignore
from jax.core import eval_jaxpr
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Primitive, Var
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe
from jax.lax import cond_p, scan, scan_p, switch
from jax.util import safe_map, split_list

# External imports
import beartype.typing as btyping
import jaxtyping as jtyping
from numpy import dtype

##########
# Types  #
##########

Any = btyping.Any
Callable = btyping.Callable
Sequence = btyping.Sequence
TypeVar = btyping.TypeVar

# JAX-specific types
PRNGKey = jtyping.PRNGKeyArray
Array = jtyping.Array

# Type variables
R = TypeVar("R")
VarOrLiteral = Var | Literal

############################################
# Staging utilities for Jaxpr interpreters #
############################################


def get_shaped_aval(x):
    """Get the shaped abstract value of a JAX array."""
    return jc.get_aval(x)


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    """Cache-enabled function to stage a flattened function to Jaxpr.

    Args:
        flat_fun: Flattened function to stage.
        in_avals: Input abstract values.

    Returns:
        ClosedJaxpr representing the function.
    """
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


def stage(f, **params):
    """Stage a function to JAX's intermediate representation (Jaxpr).

    Converts a Python function into JAX's Jaxpr format, which enables
    interpretation and transformation of the function's computation graph.
    This is essential for PJAX's ability to inspect and transform
    probabilistic computations.

    Args:
        f: Function to stage to JAX representation.
        **params: Additional parameters to pass to the wrapped function.

    Returns:
        Callable that returns a tuple of (ClosedJaxpr, execution_metadata).
        The execution metadata contains flattened arguments, input/output trees,
        and tree reconstruction functions.

    Example:
        ```python
        def my_function(x, y):
            return x + y

        staged_fn = stage(my_function)
        jaxpr, metadata = staged_fn(1.0, 2.0)
        # jaxpr contains the computational graph
        # metadata contains argument information for reconstruction
        ```
    """

    @wraps(f)
    def wrapped(
        *args, **kwargs
    ) -> tuple[ClosedJaxpr, tuple[list[Any], Any, Callable[..., Any]]]:
        debug_info = api_util.debug_info("genjax.stage", f, args, kwargs)
        fun = lu.wrap_init(f, params, debug_info=debug_info)
        if kwargs:
            flat_args, in_tree = jtu.tree_flatten((args, kwargs))
            flat_fun, out_tree = api_util.flatten_fun(fun, in_tree)
        else:
            flat_args, in_tree = jtu.tree_flatten(args)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        closed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return closed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


#########################
# Custom JAX primitives #
#########################


class InitialStylePrimitive(Primitive):
    """JAX primitive with configurable transformation implementations.

    This class extends JAX's `Primitive` to provide a convenient way to define
    custom primitives where transformation semantics are provided at the binding
    site rather than registration time. This is essential for PJAX's dynamic
    primitive creation where the same primitive can have different behaviors
    depending on the probabilistic context.

    The primitive expects implementations for JAX transformations to be provided
    as parameters during `initial_style_bind(...)` calls using these keys:

    Transformation Keys:
        - `impl`: Evaluation semantics - the concrete implementation that executes
                 when the primitive is evaluated with concrete values
        - `abstract`: Abstract semantics - used by JAX when tracing a Python program
                     to a Jaxpr; determines output shapes and dtypes from input abstract values
        - `jvp`: Forward-mode automatic differentiation - defines how to compute
                Jacobian-vector products for this primitive
        - `batch`: Vectorization semantics for `vmap` - defines how the primitive
                  behaves when vectorized over a batch dimension
        - `lowering`: Compilation semantics for `jit` - defines how to lower the
                     primitive to MLIR for XLA compilation

    Args:
        name: Name of the primitive (used in Jaxpr pretty-printing).

    Technical Details:
        Unlike standard JAX primitives where transformation rules are registered
        once, InitialStylePrimitive defers all rule definitions to binding time.
        The primitive acts as a parameterizable template where transformation
        semantics are injected dynamically, enabling PJAX's context-dependent
        reinterpretation of probabilistic operations.

    Example:
        ```python
        my_primitive = InitialStylePrimitive("my_op")

        # Transformation semantics provided at binding time
        result = my_primitive.bind(
            inputs,
            impl=lambda x: x + 1,                    # Evaluation: add 1
            abstract=lambda aval: aval,              # Same shape/dtype
            jvp=lambda primals, tangents: (primals[0] + 1, tangents[0]),
            batch=lambda args, dim: (args[0] + 1,),  # Vectorized add
            lowering=my_lowering_rule
        )
        ```
    """

    def __init__(self, name):
        super(InitialStylePrimitive, self).__init__(name)
        self.multiple_results = True

        def impl(*flat_args, **params):
            return params["impl"](*flat_args, **params)

        def abstract(*flat_avals, **params):
            return params["abstract"](*flat_avals, **params)

        def jvp(
            flat_primals: tuple[Any, ...] | list[Any],
            flat_tangents: tuple[Any, ...] | list[Any],
            **params,
        ) -> tuple[list[Any], list[Any]]:
            return params["jvp"](flat_primals, flat_tangents, **params)

        def batch(flat_vector_args: tuple[Any, ...] | list[Any], dim, **params):
            return params["batch"](flat_vector_args, dim, **params)

        def lowering(ctx: mlir.LoweringRuleContext, *mlir_args, **params):
            if "lowering_warning" in params and lowering_warning:
                warnings.warn(params["lowering_warning"])
            elif "lowering_exception" in params and enforce_lowering_exception:
                raise params["lowering_exception"]
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(ctx, *mlir_args, **params)

        # Store for elaboration.
        self.impl = impl
        self.jvp = jvp
        self.abstract = abstract
        self.batch = batch
        self.lowering = lowering

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)


class PPPrimitive(Primitive):
    """A primitive wrapper that hides metadata from JAX's Jaxpr pretty printer.

    `PPPrimitive` (Pretty Print Primitive) wraps an underlying InitialStylePrimitive
    and stores metadata parameters in a hidden field to prevent them from cluttering
    JAX's Jaxpr pretty printer output. This is essential for PJAX because probabilistic
    primitives often carry complex metadata (samplers, distributions, transformation
    rules, etc.) that would make Jaxpr representations unreadable if displayed.

    The wrapper:
    - Stores the underlying primitive and its parameters in a private field
    - Hides metadata from JAX's Jaxpr pretty printer
    - Acts as a transparent proxy for all JAX transformations
    - Preserves all transformation behavior of the wrapped primitive

    Args:
        prim: The underlying InitialStylePrimitive to wrap.
        **params: Metadata parameters to hide from pretty printer.
                 These will be merged with parameters passed during binding.

    Technical Details:
        When JAX creates a Jaxpr representation, it only shows the primitive name
        and visible parameters. By storing metadata in the PPPrimitive's internal
        state rather than as binding parameters, we get clean Jaxpr output while
        preserving all the functionality and metadata needed for transformations.

    Example:
        ```python
        base_prim = InitialStylePrimitive("sample")

        # Without PPPrimitive: cluttered Jaxpr with all metadata visible
        # sample[impl=<function>, abstract=<function>, distribution="normal", ...]

        # With PPPrimitive: clean Jaxpr output
        pretty_prim = PPPrimitive(base_prim, distribution="normal", name="x")
        # Jaxpr shows: sample

        result = pretty_prim.bind(args, mu=0.0, sigma=1.0)
        ```
    """

    def __init__(self, prim: InitialStylePrimitive, **params):
        super(PPPrimitive, self).__init__(prim.name)
        self.prim = prim
        self.multiple_results = self.prim.multiple_results
        self.params = params

        def impl(*args, **params):
            return self.prim.impl(*args, **self.params, **params)

        def abstract(*args, **params):
            return self.prim.abstract(*args, **self.params, **params)

        def jvp(*args, **params):
            return self.prim.jvp(*args, **self.params, **params)

        def batch(*args, **params):
            return self.prim.batch(*args, **self.params, **params)

        def lowering(*args, **params):
            return self.prim.lowering(*args, **self.params, **params)

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)

    @staticmethod
    def unwrap(v):
        return (v.prim, v.params) if isinstance(v, PPPrimitive) else (v, {})

    @staticmethod
    def check(primitive, other):
        if isinstance(primitive, PPPrimitive):
            return primitive.prim == other
        else:
            return primitive == other

    @staticmethod
    def rebind(
        primitive: Primitive,
        inner_params,
        params,
        *args,
    ):
        if isinstance(primitive, InitialStylePrimitive):
            return PPPrimitive(primitive, **inner_params).bind(*args, **params)
        else:
            return primitive.bind(*args, **params)


def batch_fun(fun: lu.WrappedFun, axis_data, in_dims):
    tag = jc.TraceTag()
    in_dims = in_dims() if callable(in_dims) else in_dims
    batched, out_dims = batching.batch_subtrace(fun, tag, axis_data, in_dims)
    return batched, out_dims


def initial_style_bind(
    prim,
    modular_vmap_aware=True,
    **params,
):
    """Binds a primitive to a function call."""

    def bind(f, **elaboration_kwargs):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a `PPPrimitive`
            primitive, hiding the implementation details of the eval
            (impl) rule, abstract rule, batch rule, and jvp rule."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            debug_info = jaxpr.jaxpr.debug_info

            def impl(*flat_args, **params) -> list[Any]:
                consts, flat_args = split_list(flat_args, [params["num_consts"]])
                return jc.eval_jaxpr(jaxpr.jaxpr, consts, *flat_args)

            def abstract(*flat_avals, **params):
                if modular_vmap_aware:
                    if "ctx" in params and params["ctx"] == "modular_vmap":
                        flat_avals = flat_avals[1:]  # ignore dummy
                return pe.abstract_eval_fun(
                    impl,
                    *flat_avals,
                    debug_info=debug_info,
                    **params,
                )

            def batch(flat_vector_args: tuple[Any, ...] | list[Any], dims, **params):
                axis_data = AxisData(None, None, None, None)
                batched, out_dims = batch_fun(
                    lu.wrap_init(impl, params, debug_info=debug_info),
                    axis_data,
                    dims,
                )
                return batched.call_wrapped(*flat_vector_args), out_dims()

            def jvp(
                flat_primals: tuple[Any, ...] | list[Any],
                flat_tangents: tuple[Any, ...] | list[Any],
                **params,
            ) -> tuple[list[Any], list[Any]]:
                primals_out, tangents_out = ad.jvp(
                    lu.wrap_init(impl, params, debug_info=debug_info)
                ).call_wrapped(flat_primals, flat_tangents)

                # We always normalize back to list.
                return list(primals_out), list(tangents_out)

            if "impl" in params:
                impl = params["impl"]
                params.pop("impl")

            if "abstract" in params:
                abstract = params["abstract"]
                params.pop("abstract")

            if "batch" in params:
                batch = params["batch"]
                params.pop("batch")

            if "jvp" in params:
                jvp = params["jvp"]
                params.pop("jvp")

            elaborated_prim = PPPrimitive(
                prim,
                impl=impl,
                abstract=abstract,
                batch=batch,
                jvp=jvp,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                yes_kwargs=bool(kwargs),
                **params,
            )
            outs = elaborated_prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                **elaboration_kwargs,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind


##################################
# PJAX Core: Probabilistic Primitives #
##################################


def static_dim_length(in_axes, args: tuple[Any, ...]) -> int | None:
    # perform the in_axes massaging that vmap performs internally:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    elif isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    def find_axis_size(axis: int | None, x: Any) -> int | None:
        """Find the size of the axis specified by `axis` for the argument `x`."""
        if axis is not None:
            leaf = jtu.tree_leaves(x)[0]
            return leaf.shape[axis]

    # tree_map uses in_axes as a template. To have passed vmap validation, Any non-None entry
    # must bottom out in an array-shaped leaf, and all such leafs must have the same size for
    # the specified dimension. Fetching the first is sufficient.
    axis_sizes = jtu.tree_leaves(
        jtu.tree_map(
            find_axis_size,
            in_axes,
            args,
            is_leaf=lambda x: x is None,
        )
    )
    return axis_sizes[0] if axis_sizes else None


################################
# Core PJAX Primitives          #
################################


class TerminalStyle:
    """ANSI terminal styling for pretty-printed primitives."""

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# Core PJAX primitives that represent probabilistic operations
assume_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}pjax.assume{TerminalStyle.RESET}",
)
"""Core primitive representing random sampling operations.

`assume_p` is the fundamental primitive in PJAX that represents the act of
drawing a random sample from a probability distribution. It appears in
Jaxpr when probabilistic functions are staged, and different interpreters
handle it in different ways:

- `SeedInterpreter`: Replaces with actual sampling using provided PRNG key
- `ModularVmapInterpreter`: Vectorizes the sampling operation
- Standard JAX: Raises warning/exception (requires transformation)

The primitive carries metadata about the sampler function, distribution
parameters, and optional support constraints.
"""

log_density_p = InitialStylePrimitive(
    f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}pjax.log_density{TerminalStyle.RESET}",
)
"""Core primitive representing log-density evaluation operations.

`log_density_p` represents the evaluation of log probability density at
a given value. This is dual to `assume_p` - while `assume_p` generates
samples, `log_density_p` evaluates how likely those samples are under
the distribution.

Used primarily in:
- Density evaluation for inference algorithms
- Gradient computation in variational methods
- Importance weight calculations in SMC
"""


# Zero-cost, just staging.
def make_flat(f):
    @wraps(f)
    def _make_flat(*args, **kwargs):
        debug_info = api_util.debug_info("_make_flat", f, args, kwargs)
        jaxpr, *_ = stage(f)(*args, **kwargs)

        def flat(*flat_args, **params):
            consts, args = split_list(flat_args, [params["num_consts"]])
            return eval_jaxpr(jaxpr.jaxpr, consts, *args)

        return flat, debug_info

    return _make_flat


class LoweringSamplePrimitiveToMLIRException(Exception):
    pass


# This is very cheeky.
@dataclass
class GlobalKeyCounter:
    count: int = 0


# Very large source of unique keys.
global_counter = GlobalKeyCounter()

_fake_key = jrand.key(1)


def assume_binder(
    keyful_sampler: Callable[..., Any],
    name: str | None = None,
    batch_shape: tuple[int, ...] = (),
    support: Callable[..., Any] | None = None,
):
    keyful_with_batch_shape = partial(
        keyful_sampler,
        sample_shape=batch_shape,
    )

    def assume(*args, **kwargs):
        # We're playing a trick here by allowing users to invoke assume_p
        # without a key. So we hide it inside, and we pass this as the
        # impl of `assume_p`.
        #
        # This is problematic for JIT, which will cache the statically
        # generated key. But it's obvious to the user - their returned
        # random choices won't change!
        #
        # The `seed` transformation below solves the JIT problem directly.
        def keyless(*args, **kwargs):
            global_counter.count += 1
            return keyful_with_batch_shape(
                jrand.key(global_counter.count),
                *args,
                **kwargs,
            )

        # Zero-cost, just staging.
        flat_keyful_sampler, _ = make_flat(keyful_with_batch_shape)(
            _fake_key, *args, **kwargs
        )

        # Overload batching so that the primitive is retained
        # in the Jaxpr under vmap.
        # Holy smokes recursion.
        def batch(vector_args, batch_axes, **params):
            if "ctx" in params and params["ctx"] == "modular_vmap":
                axis_size = params["axis_size"]
                vector_args = tuple(vector_args[1:])  # ignore dummy
                batch_axes = tuple(batch_axes[1:])  # ignore dummy
                n = static_dim_length(batch_axes, vector_args)
                outer_batch_dim = (
                    () if n is not None else (axis_size,) if axis_size else ()
                )
                assert isinstance(outer_batch_dim, tuple)
                new_batch_shape = outer_batch_dim + batch_shape
                v = assume_binder(
                    keyful_sampler,
                    name=name,
                    batch_shape=new_batch_shape,
                    support=support,
                )(*vector_args)
                return (v,), (0 if n or axis_size else None,)
            else:
                raise NotImplementedError()

        lowering_msg = (
            "JAX is attempting to lower the `pjax.assume_p` primitive to MLIR. "
            "This will bake a PRNG key into the MLIR code, resulting in deterministic behavior. "
            "Instead, use `seed` to transform your function into one which allows keys to be passed in. "
            "Try and do this as high in the computation graph as you can."
        )
        lowering_exception = LoweringSamplePrimitiveToMLIRException(
            lowering_msg,
        )

        return initial_style_bind(
            assume_p,
            keyful_sampler=keyful_sampler,
            flat_keyful_sampler=flat_keyful_sampler,
            batch=batch,
            support=support,
            lowering_warning=lowering_msg,
            lowering_exception=lowering_exception,
        )(keyless, name=name)(*args, **kwargs)

    return assume


def log_density_binder(
    log_density_impl: Callable[..., Any],
    name: str | None = None,
):
    def log_density(*args, **kwargs):
        # TODO: really not sure if this is right if you
        # nest vmaps...
        def batch(vector_args, batch_axes, **params):
            n = static_dim_length(batch_axes, tuple(vector_args))
            num_consts = params["num_consts"]
            in_tree = jtu.tree_unflatten(params["in_tree"], vector_args[num_consts:])
            batch_tree = jtu.tree_unflatten(params["in_tree"], batch_axes[num_consts:])
            if params["yes_kwargs"]:
                args = in_tree[0]
                kwargs = in_tree[1]
                v = log_density_binder(
                    jax.vmap(
                        lambda args, kwargs: log_density_impl(*args, **kwargs),
                        in_axes=batch_tree,
                    ),
                    name=name,
                )(args, kwargs)
            else:
                v = log_density_binder(
                    jax.vmap(
                        log_density_impl,
                        in_axes=batch_tree,
                    ),
                    name=name,
                )(*in_tree)
            outvals = (v,)
            out_axes = (0 if n else None,)
            return outvals, out_axes

        return initial_style_bind(log_density_p, batch=batch)(
            log_density_impl, name=name
        )(*args, **kwargs)

    return log_density


def wrap_sampler(
    keyful_sampler,
    name: str | None = None,
    support=None,
):
    def _(*args, **kwargs):
        batch_shape = kwargs.get("shape", ())
        if "shape" in kwargs:
            kwargs.pop("shape")
        return assume_binder(
            keyful_sampler,
            name=name,
            batch_shape=batch_shape,
            support=support,
        )(
            *args,
            **kwargs,
        )

    return _


def wrap_logpdf(
    logpdf,
    name: str | None = None,
):
    """Wrap a log-density function to work with PJAX primitives.

    Args:
        logpdf: Function that computes log probability density.
        name: Optional name for the operation (used in Jaxpr pretty-printing).

    Returns:
        Function that binds the logpdf to the log_density_p primitive.
    """

    def _(v, *args, **kwargs):
        return log_density_binder(
            logpdf,
            name=name,
        )(v, *args, **kwargs)

    return _


###################################
# Jaxpr Interpretation Infrastructure #
###################################


@dataclass
class Environment:
    """Variable environment for Jaxpr interpretation.

    Manages the mapping between JAX variables (from Jaxpr) and their concrete
    values during interpretation. This is essential for interpreters that need
    to execute Jaxpr equations step-by-step while maintaining state.

    The environment handles both:
    - Var objects: Variables with unique identifiers
    - Literal objects: Constant values embedded in the Jaxpr

    This design enables efficient interpretation of probabilistic Jaxpr by
    PJAX's specialized interpreters.
    """

    env: dict[int, Any] = field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        """
        Read a value from a variable in the environment.
        """
        v = self.get(var)
        if v is None:
            assert isinstance(var, Var)
            raise ValueError(
                f"Unbound variable in interpreter environment at count {var.count}:\nEnvironment keys (count): {list(self.env.keys())}"
            )
        return v

    def get(self, var: VarOrLiteral) -> Any:
        if isinstance(var, Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        """
        Write a value to a variable in the environment.
        """
        if isinstance(var, Literal):
            return cell
        cur_cell = self.get(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        """
        Check if a variable is in the environment.
        """
        if isinstance(var, Literal):
            return True
        return var.count in self.env

    def copy(self):
        """
        `Environment.copy` is sometimes used to create a new environment with
        the same variables and values as the original, especially in CPS
        interpreters (where a continuation closes over the application of an
        interpreter to a `Jaxpr`).
        """
        keys = list(self.env.keys())
        return Environment({k: self.env[k] for k in keys})


####################
# Seed Interpreter #
####################


@dataclass
class SeedInterpreter:
    """Interpreter that eliminates probabilistic primitives with explicit randomness.

    The `SeedInterpreter` is PJAX's core mechanism for making probabilistic
    computations compatible with standard JAX transformations. It works by
    traversing a Jaxpr and replacing `assume_p` primitives with actual sampling
    operations using explicit PRNG keys.

    Key Features:
    - **Eliminates PJAX primitives**: Converts assume_p to concrete sampling
    - **Explicit randomness**: Uses provided PRNG key for all random operations
    - **JAX compatibility**: Output can be jit'd, vmap'd, grad'd normally
    - **Deterministic**: Same key produces same results (good for debugging)
    - **Hierarchical key splitting**: Automatically manages keys for nested operations

    The interpreter handles JAX control flow primitives (cond, scan) by
    recursively applying the seed transformation to their sub-computations.

    Usage Pattern:
        This interpreter is primarily used via the `seed()` transformation:

        ```python
        # Instead of using the interpreter directly:
        # interpreter = SeedInterpreter(key)
        # result = interpreter.run_interpreter(fn, args)

        # Use the seed transformation:
        seeded_fn = seed(fn)
        result = seeded_fn(key, args)
        ```

    Technical Details:
        The interpreter maintains a PRNG key that is split at each random
        operation, ensuring proper randomness while maintaining determinism.
        For control flow, it passes seeded versions of sub-computations to
        JAX's control primitives.
    """

    key: PRNGKey

    def eval_jaxpr_seed(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            primitive, inner_params = PPPrimitive.unwrap(eqn.primitive)

            if primitive == assume_p:
                invals = safe_map(env.read, eqn.invars)
                args = subfuns + invals
                flat_keyful_sampler = inner_params["flat_keyful_sampler"]
                self.key, sub_key = jrand.split(self.key)
                outvals = flat_keyful_sampler(sub_key, *args, **inner_params)

            elif primitive == cond_p:
                invals = safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                self.key, sub_key = jrand.split(self.key)
                branches = tuple(
                    seed(jex.core.jaxpr_as_fun(branch))
                    for branch in branch_closed_jaxprs
                )
                index_val, ops_vals = invals[0], invals[1:]
                outvals = switch(
                    index_val,
                    branches,
                    sub_key,
                    *ops_vals,
                )

            # We replace the original scan with a new scan
            # that calls the interpreter on the scan body,
            # carries the key through and evolves it.
            elif primitive == scan_p:
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
                    (key, in_carry) = carry
                    (idx, in_scan) = scanned_in
                    all_values = const_vals + jtu.tree_leaves((in_carry, in_scan))
                    sub_key = jrand.fold_in(key, idx)
                    outs = seed(body_fun)(sub_key, *all_values)
                    out_carry, out_scan = split_list(outs, [num_carry])
                    return (key, out_carry), out_scan

                self.key, sub_key = jrand.split(self.key)
                fold_idxs = jnp.arange(length)
                (_, flat_carry_out), scanned_out = scan(
                    new_body,
                    (sub_key, carry_vals),
                    (fold_idxs, xs_vals),
                    length=length,
                    reverse=reverse,
                )
                outvals = jtu.tree_leaves(
                    (flat_carry_out, scanned_out),
                )

            else:
                outvals = eqn.primitive.bind(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_seed(
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def seed(
    f: Callable[..., Any],
):
    """Transform a function to accept an explicit PRNG key.

    This transformation eliminates probabilistic primitives by providing
    explicit randomness through a PRNG key, enabling the use of standard
    JAX transformations like jit and vmap.

    Args:
        f: Function containing probabilistic computations to transform.

    Returns:
        Function that takes a PRNGKey as first argument followed by
        the original function arguments.

    Example:
        >>> import jax.random as jrand
        >>> from genjax import gen, normal, seed
        >>>
        >>> @gen
        ... def model():
        ...     return normal(0.0, 1.0) @ "x"
        >>>
        >>> seeded_model = seed(model.simulate)
        >>> key = jrand.key(0)
        >>> trace = seeded_model(key, ())
    """

    @wraps(f)
    def wrapped(key: PRNGKey, *args):
        interpreter = SeedInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped


############################################
# Modular Vmap Interpreter                #
############################################


@dataclass
class ModularVmapInterpreter:
    """Vectorization interpreter that preserves probabilistic primitives.

    The `ModularVmapInterpreter` extends JAX's `vmap` to handle probabilistic
    computations correctly. Unlike standard `vmap`, which would fail on
    PJAX primitives, this interpreter knows how to vectorize probabilistic
    operations while preserving their semantic meaning.

    Key Capabilities:
    - **Probabilistic vectorization**: Correctly handles `assume_p` under vmap
    - **Batch shape inference**: Automatically adjusts distribution batch shapes
    - **Control flow support**: Handles cond/scan within vectorized computations
    - **Semantic preservation**: Maintains probabilistic meaning across batches
    - **Performance**: Avoids explicit loops for better efficiency

    How It Works:
        The interpreter uses a "dummy argument" technique to track the vectorization
        axis size and injects this information into probabilistic primitives so
        they can adjust their behavior appropriately (e.g., sampling multiple
        independent values vs. broadcasting parameters).

    Usage:
        Primarily used via the `modular_vmap()` function:

        ```python
        # Vectorize a probabilistic function
        batch_fn = modular_vmap(prob_function, in_axes=(0,))
        batch_results = batch_fn(batch_args)

        # Each element gets independent randomness
        # Distribution parameters are correctly broadcast
        ```

    Technical Details:
        The interpreter maintains PJAX primitives in the Jaxpr rather than
        eliminating them (unlike SeedInterpreter). This allows proper
        vectorization semantics for probabilistic operations.
    """

    @staticmethod
    def eval_jaxpr_modular_vmap(
        axis_size: int,
        jaxpr: Jaxpr,
        consts: list[Any],
        flat_args: list[Any],
        dummy_arg: Array,
    ):
        env = Environment()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, flat_args)
        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals

            # Probabilistic.
            if PPPrimitive.check(eqn.primitive, assume_p):
                outvals = eqn.primitive.bind(
                    dummy_arg,
                    *args,
                    axis_size=axis_size,
                    ctx="modular_vmap",
                    **params,
                )

            elif eqn.primitive == cond_p:
                invals = safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                branches = tuple(
                    partial(
                        ModularVmapInterpreter.stage_and_run,
                        axis_size,
                        jex.core.jaxpr_as_fun(branch),
                    )
                    for branch in branch_closed_jaxprs
                )
                index_val, ops_vals = invals[0], invals[1:]
                outvals = switch(
                    index_val,
                    branches,
                    dummy_arg,
                    ops_vals,
                )

            elif eqn.primitive == scan_p:
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                unroll = params["unroll"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = split_list(
                    invals, [num_consts, num_carry]
                )

                body_fun = partial(
                    ModularVmapInterpreter.stage_and_run,
                    axis_size,
                    jex.core.jaxpr_as_fun(body_jaxpr),
                )

                def new_body(carry, x):
                    (dummy, *in_carry) = carry
                    all_out = body_fun(
                        dummy,
                        (*const_vals, *in_carry, *x),
                    )
                    out_carry, out_scan = split_list(all_out, [num_carry])
                    return (dummy, *out_carry), out_scan

                (_, *out_carry), out_scan = scan(
                    new_body,
                    (dummy_arg, *carry_vals),
                    xs=xs_vals,
                    length=length,
                    reverse=reverse,
                    unroll=unroll,
                )
                outvals = jtu.tree_leaves(
                    (out_carry, out_scan),
                )

            # Deterministic and not control flow.
            else:
                outvals = eqn.primitive.bind(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]

            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    @staticmethod
    def stage_and_run(axis_size, fn, dummy_arg, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(
            fn,
        )(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        assert axis_size is not None
        flat_out = ModularVmapInterpreter.eval_jaxpr_modular_vmap(
            axis_size,
            jaxpr,
            consts,
            flat_args,
            dummy_arg,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)

    def run_interpreter(
        self,
        in_axes: int | tuple[int | None, ...] | Sequence[Any] | None,
        axis_size: int | None,
        axis_name: str | None,
        spmd_axis_name: str | None,
        fn,
        *args,
    ):
        axis_size = static_dim_length(in_axes, args) if axis_size is None else axis_size
        assert axis_size is not None

        dummy_arg = jnp.array(
            [1 for _ in range(axis_size)],
            dtype=dtype("i1"),
        )
        return jax.vmap(
            partial(
                ModularVmapInterpreter.stage_and_run,
                axis_size,
                fn,
            ),
            in_axes=(0, in_axes),
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )(dummy_arg, args)


##########################################
# Public API: Core PJAX Transformations #
##########################################


def modular_vmap(
    f: Callable[..., R],
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
) -> Callable[..., R]:
    """Vectorize a function while preserving probabilistic semantics.

    This is PJAX's probabilistic-aware version of `jax.vmap`. Unlike standard
    `vmap`, which fails on probabilistic primitives, `modular_vmap` correctly
    handles probabilistic computations by preserving their semantic meaning
    across the vectorized dimension.

    Key Differences from `jax.vmap`:
    - **Probabilistic awareness**: Handles `assume_p` and `log_density_p` primitives
    - **Batch shape handling**: Automatically adjusts distribution batch shapes
    - **Independent sampling**: Each vectorized element gets independent randomness
    - **Semantic correctness**: Maintains probabilistic meaning across batches

    Args:
        f: Function to vectorize (may contain probabilistic operations).
        in_axes: Axis specification for input arguments (same as jax.vmap).
        axis_size: Size of the mapped axis (inferred if None).
        axis_name: Name for the mapped axis (for debugging).
        spmd_axis_name: SPMD axis name for parallel computation.

    Returns:
        Vectorized function that correctly handles probabilistic computations.

    Example:
        ```python
        import jax.random as jrand
        from genjax import normal, modular_vmap

        def sample_normal(mu):
            return normal(mu, 1.0)  # Contains assume_p primitive

        # Vectorize over different means
        batch_sample = modular_vmap(sample_normal, in_axes=(0,))
        mus = jnp.array([0.0, 1.0, 2.0])
        samples = batch_sample(mus)  # Shape: (3,), independent samples

        # Compare with seed for JAX compatibility
        seeded_fn = seed(batch_sample)
        samples = seeded_fn(key, mus)  # Can be jit'd, grad'd, etc.
        ```

    Note:
        For JAX transformations (jit, grad, etc.), use `seed()` first:
        `jax.jit(seed(modular_vmap(f)))` rather than trying to jit
        the modular_vmap directly.
    """

    @wraps(f)
    def wrapped(*args):
        # Quickly throw if "normal" vmap would fail.
        jax.vmap(
            lambda *_: None,
            in_axes=in_axes,
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )(*args)

        interpreter = ModularVmapInterpreter()
        return interpreter.run_interpreter(
            in_axes,
            axis_size,
            axis_name,
            spmd_axis_name,
            f,
            *args,
        )

    return wrapped


####################################
# Configuration and Error Handling  #
####################################

# Global flags that control the behavior when PJAX primitives reach MLIR compilation
# This happens when probabilistic functions are passed to JAX transformations
# without first applying the `seed()` transformation.

enforce_lowering_exception = True
"""Whether to raise exceptions when assume_p primitives reach MLIR lowering.

When True, attempting to compile probabilistic functions (e.g., with jax.jit)
without first applying `seed()` will raise a LoweringSamplePrimitiveToMLIRException.
This prevents silent errors where PRNG keys get baked into compiled code.

Set to False for debugging or if you want warnings instead of exceptions.
"""

lowering_warning = False
"""Whether to show warnings when assume_p primitives reach MLIR lowering.

When True, shows warning messages instead of raising exceptions when
probabilistic primitives reach compilation without proper transformation.
Generally, exceptions (enforce_lowering_exception=True) are preferred
as they prevent subtle bugs.
"""
