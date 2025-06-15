import itertools as it
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import overload

import beartype.typing as btyping
import jax
import jax.core as jc
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import jaxtyping as jtyping
import penzai.pz as pz
from jax import api_util, tree_util
from jax._src.interpreters.batching import AxisData  # pyright: ignore
from jax.core import eval_jaxpr
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Primitive, Var
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe
from jax.lax import cond_p, scan, scan_p, switch
from jax.util import safe_map, split_list
from numpy import dtype
from tensorflow_probability.substrates import jax as tfp
from typing_extensions import dataclass_transform

tfd = tfp.distributions

##########
# Typing #
##########

Any = btyping.Any
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

        ```{python}
        from genjax import Pytree
        from jaxtyping import ArrayLike
        import jax.numpy as jnp


        @Pytree.dataclass
        # Enforces type annotations on instantiation.
        class MyClass(Pytree):
            my_static_field: int = Pytree.static()
            my_dynamic_field: ArrayLike


        MyClass(10, jnp.array(5.0))
        ```
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

        ```{python}
        @Pytree.dataclass
        # Enforces type annotations on instantiation.
        class MyClass(Pytree):
            my_dynamic_field: ArrayLike
            my_static_field: int = Pytree.static(default=0)


        MyClass(jnp.array(5.0))
        ```

        """
        return field(metadata={"pytree_node": False}, **kwargs)

    @staticmethod
    def field(**kwargs):
        """Declare a field of a `Pytree` dataclass to be dynamic.
        Alternatively, one can leave the annotation off in the declaration."""
        return field(**kwargs)


############################################
# Staging utilities for Jaxpr interpreters #
############################################


def get_shaped_aval(x):
    return jc.get_aval(x)


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


def stage(f, **params):
    """Returns a function that stages a function to a ClosedJaxpr."""

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
    """Contains default implementations of transformations."""

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


class ElaboratedPrimitive(Primitive):
    """
    An `ElaboratedPrimitive` is a primitive which wraps an underlying
    primitive, but is elaborated with additional parameters that only show up
    in the pretty printing of a `Jaxpr`.
    In addition, `ElaboratedPrimitive` instances hide excess metadata in
    pretty printing.
    """

    def __init__(self, prim: InitialStylePrimitive, **params):
        super(ElaboratedPrimitive, self).__init__(prim.name)
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
        return (v.prim, v.params) if isinstance(v, ElaboratedPrimitive) else (v, {})

    @staticmethod
    def check(primitive, other):
        if isinstance(primitive, ElaboratedPrimitive):
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
            return ElaboratedPrimitive(primitive, **inner_params).bind(*args, **params)
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
            """Runs a function and binds it to an `ElaboratedPrimitive`
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

            elaborated_prim = ElaboratedPrimitive(
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


########
# PJAX #
########


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


######################
# Sampling primitive #
######################


class style:
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


assume_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.assume{style.RESET}",
)
log_density_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.log_density{style.RESET}",
)


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
                )(*vector_args)
                return (v,), (0 if n or axis_size else None,)
            else:
                raise NotImplementedError()

        lowering_msg = (
            "JAX is attempting to lower the `pjax.assume_p` primitive to MLIR. "
            "This will bake a PRNG key into the MLIR code, resulting in deterministic behavior. "
            "Instead, use `genjax.seed` to transform your function into one which allows keys to be passed in. "
            "You can do this at any level of your computation above the `assume_p` invocation."
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
    def _(v, *args, **kwargs):
        return log_density_binder(
            logpdf,
            name=name,
        )(v, *args, **kwargs)

    return _


VarOrLiteral = Var | Literal


@Pytree.dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during interpretation."""

    env: dict[int, Any] = Pytree.field(default_factory=dict)

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
            primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)

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
    @wraps(f)
    def wrapped(key: PRNGKey, *args):
        interpreter = SeedInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped


############################################
# Interpreter which modularly extends vmap #
############################################


@dataclass
class ModularVmapInterpreter:
    """
    `ModularVmapInterpreter` is an interpreter which piggybacks off of
    `jax.vmap`,
    but is aware of probability distributions as first class citizens.
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
            if ElaboratedPrimitive.check(eqn.primitive, assume_p):
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


########
# APIs #
########


def modular_vmap(
    f: Callable[..., R],
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
) -> Callable[..., R]:
    """
    `modular_vmap` extends `jax.vmap` to be "aware of probability distributions"
    as a first class citizen.
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


# Control the behavior of lowering `assume_p`
# to MLIR.
# `assume_p` should be _eliminated_ by `seed`,
# otherwise, keys are baked in, yielding deterministic results.
# These flags send warnings / raise exceptions in case this happens.
enforce_lowering_exception = True
lowering_warning = True

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
    def get_args(self) -> Any:
        pass

    @abstractmethod
    def get_retval(self) -> R:
        pass

    @abstractmethod
    def get_score(self) -> Score:
        pass

    def update(self, x: X, args=None):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(self.get_args() if args is None else args, self, x)

    def __getitem__(self, addr):
        choices = self.get_choices()
        return get_choices(choices[addr])  # pyright: ignore


@Pytree.dataclass
class Tr(Trace[X, R], Pytree):
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
    x = x.get_choices() if isinstance(x, Trace) else x

    def _get_choices(x):
        if isinstance(x, Trace):
            return get_choices(x)
        else:
            return x

    return jtu.tree_map(
        _get_choices,
        x,
        is_leaf=lambda x: isinstance(x, Trace),
    )


def get_score(x: Trace[X, R]) -> Weight:
    return x.get_score()


def get_retval(x: Trace[X, R]) -> R:
    return x.get_retval()


##############
# Selections #
##############


@Pytree.dataclass
class AllSel(Pytree):
    def match(self, addr) -> "tuple[bool, AllSel]":
        return True, self


@Pytree.dataclass
class NoneSel(Pytree):
    def match(self, addr) -> "tuple[bool, NoneSel]":
        return False, self


@Pytree.dataclass
class StrSel(Pytree):
    s: str = Pytree.static()

    def match(self, addr) -> tuple[bool, AllSel | NoneSel]:
        check = addr == self.s
        return check, AllSel() if check else NoneSel()


@Pytree.dataclass
class DictSel(Pytree):
    d: "dict[str, Any]"

    def match(self, addr) -> "tuple[bool, Any]":
        check = addr in self.d
        return check, self.d[addr] if check else NoneSel()


@Pytree.dataclass
class ComplSel(Pytree):
    s: "S"

    def match(self, addr) -> "tuple[bool, ComplSel]":
        check, rest = self.s.match(addr)
        return not check, ComplSel(rest)


@Pytree.dataclass
class InSel(Pytree):
    s1: "S"
    s2: "S"

    def match(self, addr) -> "tuple[bool, InSel]":
        check1, r1 = self.s1.match(addr)
        check2, r2 = self.s2.match(addr)
        return check1 and check2, InSel(r1, r2)


@Pytree.dataclass
class OrSel(Pytree):
    s1: "S"
    s2: "S"

    def match(self, addr) -> "tuple[bool, OrSel]":
        check1, r1 = self.s1.match(addr)
        check2, r2 = self.s2.match(addr)
        return check1 or check2, OrSel(r1, r2)


@Pytree.dataclass
class S(Pytree):
    s: NoneSel | AllSel | StrSel | DictSel | ComplSel | InSel | OrSel

    def match(self, addr) -> "tuple[bool, S]":
        check, rest = self.s.match(addr)
        return check, S(rest) if not isinstance(rest, S) else rest

    def __xor__(self, other: "S") -> "S":
        return S(InSel(self, other))

    def __invert__(self) -> "S":
        return S(ComplSel(self))

    def __contains__(self, addr) -> bool:
        check, _ = self.match(addr)
        return check

    def __or__(self, other) -> "S":
        return S(OrSel(self, other))

    def __call__(self, addr) -> "tuple[bool, S]":
        return self.match(addr)


def sel(*v: tuple[()] | str | dict[str, Any] | None) -> S:
    assert len(v) <= 1
    if len(v) == 1:
        if v[0] is None:
            return S(NoneSel())
        if v[0] == ():
            return S(AllSel())
        elif isinstance(v[0], dict):
            return S(DictSel(v[0]))
        else:
            assert isinstance(v[0], str)
            return S(StrSel(v[0]))
    else:
        return S(NoneSel())


def match(addr: Addr, sel: S):
    return sel.match(addr)


class GFI(Generic[X, R], Pytree):
    def __call__(self, *args) -> "Thunk[X, R] | R":
        if handler_stack:
            return Thunk(self, args)
        else:
            tr = self.simulate(args)
            assert isinstance(tr, Tr)
            return tr.get_retval()

    def T(self, *args) -> "Thunk[X, R]":
        return Thunk(self, args)

    @abstractmethod
    def simulate(
        self,
        args,
    ) -> Trace[X, R]:
        pass

    @abstractmethod
    def generate(
        self,
        args,
        x: X | None,
    ) -> tuple[Trace[X, R], Weight]:
        pass

    @abstractmethod
    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Density, R]:
        pass

    @abstractmethod
    def update(
        self,
        args_,
        tr: Trace[X, R],
        x_: X | None,
    ) -> tuple[Tr[X, R], Weight, X | None]:
        pass

    @abstractmethod
    def regenerate(
        self,
        args_,
        tr: Trace[X, R],
        sel: S,
    ) -> tuple[Tr[X, R], Weight, X | None]:
        pass

    @abstractmethod
    def merge(self, x: X, x_: X) -> X:
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
            in_axes,
            axis_size,
            axis_name,
            spmd_axis_name,
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
        args: tuple[Any, ...],
        x: X,
    ) -> Score:
        logp, _ = self.assess(args, x)
        return logp


########################
# Generative functions #
########################


@Pytree.dataclass
class Thunk(Generic[X, R], Pytree):
    gen_fn: GFI[X, R]
    args: tuple

    def __matmul__(self, other: str):
        return trace(other, self.gen_fn, self.args)


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
        >>> trace = vectorized_normal.simulate((mus, sigma))
        >>> samples = trace.get_choices()  # Array of 3 normal samples
    """

    gen_fn: GFI[X, R]
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = Pytree.static()
    axis_size: int | None = Pytree.static()
    axis_name: str | None = Pytree.static()
    spmd_axis_name: str | None = Pytree.static()

    def simulate(
        self,
        args,
    ) -> Tr[X, R]:
        return modular_vmap(
            self.gen_fn.simulate,
            in_axes=(self.in_axes,),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args)

    def generate(
        self,
        args,
        x: X,
    ) -> tuple[Tr[X, R], Weight]:
        tr, w = modular_vmap(
            self.gen_fn.generate,
            in_axes=(self.in_axes, 0),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args, x)
        return tr, jnp.sum(w)

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Density, R]:
        return modular_vmap(
            self.gen_fn.assess,
            in_axes=(self.in_axes, 0),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args, x)

    def update(
        self,
        args_,
        tr: Tr[X, R],
        x_: X,
    ) -> tuple[Tr[X, R], Weight, X | None]:
        new_tr, w, discard = modular_vmap(
            self.gen_fn.update,
            in_axes=(self.in_axes, 0, 0),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args_, tr, x_)
        return new_tr, jnp.sum(w), discard

    def regenerate(
        self,
        args_,
        tr: Tr[X, R],
        s: S,
    ) -> tuple[Tr[X, R], Weight, X | None]:
        new_tr, w, discard = modular_vmap(
            self.gen_fn.update,
            in_axes=(self.in_axes, 0, None),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args_, tr, s)
        return new_tr, jnp.sum(w), discard

    def merge(self, x: X, x_: X) -> X:
        return modular_vmap(self.merge, in_axes=(0, 0))(x, x_)


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
        >>> import jax.numpy as jnp
        >>> from genjax import Distribution
        >>>
        >>> # Create a custom normal distribution
        >>> def sample_normal(mu, sigma):
        ...     key = jax.random.PRNGKey(0)  # In practice, use proper key management
        ...     return mu + sigma * jax.random.normal(key)
        >>>
        >>> def logpdf_normal(x, mu, sigma):
        ...     return -0.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)
        >>>
        >>> normal = Distribution(sample_normal, logpdf_normal, name="normal")
        >>> trace = normal.simulate((0.0, 1.0))  # mu=0.0, sigma=1.0
    """

    sample: Callable[..., X] = Pytree.static()
    logpdf: Callable[..., Weight] = Pytree.static()
    name: str | None = Pytree.static(default=None)

    def simulate(
        self,
        args,
    ) -> Tr[X, X]:
        x = self.sample(*args)
        log_density = self.logpdf(x, *args)
        return Tr(self, args, x, x, -log_density)

    def generate(
        self,
        args,
        x: X | None,
    ) -> tuple[Tr[X, X], Weight]:
        if x is None:
            tr = self.simulate(args)
            return tr, jnp.array(0.0)
        else:
            logp, r = self.assess(args, x)
            return Tr(self, args, x, x, -logp), logp

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Density, X]:
        logp = self.logpdf(x, *args)
        return logp, x

    def update(
        self,
        args_,
        tr: Tr[X, X],
        x_: X | None,
    ) -> tuple[Tr[X, X], Weight, X | None]:
        if x_ is None:
            x_ = get_choices(tr)
            log_density_ = self.logpdf(x_, *args_)
            return (
                Tr(self, args_, x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                tr.get_retval(),
            )
        else:
            log_density_ = self.logpdf(x_, *args_)
            return (
                Tr(self, args_, x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                tr.get_retval(),
            )

    def regenerate(
        self,
        args_,
        tr: Tr[X, X],
        s: S,
    ) -> tuple[Tr[X, X], Weight, X | None]:
        if () in s:
            tr_ = self.simulate(args_)
            w = -tr_.get_score() + tr.get_score()
            return tr_, w, get_choices(tr)
        else:
            x_ = get_choices(tr)
            log_density_ = self.logpdf(get_choices(tr), *args_)
            return (
                Tr(self, args_, x_, x_, -log_density_),
                log_density_ + tr.get_score(),
                None,
            )

    def merge(self, x: X, x_: X) -> X:
        raise Exception(
            "Can't merge: the underlying sample space `X` for the type `Distribution` doesn't support merging."
        )


def distribution[X](
    sampler: Callable[..., X],
    logpdf: Callable[..., Any],
    /,
    name: str | None = None,
) -> Distribution[Any]:
    return Distribution(
        sampler,
        logpdf,
        name,
    )


# Mostly, just use TFP.
# This wraps PJAX's `assume_p` correctly.
def tfp_distribution[X](
    dist: Callable[..., "tfd.Distribution"],
    /,
    name: str | None = None,
) -> Distribution[Any]:
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


@dataclass
class Simulate:
    score: Weight
    trace_map: dict[str, Any]

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
    ) -> R:
        tr = gen_fn.simulate(args)
        self.score += tr.get_score()
        self.trace_map[addr] = tr
        return tr.get_retval()


@dataclass
class Generate:
    choice_map: dict[str, Any]
    score: Weight
    weight: Weight
    trace_map: dict[str, Any]

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
    ) -> R:
        x = (
            get_choices(
                self.choice_map[addr],
            )
            if addr in self.choice_map
            else None
        )
        tr, weight = gen_fn.generate(args, x)
        self.weight += weight
        self.score += tr.get_score()
        self.trace_map[addr] = tr
        return tr.get_retval()


@dataclass
class Assess:
    choice_map: dict[str, Any]
    logp: Density

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
    ) -> R:
        x = self.choice_map[addr]
        x = get_choices(x)
        logp, r = gen_fn.assess(args, x)
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

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args_,
    ) -> R:
        subtrace = self.trace.get_choices()[addr]
        x = (
            self.choice_map[addr]
            if addr in self.choice_map
            else self.trace.get_choices()[addr]
        )
        x = get_choices(x)
        tr, w, discard = gen_fn.update(args_, subtrace, x)
        self.trace_map[addr] = tr
        self.discard[addr] = discard
        self.score += tr.get_score()
        self.weight += w
        return tr.get_retval()


@dataclass
class Regenerate(Generic[R]):
    trace: Tr[dict[str, Any], R]
    s: S
    trace_map: dict[str, Any]
    discard: dict[str, Any]
    score: Score
    weight: Weight

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args_,
    ) -> R:
        subtrace = self.trace.get_choices()[addr]
        subs = self.s[addr]
        tr, w, discard = gen_fn.regenerate(args_, subtrace, subs)
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
) -> R:
    handler = handler_stack[-1]
    retval = handler(addr, gen_fn, args)
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
        >>> from genjax import gen, normal
        >>>
        >>> @gen
        >>> def linear_regression(xs):
        ...     slope = normal(0.0, 1.0) @ "slope"
        ...     intercept = normal(0.0, 1.0) @ "intercept"
        ...     noise = normal(0.0, 0.1) @ "noise"
        ...     return normal(slope * xs + intercept, noise) @ "y"
        >>>
        >>> trace = linear_regression.simulate((jnp.array([1.0, 2.0, 3.0]),))
        >>> choices = trace.get_choices()  # dict with keys "slope", "intercept", "noise", "y"
    """

    source: Callable[..., R] = Pytree.static()

    def simulate(
        self,
        args,
    ) -> Tr[dict[str, Any], R]:
        handler_stack.append(Simulate(jnp.array(0.0), {}))
        r = self.source(*args)
        handler = handler_stack.pop()
        assert isinstance(handler, Simulate)
        score, trace_map = handler.score, handler.trace_map
        return Tr(self, args, trace_map, r, score)

    def generate(
        self,
        args,
        x: dict[str, Any] | None,
    ) -> tuple[Tr[dict[str, Any], R], Weight]:
        if x is None:
            tr = self.simulate(args)
            return tr, jnp.array(0.0)
        else:
            handler_stack.append(Generate(x, jnp.array(0.0), jnp.array(0.0), {}))
            r = self.source(*args)
            handler = handler_stack.pop()
            assert isinstance(handler, Generate)
            score, weight, trace_map = handler.score, handler.weight, handler.trace_map
            return Tr(self, args, trace_map, r, score), weight

    def assess(
        self,
        args,
        x: dict[str, Any],
    ) -> tuple[Density, R]:
        handler_stack.append(Assess(x, jnp.array(0.0)))
        r = self.source(*args)
        handler = handler_stack.pop()
        assert isinstance(handler, Assess)
        logp = handler.logp
        return logp, r

    def update(
        self,
        args_,
        tr: Tr[dict[str, Any], R],
        x_: dict[str, Any] | None,
    ) -> tuple[Tr[dict[str, Any], R], Weight, dict[str, Any] | None]:
        x_ = {} if x_ is None else x_
        handler_stack.append(Update(tr, x_, {}, {}, jnp.array(0.0), jnp.array(0.0)))
        r = self.source(*args_)
        handler = handler_stack.pop()
        assert isinstance(handler, Update)
        trace_map, score, w, discard = (
            handler.trace_map,
            handler.score,
            handler.weight,
            handler.discard,
        )
        return Tr(self, args_, trace_map, r, score), w, discard

    def regenerate(
        self,
        args_,
        tr: Tr[dict[str, Any], R],
        s: S,
    ) -> tuple[Tr[dict[str, Any], R], Weight, dict[str, Any] | None]:
        handler_stack.append(Regenerate(tr, s, {}, {}, jnp.array(0.0), jnp.array(0.0)))
        r = self.source(*args_)
        handler = handler_stack.pop()
        assert isinstance(handler, Update)
        trace_map, score, w, discard = (
            handler.trace_map,
            handler.score,
            handler.weight,
            handler.discard,
        )
        return Tr(self, args_, trace_map, r, score), w, discard

    def merge(
        self,
        x: dict[str, Any],
        x_: dict[str, Any],
    ) -> dict[str, Any]:
        return x | x_


def gen(fn: Callable[..., R]) -> Fn[R]:
    return Fn(source=fn)


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
        >>> from genjax import gen, normal, Scan
        >>>
        >>> @gen
        >>> def step(carry, x):
        ...     noise = normal(0.0, 0.1) @ "noise"
        ...     new_carry = carry + x + noise
        ...     return new_carry, new_carry  # output equals new carry
        >>>
        >>> scan_fn = Scan(step)
        >>> init_carry = 0.0
        >>> xs = jnp.array([1.0, 2.0, 3.0])
        >>> trace = scan_fn.simulate((init_carry, xs))
        >>> final_carry, outputs = trace.get_retval()
    """

    callee: GFI[X, R]
    length: int = Pytree.static()

    def merge(self, x: X, x_: X):
        return self.callee.merge(x, x_)

    def simulate(
        self,
        args,
    ) -> ScanTr[X, R]:
        init_carry, xs = args

        def scan_fn(carry, x):
            trace = self.callee.simulate((carry, x))
            new_carry = trace.get_retval()[0]  # (C, Out) -> C
            out = trace.get_retval()[1]  # (C, Out) -> Out
            return new_carry, (trace, out)

        final_carry, (traces, outs) = scan(
            scan_fn,
            init_carry,
            xs,
            length=self.length,
        )

        return ScanTr(self, args, traces, final_carry, outs)

    def generate(
        self,
        args,
        x: X | None,
    ) -> tuple[ScanTr[X, R], Weight]:
        carry_args, scanned_args = args

        def scan_fn(carry, scanned):
            (scanned_args, x) = scanned
            trace, weight = self.callee.generate((carry, scanned_args), x)
            new_carry = trace.get_retval()[0]
            scanned_out = trace.get_retval()[1]
            return new_carry, (trace, scanned_out, weight)

        final_carry, (traces, scanned_outs, weights) = scan(
            scan_fn,
            carry_args,
            (scanned_args, x),
            length=self.length,
        )

        total_weight = jnp.sum(weights)

        return ScanTr(self, args, traces, final_carry, scanned_outs), total_weight

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Density, R]:
        carry_args, scanned_args = args

        def scan_fn(carry, scanned):
            (scanned_args, x) = scanned
            density, (new_carry, scanned_out) = self.callee.assess(
                (carry, scanned_args), x
            )
            return new_carry, (scanned_out, density)

        final_carry, (scanned_outs, density) = scan(
            scan_fn,
            carry_args,
            (scanned_args, x),
            length=self.length,
        )
        return jnp.sum(density), (final_carry, scanned_outs)

    def update(
        self,
        args_,
        tr: ScanTr[X, R],
        x_: X,
    ) -> tuple[ScanTr[X, R], Weight, X | None]:
        carry_args, scanned_args = args_
        old_traces = tr.traces

        def scan_fn(carry, scanned_in):
            input_x, old_trace, new_choice = scanned_in
            new_trace, weight, discard = self.callee.update(
                (carry, input_x), old_trace, new_choice
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
            length=self.length,
        )

        total_weight = jnp.sum(weights)

        new_tr = ScanTr(self, args_, new_traces, final_carry, outs)
        return new_tr, total_weight, discards

    def regenerate(
        self,
        args_,
        tr: ScanTr[X, R],
        s: S,
    ) -> tuple[ScanTr[X, R], Weight, X | None]:
        init_carry, xs = args_
        old_traces = tr.traces

        def scan_fn(carry, xs_old_trace):
            input_x, old_trace = xs_old_trace
            new_trace, weight, discard = self.callee.regenerate(
                (carry, input_x),
                old_trace,
                s,  # selection applies to all steps
            )
            new_carry = new_trace.get_retval()[0]
            out = new_trace.get_retval()[1]
            return new_carry, (new_trace, weight, discard, out)

        # Pack xs and old_traces together for scan
        scan_inputs = (xs, old_traces)

        final_carry, (new_traces, weights, discards, outs) = scan(
            scan_fn, init_carry, scan_inputs, length=self.length
        )

        total_weight = jnp.sum(weights)
        # discards will be vectorized, so we need to handle them appropriately
        any_discards = jnp.any(jtu.tree_map(lambda x: x is not None, discards))

        new_tr = ScanTr(self, args_, new_traces, final_carry, outs)
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
        callee = self.gen_fn.callee
        chm, chm_ = map(get_choices, self.trs)
        chm = jtu.tree_map(
            lambda v: jnp.where(self.check, v, jnp.nan),
            chm,
        )
        chm_ = jtu.tree_map(
            lambda v: jnp.where(self.check, jnp.nan, v),
            chm_,
        )
        return callee.merge(chm, chm_)

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

    def merge(self, x: X, x_: X):
        return self.callee.merge(x, x_)

    def simulate(
        self,
        args,
    ) -> CondTr[X, R]:
        (check, *args) = args
        tr = self.callee.simulate(args)
        tr_ = self.callee_.simulate(args)
        return CondTr(self, check, [tr, tr_])

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Density, R]:
        (check, *args) = args
        logp, r = self.callee.assess(args, x)
        logp_, r_ = self.callee_.assess(args, x)
        total_density = jnp.where(check, logp, logp_)
        retval = jnp.where(check, r, r_)
        return total_density, retval

    def generate(
        self,
        args,
        x: X | None,
    ) -> tuple[CondTr[X, R], Weight]:
        (check, *args) = args
        if x is None:
            tr = self.callee.simulate(args)
            tr_ = self.callee_.simulate(args)
            return CondTr(self, check, [tr, tr_]), jnp.array(0.0)
        else:
            tr, w = self.callee.generate(args, x)
            tr_, w_ = self.callee_.generate(args, x)
            total_weight = jnp.where(check, w, w_)
            return CondTr(self, check, [tr, tr_]), total_weight

    def update(
        self,
        args,
        tr: CondTr[X, R],
        x: X,
    ) -> tuple[Trace[X, R], Weight, X]:
        (check, *args) = args
        tr, w, discard = self.callee.update(args, tr.trs[0], x)
        tr_, w_, discard_ = self.callee_.update(args, tr.trs[1], x)
        return (
            CondTr(self, check, [tr, tr_]),
            jnp.select(check, [w, w_]),
            self.callee.merge(discard, discard_),
        )

    def regenerate(
        self,
        args,
        tr: CondTr[X, R],
        s: S,
    ) -> tuple[Trace[X, R], Weight, X | None]:
        (check, *args) = args
        tr, w, discard = self.callee.regenerate(args, tr.trs[0], s)
        tr_, w_, discard_ = self.callee_.regenerate(args, tr.trs[1], s)
        discard = (
            discard
            if discard_ is None
            else discard_
            if discard is None
            else self.callee.merge(discard, discard_)
        )
        return (
            CondTr(self, check, [tr, tr_]),
            jnp.where(check, [w, w_]),
            discard,
        )
