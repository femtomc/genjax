"""ADEV: Sound Automatic Differentiation of Expected Values

This module implements ADEV (Automatic Differentiation of Expectation Values), a system
for computing sound, unbiased gradient estimators of expectations involving stochastic
functions. Based on the research presented in "ADEV: Sound Automatic Differentiation of
Expected Values of Probabilistic Programs" (Lew et al., POPL 2023, arXiv:2212.06386).

ADEV is a source-to-source transformation that extends forward-mode automatic
differentiation to correctly handle probabilistic computations. The key insight is
transforming a probabilistic program into a new program whose expected return value
is the derivative of the original program's expectation.

Theoetical Foundation:
ADEV uses a continuation-passing style (CPS) transformation that reflects the law of
iterated expectation: E[f(X)] = E[E[f(X) | Z]] where X depends on Z. This enables
modular composition of different gradient estimation strategies while maintaining
soundness guarantees proven via logical relations.

Key Concepts:
    - ADEVPrimitive: Stochastic primitives with custom gradient estimation strategies
    - Dual Numbers: Pairs (primal, tangent) for forward-mode automatic differentiation
    - Continuations: Higher-order functions representing "the rest of the computation"
        - Pure continuation: Operates on primal values only (no differentiation)
        - Dual continuation: Operates on dual numbers (differentiation applied)
    - CPS Transformation: Allows modular selection of gradient strategies per distribution

Gradient Estimation Strategies:
    - REINFORCE: Score function estimator ∇E[f(X)] = E[f(X) * ∇log p(X)]
    - Reparameterization: Pathwise estimator ∇E[f(g(ε))] = E[∇f(g(ε)) * ∇g(ε)]
    - Enumeration: Exact computation for finite discrete distributions
    - Measure-Valued Derivatives: Advanced discrete gradient estimators

Example:
    ```python
    from genjax.adev import expectation, normal_reparam

    @expectation
    def objective(theta):
        x = normal_reparam(theta, 1.0)  # Reparameterizable distribution
        return x**2

    grad = objective.grad_estimate(0.5)  # Unbiased gradient estimate
    ```

References:
    Lew, A. K., Huot, M., Staton, S., & Mansinghka, V. K. (2023). ADEV: Sound
    Automatic Differentiation of Expected Values of Probabilistic Programs.
    Proceedings of the ACM on Programming Languages, 7(POPL), 121-148.
"""

import itertools as it
from abc import abstractmethod
from functools import wraps

import jax
import jax._src.core
import jax.dtypes
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype.vale import Is
from genjax.core import Pytree, Const, const, distribution, Any, Callable, Annotated
from genjax.pjax import (
    PPPrimitive,
    Environment,
    sample_binder,
    sample_p,
    modular_vmap,
    stage,
)
from jax import util as jax_util
from jax.extend import source_info_util as src_util
from jax.extend.core import Jaxpr, Var, jaxpr_as_fun
from jax.interpreters import ad as jax_autodiff
from jaxtyping import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from genjax.distributions import (
    bernoulli,
    categorical,
    geometric,
    normal,
    multivariate_normal,
)

tfd = tfp.distributions

DualTree = Annotated[
    Any,
    Is[lambda v: Dual.static_check_dual_tree(v)],
]
"""
`DualTree` is the type of `Pytree` argument values with `Dual` leaves.
"""

###################
# ADEV primitives #
###################


class ADEVPrimitive(Pytree):
    """Base class for stochastic primitives with custom gradient estimation strategies.

    An ADEVPrimitive represents a stochastic operation (like sampling from a distribution)
    that can provide custom gradient estimates through the ADEV system. Each primitive
    implements both forward sampling and a strategy for computing Jacobian-Vector Product
    (JVP) estimates during automatic differentiation.

    The key insight is that different stochastic operations benefit from different
    gradient estimation strategies (REINFORCE, reparameterization, enumeration, etc.),
    and ADEVPrimitive allows each operation to specify its optimal strategy.

    Methods:
        sample: Forward sampling operation
        prim_jvp_estimate: Custom gradient estimation strategy
        __call__: Convenience method that wraps sample with ADEV infrastructure
    """

    @abstractmethod
    def sample(self, *args) -> Any:
        """Forward sampling operation.

        Args:
            *args: Parameters for the stochastic operation

        Returns:
            Sample from the distribution/stochastic process
        """
        pass

    @abstractmethod
    def prim_jvp_estimate(
        self,
        dual_tree: tuple[DualTree, ...],
        konts: tuple[
            Callable[..., Any],  # Pure continuation (kpure)
            Callable[..., Any],  # Dual continuation (kdual)
        ],
    ) -> "Dual":
        """Custom JVP gradient estimation strategy.

        This method implements the core gradient estimation logic for this primitive.
        It receives dual numbers (primal + tangent values) for the arguments and
        two continuations representing the rest of the computation.

        Args:
            dual_tree: Arguments as dual numbers (primal, tangent) pairs
            konts: Pair of continuations:
                - konts[0] (kpure): Pure continuation - no ADEV transformation
                - konts[1] (kdual): Dual continuation - ADEV transformation applied

        Returns:
            Dual number representing the gradient estimate for this operation

        Note:
            The choice of continuation reflects ADEV's CPS transformation approach:
            - Pure continuation: Evaluates the remaining computation on primal values
            - Dual continuation: Applies ADEV transformation to remaining computation

            Different gradient strategies utilize these continuations differently:
            - REINFORCE: Uses dual continuation to evaluate f(X), computes ∇log p(X)
            - Reparameterization: Uses dual continuation with reparameterized samples
            - Enumeration: May use both to compute weighted exact expectations

            This design enables modular composition as described in the ADEV paper.
        """
        pass

    def __call__(self, *args):
        """Convenience method for sampling with ADEV infrastructure."""
        return sample_primitive(self, *args)


####################
# Sample intrinsic #
####################


def sample_primitive(adev_prim: ADEVPrimitive, *args):
    """Integrate an ADEV primitive with the PJAX infrastructure.

    This function wraps an ADEVPrimitive so it can be used within GenJAX's
    probabilistic programming system. It ensures the primitive works correctly
    with JAX transformations (jit, vmap, grad) and addressing (@) operators.

    The key insight is that ADEV primitives need to be integrated with PJAX's
    sample_binder to get proper parameter setup (like flat_keyful_sampler) that
    enables compatibility with the seed transformation and other GenJAX features.

    Args:
        adev_prim: The ADEV primitive to integrate
        *args: Arguments to pass to the primitive's sample method

    Returns:
        Sample from the primitive, properly integrated with PJAX infrastructure

    Note:
        This function was crucial for fixing the flat_keyful_sampler error -
        previously ADEV primitives bypassed sample_binder and lacked proper
        parameter setup for JAX transformations.
    """

    def _adev_prim_call(key, adev_prim, *args, **kwargs):
        """Wrapper function that conforms to sample_binder's expected signature."""
        return adev_prim.sample(*args)

    return sample_binder(_adev_prim_call)(adev_prim, *args)


####################
# ADEV interpreter #
####################


@Pytree.dataclass
class Dual(Pytree):
    """Dual number for forward-mode automatic differentiation.

    A Dual number represents both a value (primal) and its derivative (tangent)
    with respect to some input. This is the fundamental data structure for
    forward-mode AD in the ADEV system.

    Attributes:
        primal: The actual value
        tangent: The derivative/gradient information

    Example:
        >>> x = Dual(3.0, 1.0)  # x = 3, dx/dx = 1
        >>> y = Dual(x.primal**2, 2*x.primal*x.tangent)  # y = x^2, dy/dx = 2x
    """

    primal: Any
    tangent: Any

    @staticmethod
    def tree_pure(v):
        """Convert a tree to have Dual leaves with zero tangents.

        This is used to "lift" regular values into the dual number system
        by pairing them with zero tangents, indicating no sensitivity.

        Args:
            v: Pytree that may contain mix of Dual and regular values

        Returns:
            Pytree where all leaves are Dual numbers
        """

        def _inner(v):
            if isinstance(v, Dual):
                return v
            else:
                return Dual(v, jnp.zeros_like(v))

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def dual_tree(primals, tangents):
        """Combine primal and tangent trees into a tree of Dual numbers.

        Args:
            primals: Tree of primal values
            tangents: Tree of tangent values (same structure as primals)

        Returns:
            Tree of Dual numbers combining corresponding primals and tangents
        """
        return jtu.tree_map(lambda v1, v2: Dual(v1, v2), primals, tangents)

    @staticmethod
    def tree_primal(v):
        """Extract primal values from a tree of Dual numbers.

        Args:
            v: Tree that may contain Dual numbers

        Returns:
            Tree with Dual numbers replaced by their primal values
        """

        def _inner(v):
            if isinstance(v, Dual):
                return v.primal
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_tangent(v):
        """Extract tangent values from a tree of Dual numbers.

        Args:
            v: Tree that may contain Dual numbers

        Returns:
            Tree with Dual numbers replaced by their tangent values
        """

        def _inner(v):
            if isinstance(v, Dual):
                return v.tangent
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_leaves(v):
        """Get leaves of a tree, treating Dual numbers as atomic.

        Args:
            v: Tree structure

        Returns:
            List of Dual leaves
        """
        v = Dual.tree_pure(v)
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_unzip(v):
        """Separate a tree of Dual numbers into primal and tangent trees.

        Args:
            v: Tree containing Dual numbers

        Returns:
            Tuple of (primal_leaves, tangent_leaves) as flat lists
        """
        primals = jtu.tree_leaves(Dual.tree_primal(v))
        tangents = jtu.tree_leaves(Dual.tree_tangent(v))
        return tuple(primals), tuple(tangents)

    @staticmethod
    def static_check_is_dual(v) -> bool:
        """Check if a value is a Dual number."""
        return isinstance(v, Dual)

    @staticmethod
    def static_check_dual_tree(v) -> bool:
        """Check if all leaves in a tree are Dual numbers."""
        return all(
            map(
                lambda v: isinstance(v, Dual),
                jtu.tree_leaves(v, is_leaf=Dual.static_check_is_dual),
            )
        )


@Pytree.dataclass
class ADEV(Pytree):
    """Interpreter for ADEV's continuation-passing style automatic differentiation.

    The ADEV interpreter processes JAX computation graphs (Jaxpr) and transforms them
    to support stochastic automatic differentiation. It implements a continuation-passing
    style (CPS) transformation that reflects the law of iterated expectation, allowing
    different gradient estimation strategies for each stochastic operation.

    Key responsibilities:
    1. Propagate dual numbers through deterministic JAX operations
    2. Apply CPS transformation at stochastic operations (sample_p primitives)
    3. Create continuation closures for gradient estimation strategies
    4. Handle control flow (conditionals, loops) within the AD system

    The CPS transformation is crucial: when encountering a stochastic operation,
    the interpreter creates two continuations representing the rest of the computation:
    - Pure continuation: For sampling-based gradient estimates
    - Dual continuation: For the ADEV-transformed remainder

    This allows each ADEVPrimitive to choose its optimal gradient strategy while
    maintaining composability across the entire computation graph.
    """

    @staticmethod
    def flat_unzip(duals: list[Any]):
        primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in duals)
        return list(primals), list(tangents)

    @staticmethod
    def eval_jaxpr_adev(
        jaxpr: Jaxpr,
        consts: list[ArrayLike],
        flat_duals: list[Dual],
    ):
        dual_env = Environment()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, Dual.tree_pure(consts))
        jax_util.safe_map(dual_env.write, jaxpr.invars, flat_duals)

        # TODO: Pure evaluation.
        def eval_jaxpr_iterate_pure(eqns, pure_env, invars, flat_args):
            jax_util.safe_map(pure_env.write, invars, flat_args)
            for eqn in eqns:
                in_vals = jax_util.safe_map(pure_env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals
                if eqn.primitive is sample_p:
                    pass
                else:
                    outs = eqn.primitive.bind(*args, **params)
                    if not eqn.primitive.multiple_results:
                        outs = [outs]
                    jax_util.safe_map(pure_env.write, eqn.outvars, outs)

            return jax_util.safe_map(pure_env.read, jaxpr.outvars)

        # Dual evaluation.
        def eval_jaxpr_iterate_dual(
            eqns,
            dual_env: Environment,
            invars: list[Var],
            flat_duals: list[Dual],
        ):
            jax_util.safe_map(dual_env.write, invars, flat_duals)

            for eqn_idx, eqn in enumerate(eqns):
                with src_util.user_context(eqn.source_info.traceback):
                    in_vals = jax_util.safe_map(dual_env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    duals = subfuns + in_vals

                    primitive, inner_params = PPPrimitive.unwrap(eqn.primitive)
                    # Our sample_p primitive.
                    if primitive is sample_p:
                        dual_env = dual_env.copy()
                        pure_env = Dual.tree_primal(dual_env)

                        # Create pure continuation (kpure): represents E[f(X) | X=x]
                        # Operates on primal values only, no differentiation applied
                        def _sample_pure_kont(*args):
                            return eval_jaxpr_iterate_pure(
                                eqns[eqn_idx + 1 :],
                                pure_env,
                                eqn.outvars,
                                [*args],
                            )

                        # Create dual continuation (kdual): represents ∇E[f(X) | X=x]
                        # Applies ADEV transformation to the remaining computation
                        def _sample_dual_kont(*duals: Dual):
                            return eval_jaxpr_iterate_dual(
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                list(duals),
                            )

                        in_tree = inner_params["in_tree"]
                        num_consts = inner_params["num_consts"]

                        flat_primals, flat_tangents = ADEV.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals[num_consts:]))
                        )
                        adev_prim, *primals = jtu.tree_unflatten(in_tree, flat_primals)
                        _, *tangents = jtu.tree_unflatten(in_tree, flat_tangents)
                        dual_tree = Dual.dual_tree(primals, tangents)

                        return adev_prim.prim_jvp_estimate(
                            tuple(dual_tree),
                            (_sample_pure_kont, _sample_dual_kont),
                        )

                    # Handle branching.
                    elif eqn.primitive is jax.lax.cond_p:
                        pure_env = Dual.tree_primal(dual_env)

                        # Create dual continuation for the computation after the cond_p.
                        def _cond_dual_kont(dual_tree: list[Any]):
                            dual_leaves = Dual.tree_pure(dual_tree)
                            return eval_jaxpr_iterate_dual(
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                dual_leaves,
                            )

                        branch_adev_functions = list(
                            map(
                                lambda fn: ADEV.forward_mode(
                                    jaxpr_as_fun(fn),
                                    _cond_dual_kont,
                                ),
                                params["branches"],
                            )
                        )

                        # NOTE: JAX stores conditional branches in reverse order in the params.
                        # We reverse them here to match the expected order for jax.lax.cond.
                        # This is a JAX implementation detail that may change in future versions.
                        return jax.lax.cond(
                            Dual.tree_primal(in_vals[0]),
                            *it.chain(reversed(branch_adev_functions), in_vals[1:]),
                        )

                    # Default JVP rule for other JAX primitives.
                    else:
                        flat_primals, flat_tangents = ADEV.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals))
                        )
                        if len(flat_primals) == 0:
                            primal_outs = eqn.primitive.bind(*flat_primals, **params)
                            tangent_outs = jtu.tree_map(jnp.zeros_like, primal_outs)
                        else:
                            jvp = jax_autodiff.primitive_jvps.get(eqn.primitive)
                            if not jvp:
                                msg = f"differentiation rule for '{eqn.primitive}' not implemented"
                                raise NotImplementedError(msg)
                            primal_outs, tangent_outs = jvp(
                                flat_primals, flat_tangents, **params
                            )

                if not eqn.primitive.multiple_results:
                    primal_outs = [primal_outs]
                    tangent_outs = [tangent_outs]

                jax_util.safe_map(
                    dual_env.write,
                    eqn.outvars,
                    Dual.dual_tree(primal_outs, tangent_outs),
                )
            (out_dual,) = jax_util.safe_map(dual_env.read, jaxpr.outvars)
            if not isinstance(out_dual, Dual):
                out_dual = Dual(out_dual, jnp.zeros_like(out_dual))
            return out_dual

        return eval_jaxpr_iterate_dual(jaxpr.eqns, dual_env, jaxpr.invars, flat_duals)

    @staticmethod
    def forward_mode(f, kont=lambda v: v):
        def _inner(*duals: DualTree):
            primals = Dual.tree_primal(duals)
            closed_jaxpr, (_, _, out_tree) = stage(f)(*primals)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            dual_leaves = Dual.tree_leaves(Dual.tree_pure(duals))
            out_duals = ADEV.eval_jaxpr_adev(
                jaxpr,
                consts,
                dual_leaves,
            )
            out_tree_def = out_tree()
            tree_primals, tree_tangents = Dual.tree_unzip(out_duals)
            out_dual_tree = Dual.dual_tree(
                jtu.tree_unflatten(out_tree_def, tree_primals),
                jtu.tree_unflatten(out_tree_def, tree_tangents),
            )
            vs = kont(out_dual_tree)
            return vs

        # Force coercion to JAX arrays.
        def maybe_array(v):
            return jnp.array(v, copy=False)

        def _dual(*duals: DualTree):
            duals = jtu.tree_map(maybe_array, duals)
            return _inner(*duals)

        return _dual


#################
# ADEV programs #
#################


@Pytree.dataclass
class ADEVProgram(Pytree):
    """Internal representation of a stochastic program for ADEV gradient estimation.

    An ADEVProgram wraps a source function containing stochastic operations and
    provides the infrastructure for computing Jacobian-Vector Product (JVP) estimates
    through the ADEV system. This class serves as an intermediate representation
    between user-defined @expectation functions and the low-level ADEV interpreter.

    The ADEVProgram handles the integration between:
    1. User source code containing ADEV primitives
    2. The ADEV interpreter's CPS transformation
    3. Continuation-based gradient estimation strategies

    Attributes:
        source: The original function containing stochastic operations

    Note:
        This class is typically not used directly by users. It's created internally
        by the @expectation decorator and managed by the Expectation class.
    """

    source: Const[Callable[..., Any]]

    def jvp_estimate(
        self,
        duals: tuple[DualTree, ...],  # Pytree with Dual leaves.
        dual_kont: Callable[..., Any],
    ) -> Dual:
        """Compute JVP estimate for the stochastic program.

        This method applies the ADEV forward-mode transformation to compute
        an unbiased estimate of the Jacobian-Vector Product for expectations
        involving stochastic operations. It uses the continuation-passing style
        transformation to integrate different gradient estimation strategies.

        Args:
            duals: Input arguments as dual numbers (primal, tangent) pairs
            dual_kont: Continuation representing the computation after this program

        Returns:
            Dual number containing the JVP estimate (primal value + gradient estimate)

        Note:
            This method coordinates between the user's source function and the
            ADEV interpreter to apply the appropriate gradient estimation strategies
            for each stochastic primitive encountered during execution.
        """

        def adev_jvp(f):
            @wraps(f)
            def wrapped(*duals: DualTree):
                return ADEV.forward_mode(self.source.value, dual_kont)(*duals)

            return wrapped

        return adev_jvp(self.source.value)(*duals)


###############
# Expectation #
###############


@Pytree.dataclass
class Expectation(Pytree):
    """Represents an expectation with automatic differentiation support.

    An Expectation object encapsulates a stochastic computation and provides methods
    to compute unbiased gradient estimates of expectation values. This is the primary
    user-facing interface for ADEV (Automatic Differentiation of Expectation Values).

    The key insight is that for expectations E[f(X)] where X is a random variable,
    we can compute unbiased gradient estimates ∇E[f(X)] using various strategies:
    - REINFORCE: ∇E[f(X)] = E[f(X) * ∇log p(X)]
    - Reparameterization: ∇E[f(X)] = E[∇f(g(ε))] where X = g(ε), ε ~ fixed distribution
    - Enumeration: Exact computation for discrete distributions with finite support

    Attributes:
        prog: Internal ADEVProgram containing the source computation

    Example:
        >>> from genjax.adev import expectation, normal_reparam
        >>> import jax.numpy as jnp
        >>>
        >>> @expectation
        ... def loss_function(theta):
        ...     x = normal_reparam(theta, 1.0)
        ...     return x**2
        >>>
        >>> # Compute gradient estimate
        >>> grad = loss_function.grad_estimate(0.5)
        >>> jnp.isfinite(grad)  # doctest: +ELLIPSIS
        Array(True, dtype=bool...)
        >>>
        >>> # Compute expectation value
        >>> value = loss_function.estimate(0.5)
        >>> jnp.isfinite(value)  # doctest: +ELLIPSIS
        Array(True, dtype=bool...)
    """

    prog: ADEVProgram

    def jvp_estimate(self, *duals: DualTree):
        """Compute Jacobian-Vector Product estimate for the expectation.

        This method provides the core JVP computation for ADEV. It applies the
        continuation-passing style transformation with an identity continuation,
        meaning this expectation represents the "final" computation in the chain.

        Args:
            *duals: Input arguments as dual numbers (primal, tangent) pairs

        Returns:
            Dual number with primal value E[f(X)] and tangent containing ∇E[f(X)]

        Note:
            This is the foundational method that enables both grad_estimate and
            integration with JAX's automatic differentiation system.
        """

        # Identity continuation - this expectation is the final computation
        def _identity(v):
            return v

        return self.prog.jvp_estimate(duals, _identity)

    def grad_estimate(self, *primals):
        """Compute unbiased gradient estimate of the expectation.

        This method provides the primary interface for computing gradients of
        expectation values. It leverages JAX's grad transformation combined with
        ADEV's custom JVP rules to produce unbiased gradient estimates.

        Args:
            *primals: Input values to compute gradients with respect to

        Returns:
            If single argument: Single gradient estimate array
            If multiple arguments: Tuple of gradient estimates

        Example:
            >>> from genjax.adev import expectation, normal_reparam
            >>> import jax.numpy as jnp
            >>>
            >>> @expectation
            ... def objective(mu, sigma):
            ...     x = normal_reparam(mu, sigma)
            ...     return x**2
            >>>
            >>> # Compute gradient with respect to both parameters
            >>> grad_mu, grad_sigma = objective.grad_estimate(1.0, 0.5)
            >>> jnp.isfinite(grad_mu)  # doctest: +ELLIPSIS
            Array(True, dtype=bool...)
            >>> jnp.isfinite(grad_sigma)  # doctest: +ELLIPSIS
            Array(True, dtype=bool...)

        Note:
            The gradient estimates are unbiased, meaning E[∇̂f] = ∇E[f], but they
            may have variance. The choice of gradient estimation strategy (REINFORCE,
            reparameterization, etc.) affects this variance.
        """

        def _invoke_closed_over(primals):
            return invoke_closed_over(self, primals)

        grad_result = jax.grad(_invoke_closed_over)(primals)

        # Return single gradient for single argument, tuple for multiple arguments
        if len(primals) == 1:
            return grad_result[0]
        else:
            return grad_result

    def estimate(self, *args):
        """Compute the expectation value (forward pass only).

        This method evaluates E[f(X)] without computing gradients. It's useful
        when you only need the expectation value itself, not its derivatives.

        Args:
            *args: Arguments to the expectation function

        Returns:
            The expectation value E[f(X)] as computed by the stochastic program

        Example:
            >>> from genjax.adev import expectation, normal_reparam
            >>> import jax.numpy as jnp
            >>>
            >>> @expectation
            ... def mean_squared(mu):
            ...     x = normal_reparam(mu, 1.0)
            ...     return x**2
            >>>
            >>> # Just compute E[X^2] where X ~ Normal(mu, 1)
            >>> expectation_value = mean_squared.estimate(2.0)
            >>> jnp.isfinite(expectation_value)  # doctest: +ELLIPSIS
            Array(True, dtype=bool...)
            >>> expectation_value > 0  # Should be positive for squared values  # doctest: +ELLIPSIS
            Array(True, dtype=bool...)

        Note:
            This method uses zero tangents in the dual number computation,
            effectively performing only the forward pass through the stochastic
            computation graph.
        """
        tangents = jtu.tree_map(lambda _: 0.0, args)
        return self.jvp_estimate(*Dual.dual_tree(args, tangents)).primal


def expectation(source: Callable[..., Any]) -> Expectation:
    """Decorator to create an Expectation object from a stochastic function.

    This decorator transforms a function containing stochastic operations into an
    Expectation object that supports automatic differentiation of expectation values.
    The decorated function should use ADEV-compatible distributions (those with
    gradient estimation strategies like normal_reparam, normal_reinforce, etc.).

    Args:
        source: Function containing stochastic operations using ADEV primitives

    Returns:
        Expectation object with grad_estimate, jvp_estimate, and estimate methods

    Example:
        >>> from genjax.adev import expectation, normal_reparam
        >>>
        >>> # Basic usage
        >>> @expectation
        ... def quadratic_loss(theta):
        ...     x = normal_reparam(theta, 1.0)  # Reparameterizable distribution
        ...     return (x - 2.0)**2
        >>>
        >>> # Compute gradient
        >>> gradient = quadratic_loss.grad_estimate(1.0)
        >>> import jax.numpy as jnp
        >>> jnp.isfinite(gradient)  # doctest: +ELLIPSIS
        Array(True, dtype=bool...)
        >>>
        >>> # Compute expectation value
        >>> loss_value = quadratic_loss.estimate(1.0)
        >>> jnp.isfinite(loss_value)  # doctest: +ELLIPSIS
        Array(True, dtype=bool...)

        More complex example with multiple variables:
        @expectation
        def complex_objective(mu, sigma):
            x = normal_reparam(mu, sigma)
            y = normal_reinforce(0.0, 1.0)  # REINFORCE strategy
            return jnp.sin(x) * jnp.cos(y)

        grad_mu, grad_sigma = complex_objective.grad_estimate(0.5, 1.2)
        ```

    Note:
        The function should only use ADEV-compatible distributions that have
        gradient estimation strategies. Regular distributions (normal, beta, etc.)
        won't provide gradient estimates - use their ADEV variants instead
        (normal_reparam, normal_reinforce, flip_enum, etc.).

        The resulting Expectation object's interfaces (grad_estimate, estimate, etc.)
        are compatible with JAX transformations like jit and modular_vmap. The
        Expectation object itself is also a Pytree, so it can be passed as an
        argument to JAX-transformed functions. Use modular_vmap instead of regular
        vmap for proper handling of probabilistic primitives within ADEV programs.
    """
    prog = ADEVProgram(const(source))
    return Expectation(prog)


#########################################
# Register custom forward mode with JAX #
#########################################


# These functions register ADEV's jvp_estimate as a custom JVP rule for JAX.
# This enables JAX to automatically synthesize grad implementations for
# Expectation objects, following the approach described in:
# "You Only Linearize Once" (arXiv:2204.10923)


@jax.custom_jvp
def invoke_closed_over(instance, args):
    """Primal forward-mode function for Expectation objects with custom JVP rule.

    This function serves as the primal computation for JAX's custom JVP rule
    registration. It's defined externally to the Expectation class to avoid
    complications with defining custom JVP rules on Pytree classes.

    Args:
        instance: Expectation object to evaluate
        args: Arguments to pass to the expectation

    Returns:
        The expectation value computed by instance.estimate(*args)

    Note:
        This function is decorated with @jax.custom_jvp to register ADEV's
        jvp_estimate as the custom JVP rule. This allows JAX to automatically
        synthesize grad implementations that use ADEV's unbiased gradient estimators.
    """
    return instance.estimate(*args)


def invoke_closed_over_jvp(primals: tuple, tangents: tuple):
    """Custom JVP rule that delegates to ADEV's jvp_estimate method.

    This function registers ADEV's jvp_estimate as the JVP rule for Expectation
    objects, enabling JAX to automatically synthesize grad implementations.
    When JAX encounters invoke_closed_over in a computation that requires
    differentiation, it will use this rule instead of trying to differentiate
    through the stochastic computation.

    Args:
        primals: Tuple of (instance, args) representing the primal values
        tangents: Tuple of (_, tangents) representing the tangent vectors

    Returns:
        Tuple of (primal_output, tangent_output) where:
        - primal_output: The expectation value E[f(X)]
        - tangent_output: ADEV's unbiased gradient estimate ∇E[f(X)]

    Note:
        This converts between JAX's JVP representation (separate primals/tangents)
        and ADEV's dual number representation, then delegates to jvp_estimate
        for the actual gradient computation using ADEV's CPS transformation.
    """
    (instance, primals) = primals
    (_, tangents) = tangents
    duals = Dual.dual_tree(primals, tangents)
    out_dual = instance.jvp_estimate(*duals)
    (v,), (tangent,) = Dual.tree_unzip(out_dual)
    return v, tangent


# Register ADEV's jvp_estimate as the custom JVP rule for JAX
# This allows JAX to automatically synthesize grad implementations for Expectation objects
# symbolic_zeros=False ensures tangents are computed even for zero inputs
invoke_closed_over.defjvp(invoke_closed_over_jvp, symbolic_zeros=False)

################################
# Gradient strategy primitives #
################################


@Pytree.dataclass
class REINFORCE(ADEVPrimitive):
    """REINFORCE (score function) gradient estimator primitive.

    Implements the REINFORCE gradient estimator (Williams, 1992), also known as
    the score function estimator or likelihood ratio method. This estimator is
    one of the key gradient estimation strategies supported by the ADEV framework.

    Theoretical Foundation:
    The REINFORCE estimator is based on the score function identity:
        ∇_θ E[f(X)] = E[f(X) * ∇_θ log p(X; θ)]

    where ∇_θ log p(X; θ) is the score function. This identity holds for any
    distribution p(X; θ) with differentiable log-density, making REINFORCE
    universally applicable but potentially high-variance.

    ADEV Implementation:
    Within ADEV's CPS framework, REINFORCE:
    1. Samples X ~ p(·; θ) using the current parameters
    2. Evaluates f(X) using the dual continuation (kdual)
    3. Computes the score function ∇_θ log p(X; θ) via JAX's JVP
    4. Returns f(X) + f(X) * ∇_θ log p(X; θ) as the gradient estimate

    Attributes:
        sample_function: Function to sample from the distribution
        differentiable_logpdf: Function to compute log-probability density

    Note:
        While general-purpose, REINFORCE can exhibit high variance. Reparameterization
        is preferred when available, as proven more efficient in the ADEV paper.
    """

    sample_function: Const[Callable[..., Any]]
    differentiable_logpdf: Const[Callable[..., Any]]

    def sample(self, *args):
        """Forward sampling using the provided sample function."""
        return self.sample_function.value(*args)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        """REINFORCE gradient estimation using the score function identity.

        Implements the score function estimator: ∇_θ E[f(X)] = E[f(X) * ∇_θ log p(X; θ)]

        This method applies the ADEV CPS transformation for REINFORCE:
        1. Sample X ~ p(·; θ) from the distribution with current parameters
        2. Evaluate f(X) using the dual continuation (kdual) to get the function value
        3. Compute the score function ∇_θ log p(X; θ) using JAX's forward-mode AD
        4. Combine via the REINFORCE identity: f(X) + f(X) * ∇_θ log p(X; θ)

        The dual continuation captures the ADEV-transformed "rest of the computation"
        after this stochastic choice, enabling modular composition with other
        gradient estimation strategies as described in the ADEV paper.
        """
        (_, kdual) = konts
        primals = Dual.tree_primal(dual_tree)
        tangents = Dual.tree_tangent(dual_tree)

        # Sample from the distribution
        v = self.sample(*primals)

        # Evaluate f(X) using dual continuation
        dual_tree = Dual.tree_pure(v)
        out_dual = kdual(dual_tree)
        (out_primal,), (out_tangent,) = Dual.tree_unzip(out_dual)

        # Compute score function: ∇log p(X)
        # For discrete values, use float0 tangent type as required by JAX
        v_tangent = (
            jnp.zeros(v.shape, dtype=jax.dtypes.float0)
            if v.dtype in (jnp.bool_, jnp.int32, jnp.int64)
            else jnp.zeros_like(v)
        )
        _, lp_tangent = jax.jvp(
            self.differentiable_logpdf.value,
            (v, *primals),
            (v_tangent, *tangents),
        )

        # REINFORCE identity: ∇E[f(X)] = f(X) + f(X) * ∇log p(X)
        # This gives an unbiased estimate of the gradient as proven in the ADEV paper
        return Dual(out_primal, out_tangent + (out_primal * lp_tangent))


def reinforce(sample_func, logpdf_func):
    """Factory function for creating REINFORCE gradient estimators.

    Args:
        sample_func: Function to sample from distribution
        logpdf_func: Function to compute log-probability density

    Returns:
        REINFORCE primitive for the given distribution

    Example:
        >>> normal_reinforce_prim = reinforce(normal.sample, normal.logpdf)
    """
    return REINFORCE(const(sample_func), const(logpdf_func))


######################################
# Discrete gradient estimator primitives #
######################################


@Pytree.dataclass
class FlipEnum(ADEVPrimitive):
    """Exact enumeration gradient estimator for Bernoulli distributions.

    For discrete distributions with finite support, we can compute exact gradients
    by enumerating all possible outcomes and weighting by their probabilities.
    This gives zero-variance gradient estimates for the flip/Bernoulli case.

    The estimator computes: ∇E[f(X)] = p*f(True) + (1-p)*f(False)
    """

    def sample(self, *args):
        (probs,) = args
        return 1 == bernoulli.sample(probs)

    def prim_jvp_estimate(
        self,
        dual_tree: tuple[DualTree, ...],
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        true_dual = kdual(Dual(jnp.array(True), jnp.zeros_like(jnp.array(True))))
        false_dual = kdual(Dual(jnp.array(False), jnp.zeros_like(jnp.array(False))))
        (true_primal,), (true_tangent,) = Dual.tree_unzip(true_dual)
        (false_primal,), (false_tangent,) = Dual.tree_unzip(false_dual)

        def _inner(p, tl, fl):
            return p * tl + (1 - p) * fl

        out_primal, out_tangent = jax.jvp(
            _inner,
            (p_primal, true_primal, false_primal),
            (p_tangent, true_tangent, false_tangent),
        )
        return Dual(out_primal, out_tangent)


flip_enum = FlipEnum()


@Pytree.dataclass
class FlipMVD(ADEVPrimitive):
    """Measure-Valued Derivative (MVD) gradient estimator for Bernoulli distributions.

    Implements the measure-valued derivative approach for gradient estimation with
    discrete distributions. MVD is a flexible gradient estimation technique that
    decomposes the derivative of a probability density into positive and negative
    components: ∇_θ p(x; θ) = c_θ(p^+(x; θ) - p^-(x; θ)).

    Theoretical Foundation:
    For discrete distributions like Bernoulli, MVD enables gradient estimation
    without requiring differentiability assumptions. The estimator works by:
    1. Sampling from the original distribution
    2. Evaluating the function on both the sampled value and its complement
    3. Using a signed difference to create an unbiased gradient estimate

    MVD Implementation for Bernoulli:
    The key insight is using the "phantom estimator" approach where:
    - The sampled outcome determines the sign via (-1)^v
    - Both the actual outcome and its complement are evaluated
    - The difference (other - b_primal) captures the discrete gradient

    Advantages:
    - Works with discrete distributions where REINFORCE may have issues
    - No differentiability requirements on the objective function
    - Provides unbiased gradient estimates for discrete parameters

    Disadvantages:
    - Computationally expensive (requires multiple evaluations)
    - Higher variance than reparameterization when applicable
    - Only applies to single parameters at a time

    Note:
        This is a "phantom estimator" that evaluates the function on auxiliary
        samples (the complement outcome) to construct the gradient estimate.
        The (-1)^v term creates the appropriate sign for the discrete difference.
    """

    def sample(self, *args):
        """Sample from Bernoulli distribution."""
        p = (args,)
        return 1 == bernoulli.sample(probs=p)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        """Measure-valued derivative gradient estimation for Bernoulli.

        Implements the MVD approach using phantom estimation:
        1. Sample v ~ Bernoulli(p) to get the primary outcome
        2. Evaluate f(v) using the dual continuation (kdual)
        3. Evaluate f(¬v) using the pure continuation (kpure) as phantom estimate
        4. Combine with signed difference: (-1)^v * (f(¬v) - f(v))

        The (-1)^v term ensures the correct sign for the discrete gradient:
        - When v=1: -1 * (f(0) - f(1)) = f(1) - f(0)
        - When v=0: +1 * (f(1) - f(0)) = f(1) - f(0)

        This creates an unbiased estimator of ∇_p E[f(X)] for X ~ Bernoulli(p).
        """
        (kpure, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)  # Fix: was tree_primal

        # Sample from Bernoulli(p)
        v = bernoulli.sample(probs=p_primal)
        b = v == 1

        # Evaluate f(v) using dual continuation
        # For discrete values, use float0 tangent type as required by JAX
        b_tangent_zero = (
            jnp.zeros(b.shape, dtype=jax.dtypes.float0)
            if b.dtype in (jnp.bool_, jnp.int32, jnp.int64)
            else jnp.zeros_like(b)
        )
        b_dual = kdual(Dual(b, b_tangent_zero))
        (b_primal,), (b_tangent,) = Dual.tree_unzip(b_dual)

        # Evaluate f(¬v) using pure continuation (phantom estimate)
        other_result = kpure(jnp.logical_not(b))

        # Extract scalar value using JAX-compatible tree operations
        # kpure may return a pytree structure, so we flatten and take the first element
        other_flat, _ = jtu.tree_flatten(other_result)
        other = other_flat[0]  # Assume there's always at least one element

        # MVD estimator: (-1)^v * (f(¬v) - f(v))
        # This creates the signed discrete difference for gradient estimation
        est = ((-1) ** v) * (other - b_primal)

        return Dual(b_primal, b_tangent + est * p_tangent)


flip_mvd = FlipMVD()


@Pytree.dataclass
class FlipEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (p,) = args
        return 1 == bernoulli.sample(probs=p)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        ret_primals, ret_tangents = modular_vmap(kdual)(
            (jnp.array([True, False]),),
            (jnp.zeros_like(jnp.array([True, False]))),
        )

        def _inner(p, ret):
            return jnp.sum(jnp.array([p, 1 - p]) * ret)

        return Dual(
            *jax.jvp(
                _inner,
                (p_primal, ret_primals),
                (p_tangent, ret_tangents),
            )
        )


flip_enum_parallel = FlipEnumParallel()


@Pytree.dataclass
class CategoricalEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (probs,) = args
        return categorical.sample(probs)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (probs_primal,) = Dual.tree_primal(dual_tree)
        (probs_tangent,) = Dual.tree_tangent(dual_tree)
        idxs = jnp.arange(len(probs_primal))
        ret_primals, ret_tangents = modular_vmap(kdual)(
            (idxs,), (jnp.zeros_like(idxs),)
        )

        def _inner(probs, primals):
            return jnp.sum(jax.nn.softmax(probs) * primals)

        return Dual(
            *jax.jvp(
                _inner,
                (probs_primal, ret_primals),
                (probs_tangent, ret_tangents),
            )
        )


categorical_enum_parallel = CategoricalEnumParallel()

########################################
# REINFORCE distribution estimators   #
########################################

flip_reinforce = distribution(
    reinforce(
        bernoulli.sample,
        bernoulli.logpdf,
    ),
    bernoulli.logpdf,
)

geometric_reinforce = distribution(
    reinforce(
        geometric.sample,
        geometric.logpdf,
    ),
    geometric.logpdf,
)

normal_reinforce = distribution(
    reinforce(
        normal.sample,
        normal.logpdf,
    ),
    normal.logpdf,
)


########################################
# Reparameterization estimators       #
########################################


@Pytree.dataclass
class NormalREPARAM(ADEVPrimitive):
    """Reparameterization (pathwise) gradient estimator for normal distributions.

    Implements the reparameterization trick, also known as the pathwise estimator,
    which is one of the core gradient estimation strategies in the ADEV framework.
    This provides low-variance gradient estimates for reparameterizable distributions.

    Theoretical Foundation:
    For a reparameterizable distribution p(X; θ) = p(g(ε; θ)) where ε ~ p(ε) is
    parameter-free, the pathwise estimator is:
        ∇_θ E[f(X)] = E[∇_θ f(g(ε; θ))]

    For Normal(μ, σ): X = g(ε; μ, σ) = μ + σ * ε, where ε ~ Normal(0, 1)
    This reparameterization allows gradients to flow directly through the
    parameters μ and σ via standard automatic differentiation (chain rule).

    ADEV Implementation:
    Within ADEV's CPS framework, reparameterization:
    1. Samples parameter-free noise ε ~ Normal(0, 1)
    2. Applies the transformation X = μ + σ * ε with JAX's JVP for gradients
    3. Passes the reparameterized sample through the dual continuation (kdual)

    This strategy typically exhibits lower variance than REINFORCE, as noted in
    the ADEV paper and empirical studies (Kingma & Welling, 2014).
    """

    def sample(self, *args):
        loc, scale_diag = args
        return normal.sample(loc, scale_diag)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        """Reparameterization gradient estimation using the pathwise estimator.

        Implements: ∇_θ E[f(X)] = E[∇_θ f(g(ε; θ))] where X = g(ε; θ)

        This method applies the ADEV CPS transformation for reparameterization:
        1. Sample parameter-free noise ε ~ Normal(0, 1)
        2. Apply reparameterization X = μ + σ * ε with gradients via JAX JVP
        3. Pass the dual number (X, ∇X) to the dual continuation (kdual)

        The dual continuation captures the ADEV-transformed remainder of the
        computation, enabling low-variance gradient flow as described in the ADEV paper.
        """
        _, kdual = konts
        (mu_primal, sigma_primal) = Dual.tree_primal(dual_tree)
        (mu_tangent, sigma_tangent) = Dual.tree_tangent(dual_tree)

        # Sample parameter-free noise
        eps = normal.sample(0.0, 1.0)

        # Reparameterization: X = μ + σ * ε with gradient flow
        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return kdual(Dual(primal_out, tangent_out))


normal_reparam = distribution(
    NormalREPARAM(),
    normal.logpdf,
)


@Pytree.dataclass
class MultivariateNormalREPARAM(ADEVPrimitive):
    """Multivariate reparameterization (pathwise) gradient estimator.

    Extends the reparameterization trick to multivariate normal distributions,
    implementing the pathwise estimator for high-dimensional parameter spaces
    as supported by the ADEV framework.

    Theoretical Foundation:
    For MultivariateNormal(μ, Σ), the reparameterization is:
        X = g(ε; μ, Σ) = μ + L @ ε
    where L = cholesky(Σ) and ε ~ Normal(0, I).

    The pathwise estimator then gives:
        ∇_{μ,Σ} E[f(X)] = E[∇_{μ,Σ} f(μ + L @ ε)]

    ADEV Implementation:
    This primitive enables efficient gradient flow with respect to both the mean
    vector μ and covariance matrix Σ, crucial for scalable variational inference
    in high-dimensional spaces. The Cholesky decomposition ensures positive
    definiteness while enabling automatic differentiation through the covariance
    structure.

    This implementation follows the ADEV paper's approach to modular gradient
    estimation, allowing seamless integration with other stochastic primitives
    in complex probabilistic programs.
    """

    def sample(self, *args):
        loc, covariance_matrix = args
        return multivariate_normal.sample(loc, covariance_matrix)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        """Multivariate reparameterization using Cholesky decomposition.

        Implements: ∇E[f(X)] = E[∇f(μ + L @ ε)] where L = cholesky(Σ)

        This method applies the pathwise estimator for multivariate normal distributions:
        1. Sample standard multivariate normal noise ε ~ Normal(0, I)
        2. Apply Cholesky reparameterization X = μ + L @ ε with gradient flow
        3. Pass the dual number (X, ∇X) to the dual continuation (kdual)

        The Cholesky decomposition ensures efficient and numerically stable
        gradients with respect to the covariance matrix Σ, as described in
        the ADEV framework for modular gradient estimation strategies.
        """
        _, kdual = konts
        (loc_primal, cov_primal) = Dual.tree_primal(dual_tree)
        (loc_tangent, cov_tangent) = Dual.tree_tangent(dual_tree)

        # Sample standard multivariate normal: ε ~ Normal(0, I)
        eps = multivariate_normal.sample(
            jnp.zeros_like(loc_primal), jnp.eye(loc_primal.shape[-1])
        )

        # Multivariate reparameterization: X = μ + L @ ε where L = cholesky(Σ)
        # This provides efficient gradients for both mean and covariance parameters
        def _inner(loc, cov):
            L = jnp.linalg.cholesky(cov)
            return loc + L @ eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (loc_primal, cov_primal),
            (loc_tangent, cov_tangent),
        )
        return kdual(Dual(primal_out, tangent_out))


multivariate_normal_reparam = distribution(
    MultivariateNormalREPARAM(),
    multivariate_normal.logpdf,
)

multivariate_normal_reinforce = distribution(
    reinforce(
        multivariate_normal.sample,
        multivariate_normal.logpdf,
    ),
    multivariate_normal.logpdf,
)


###########
# Exports #
###########

__all__ = [
    # Core ADEV classes
    "Dual",
    "ADEVPrimitive",
    "expectation",
    # Discrete gradient estimators
    "flip_enum",
    "flip_enum_parallel",
    "flip_mvd",
    "categorical_enum_parallel",
    # Continuous gradient estimators (distributions)
    "flip_reinforce",
    "geometric_reinforce",
    "normal_reinforce",
    "normal_reparam",
    "multivariate_normal_reparam",
    "multivariate_normal_reinforce",
    # Gradient strategy factories
    "reinforce",
]
