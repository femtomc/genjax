"""Gaussian Process implementation as a GenJAX generative function.

This module provides a native GenJAX implementation of Gaussian processes,
treating GPs as generative functions with exact inference as the internal proposal.
"""

import jax
import jax.numpy as jnp
import genjax
from genjax.core import GFI, Trace, Score, Density, Weight, Selection, Pytree
from beartype.typing import Optional, Any, Tuple, Dict

from .kernels import Kernel
from .mean import MeanFunction

# Use simplified Array type to avoid jaxtyping issues
Array = jnp.ndarray


@Pytree.dataclass
class GPTrace(Trace[Array, Array]):
    """Trace from GP generative function."""

    # Input locations
    x_train: Optional[Array]
    y_train: Optional[Array]
    x_test: Array

    # Generated values (the choices)
    y_test: Array

    # GP parameters
    kernel: Kernel
    mean_fn: MeanFunction
    noise_variance: Array

    # Score (log probability)
    score: Score

    # Standard trace fields
    args: Any
    kwargs: Dict[str, Any]
    gen_fn: "GP"

    def get_choices(self) -> Array:
        """Return the random choices (test outputs)."""
        # For GP, the choices are just the y_test values
        # No need for Fixed handling since GP doesn't use hierarchical addresses
        return self.y_test

    def get_retval(self) -> Array:
        """Return value is the same as choices for GP."""
        return self.y_test

    def get_score(self) -> Score:
        """Return the score (log 1/P)."""
        return self.score

    def get_args(self) -> Any:
        """Return the arguments."""
        return self.args

    def get_gen_fn(self) -> "GP":
        """Return the generative function."""
        return self.gen_fn

    def get_fixed_choices(self) -> Array:
        """Get choices preserving Fixed wrappers.

        For GP, this is the same as get_choices since we don't use
        hierarchical addresses or Fixed wrappers.
        """
        return self.y_test


@Pytree.dataclass
class GP(GFI[Array, Array]):
    """Gaussian Process as a generative function.

    Uses exact Gaussian conditioning as the internal proposal distribution,
    making it efficient for inference.
    """

    kernel: Kernel
    mean_fn: MeanFunction
    noise_variance: float
    jitter: float

    def _compute_posterior(
        self,
        x_train: Optional[Array],
        y_train: Optional[Array],
        x_test: Array,
    ) -> Tuple[Array, Array]:
        """Compute GP posterior mean and covariance."""
        if (
            x_train is None
            or y_train is None
            or (hasattr(x_train, "shape") and x_train.shape[0] == 0)
            or (hasattr(y_train, "shape") and y_train.shape[0] == 0)
        ):
            # Prior distribution
            mean = self.mean_fn(x_test)
            cov = self.kernel(x_test, x_test)
            return mean, cov

        # Compute kernel matrices
        K_train = self.kernel(x_train, x_train)
        K_train_test = self.kernel(x_train, x_test)
        K_test = self.kernel(x_test, x_test)

        # Add noise to training covariance
        K_train_noisy = K_train + (self.noise_variance + self.jitter) * jnp.eye(
            x_train.shape[0]
        )

        # Compute Cholesky decomposition
        L = jax.scipy.linalg.cholesky(K_train_noisy, lower=True)

        # Compute posterior mean
        residual = y_train - self.mean_fn(x_train)
        alpha = jax.scipy.linalg.solve_triangular(L, residual, lower=True)
        beta = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)
        posterior_mean = self.mean_fn(x_test) + K_train_test.T @ beta

        # Compute posterior covariance
        v = jax.scipy.linalg.solve_triangular(L, K_train_test, lower=True)
        posterior_cov = K_test - v.T @ v

        return posterior_mean, posterior_cov

    def simulate(
        self,
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        **kwargs,
    ) -> GPTrace:
        """Forward sampling from the GP.

        Note: When used inside a @gen function, this will use PJAX primitives
        for sampling. When used with seed() transformation, requires a key.
        """
        # Compute posterior
        mean, cov = self._compute_posterior(x_train, y_train, x_test)

        # Add jitter for numerical stability
        cov_jittered = cov + self.jitter * jnp.eye(x_test.shape[0])

        # Sample from multivariate normal
        y_test = genjax.multivariate_normal.sample(mean, cov_jittered)

        # Compute log probability
        log_prob = genjax.multivariate_normal.logpdf(y_test, mean, cov_jittered)

        # Create trace
        trace = GPTrace(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            noise_variance=jnp.array(self.noise_variance),
            score=-log_prob,  # Score is negative log probability
            args=(x_test,),
            kwargs={"x_train": x_train, "y_train": y_train},
            gen_fn=self,
        )

        return trace

    def assess(
        self,
        y_test: Array,
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[Density, Array]:
        """Evaluate density at given choices."""
        # Compute posterior
        mean, cov = self._compute_posterior(x_train, y_train, x_test)

        # Add jitter for numerical stability
        cov_jittered = cov + self.jitter * jnp.eye(x_test.shape[0])

        # Compute log probability
        log_prob = genjax.multivariate_normal.logpdf(y_test, mean, cov_jittered)

        return log_prob, y_test

    def generate(
        self,
        constraints: Optional[Array],
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[GPTrace, Weight]:
        """Constrained generation via importance sampling.

        For GPs with exact inference, when all outputs are constrained,
        the weight is 0 (proposal equals model). When partially constrained,
        we use the conditional distribution.
        """
        # Compute posterior
        mean, cov = self._compute_posterior(x_train, y_train, x_test)

        # Add jitter for numerical stability
        cov_jittered = cov + self.jitter * jnp.eye(x_test.shape[0])

        if constraints is None:
            # No constraints - regular sampling
            y_test = genjax.multivariate_normal.sample(mean, cov_jittered)
            log_prob = genjax.multivariate_normal.logpdf(y_test, mean, cov_jittered)
            weight = jnp.array(0.0)  # Weight is 0 for importance sampling
        else:
            # All outputs constrained (for now)
            # TODO: Handle partial constraints
            y_test = constraints
            log_prob = genjax.multivariate_normal.logpdf(y_test, mean, cov_jittered)
            weight = jnp.array(0.0)  # Exact inference, so weight is 0

        # Create trace
        trace = GPTrace(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            noise_variance=jnp.array(self.noise_variance),
            score=-log_prob,  # Score is negative log probability
            args=(x_test,),
            kwargs={"x_train": x_train, "y_train": y_train},
            gen_fn=self,
        )

        return trace, weight  # Weight is 0 in log space

    def update(
        self,
        trace: GPTrace,
        constraints: Optional[Array],
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[GPTrace, Weight, Optional[Array]]:
        """Update move for MCMC/SMC.

        For GPs, this is essentially a regenerate of all test points.
        """
        old_y_test = trace.y_test
        old_score = trace.score

        # Generate new trace
        new_trace, _ = self.generate(constraints, x_test, x_train, y_train)

        # Compute weight
        weight = -new_trace.score + old_score

        # Discard is the old values if constraints were provided
        discard = old_y_test if constraints is not None else None

        return new_trace, weight, discard

    def regenerate(
        self,
        trace: GPTrace,
        selection: Selection,
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
    ) -> Tuple[GPTrace, Weight, Optional[Array]]:
        """Regenerate selected addresses.

        For GPs without hierarchical structure, we regenerate all test points.
        """
        # For now, implement as full regeneration
        # TODO: Implement partial regeneration based on selection
        return self.update(trace, None, x_test, x_train, y_train)

    def merge(
        self,
        choices1: Array,
        choices2: Array,
    ) -> Array:
        """Merge two choice maps (second takes precedence)."""
        _ = choices1  # Unused but required by interface
        return choices2

    def filter(
        self,
        choices: Array,
        selection: Selection,
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Filter choices by selection.

        For GPs without hierarchical structure, this is simplified.
        """
        # For now, treat as all-or-nothing
        # TODO: Implement proper filtering based on selection
        return choices, None

    def log_density(
        self,
        y_test: Array,
        x_test: Array,
        x_train: Optional[Array] = None,
        y_train: Optional[Array] = None,
        **kwargs,
    ) -> Score:
        """Compute log density."""
        density, _ = self.assess(y_test, x_test, x_train, y_train)
        return density
