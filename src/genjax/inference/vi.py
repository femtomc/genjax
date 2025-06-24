"""
Variational inference algorithms and utilities for GenJAX.

This module provides implementations of common variational inference patterns
that can be composed with generative functions through the GFI (Generative Function Interface).
Follows GenJAX patterns for vectorized computation and programmable inference.
"""

import jax.numpy as jnp
from jax.lax import scan
from typing import Any

from genjax.core import GFI, Pytree, gen, X, R, const, Const
from genjax.adev import (
    expectation,
    multivariate_normal_reparam,
    multivariate_normal_reinforce,
)


@Pytree.dataclass
class VariationalApproximation(Pytree):
    """Result of variational inference containing parameters, history, and diagnostics.

    Args:
        final_params: Final optimized variational parameters.
        param_history: History of parameter values during optimization.
        loss_history: History of loss values during optimization.
        n_iterations: Number of optimization iterations performed.
    """

    final_params: jnp.ndarray
    param_history: jnp.ndarray
    loss_history: jnp.ndarray
    n_iterations: Const[int]

    def get_final_params(self) -> jnp.ndarray:
        """Get the final optimized parameters."""
        return self.final_params

    def get_loss_history(self) -> jnp.ndarray:
        """Get the history of loss values during optimization."""
        return self.loss_history

    def get_param_history(self) -> jnp.ndarray:
        """Get the history of parameter values during optimization."""
        return self.param_history


def elbo_factory(
    target_gf: GFI[X, R],
    variational_family: GFI[X, R],
    constraint: X,
    target_args: tuple = (),
) -> Any:
    """Factory function to create ELBO objective for variational inference.

    Args:
        target_gf: Target generative function (model).
        variational_family: Variational family (parameterized by theta, with access to constraints).
        constraint: Observed data/constraints.
        target_args: Arguments for target generative function.

    Returns:
        ELBO function that takes variational parameters and returns expectation.
    """

    @expectation
    def elbo(*variational_params):
        """
        Evidence Lower Bound (ELBO) objective function.

        ELBO = E_q[log p(x,z) - log q(z|theta)]
        where q is the variational family and p is the target model.
        """
        # Sample from variational family (with access to constraints and parameters)
        tr = variational_family.simulate(constraint, *variational_params)

        # Get variational score: log(1/q(z|theta)) = -log(q(z|theta))
        q_score = tr.get_score()

        # Evaluate target density: log p(x,z)
        merged_choices, _ = target_gf.merge(constraint, tr.get_choices())
        p_density, _ = target_gf.assess(merged_choices, *target_args)

        # ELBO = log p(x,z) - log q(z|theta) = p_density - (-log q) = p_density + q_score
        return p_density + q_score

    return elbo


def optimize_vi(
    elbo_fn: Any,
    init_params: jnp.ndarray,
    learning_rate: float = 1e-3,
    n_iterations: int = 1000,
    track_history: bool = True,
) -> VariationalApproximation:
    """Optimize variational parameters using gradient ascent on ELBO.

    Args:
        elbo_fn: ELBO expectation function (from elbo_factory, constraints already bound).
        init_params: Initial variational parameters.
        learning_rate: Step size for gradient ascent.
        n_iterations: Number of optimization iterations
        track_history: Whether to track parameter and loss history

    Returns:
        VariationalApproximation with optimized parameters and diagnostics
    """

    def update_step(carry, _iteration):
        params = carry

        # Compute gradient of ELBO w.r.t. parameters (constraints already bound in elbo_fn)
        param_grad = elbo_fn.grad_estimate(params)

        # Gradient ascent step (maximizing ELBO)
        new_params = params + learning_rate * param_grad

        if track_history:
            return new_params, (new_params, 0.0)  # Placeholder loss for now
        else:
            return new_params, 0.0

    # Run optimization loop
    final_params, history = scan(
        update_step,
        init_params,
        jnp.arange(n_iterations),
    )

    if track_history:
        param_history, loss_history = history
    else:
        param_history = jnp.array([])
        loss_history = history

    return VariationalApproximation(
        final_params=final_params,
        param_history=param_history,
        loss_history=loss_history,
        n_iterations=const(n_iterations),
    )


def mean_field_normal_family(
    n_dims: int,
    gradient_estimator: str = "reparam",
) -> GFI:
    """
    Create a mean-field normal variational family.

    Args:
        n_dims: Dimensionality of the latent space
        gradient_estimator: Type of gradient estimator ("reparam" or "reinforce")

    Returns:
        Generative function representing mean-field normal variational family
    """

    if gradient_estimator == "reparam":
        mvnormal_fn = multivariate_normal_reparam
    elif gradient_estimator == "reinforce":
        mvnormal_fn = multivariate_normal_reinforce
    else:
        raise ValueError(f"Unknown gradient estimator: {gradient_estimator}")

    @gen
    def variational_family(constraint, params):
        """
        Mean-field normal variational family.

        Args:
            constraint: Observed data/constraints (not used in mean-field, but available)
            params: Array of shape (2*n_dims,) containing [means, log_stds]
        """
        means = params[:n_dims]
        log_stds = params[n_dims:]
        stds = jnp.exp(log_stds)

        # Create diagonal covariance matrix for mean-field assumption
        cov = jnp.diag(stds**2)

        # Single multivariate normal sample with static address
        x = mvnormal_fn(means, cov) @ "x"
        return x

    return variational_family


def full_covariance_normal_family(
    n_dims: int,
    gradient_estimator: str = "reparam",
) -> GFI:
    """
    Create a full-covariance multivariate normal variational family.

    Args:
        n_dims: Dimensionality of the latent space
        gradient_estimator: Type of gradient estimator ("reparam" or "reinforce")

    Returns:
        Generative function representing full-covariance normal variational family
    """

    if gradient_estimator == "reparam":
        mvnormal_fn = multivariate_normal_reparam
    elif gradient_estimator == "reinforce":
        mvnormal_fn = multivariate_normal_reinforce
    else:
        raise ValueError(f"Unknown gradient estimator: {gradient_estimator}")

    @gen
    def variational_family(constraint, params):
        """
        Full-covariance multivariate normal variational family.

        Args:
            constraint: Observed data/constraints (not used in this family, but available)
            params: Dictionary with 'mean' (n_dims,) and 'chol_cov' (n_dims, n_dims)
        """
        mean = params["mean"]
        chol_cov = params["chol_cov"]

        # Convert Cholesky factor to full covariance matrix
        # Cov = L @ L^T where L is the Cholesky factor
        cov = chol_cov @ chol_cov.T

        # Single multivariate normal sample with static address
        x = mvnormal_fn(mean, cov) @ "x"
        return x

    return variational_family


def elbo_vi(
    target_gf: GFI[X, R],
    variational_family: GFI[X, R],
    init_params: jnp.ndarray,
    constraint: X,
    target_args: tuple = (),
    learning_rate: float = 1e-3,
    n_iterations: int = 1000,
    track_history: bool = True,
) -> VariationalApproximation:
    """
    Complete ELBO-based variational inference pipeline.

    Combines ELBO construction and optimization into a single convenient function.

    Args:
        target_gf: Target generative function (model)
        variational_family: Variational family (parameterized, with access to constraints)
        init_params: Initial variational parameters
        constraint: Observed data/constraints
        target_args: Arguments for target generative function
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations
        track_history: Whether to track parameter and loss history

    Returns:
        VariationalApproximation with optimized parameters and diagnostics
    """

    # Create ELBO objective with constraints bound
    elbo_fn = elbo_factory(target_gf, variational_family, constraint, target_args)

    # Optimize
    return optimize_vi(
        elbo_fn=elbo_fn,
        init_params=init_params,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        track_history=track_history,
    )
