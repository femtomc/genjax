import jax.numpy as jnp
import jax.tree_util as jtu

from examples.cone.core import (
    DEFAULT_DATA,
    estimate_objective_statistics,
    make_expressive_objective,
    make_naive_iwae_objective,
    naive_elbo_objective,
    optimize_objective,
    run_table4_suite,
)


def _all_finite(tree) -> bool:
    leaves = jtu.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_cone_objective_gradients_are_finite():
    iwae3 = make_naive_iwae_objective(3)
    expressive = make_expressive_objective(2, 2)

    naive_grad = naive_elbo_objective.grad_estimate(DEFAULT_DATA, (0.0, 0.0, 1.0, 1.0))[1]
    iwae_grad = iwae3.grad_estimate(DEFAULT_DATA, (0.0, 0.0, 1.0, 1.0))[1]
    expressive_grad = expressive.grad_estimate(DEFAULT_DATA, (0.0, 0.0))[1]

    assert _all_finite(naive_grad)
    assert _all_finite(iwae_grad)
    assert _all_finite(expressive_grad)


def test_short_elbo_optimization_improves_objective():
    init_params = (0.0, 0.0, 1.0, 1.0)

    init_mean, _ = estimate_objective_statistics(
        naive_elbo_objective,
        init_params,
        n_samples=256,
        seed_value=10,
    )

    result = optimize_objective(
        naive_elbo_objective,
        init_params,
        n_steps=150,
        batch_size=32,
        learning_rate=1e-3,
        seed_value=11,
    )

    final_mean, _ = estimate_objective_statistics(
        naive_elbo_objective,
        result.params,
        n_samples=256,
        seed_value=12,
    )

    assert final_mean > init_mean


def test_table4_suite_smoke():
    results = run_table4_suite(
        n_steps=20,
        batch_size=16,
        learning_rate=1e-3,
        eval_samples=32,
    )

    assert len(results) == 5
    for result in results:
        assert jnp.isfinite(result.mean)
        assert jnp.isfinite(result.variance)
