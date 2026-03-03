import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from examples.air.core import (
    SMALL_CONFIG,
    VALID_ESTIMATORS,
    init_air_params,
    make_air_objective,
    run_estimator_suite,
    sample_prior_dataset,
    train_air,
)


def _all_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jtu.tree_leaves(tree))


def test_air_objective_gradients_are_finite():
    config = SMALL_CONFIG
    params = init_air_params(jax.random.key(0), config=config)
    dataset = sample_prior_dataset(
        params.decoder,
        config=config,
        n_samples=8,
        seed_value=1,
    )
    observation = dataset.observations[0]

    for estimator in VALID_ESTIMATORS:
        objective, _, _ = make_air_objective(
            estimator,
            config=config,
            num_particles=1,
        )
        grad = objective.grad_estimate(observation, params)[1]
        assert _all_finite(grad)


def test_air_short_training_smoke():
    config = SMALL_CONFIG
    params = init_air_params(jax.random.key(2), config=config)
    dataset = sample_prior_dataset(
        params.decoder,
        config=config,
        n_samples=48,
        seed_value=3,
    )

    result = train_air(
        dataset.observations,
        dataset.true_counts,
        estimator="enum",
        config=config,
        num_particles=1,
        init_params=params,
        num_epochs=2,
        batch_size=8,
        learning_rate=1e-4,
        evaluate_accuracy_every=1,
        seed_value=4,
    )

    assert result.loss_history.shape == (2,)
    assert result.accuracy_history.shape == (2,)
    assert result.epoch_times.shape == (2,)
    assert jnp.all(jnp.isfinite(result.loss_history))
    assert jnp.all(jnp.isfinite(result.accuracy_history))


def test_air_estimator_suite_smoke():
    config = SMALL_CONFIG
    params = init_air_params(jax.random.key(5), config=config)
    dataset = sample_prior_dataset(
        params.decoder,
        config=config,
        n_samples=32,
        seed_value=6,
    )

    results = run_estimator_suite(
        dataset.observations,
        dataset.true_counts,
        estimators=("enum", "mvd"),
        config=config,
        num_particles=1,
        num_epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        eval_objective_samples=2,
        seed_value=7,
    )

    assert len(results) == 2
    for result in results:
        assert jnp.isfinite(result.final_loss)
        assert jnp.isfinite(result.final_accuracy)
        assert jnp.isfinite(result.objective_mean)
        assert jnp.isfinite(result.objective_variance)
