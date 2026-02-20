from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.tree_util as jtu

from genjax import gen, normal, normal_reparam
from genjax.adev import expectation
from genjax.pjax import seed


DEFAULT_DATA = {"z": 5.0}


@gen
def cone_model():
    """Noisy cone model used in the PLDI'24 artifact."""
    x = normal_reparam(0.0, 10.0) @ "x"
    y = normal_reparam(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    normal_reparam(rs, 0.1 + (rs / 100.0)) @ "z"


@gen
def naive_variational_family(data, phi):
    """Mean-field Gaussian guide over (x, y)."""
    mu_x, mu_y, log_sigma_x, log_sigma_y = phi
    normal_reparam(mu_x, jnp.exp(log_sigma_x)) @ "x"
    normal_reparam(mu_y, jnp.exp(log_sigma_y)) @ "y"


@gen
def expressive_variational_family(data, phi):
    """Auxiliary-variable guide.

    We use a Normal auxiliary variable because current ADEV exports
    reparameterized Normal primitives directly.
    """
    z_obs = data["z"]
    u = normal_reparam(0.0, 1.0) @ "u"
    theta = 2.0 * jnp.pi * jax.nn.sigmoid(u)

    log_sigma_x, log_sigma_y = phi
    normal_reparam(jnp.sqrt(z_obs) * jnp.cos(theta), jnp.exp(log_sigma_x)) @ "x"
    normal_reparam(jnp.sqrt(z_obs) * jnp.sin(theta), jnp.exp(log_sigma_y)) @ "y"


def _naive_log_weight(data: dict[str, float], phi: tuple[float, ...]) -> jnp.ndarray:
    tr = naive_variational_family.simulate(data, phi)
    merged_choices, _ = cone_model.merge(data, tr.get_choices())
    model_logp, _ = cone_model.assess(merged_choices)
    return model_logp + tr.get_score()


@expectation
def naive_elbo_objective(data, phi):
    return _naive_log_weight(data, phi)


def make_naive_iwae_objective(num_particles: int):
    if num_particles <= 0:
        raise ValueError("num_particles must be >= 1")

    log_num_particles = jnp.log(float(num_particles))

    @expectation
    def objective(data, phi):
        log_weights = []
        for _ in range(num_particles):
            log_weights.append(_naive_log_weight(data, phi))
        log_weights = jnp.stack(log_weights)
        return jsp.logsumexp(log_weights) - log_num_particles

    return objective


def _expressive_log_q_xy_given_u(
    data: dict[str, float],
    phi: tuple[float, float],
    x: jnp.ndarray,
    y: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    z_obs = data["z"]
    theta = 2.0 * jnp.pi * jax.nn.sigmoid(u)
    log_sigma_x, log_sigma_y = phi
    sigma_x = jnp.exp(log_sigma_x)
    sigma_y = jnp.exp(log_sigma_y)

    log_q_x = normal.logpdf(x, jnp.sqrt(z_obs) * jnp.cos(theta), sigma_x)
    log_q_y = normal.logpdf(y, jnp.sqrt(z_obs) * jnp.sin(theta), sigma_y)
    return log_q_x + log_q_y


def make_expressive_objective(num_inner_particles: int, num_outer_particles: int):
    """Build nested-importance objectives for HVI/IWHVI/DIWHVI variants.

    - num_inner_particles = N
    - num_outer_particles = K
    """
    if num_inner_particles <= 0 or num_outer_particles <= 0:
        raise ValueError("num_inner_particles and num_outer_particles must be >= 1")

    log_num_inner = jnp.log(float(num_inner_particles))
    log_num_outer = jnp.log(float(num_outer_particles))

    def _single_log_weight(data, phi):
        tr = expressive_variational_family.simulate(data, phi)
        choices = tr.get_choices()

        x = choices["x"]
        y = choices["y"]
        u0 = choices["u"]

        model_logp, _ = cone_model.assess({"x": x, "y": y, "z": data["z"]})

        log_q_candidates = [_expressive_log_q_xy_given_u(data, phi, x, y, u0)]
        for _ in range(num_inner_particles - 1):
            # Additional auxiliary proposals for the marginal-density estimate.
            u_aux = normal_reparam(0.0, 1.0)
            log_q_candidates.append(
                _expressive_log_q_xy_given_u(data, phi, x, y, u_aux)
            )

        log_q_est = jsp.logsumexp(jnp.stack(log_q_candidates)) - log_num_inner
        return model_logp - log_q_est

    @expectation
    def objective(data, phi):
        log_weights = []
        for _ in range(num_outer_particles):
            log_weights.append(_single_log_weight(data, phi))
        log_weights = jnp.stack(log_weights)
        return jsp.logsumexp(log_weights) - log_num_outer

    return objective


@dataclass
class OptimizationResult:
    params: Any
    loss_history: jnp.ndarray


@dataclass
class ObjectiveResult:
    name: str
    params: Any
    mean: float
    variance: float


def _compile_batched_loss_and_grad(objective):
    seeded_loss_and_grad = seed(
        lambda data, params: (
            objective.estimate(data, params),
            objective.grad_estimate(data, params)[1],
        )
    )
    return jax.jit(jax.vmap(seeded_loss_and_grad, in_axes=(0, None, None)))


def _compile_batched_estimate(objective):
    seeded_estimate = seed(lambda data, params: objective.estimate(data, params))
    return jax.jit(jax.vmap(seeded_estimate, in_axes=(0, None, None)))


def optimize_objective(
    objective,
    init_params: Any,
    *,
    data: dict[str, float] | None = None,
    n_steps: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed_value: int = 0,
) -> OptimizationResult:
    if data is None:
        data = DEFAULT_DATA

    batch_loss_and_grad = _compile_batched_loss_and_grad(objective)

    params = init_params
    losses = []
    key = jax.random.key(seed_value)

    for _ in range(n_steps):
        key, sub_key = jax.random.split(key)
        keys = jax.random.split(sub_key, batch_size)

        batch_losses, batch_grads = batch_loss_and_grad(keys, data, params)
        mean_loss = jnp.mean(batch_losses)
        mean_grads = jtu.tree_map(lambda g: jnp.mean(g, axis=0), batch_grads)
        params = jtu.tree_map(lambda p, g: p + learning_rate * g, params, mean_grads)

        losses.append(mean_loss)

    loss_history = jnp.stack(losses) if losses else jnp.array([])
    return OptimizationResult(params=params, loss_history=loss_history)


def estimate_objective_statistics(
    objective,
    params: Any,
    *,
    data: dict[str, float] | None = None,
    n_samples: int = 5000,
    seed_value: int = 0,
) -> tuple[float, float]:
    if data is None:
        data = DEFAULT_DATA

    batch_estimate = _compile_batched_estimate(objective)
    keys = jax.random.split(jax.random.key(seed_value), n_samples)
    samples = batch_estimate(keys, data, params)
    return float(jnp.mean(samples)), float(jnp.var(samples))


def sample_model_prior(*, n_samples: int = 5000, seed_value: int = 0):
    keys = jax.random.split(jax.random.key(seed_value), n_samples)
    seeded_simulate = seed(cone_model.simulate)
    traces = jax.jit(jax.vmap(seeded_simulate, in_axes=(0,)))(keys)
    return traces.get_choices()


def sample_guide(
    guide,
    params: Any,
    *,
    data: dict[str, float] | None = None,
    n_samples: int = 50000,
    seed_value: int = 0,
):
    if data is None:
        data = DEFAULT_DATA

    keys = jax.random.split(jax.random.key(seed_value), n_samples)
    seeded_simulate = seed(guide.simulate)
    traces = jax.jit(jax.vmap(seeded_simulate, in_axes=(0, None, None)))(
        keys,
        data,
        params,
    )
    return traces.get_choices()


def run_table4_suite(
    *,
    n_steps: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    eval_samples: int = 5000,
) -> list[ObjectiveResult]:
    specs = [
        (
            "ELBO",
            naive_elbo_objective,
            (0.0, 0.0, 1.0, 1.0),
            0,
        ),
        (
            "IWAE(K = 5)",
            make_naive_iwae_objective(5),
            (3.0, 0.0, 1.0, 1.0),
            1,
        ),
        (
            "HVI-ELBO(N = 1)",
            make_expressive_objective(1, 1),
            (0.0, 0.0),
            2,
        ),
        (
            "IWHVI(N = 5, K = 1)",
            make_expressive_objective(5, 1),
            (0.0, 0.0),
            3,
        ),
        (
            "IWHVI(N = 5, K = 5) (also called DIWHVI)",
            make_expressive_objective(5, 5),
            (0.0, 0.0),
            4,
        ),
    ]

    results = []
    for name, objective, init_params, seed_offset in specs:
        training = optimize_objective(
            objective,
            init_params,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed_value=seed_offset,
        )
        mean, var = estimate_objective_statistics(
            objective,
            training.params,
            n_samples=eval_samples,
            seed_value=100 + seed_offset,
        )
        results.append(
            ObjectiveResult(
                name=name,
                params=training.params,
                mean=mean,
                variance=var,
            )
        )

    return results
