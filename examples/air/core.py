from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.tree_util as jtu
from jax.example_libraries import optimizers
import numpy as np

from genjax import (
    flip,
    flip_enum,
    flip_mvd,
    flip_reinforce,
    gen,
    multivariate_normal_diag_reparam,
)
from genjax.core import distribution
from genjax.adev import expectation
from genjax.pjax import seed

EstimatorName = Literal["enum", "mvd", "reinforce", "hybrid"]
VALID_ESTIMATORS: tuple[EstimatorName, ...] = (
    "enum",
    "mvd",
    "reinforce",
    "hybrid",
)

# Wrapping ADEV discrete primitives as GenJAX distributions enables
# addressable `@ "site"` usage inside `@gen` programs.
flip_enum_dist = distribution(flip_enum, flip.logpdf)
flip_mvd_dist = distribution(flip_mvd, flip.logpdf)


@dataclass(frozen=True)
class AIRConfig:
    """Configuration for AIR model/guide architecture.

    Defaults match the original PLDI'24 AIR setup: three object slots,
    50x50 canvas, 20x20 attention window, 50-dim object code, and 256 hidden
    units in the guide LSTM.
    """

    num_steps: int = 3
    canvas_size: int = 50
    window_size: int = 20
    z_what_size: int = 50
    hidden_size: int = 256
    decoder_hidden_size: int = 200
    encoder_hidden_size: int = 200
    obs_noise_scale: float = 0.3
    z_where_prior_loc: tuple[float, float, float] = (3.0, 0.0, 0.0)
    z_where_prior_scale: tuple[float, float, float] = (0.2, 1.0, 1.0)
    z_pres_prior: tuple[float, ...] = (0.05, 0.05**2.3, 0.05**5)
    eps: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.canvas_size <= 0 or self.window_size <= 0:
            raise ValueError("canvas_size and window_size must be positive")
        if self.z_what_size <= 0:
            raise ValueError("z_what_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if len(self.z_where_prior_loc) != 3 or len(self.z_where_prior_scale) != 3:
            raise ValueError("z_where prior loc/scale must have length 3")
        if len(self.z_pres_prior) != self.num_steps:
            raise ValueError(
                "z_pres_prior length must match num_steps "
                f"(got {len(self.z_pres_prior)} vs {self.num_steps})"
            )


DEFAULT_CONFIG = AIRConfig()
SMALL_CONFIG = AIRConfig(
    canvas_size=24,
    window_size=12,
    z_what_size=16,
    hidden_size=64,
    decoder_hidden_size=64,
    encoder_hidden_size=64,
)


class LinearParams(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray


class DecoderParams(NamedTuple):
    dense_1: LinearParams
    dense_2: LinearParams


class EncoderParams(NamedTuple):
    dense_1: LinearParams
    dense_2: LinearParams


class PredictParams(NamedTuple):
    dense: LinearParams


class LSTMParams(NamedTuple):
    w_ih: jnp.ndarray
    w_hh: jnp.ndarray
    b_ih: jnp.ndarray
    b_hh: jnp.ndarray


class AIRParams(NamedTuple):
    decoder: DecoderParams
    rnn: LSTMParams
    encoder: EncoderParams
    predict: PredictParams


@dataclass
class AIRDataset:
    observations: jnp.ndarray
    true_counts: jnp.ndarray


@dataclass
class AIRTrainingResult:
    params: AIRParams
    loss_history: jnp.ndarray
    accuracy_history: jnp.ndarray
    epoch_times: jnp.ndarray


@dataclass
class AIRSuiteResult:
    estimator: EstimatorName
    num_particles: int
    final_loss: float
    final_accuracy: float
    objective_mean: float
    objective_variance: float
    params: AIRParams


def _as_float32(x) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _init_weight(key: jax.Array, out_dim: int, in_dim: int) -> jnp.ndarray:
    limit = np.sqrt(6.0 / float(in_dim + out_dim))
    return jax.random.uniform(
        key,
        shape=(out_dim, in_dim),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> LinearParams:
    w_key, _ = jax.random.split(key)
    weight = _init_weight(w_key, out_dim, in_dim)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return LinearParams(weight=weight, bias=bias)


def _linear(layer, x):
    return jnp.matmul(x, layer.weight.T) + layer.bias


def _init_lstm(key: jax.Array, in_dim: int, hidden_dim: int) -> LSTMParams:
    k1, k2 = jax.random.split(key)
    w_ih = _init_weight(k1, 4 * hidden_dim, in_dim)
    w_hh = _init_weight(k2, 4 * hidden_dim, hidden_dim)
    b_ih = jnp.zeros((4 * hidden_dim,), dtype=jnp.float32)
    b_hh = jnp.zeros((4 * hidden_dim,), dtype=jnp.float32)
    return LSTMParams(w_ih=w_ih, w_hh=w_hh, b_ih=b_ih, b_hh=b_hh)


def init_air_params(
    key: jax.Array,
    *,
    config: AIRConfig = DEFAULT_CONFIG,
) -> AIRParams:
    """Initialize AIR parameters with Glorot-style linear layers."""

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    decoder = DecoderParams(
        dense_1=_init_linear(k1, config.z_what_size, config.decoder_hidden_size),
        dense_2=_init_linear(k2, config.decoder_hidden_size, config.window_size**2),
    )

    rnn_input_size = (
        config.canvas_size**2 + 3 + config.z_what_size + 1
    )  # obs + z_where + z_what + z_pres
    rnn = _init_lstm(k3, rnn_input_size, config.hidden_size)

    encoder = EncoderParams(
        dense_1=_init_linear(k4, config.window_size**2, config.encoder_hidden_size),
        dense_2=_init_linear(k5, config.encoder_hidden_size, 2 * config.z_what_size),
    )

    predict = PredictParams(
        dense=_init_linear(
            jax.random.fold_in(key, 999),
            config.hidden_size,
            1 + 3 + 3,
        )
    )

    return AIRParams(decoder=decoder, rnn=rnn, encoder=encoder, predict=predict)


def decoder_apply(params, z_what):
    v = _linear(params.dense_1, z_what)
    v = jax.nn.leaky_relu(v)
    v = _linear(params.dense_2, v)
    return jax.nn.sigmoid(v)


def encoder_apply(params, x_att):
    v = _linear(params.dense_1, x_att)
    v = jax.nn.leaky_relu(v)
    v = _linear(params.dense_2, v)
    z_what_size = v.shape[0] // 2
    loc = v[:z_what_size]
    scale = jax.nn.softplus(v[z_what_size:]) + 1e-4
    return loc, scale


def predict_apply(params, h):
    a = _linear(params.dense, h)
    z_pres_p = jax.nn.sigmoid(a[0:1])
    z_where_loc = a[1:4]
    z_where_scale = jax.nn.softplus(a[4:]) + 1e-4
    return z_pres_p, z_where_loc, z_where_scale


def lstm_cell_apply(params, x, h, c):
    gates = (
        jnp.matmul(x, params.w_ih.T)
        + params.b_ih
        + jnp.matmul(h, params.w_hh.T)
        + params.b_hh
    )
    i, f, g, o = jnp.split(gates, 4)
    i = jax.nn.sigmoid(i)
    f = jax.nn.sigmoid(f)
    g = jnp.tanh(g)
    o = jax.nn.sigmoid(o)
    new_c = f * c + i * g
    new_h = o * jnp.tanh(new_c)
    return new_h, new_c


def affine_grid_generator(height: int, width: int, theta):
    """Generate a normalized affine sampling grid.

    Args:
        height: output grid height
        width: output grid width
        theta: [batch, 2, 3] affine matrices

    Returns:
        [batch, 2, height, width] grid in [-1, 1]
    """

    num_batch = theta.shape[0]

    x = jnp.linspace(-1.0, 1.0, width)
    y = jnp.linspace(-1.0, 1.0, height)
    x_t, y_t = jnp.meshgrid(x, y)

    x_t_flat = jnp.reshape(x_t, (-1,))
    y_t_flat = jnp.reshape(y_t, (-1,))

    ones = jnp.ones_like(x_t_flat)
    sampling_grid = jnp.stack([x_t_flat, y_t_flat, ones], axis=0)

    sampling_grid = jnp.expand_dims(sampling_grid, axis=0)
    sampling_grid = jnp.tile(sampling_grid, (num_batch, 1, 1))

    batch_grids = jnp.matmul(theta, sampling_grid)
    return jnp.reshape(batch_grids, (num_batch, 2, height, width))


def gather_nd_unbatched(params, indices):
    return params[tuple(jnp.moveaxis(indices, -1, 0))]


def gather_nd(params, indices, *, batch: bool = False):
    if not batch:
        return gather_nd_unbatched(params, indices)
    return jax.vmap(gather_nd_unbatched, in_axes=(0, 0))(params, indices)


def get_pixel_value(img, x, y):
    # Indices are non-differentiable in this STN implementation.
    x = jax.lax.stop_gradient(x).astype(jnp.int32)
    y = jax.lax.stop_gradient(y).astype(jnp.int32)

    batch_size, height, width = x.shape

    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)
    batch_idx = jnp.reshape(batch_idx, (batch_size, 1, 1))
    b = jnp.tile(batch_idx, (1, height, width))

    return img[b, y, x]


def bilinear_sampler(img, x, y):
    """Bilinear sampling for STN-style object/image transforms."""

    h = jnp.shape(img)[1]
    w = jnp.shape(img)[2]
    max_y = h - 1
    max_x = w - 1
    zero = jnp.zeros([], dtype=jnp.int32)

    x = 0.5 * ((x + 1.0) * max_x - 1.0)
    y = 0.5 * ((y + 1.0) * max_y - 1.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1

    x0 = jnp.clip(x0, zero, max_x)
    x1 = jnp.clip(x1, zero, max_x)
    y0 = jnp.clip(y0, zero, max_y)
    y1 = jnp.clip(y1, zero, max_y)

    ia = get_pixel_value(img, x0, y0)
    ib = get_pixel_value(img, x0, y1)
    ic = get_pixel_value(img, x1, y0)
    id_ = get_pixel_value(img, x1, y1)

    x0f = x0.astype(jnp.float32)
    x1f = x1.astype(jnp.float32)
    y0f = y0.astype(jnp.float32)
    y1f = y1.astype(jnp.float32)

    wa = (x1f - x) * (y1f - y)
    wb = (x1f - x) * (y - y0f)
    wc = (x - x0f) * (y1f - y)
    wd = (x - x0f) * (y - y0f)

    wa = jnp.expand_dims(wa, axis=3)
    wb = jnp.expand_dims(wb, axis=3)
    wc = jnp.expand_dims(wc, axis=3)
    wd = jnp.expand_dims(wd, axis=3)

    return wa * ia + wb * ib + wc * ic + wd * id_


def expand_z_where(z_where):
    """Map [s, x, y] -> [[s, 0, x], [0, s, y]] for STN transforms."""

    expansion_indices = jnp.array([1, 0, 2, 0, 1, 3])
    z_where = jnp.expand_dims(z_where, axis=0)
    out = jnp.concatenate((jnp.zeros((1, 1), dtype=z_where.dtype), z_where), axis=1)
    return jnp.reshape(out[:, expansion_indices], (1, 2, 3))


def z_where_inv(z_where):
    out = jnp.array([1.0, -z_where[1], -z_where[2]], dtype=jnp.float32)
    return out / z_where[0]


def object_to_image(config: AIRConfig, z_where, obj):
    theta = expand_z_where(z_where)
    grid = affine_grid_generator(config.canvas_size, config.canvas_size, theta)
    x_s = grid[:, 0, :, :]
    y_s = grid[:, 1, :, :]

    out = bilinear_sampler(
        jnp.reshape(obj, (1, config.window_size, config.window_size, 1)),
        x_s,
        y_s,
    )
    return jnp.reshape(out, (config.canvas_size, config.canvas_size))


def image_to_object(config: AIRConfig, z_where, image):
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid_generator(config.window_size, config.window_size, theta_inv)
    x_s = grid[:, 0, :, :]
    y_s = grid[:, 1, :, :]

    out = bilinear_sampler(
        jnp.reshape(image, (1, config.canvas_size, config.canvas_size, 1)),
        x_s,
        y_s,
    )
    return jnp.reshape(out, (config.window_size**2,))


def make_air_model(*, config: AIRConfig = DEFAULT_CONFIG):
    z_where_prior_loc = _as_float32(config.z_where_prior_loc)
    z_where_prior_scale = _as_float32(config.z_where_prior_scale)

    z_what_prior_loc = jnp.zeros((config.z_what_size,), dtype=jnp.float32)
    z_what_prior_scale = jnp.ones((config.z_what_size,), dtype=jnp.float32)

    z_pres_prior = _as_float32(config.z_pres_prior)
    obs_scale = config.obs_noise_scale * jnp.ones(
        (config.canvas_size, config.canvas_size),
        dtype=jnp.float32,
    )

    @gen
    def model(decoder_params):
        x = jnp.zeros((config.canvas_size, config.canvas_size), dtype=jnp.float32)

        for t in range(config.num_steps):
            z_pres = flip_reinforce(z_pres_prior[t]) @ f"z_pres_{t}"
            z_pres_scalar = jnp.where(
                z_pres,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            )

            z_where = (
                multivariate_normal_diag_reparam(z_where_prior_loc, z_where_prior_scale)
                @ f"z_where_{t}"
            )
            z_what = (
                multivariate_normal_diag_reparam(z_what_prior_loc, z_what_prior_scale)
                @ f"z_what_{t}"
            )

            y_att = decoder_apply(decoder_params, z_what)
            y = object_to_image(config, z_where, y_att)
            x = x + (y * z_pres_scalar)

        multivariate_normal_diag_reparam(x, obs_scale) @ "obs"
        return x

    return model


def make_air_guide(
    estimator: EstimatorName,
    *,
    config: AIRConfig = DEFAULT_CONFIG,
):
    if estimator not in VALID_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}")

    @gen
    def guide(obs, params):
        h = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
        c = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
        z_where = jnp.zeros((3,), dtype=jnp.float32)
        z_what = jnp.zeros((config.z_what_size,), dtype=jnp.float32)
        z_pres = jnp.ones((1,), dtype=jnp.float32)

        obs_flat = jnp.reshape(obs, (-1,))

        for t in range(config.num_steps):
            rnn_input = jnp.concatenate(
                [obs_flat, z_where, z_what, z_pres],
                axis=0,
            )
            h, c = lstm_cell_apply(params.rnn, rnn_input, h, c)

            z_pres_p, z_where_loc, z_where_scale = predict_apply(params.predict, h)
            z_pres_prob = (
                config.eps + (z_pres_p[0] * z_pres[0])
            ) / (1.0 + (1.01 * config.eps))

            if estimator == "enum":
                z_pres_site = flip_enum_dist(z_pres_prob) @ f"z_pres_{t}"
            elif estimator == "mvd":
                z_pres_site = flip_mvd_dist(z_pres_prob) @ f"z_pres_{t}"
            elif estimator == "reinforce":
                z_pres_site = flip_reinforce(z_pres_prob) @ f"z_pres_{t}"
            elif estimator == "hybrid":
                if t < config.num_steps - 1:
                    z_pres_site = flip_mvd_dist(z_pres_prob) @ f"z_pres_{t}"
                else:
                    z_pres_site = flip_enum_dist(z_pres_prob) @ f"z_pres_{t}"
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            z_pres = jnp.array(
                [
                    jnp.where(
                        z_pres_site,
                        jnp.array(1.0, dtype=jnp.float32),
                        jnp.array(0.0, dtype=jnp.float32),
                    )
                ],
                dtype=jnp.float32,
            )

            z_where = (
                multivariate_normal_diag_reparam(z_where_loc, z_where_scale)
                @ f"z_where_{t}"
            )

            x_att = image_to_object(config, z_where, obs)
            z_what_loc, z_what_scale = encoder_apply(params.encoder, x_att)
            z_what = (
                multivariate_normal_diag_reparam(z_what_loc, z_what_scale)
                @ f"z_what_{t}"
            )

    return guide


def make_air_objective(
    estimator: EstimatorName,
    *,
    config: AIRConfig = DEFAULT_CONFIG,
    num_particles: int = 1,
):
    if num_particles <= 0:
        raise ValueError("num_particles must be >= 1")

    model = make_air_model(config=config)
    guide = make_air_guide(estimator, config=config)

    log_num_particles = jnp.log(float(num_particles))

    def single_log_weight(obs, params):
        tr = guide.simulate(obs, params)

        merged_choices, _ = model.merge(
            {"obs": obs},
            tr.get_choices(),
        )
        model_logp, _ = model.assess(merged_choices, params.decoder)

        return jnp.sum(model_logp) + tr.get_score()

    @expectation
    def objective(obs, params):
        if num_particles == 1:
            return single_log_weight(obs, params)

        log_weights = []
        for _ in range(num_particles):
            log_weights.append(single_log_weight(obs, params))

        stacked = jnp.stack(log_weights)
        return jsp.logsumexp(stacked) - log_num_particles

    return objective, model, guide


def _compile_batched_loss_and_grad(objective):
    seeded_loss_and_grad = seed(
        lambda obs, params: (
            objective.prog.source.value(obs, params),
            objective.grad_estimate(obs, params)[1],
        )
    )
    return jax.jit(jax.vmap(seeded_loss_and_grad, in_axes=(0, 0, None)))


def _compile_batched_estimate(objective):
    seeded_estimate = seed(lambda obs, params: objective.prog.source.value(obs, params))
    return jax.jit(jax.vmap(seeded_estimate, in_axes=(0, 0, None)))


def sample_prior_dataset(
    decoder_params: DecoderParams,
    *,
    config: AIRConfig = DEFAULT_CONFIG,
    n_samples: int = 1024,
    seed_value: int = 0,
) -> AIRDataset:
    """Sample synthetic AIR observations from the model prior."""

    model = make_air_model(config=config)
    seeded_simulate = seed(model.simulate)

    keys = jax.random.split(jax.random.key(seed_value), n_samples)
    traces = jax.jit(jax.vmap(seeded_simulate, in_axes=(0, None)))(keys, decoder_params)

    choices = traces.get_choices()
    observations = jnp.reshape(
        choices["obs"],
        (n_samples, config.canvas_size, config.canvas_size),
    )

    true_counts = jnp.zeros((n_samples,), dtype=jnp.int32)
    for t in range(config.num_steps):
        true_counts = true_counts + choices[f"z_pres_{t}"].astype(jnp.int32)

    return AIRDataset(observations=observations, true_counts=true_counts)


def load_multi_mnist_npz(
    path: str | Path,
    *,
    max_examples: int | None = None,
) -> AIRDataset:
    """Load Multi-MNIST-style data saved as ``multi_mnist_uint8.npz``.

    The NPZ file should contain:
    - ``x``: uint8 images [N, 50, 50]
    - ``y``: object-array labels where each element is a sequence of digits
    """

    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        x = data["x"]
        y = data["y"]

    if max_examples is not None:
        x = x[:max_examples]
        y = y[:max_examples]

    observations = _as_float32(x) / 255.0
    true_counts = jnp.asarray([len(labels) for labels in y], dtype=jnp.int32)

    return AIRDataset(observations=observations, true_counts=true_counts)


def prepare_air_dataset(
    *,
    dataset: Literal["synthetic", "multi-mnist"] = "synthetic",
    config: AIRConfig = DEFAULT_CONFIG,
    n_samples: int = 1024,
    seed_value: int = 0,
    data_path: str | Path | None = None,
    decoder_params: DecoderParams | None = None,
) -> AIRDataset:
    """Prepare training data for the AIR case study."""

    if dataset == "multi-mnist":
        if data_path is None:
            raise ValueError("data_path is required when dataset='multi-mnist'")
        return load_multi_mnist_npz(data_path, max_examples=n_samples)

    if decoder_params is None:
        init_params = init_air_params(jax.random.key(seed_value), config=config)
        decoder_params = init_params.decoder

    return sample_prior_dataset(
        decoder_params,
        config=config,
        n_samples=n_samples,
        seed_value=seed_value,
    )


def estimate_count_accuracy(
    guide,
    params: AIRParams,
    observations: jnp.ndarray,
    true_counts: jnp.ndarray,
    *,
    config: AIRConfig = DEFAULT_CONFIG,
    seed_value: int = 0,
    batch_size: int = 1024,
) -> tuple[float, np.ndarray]:
    """Estimate object-count accuracy of the guide on labeled observations.

    Evaluation is batched to avoid compiling/running a single giant vmap over the
    full dataset, which can exhaust GPU memory for larger AIR datasets.
    """

    num_examples = observations.shape[0]
    if num_examples == 0:
        return 0.0, np.zeros((config.num_steps + 1, config.num_steps + 1), dtype=int)

    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    seeded_simulate = seed(guide.simulate)
    batched_simulate = jax.jit(jax.vmap(seeded_simulate, in_axes=(0, 0, None)))

    max_count = config.num_steps
    confusion = np.zeros((max_count + 1, max_count + 1), dtype=int)
    total_correct = 0

    for start in range(0, num_examples, batch_size):
        stop = min(start + batch_size, num_examples)
        obs_batch = observations[start:stop]
        true_batch = true_counts[start:stop]

        sub_key = jax.random.fold_in(jax.random.key(seed_value), start)
        keys = jax.random.split(sub_key, stop - start)
        traces = batched_simulate(keys, obs_batch, params)

        choices = traces.get_choices()
        inferred_counts = jnp.zeros((stop - start,), dtype=jnp.int32)
        for t in range(config.num_steps):
            inferred_counts = inferred_counts + choices[f"z_pres_{t}"].astype(jnp.int32)

        true_np = np.asarray(true_batch, dtype=int)
        inferred_np = np.asarray(inferred_counts, dtype=int)

        total_correct += int(np.sum(true_np == inferred_np))
        for true_c, infer_c in zip(true_np, inferred_np):
            if 0 <= true_c <= max_count and 0 <= infer_c <= max_count:
                confusion[true_c, infer_c] += 1

    accuracy = float(total_correct / num_examples)
    return accuracy, confusion


def train_air(
    observations: jnp.ndarray,
    true_counts: jnp.ndarray,
    *,
    estimator: EstimatorName,
    config: AIRConfig = DEFAULT_CONFIG,
    num_particles: int = 1,
    init_params: AIRParams | None = None,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    evaluate_accuracy_every: int = 1,
    eval_batch_size: int = 256,
    seed_value: int = 0,
) -> AIRTrainingResult:
    """Train AIR with a chosen discrete gradient estimator."""

    if estimator not in VALID_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}")

    if observations.ndim != 3:
        raise ValueError("observations must have shape [N, H, W]")
    if observations.shape[0] != true_counts.shape[0]:
        raise ValueError("observations and true_counts must have matching length")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be >= 1")

    if init_params is None:
        init_params = init_air_params(jax.random.key(seed_value), config=config)

    objective, _, guide = make_air_objective(
        estimator,
        config=config,
        num_particles=num_particles,
    )
    batch_loss_and_grad = _compile_batched_loss_and_grad(objective)

    effective_batch_size = min(batch_size, observations.shape[0])
    num_batches = observations.shape[0] // effective_batch_size
    if num_batches <= 0:
        raise ValueError(
            "Need at least one full batch; increase dataset size or lower batch_size"
        )

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(init_params)

    @jax.jit
    def update_step(
        step: jnp.int32,
        opt_state,
        keys: jnp.ndarray,
        batch_obs: jnp.ndarray,
    ):
        params = get_params(opt_state)
        batch_losses, batch_grads = batch_loss_and_grad(keys, batch_obs, params)
        mean_loss = jnp.mean(batch_losses)
        mean_grads = jtu.tree_map(lambda g: jnp.mean(g, axis=0), batch_grads)

        # The objective is maximized; Adam is a minimizer, so negate gradients.
        ascent_grads = jtu.tree_map(lambda g: -g, mean_grads)
        next_opt_state = opt_update(step, ascent_grads, opt_state)
        return next_opt_state, mean_loss

    key = jax.random.key(seed_value)
    global_step = 0

    loss_history = []
    accuracy_history = []
    epoch_times = []

    cumulative_time = 0.0
    for epoch in range(num_epochs):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, observations.shape[0])

        epoch_start = time.perf_counter()
        epoch_losses = []

        for batch_idx in range(num_batches):
            start = batch_idx * effective_batch_size
            stop = start + effective_batch_size
            idx = permutation[start:stop]
            batch_obs = observations[idx]

            key, sub_key = jax.random.split(key)
            keys = jax.random.split(sub_key, effective_batch_size)

            opt_state, batch_loss = update_step(
                jnp.asarray(global_step, dtype=jnp.int32),
                opt_state,
                keys,
                batch_obs,
            )
            global_step += 1
            epoch_losses.append(batch_loss)

        cumulative_time += time.perf_counter() - epoch_start
        epoch_times.append(cumulative_time)

        epoch_loss = jnp.mean(jnp.stack(epoch_losses))
        loss_history.append(epoch_loss)

        params = get_params(opt_state)
        if evaluate_accuracy_every > 0 and (
            epoch % evaluate_accuracy_every == 0 or epoch == num_epochs - 1
        ):
            accuracy, _ = estimate_count_accuracy(
                guide,
                params,
                observations,
                true_counts,
                config=config,
                seed_value=seed_value + 1000 + epoch,
                batch_size=eval_batch_size,
            )
            accuracy_history.append(accuracy)
        else:
            accuracy_history.append(float("nan"))

    final_params = get_params(opt_state)
    return AIRTrainingResult(
        params=final_params,
        loss_history=jnp.asarray(loss_history),
        accuracy_history=jnp.asarray(accuracy_history),
        epoch_times=jnp.asarray(epoch_times),
    )


def estimate_objective_statistics(
    objective,
    params: AIRParams,
    observations: jnp.ndarray,
    *,
    n_mc_samples: int = 8,
    seed_value: int = 0,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Estimate mean/variance of objective values over data and RNG draws.

    Evaluation is batched over observations to keep memory bounded on GPU.
    """

    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be >= 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    batch_estimate = _compile_batched_estimate(objective)

    key = jax.random.key(seed_value)
    means = []
    num_examples = observations.shape[0]

    for _ in range(n_mc_samples):
        key, sub_key = jax.random.split(key)

        total = 0.0
        for start in range(0, num_examples, batch_size):
            stop = min(start + batch_size, num_examples)
            obs_batch = observations[start:stop]

            batch_key = jax.random.fold_in(sub_key, start)
            keys = jax.random.split(batch_key, stop - start)
            values = batch_estimate(keys, obs_batch, params)
            total = total + jnp.sum(values)

        means.append(total / num_examples)

    sample_means = jnp.stack(means)
    return float(jnp.mean(sample_means)), float(jnp.var(sample_means))


def run_estimator_suite(
    observations: jnp.ndarray,
    true_counts: jnp.ndarray,
    *,
    estimators: Sequence[EstimatorName] = VALID_ESTIMATORS,
    config: AIRConfig = DEFAULT_CONFIG,
    num_particles: int = 1,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    eval_objective_samples: int = 8,
    eval_batch_size: int = 256,
    seed_value: int = 0,
) -> list[AIRSuiteResult]:
    """Train/evaluate multiple AIR estimators and summarize final metrics."""

    results: list[AIRSuiteResult] = []

    root_key = jax.random.key(seed_value)
    for idx, estimator in enumerate(estimators):
        if estimator not in VALID_ESTIMATORS:
            raise ValueError(f"Unknown estimator: {estimator}")

        init_key = jax.random.fold_in(root_key, idx)
        init_params = init_air_params(init_key, config=config)

        training = train_air(
            observations,
            true_counts,
            estimator=estimator,
            config=config,
            num_particles=num_particles,
            init_params=init_params,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            evaluate_accuracy_every=1,
            eval_batch_size=eval_batch_size,
            seed_value=seed_value + idx,
        )

        objective, _, guide = make_air_objective(
            estimator,
            config=config,
            num_particles=num_particles,
        )

        objective_mean, objective_var = estimate_objective_statistics(
            objective,
            training.params,
            observations,
            n_mc_samples=eval_objective_samples,
            seed_value=seed_value + 100 + idx,
            batch_size=eval_batch_size,
        )

        final_accuracy, _ = estimate_count_accuracy(
            guide,
            training.params,
            observations,
            true_counts,
            config=config,
            seed_value=seed_value + 200 + idx,
            batch_size=eval_batch_size,
        )

        results.append(
            AIRSuiteResult(
                estimator=estimator,
                num_particles=num_particles,
                final_loss=float(training.loss_history[-1]),
                final_accuracy=final_accuracy,
                objective_mean=objective_mean,
                objective_variance=objective_var,
                params=training.params,
            )
        )

    return results


def save_training_history_csv(result: AIRTrainingResult, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "epoch_wall_time_s"])
        for epoch, (loss, acc, wall_time) in enumerate(
            zip(result.loss_history, result.accuracy_history, result.epoch_times)
        ):
            writer.writerow(
                [
                    epoch,
                    float(loss),
                    float(acc),
                    float(wall_time),
                ]
            )


def save_suite_results_csv(results: Sequence[AIRSuiteResult], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "estimator",
                "num_particles",
                "final_loss",
                "final_accuracy",
                "objective_mean",
                "objective_variance",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.estimator,
                    result.num_particles,
                    result.final_loss,
                    result.final_accuracy,
                    result.objective_mean,
                    result.objective_variance,
                ]
            )
