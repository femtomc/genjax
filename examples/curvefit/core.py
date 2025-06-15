from genjax import gen, flip, uniform, normal
from genjax import modular_vmap as vmap
from genjax import seed
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
from genjax import Pytree
import jax

pi = jnp.pi
tfd = tfp.distributions

exponential = tfp_distribution(tfd.Exponential)


@Pytree.dataclass
class Lambda(Pytree):
    f: any = Pytree.static()
    dynamic_vals: jnp.ndarray
    static_vals: tuple = Pytree.static(default=())

    def __call__(self, *x):
        return self.f(*x, *self.static_vals, self.dynamic_vals)


### Model + inference code ###
@gen
def point(x, curve):
    y_det = curve(x)
    is_outlier = flip(0.08) @ "is_out"
    y_out = uniform(-3.0, 3.0) @ "y_out"
    y = jnp.where(is_outlier, y_out, y_det)
    y_observed = normal(y, 0.2) @ "obs"
    return y_observed


def sinfn(x, a):
    return jnp.sin(2.0 * pi * a[0] * x + a[1])


@gen
def sine():
    freq = exponential(10.0) @ "freq"
    offset = uniform(0.0, 2.0 * pi) @ "off"
    return Lambda(sinfn, jnp.array([freq, offset]))


@gen
def onepoint_curve(x):
    curve = sine() @ "curve"
    y = point(x, curve) @ "y"
    return curve, (x, y)


@gen
def npoint_curve(n):
    curve = sine() @ "curve"
    xs = jnp.arange(0, n)
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)


def _infer_latents(key, ys, n_samples):
    constraints = {"ys": {"obs": ys}}
    samples, weights = seed(
        vmap(
            lambda constraints: npoint_curve.generate((len(ys),), constraints),
            axis_size=n_samples,
            in_axes=None,
        )
    )(key, constraints)
    return samples, weights


infer_latents = jax.jit(_infer_latents, static_argnums=(2,))


def get_points_for_inference():
    trace = npoint_curve.simulate((10,))
    return trace.get_retval()
