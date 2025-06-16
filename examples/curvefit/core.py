from genjax import gen, flip, uniform, normal
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


def npoint_curve_factory(n: int):
    """Factory function to create npoint_curve with static n parameter."""

    @gen
    def npoint_curve():
        curve = sine() @ "curve"
        xs = jnp.arange(0, n)  # n is now static from factory closure
        ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
        return curve, (xs, ys)

    return npoint_curve


def _infer_latents(key, ys, n_samples):
    """
    Infer latent curve parameters using genjax.smc.default_importance_sampling.

    Uses factory pattern for npoint_curve to handle static n parameter, and proper
    closure pattern from test_smc.py for default_importance_sampling with seed.
    """
    from genjax.smc import default_importance_sampling

    constraints = {"ys": {"obs": ys}}
    n_points = len(ys)

    # Create model with static n using factory pattern
    npoint_curve_model = npoint_curve_factory(n_points)

    # Create closure for default_importance_sampling that captures static arguments
    # This pattern follows test_smc.py lines 242-248
    def default_importance_sampling_closure(target_gf, target_args, constraints):
        return default_importance_sampling(
            target_gf,
            target_args,
            n_samples,  # n_samples captured as static
            constraints,
        )

    # Apply seed to the closure - pattern from test_smc.py lines 251-256
    result = seed(default_importance_sampling_closure)(
        key,
        npoint_curve_model,  # target generative function (from factory)
        (),  # target args (empty since n is captured in factory)
        constraints,  # constraints
    )

    # Extract samples (traces) and weights for compatibility
    return result.traces, result.log_weights


# For backward compatibility and JIT compilation
infer_latents = jax.jit(_infer_latents, static_argnums=(2,))


def get_points_for_inference():
    npoint_curve_model = npoint_curve_factory(10)
    trace = npoint_curve_model.simulate(())
    return trace.get_retval()
