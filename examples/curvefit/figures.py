from genjax import gen, flip, uniform, normal, get_choices
from genjax import modular_vmap as vmap
from genjax import seed
import jax.numpy as jnp
import jax.random as jrand

from tensorflow_probability.substrates import jax as tfp

from genjax import tfp_distribution
import matplotlib.pyplot as plt
import numpy as np
import time
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

### Save Plots ###


## Onepoint trace visualization ##
def visualize_onepoint_trace(trace, ylim=(-1.5, 1.5)):
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(-1, 10, 300)
    fig = plt.figure(figsize=(2, 2))
    plt.plot(xvals, jax.vmap(curve)(xvals), color="black")
    color = "red" if get_choices(trace)["y"]["is_out"] else "green"
    plt.scatter(pt[0], pt[1], color=color, s=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.ylim(ylim)
    plt.tight_layout(pad=0.5)
    return fig


def save_onepoint_trace_viz():
    print("Making and saving onepoint trace visualization.")
    trace = onepoint_curve.simulate((0.0,))
    fig = visualize_onepoint_trace(trace)
    fig.savefig("examples/curvefit/figs/010_onepoint_trace.pdf")


## Multipoint trace visualization ##
def visualize_multipoint_trace(
    trace,
    figsize=(4, 2),
    yrange=None,
    show_ticks=True,
    ax=None,
    min_and_max_x=(-1, 11),
):
    curve, (xs, ys) = trace.get_retval()
    xvals = jnp.linspace(*min_and_max_x, 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.plot(xvals, jax.vmap(curve)(xvals), color="black")
    ax.scatter(
        xs,
        ys,
        color=[
            "red" if get_choices(trace)["ys"]["is_out"][i] else "green"
            for i in range(len(xs))
        ],
        s=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if yrange is not None:
        ax.set_ylim(yrange)
    if fig is not None:
        fig.tight_layout(pad=0.5)
    return fig


def save_multipoint_trace_viz():
    print("Making and saving multipoint trace visualization.")
    trace = npoint_curve.simulate((10,))
    fig = visualize_multipoint_trace(trace, yrange=(-1.5, 1.5))
    fig.savefig("examples/curvefit/figs/020_multipoint_trace.pdf")


## 4 Multipoint trace visualization ##
def make_fig_with_centered_number(number):
    fig = plt.figure(figsize=(1.6, 0.8))
    plt.text(0.5, 0.5, f"{number:.4f}", fontsize=20, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig


def save_four_multipoint_trace_vizs():
    print("Making and saving visualizations of traces generated from vmap(simulate).")
    traces = vmap(
        lambda: npoint_curve.simulate((10,)),
        axis_size=4,
        in_axes=None,
    )()
    for i in range(4):
        trace = jax.tree.map(lambda x: x[i], traces)
        fig = visualize_multipoint_trace(
            trace, figsize=(1, 0.5), yrange=(-3, 3), show_ticks=False
        )
        fig.savefig(
            f"examples/curvefit/figs/03{i}_batched_multipoint_trace.pdf", pad_inches=0
        )

    print("Making and saving visualizations of trace densities.")
    densities = vmap(
        lambda chm: npoint_curve.log_density((10,), chm),
        in_axes=0,
    )(traces.get_choices())
    for i in range(4):
        fig = make_fig_with_centered_number(densities[i])
        fig.savefig(
            f"examples/curvefit/figs/04{i}_batched_multipoint_trace_density.pdf"
        )


## Inference-related figures ##
def get_points_for_inference():
    trace = npoint_curve.simulate((10,))
    return trace.get_retval()


def save_inference_viz(n_curves_to_plot=100):
    print("Making and saving inference visualization.")

    xvals = jnp.linspace(-1, 11, 300)
    _, (xs, ys) = get_points_for_inference()
    samples, weights = infer_latents(jrand.key(1), ys, int(10_000_000))
    order = jnp.argsort(weights, descending=True)
    samples, weights = jax.tree.map(
        lambda x: x[order[:n_curves_to_plot]], (samples, weights)
    )
    curves = [
        jax.tree.map(lambda x: x[i], samples.get_retval()[0])
        for i in range(n_curves_to_plot)
    ]
    alphas = jnp.sqrt(jax.nn.softmax(weights))
    fig = plt.figure(figsize=(2, 2))
    for i, curve in enumerate(curves):
        plt.plot(xvals, curve(xvals), color="blue", alpha=float(alphas[i]))
        plt.scatter(xs, ys, color="black", s=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig("examples/curvefit/figs/050_inference_viz.pdf")


## Inference time & quality scaling plots ##
def get_inference_scaling_data():
    n_samples = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    mean_lml_ests = []
    mean_times = []
    _, (xs, ys) = get_points_for_inference()
    for n in n_samples:
        times = []
        lml_ests = []
        infer_latents(jrand.key(1), ys, n)
        for _ in range(200):
            s = time.time()
            samples, weights = infer_latents(jrand.key(1), ys, n)
            jax.block_until_ready(weights)
            times.append(time.time() - s)
            lml_ests.append(jax.scipy.special.logsumexp(weights) - jnp.log(n))
        print(n, np.min(times), np.std(times))
        mean_times.append(np.min(times) * 1000)
        mean_lml_ests.append(np.mean(lml_ests))
    return n_samples, mean_lml_ests, mean_times


def save_inference_scaling_viz():
    n_samples, mean_lml_ests, mean_times = get_inference_scaling_data()

    gold_standard_lml_est = mean_lml_ests[-1]
    lml_est_errors = [gold_standard_lml_est - lml_est for lml_est in mean_lml_ests]

    n_samples, lml_est_errors, mean_times = (
        n_samples[:-1],
        lml_est_errors[:-1],
        mean_times[:-1],
    )

    ## Plot 1: Inference time scaling ##
    print("Making and saving inference time scaling visualization.")
    fig = plt.figure(figsize=(3, 1.5))
    plt.plot(n_samples, mean_times, marker="o", color="black")
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Inference\ntime (ms)")
    plt.ylim((0, 3))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig("examples/curvefit/figs/060_inference_time_scaling.pdf")

    ## Plot 2: Inference quality scaling ##
    print("Making and saving inference quality scaling visualization.")
    fig = plt.figure(figsize=(3, 2.5))
    plt.plot(mean_times, lml_est_errors, marker="o", color="black", label="IS")
    plt.xscale("log")
    plt.xlabel("Mean wall clock time (ms)")
    plt.ylabel("Error in est.\nof log P(obs)")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout(pad=0.2)
    fig.savefig("examples/curvefit/figs/061_inference_quality_scaling.pdf")


### Script to generate + save the figures ###
if __name__ == "__main__":
    save_onepoint_trace_viz()
    save_multipoint_trace_viz()
    save_four_multipoint_trace_vizs()
    save_inference_viz()
    save_inference_scaling_viz()
