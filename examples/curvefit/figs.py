import matplotlib.pyplot as plt
import numpy as np
import time
import jax.numpy as jnp
import jax.random as jrand
import jax

from core import (
    onepoint_curve,
    npoint_curve_factory,
    infer_latents,
    get_points_for_inference,
)
from jax import vmap


## Onepoint trace visualization ##
def visualize_onepoint_trace(trace, ylim=(-1.5, 1.5)):
    curve, pt = trace.get_retval()
    xvals = jnp.linspace(-1, 10, 300)
    fig = plt.figure(figsize=(2, 2))
    plt.plot(xvals, jax.vmap(curve)(xvals), color="black")
    color = "green"  # Point color for visualization
    plt.scatter(pt[0], pt[1], color=color, s=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.ylim(ylim)
    plt.tight_layout(pad=0.5)
    return fig


def save_onepoint_trace_viz():
    print("Making and saving onepoint trace visualization.")
    trace = onepoint_curve.simulate(0.0)
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
        color="green",  # Consistent point color for visualization
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
    npoint_curve = npoint_curve_factory(10)
    trace = npoint_curve.simulate()
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
    npoint_curve = npoint_curve_factory(10)
    traces = vmap(
        lambda: npoint_curve.simulate(),
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
        lambda chm: npoint_curve.log_density((), chm),
        in_axes=0,
    )(traces.get_choices())
    for i in range(4):
        density_val = jnp.asarray(densities[i]).item()
        fig = make_fig_with_centered_number(density_val)
        fig.savefig(
            f"examples/curvefit/figs/04{i}_batched_multipoint_trace_density.pdf"
        )


## Inference-related figures ##
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
