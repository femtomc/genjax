import jax
import core
import numpy as np
import jax.random as jrand
from data import get_blinker_n
from examples.utils import timing
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({"font.size": 26})


def task(n: int):
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(1), core.GibbsSampler(get_blinker_n(n), 0.03), 250, 1
    )
    final_pred_post = run_summary.predictive_posterior_scores[-1]
    return final_pred_post


def timing_figure(
    ns=[10, 100, 200, 300, 400],
    color="skyblue",
):
    times = []
    for n in ns:
        _, (mean, _) = timing(
            lambda: task(n),
            repeats=1,
            inner_repeats=1,
        )
        times.append(mean)

    arr = np.array(times)
    # arr = arr / arr[0]

    fig = plt.figure(figsize=(12, 6))
    plt.bar(
        ns,
        arr,
        color="skyblue",
        edgecolor="black",
        width=50,
    )

    # Add labels and title
    plt.xlabel("Linear dimension of image (N)")
    plt.ylabel("Timing (s)")
    plt.grid(True, alpha=0.3)
    plt.xticks(ns)
    plt.xlim(min(ns) - 50, max(ns) + 100)
    plt.tight_layout()
    fig.savefig("examples/gol/figs/timing_scaling.pdf")


if __name__ == "__main__":
    with jax.default_device(jax.devices("gpu")[0]):
        timing_figure(color="orange")

    with jax.default_device(jax.devices("cpu")[0]):
        timing_figure(color="skyblue")
