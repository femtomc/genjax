import jax
import jax.numpy as jnp
import jax.random as jrand
import time
from genjax import seed
from genjax import modular_vmap as vmap
from genjax import beta, flip, gen


def timing(fn, repeats=200, inner_repeats=200, number=100):
    times = []
    for i in range(repeats):
        possible = []
        for i in range(inner_repeats):
            start_time = time.perf_counter()
            fn()
            interval = time.perf_counter() - start_time
            possible.append(interval)
        times.append(jnp.array(possible).min())
    times = jnp.array(times)
    return times, (jnp.mean(times), jnp.std(times))


@gen
def beta_ber():
    # define the hyperparameters that control the Beta prior
    alpha0 = jnp.array(10.0)
    beta0 = jnp.array(10.0)
    # sample f from the Beta prior
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip(f) @ "obs"


def genjax_timing(
    num_obs=50,
    repeats=50,
    num_samples=1000,
):
    data = {"obs": jnp.ones(num_obs)}

    def importance_(data):
        _, w = beta_ber.generate((), data)
        return w

    imp_jit = jax.jit(
        seed(
            vmap(
                importance_,
                axis_size=num_samples,
                in_axes=None,
            )
        ),
    )
    key = jrand.key(1)
    _ = imp_jit(key, data)
    _ = imp_jit(key, data)
    times, (time_mu, time_std) = timing(
        lambda: imp_jit(key, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def numpyro_timing(
    num_obs=50,
    repeats=200,
    num_samples=1000,
):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.handlers import block, replay, seed
    from numpyro.infer.util import (
        log_density,
    )

    key = jax.random.PRNGKey(314159)
    data = jnp.ones(num_obs)

    def model(data):
        # define the hyperparameters that control the Beta prior
        alpha0 = 10.0
        beta0 = 10.0
        # sample f from the Beta prior
        f = numpyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        # loop over the observed data
        with numpyro.plate("data", size=len(data)):
            # observe datapoint i using the Bernoulli
            # likelihood Bernoulli(f)
            numpyro.sample("obs", dist.Bernoulli(f), obs=data)

    guide = block(model, hide=["obs"])

    def importance(model, guide):
        def fn(key, *args, **kwargs):
            key, sub_key = jax.random.split(key)
            seeded_guide = seed(guide, sub_key)
            guide_log_density, guide_trace = log_density(
                seeded_guide,
                args,
                kwargs,
                {},
            )
            seeded_model = seed(model, key)
            replay_model = replay(seeded_model, guide_trace)
            model_log_density, model_trace = log_density(
                replay_model,
                args,
                kwargs,
                {},
            )
            return model_log_density - guide_log_density

        return fn

    vectorized_importance_weights = jax.jit(
        jax.vmap(importance(model, guide), in_axes=(0, None)),
    )

    sub_keys = jax.random.split(key, num_samples)

    # Run to warm up the JIT.
    _ = vectorized_importance_weights(sub_keys, data)
    _ = vectorized_importance_weights(sub_keys, data)

    times, (time_mu, time_std) = timing(
        lambda: vectorized_importance_weights(sub_keys, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def handcoded_timing(
    num_obs=50,
    repeats=50,
    num_samples=1000,
):
    data = jnp.ones(num_obs)

    def importance_(data):
        alpha0 = 10.0
        beta0 = 10.0
        f = beta.sample(alpha0, beta0)
        w = flip.logpdf(data, f)
        return jnp.sum(w)

    imp_jit = jax.jit(
        seed(
            vmap(
                importance_,
                axis_size=num_samples,
                in_axes=None,
            )
        ),
    )

    key = jrand.key(1)
    _ = imp_jit(key, data)
    _ = imp_jit(key, data)
    times, (time_mu, time_std) = timing(
        lambda: imp_jit(key, data).block_until_ready(),
        repeats=repeats,
    )
    return times, (time_mu, time_std)


def timing_comparison_fig(
    num_obs=50,
    repeats=200,
    num_samples=1000,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")

    gj_times, (gj_mu, gj_std) = genjax_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )
    np_times, (np_mu, np_std) = numpyro_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )
    hc_times, (hc_mu, hc_std) = handcoded_timing(
        repeats=repeats,
        num_obs=num_obs,
        num_samples=num_samples,
    )
    print(gj_mu, hc_mu, np_mu)

    fig, ax = plt.subplots(figsize=(5, 3), dpi=240)
    ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Counts")

    # GenJAX
    ax.hist(
        gj_times,
        bins=20,
        color="deepskyblue",
        label="GenJAX",
    )
    ax.axvline(
        x=gj_mu,
        color="black",
        linestyle="-",
        linewidth=2,
    )  # Outline
    ax.axvline(
        x=gj_mu,
        color="deepskyblue",
        linestyle="--",
        linewidth=1,
    )

    # NumPyro
    ax.hist(
        np_times,
        bins=20,
        color="coral",
        label="NumPyro",
    )
    ax.axvline(
        x=np_mu,
        color="black",
        linestyle="-",
        linewidth=2,
    )  # Outline
    ax.axvline(
        x=np_mu,
        color="coral",
        linestyle="--",
        linewidth=1,
    )

    # Handcoded
    ax.hist(
        hc_times,
        bins=20,
        color="gold",
        label="Handcoded",
    )
    ax.axvline(
        x=hc_mu,
        color="black",
        linestyle="-",
        linewidth=2,
    )  # Outline
    ax.axvline(
        x=hc_mu,
        color="gold",
        linestyle="--",
        linewidth=1,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.tight_layout()
    plt.savefig("examples/betaber/figs/comparison.pdf")


if __name__ == "__main__":
    timing_comparison_fig()
