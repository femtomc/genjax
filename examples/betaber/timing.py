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


if __name__ == "__main__":
    _, (mu, _) = genjax_timing(num_samples=1000)
    _, (mu_, _) = handcoded_timing(num_samples=1000)
    print(mu, mu_)
