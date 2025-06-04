import time

import core
import jax.numpy as jnp
import jax.random as jrand
from dataloading import get_blinker_4x4, get_mit_logo


blinker_small = get_blinker_4x4()


def timing(fn, repeats=200, inner_repeats=200):
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


def save_blinker_gibbs_figure():
    print("Running Gibbs sampler on blinker_small.")
    t = time.time()
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(1), core.GibbsSampler(blinker_small, 0.03), 250, 1
    )
    print(f"Gibbs run completed in {(time.time() - t):6f}s. Making figure.")
    fig = core.get_gol_sampler_lastframe_figure(
        blinker_small,
        run_summary,
        1,
    )
    fig.savefig("examples/gol/figs/gibbs_on_blinker.pdf")


def save_logo_gibbs_figure(chain_length=250):
    print("Running Gibbs sampler on MIT logo.")
    t = time.time()
    logo = get_mit_logo()
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(1), core.GibbsSampler(logo, 0.03), chain_length, 1
    )
    final_pred_post = run_summary.predictive_posterior_scores[-1]
    final_n_bit_flips = run_summary.n_incorrect_bits_in_reconstructed_image(logo)
    print(f"""
          Gibbs run completed in {(time.time() - t):6f}s.
          Final predictive posterior was {final_pred_post}.
          Final number of incorrect bits was {final_n_bit_flips}.
          Making figure.
          """)
    fig = core.get_gol_sampler_lastframe_figure(get_mit_logo(), run_summary, 1)
    fig.savefig(f"examples/gol/figs/gibbs_on_logo_{chain_length}.pdf")


if __name__ == "__main__":
    save_blinker_gibbs_figure()
    save_logo_gibbs_figure(chain_length=0)
    save_logo_gibbs_figure(chain_length=250)
