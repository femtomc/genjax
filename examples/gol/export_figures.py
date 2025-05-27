import time

import core
import jax.random as jrand
from dataloading import get_blinker_4x4, get_mit_logo, get_popl_logo
from jax import jit

from genjax import seed

blinker_small = get_blinker_4x4()


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


def save_logo_gibbs_figure():
    print("Running Gibbs sampler on MIT logo.")
    t = time.time()
    logo = get_mit_logo()
    run_summary = core.run_sampler_and_get_summary(
        jrand.key(1), core.GibbsSampler(logo, 0.03), 250, 1
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
    fig.savefig("examples/gol/figs/gibbs_on_logo.pdf")


if __name__ == "__main__":
    save_blinker_gibbs_figure()
    save_logo_gibbs_figure()
